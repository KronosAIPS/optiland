"""Even and odd aspheres for Non-Sequential Raytracing.

A conic base plus a radial polynomial, the surface kinds of KronosNSRT issue 30
(``docs/theory/07_geometry.md`` section 7.6, requirement R-07-7). All
operations in LOCAL coordinates; the vertex is the local origin and the axis is
local +z.

The sag::

    z(r) = c r^2 / (1 + sqrt(1 - (1 + K) c^2 r^2)) + P(r)

    even: P(r) = sum_i A_i r^(2 (i + 1))     (coefficients[0] multiplies r^2)
    odd:  P(r) = sum_i A_i r^(i + 1)         (coefficients[0] multiplies r^1)

The coefficient convention is the library's sequential ``EvenAsphere`` and
``OddAsphere`` (``optiland.geometries``), so a surface moves between the two
engines with the same numbers.

The intersection
----------------
1. **Three candidates per ray.**

   a. *The base conic's two roots*, from
      :meth:`ConicGeometry._quadratic_roots` (the stable root form of section
      7.4). A seed is usable where the quadratic was well posed, the root is
      finite, and the point lies on the sheet the sag function describes --
      the conic kind's own sheet test.
   b. *A first-crossing scan.* The ray's segment inside the region that
      holds every crossing inside the aperture -- the aperture cylinder cut
      by the slab ``z_lo <= z <= z_hi`` containing the sag over the aperture
      (:meth:`sag_range`) -- is sampled at ``scan_samples + 1`` points
      (default 33); the first sign change of ``f`` gives a bracket and a
      regula-falsi seed. It finds the roots the conic seeds cannot: a ray
      the base conic misses, and a nearer crossing of a strongly aspheric
      (non-monotone) surface where a conic seed converges to a farther one.
      Two crossings closer together than one sample interval are left to the
      conic seeds.

   A candidate without a seed is a miss with reason ``no_seed``.
2. **Newton on** ``f(t) = z(t) - sag(r(t))``, ``f'(t) = d_z - sigma (x d_x +
   y d_y)`` with ``sigma = (d sag / d r) / r``, run from each seed for a fixed
   number of masked iterations (``max_iterations``, default 16). No lane is
   read to the host and no loop exits early: a converged lane is frozen by
   ``where`` and rides along, so the iteration count is a constant a CUDA
   graph can record (the graph-replay contract of
   ``backends/graph_replay.py``).
3. **The residual at a base-conic seed is the polynomial alone.** The seed
   solves the base conic exactly in exact arithmetic, so the first residual is
   taken as ``-P(r)``; the conic part's rounding residual is not re-evaluated
   there. With every coefficient zero the first residual is exactly zero, the
   lane is converged before any step, and the kind returns the conic kind's
   numbers bit for bit (tested).
4. **Convergence:** ``|f| <= k_res ulp(max(|x|, |y|, |z|, 1)) |grad G|``,
   ``G = z - sag``: ``k_res`` (default 32) ulps of the working dtype at the
   point's coordinate scale, times the gradient norm that converts a position
   rounding into a residual (``docs/theory/08_precision.md`` section 8.7; the
   same 32-ulp rule as the NURBS prototype of the CAD study). A lane that
   converges takes one more Newton step (a polish: quadratic convergence
   takes it from the tolerance to the rounding floor) and freezes; a residual
   of exactly zero makes that step zero.
5. **The guard and the fallbacks** (R-07-7), in this order, every iteration:

   - *Domain test before any sag evaluation.* ``1 - (1 + K) c^2 r^2`` must
     exceed ``4 u`` (:func:`_tol.radicand_min`); an iterate outside is never
     square-rooted (the radicand is replaced by 1 on that lane before the
     root, so no NaN reaches either pass).
   - *Bracket.* Every evaluated iterate with ``f < 0`` or ``f > 0`` becomes
     the corresponding end of a bracket (the scan candidate starts with one).
     Once both ends exist, a Newton iterate strictly inside the bracket is
     taken whatever the slope, and one outside is replaced by the bracket's
     midpoint (bisection, one bit per step, unconditional).
   - *Tangent guard*, without a bracket: ``|f'| >= eta |grad G|`` with ``eta =
     1e-3`` by default. ``f'`` is ``cos(theta_i) |grad G|`` up to sign, so
     this is a test on the cosine of incidence; a step that fails it ends the
     lane as ``grazing``.
   - *Domain damping.* An iterate that left the domain is pulled halfway back
     to the last iterate inside it when no bracket exists (the domain is an
     interval along the ray, ``r^2(t)`` being convex, so a bracket's midpoint
     is always inside); a seed outside the domain ends as ``domain``.
   - *Hard cap.* A lane still unconverged after ``max_iterations`` steps is a
     miss with reason ``not_converged`` (``aperture`` when its last iterate
     lies outside the aperture): its last iterate is never returned.

6. **Acceptance** of a converged root: finite, ``t > eps`` (the caller's
   self-intersection threshold), inside the circular aperture, and -- when the
   refinement took at least one step -- the tangent guard holding at the root
   itself (a root found at ``cos(theta_i) < eta`` is a ``grazing`` miss, the
   status test T-07-6 asks for). The two conic candidates are picked exactly
   as the conic kind picks its roots; the scan candidate replaces the pick
   only where it is a distinct nearer root (nearer by more than the two
   roots' own uncertainties along the ray) or where the conic candidates found
   none.

Every miss carries its reason in :attr:`last_status` (see
:data:`MISS_REASONS`), a device array of the last call, read to the host only
by :meth:`miss_reason_counts`.

The adjoint
-----------
The iterations run under ``torch.no_grad()``. At the accepted root ``t*`` one
attached Newton step with the derivative detached is added in its zero-valued
form::

    t = t* - (f(t*; theta) - stopgrad f(t*; theta)) / stopgrad f'(t*)

Its value is ``t*`` to the bit; its derivative is the implicit-function
theorem's ``dt*/dtheta = -(df/dtheta) / f'`` for every attached input: the
curvature, the conic constant, the coefficients, and the ray origin and
direction (through which the placement reaches it). The normal is then
evaluated attached at ``p(t)``, so it carries both the moved point and the
explicit parameter dependence (``docs/theory/09_differentiation.md``, the
class of an implicit root; the same pattern as the NURBS prototype's attached
step, there measured to 8.8e-16 median against a closed form).
"""

from __future__ import annotations

import contextlib
import math

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential._utils import (
    as_float,
    as_param,
    is_tensor,
    resident_scalar,
)
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry

#: Per-ray status codes of :attr:`_AsphereGeometry.last_status`. A missed
#: ray's code is the largest of its three candidates' codes. The geometric
#: misses (no seed, behind the ray, outside the aperture) come first and the
#: refinement's own give-ups (domain, grazing, not converged) last, so a miss
#: that could have been a lost hit is always reported as one.
HIT = 0
NO_SEED = 1
BEHIND = 2
APERTURE = 3
DOMAIN = 4
GRAZING = 5
NOT_CONVERGED = 6

MISS_REASONS = {
    HIT: "hit",
    NO_SEED: "no_seed",
    BEHIND: "behind",
    APERTURE: "aperture",
    DOMAIN: "domain",
    GRAZING: "grazing",
    NOT_CONVERGED: "not_converged",
}

#: Fixed Newton iteration count (the hard cap of R-07-7). The theory's measured
#: sequence (section 7.6) needs 4 to 6 steps at cos(theta_i) >= 0.1 and 12 at
#: 1e-3 from the base-conic seed of its chapter-7 surface.
DEFAULT_MAX_ITERATIONS = 16

#: The tangent guard's threshold on the cosine of incidence (R-07-7 default).
DEFAULT_GUARD_ETA = 1e-3

#: Residual tolerance in ulps of the coordinate scale (times |grad G|).
DEFAULT_RESIDUAL_K = 32

#: Intervals of the first-crossing scan along the ray (module docstring, item
#: 1): two crossings closer together than the ray's segment in the scan region
#: over this count can escape the scan and are then left to the conic seeds.
DEFAULT_SCAN_SAMPLES = 32


def _no_grad():
    """``torch.no_grad()`` on the torch backend, a null context otherwise."""
    if be.get_backend() == "torch":
        import torch  # noqa: PLC0415

        return torch.no_grad()
    return contextlib.nullcontext()


def _requires_grad(value) -> bool:
    return is_tensor(value) and bool(value.requires_grad)


class _AsphereGeometry(AnalyticGeometry):
    """Conic base plus a radial polynomial; see the module docstring.

    Attributes:
        radius: Vertex radius of curvature [mm] (0 or inf: a flat base).
        conic: Conic constant K of the base.
        aperture_radius: Circular semi-aperture [mm].
        coefficients: The polynomial coefficients, a tuple of floats or
            tensors, or a 1-D tensor (the convention of the module docstring).
        max_iterations: Fixed Newton iteration count.
        guard_eta: Tangent-guard threshold on the cosine of incidence.
        residual_k: Residual tolerance in ulps of the coordinate scale.
        last_status: Per-ray status of the last :meth:`ray_intersect` call
            (codes in :data:`MISS_REASONS`), in the working backend and dtype.
        last_steps: Per-ray count of refinement steps the accepted (or, for a
            miss, the first) candidate took in the last call.
    """

    _odd = False

    def __init__(
        self,
        radius: float,
        conic: float,
        aperture_radius: float,
        coefficients=(),
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        guard_eta: float = DEFAULT_GUARD_ETA,
        residual_k: int = DEFAULT_RESIDUAL_K,
        scan_samples: int = DEFAULT_SCAN_SAMPLES,
    ) -> None:
        """Initialize the asphere.

        Args:
            radius: Vertex radius of curvature of the base conic [mm].
            conic: Conic constant K of the base.
            aperture_radius: Circular semi-aperture [mm].
            coefficients: Polynomial coefficients (see the module docstring for
                which power each multiplies). A sequence of floats or tensors,
                or a 1-D tensor; a tensor with ``requires_grad`` stays attached.
            max_iterations: Fixed Newton iteration count, at least 1.
            guard_eta: Tangent-guard threshold, in (0, 1).
            residual_k: Residual tolerance in ulps, at least 1.

        Raises:
            ValueError: If the base conic's sag is undefined anywhere inside
                the aperture (``(1 + K) c^2 a^2 >= 1``), or an option is out of
                range.
        """
        self.radius = as_param(radius)
        self.conic = as_param(conic)
        self.aperture_radius = as_param(aperture_radius)
        if is_tensor(coefficients):
            if coefficients.ndim != 1:
                raise ValueError("Asphere coefficients must be a 1-D sequence.")
            self.coefficients = coefficients
        else:
            self.coefficients = tuple(as_param(a) for a in coefficients)
        if int(max_iterations) < 1:
            raise ValueError("max_iterations must be at least 1.")
        if not 0.0 < float(guard_eta) < 1.0:
            raise ValueError("guard_eta must lie in (0, 1).")
        if int(residual_k) < 1:
            raise ValueError("residual_k must be at least 1.")
        if int(scan_samples) < 1:
            raise ValueError("scan_samples must be at least 1.")
        self.max_iterations = int(max_iterations)
        self.guard_eta = float(guard_eta)
        self.residual_k = int(residual_k)
        self.scan_samples = int(scan_samples)
        self._base = ConicGeometry(self.radius, self.conic, self.aperture_radius)
        self.last_status = None
        self.last_steps = None
        self._check_domain()

    # -- parameters ----------------------------------------------------------

    @property
    def num_coefficients(self) -> int:
        """Number of polynomial coefficients."""
        return int(self.coefficients.shape[0]) if is_tensor(self.coefficients) else len(
            self.coefficients
        )

    def coefficient_values(self) -> list[float]:
        """The coefficients as detached Python floats (host bookkeeping)."""
        if is_tensor(self.coefficients):
            return [float(v) for v in self.coefficients.detach().cpu().numpy()]
        return [as_float(a) for a in self.coefficients]

    def _sync_base(self) -> ConicGeometry:
        """The base conic, carrying this surface's current parameters."""
        base = self._base
        base.radius = self.radius
        base.conic = self.conic
        base.aperture_radius = self.aperture_radius
        return base

    def _curvature(self):
        return self._sync_base()._curvature()

    def _check_domain(self) -> None:
        c = as_float(self._curvature())
        kp = 1.0 + as_float(self.conic)
        a = as_float(self.aperture_radius)
        if kp * c * c * a * a >= 1.0:
            raise ValueError(
                "The base conic's sag is undefined inside the aperture: "
                f"(1 + K) c^2 a^2 = {kp * c * c * a * a:.6g} >= 1. Reduce the "
                "aperture radius below 1 / (c sqrt(1 + K))."
            )

    def _any_param_requires_grad(self) -> bool:
        if any(_requires_grad(v) for v in (self.radius, self.conic)):
            return True
        if is_tensor(self.coefficients):
            return _requires_grad(self.coefficients)
        return any(_requires_grad(a) for a in self.coefficients)

    # -- the surface ---------------------------------------------------------

    def _poly(self, r2):
        """``P`` and ``sigma_P = (dP/dr) / r`` at ``r^2 = r2`` (Horner)."""
        n = self.num_coefficients
        ones = be.ones_like(r2)
        if n == 0:
            zeros = be.zeros_like(r2)
            return zeros, zeros
        a = self.coefficients
        if not self._odd:
            # P = r2 (a0 + r2 (a1 + ...)); dP/dr = 2 r sum (i+1) a_i r2^i.
            p = a[n - 1] * ones
            d = (n * a[n - 1]) * ones
            for i in range(n - 2, -1, -1):
                p = a[i] + r2 * p
                d = (i + 1) * a[i] + r2 * d
            return r2 * p, 2.0 * d
        # Odd: P = r (a0 + r (a1 + ...)); dP/dr = sum (i+1) a_i r^i. The
        # radius is formed from a sanitized square so r = 0 carries no
        # infinite derivative into the backward pass.
        pos = r2 > 0.0
        r = be.where(pos, be.where(pos, r2, ones) ** 0.5, be.zeros_like(r2))
        p = a[n - 1] * ones
        d = (n * a[n - 1]) * ones
        for i in range(n - 2, -1, -1):
            p = a[i] + r * p
            d = (i + 1) * a[i] + r * d
        sigma = d / be.where(pos, r, ones)
        # At r = 0 the slope term sigma * x vanishes whatever sigma is; the
        # value is kept finite there.
        return r * p, sigma

    def sag(self, x, y):
        """Surface sag z(x, y) [mm] (the domain must hold; no clamping)."""
        r2 = x**2 + y**2
        c = self._curvature()
        under = 1.0 - (1.0 + self.conic) * c**2 * r2
        poly, _ = self._poly(r2)
        return c * r2 / (1.0 + under**0.5) + poly

    def _evaluate(self, x, y, z, dx, dy, dz, rmin):
        """Residual, derivative and gradient norm at the points (x, y, z).

        Returns:
            ``(f, fp, gnorm, poly, dom)``: ``f = z - sag``, ``fp = df/dt``,
            ``|grad G|``, the polynomial part of the sag, and the domain mask.
            Lanes outside the domain carry finite placeholder values.
        """
        r2 = x**2 + y**2
        c = self._curvature()
        under = 1.0 - (1.0 + self.conic) * c**2 * r2
        dom = under > rmin
        w = be.where(dom, under, be.ones_like(under)) ** 0.5
        poly, sigma_p = self._poly(r2)
        sag = c * r2 / (1.0 + w) + poly
        sigma = c / w + sigma_p
        f = z - sag
        fp = dz - sigma * (x * dx + y * dy)
        gnorm = (sigma * sigma * r2 + 1.0) ** 0.5
        return f, fp, gnorm, poly, dom

    def _normal_local(self, x, y):
        """Unnormalised normal ``(-s_x, -s_y, 1)``: the conic kind's, minus
        the polynomial slope (exactly the conic's when every coefficient is
        zero)."""
        n_conic = self._sync_base()._normal_local(x, y)
        _, sigma_p = self._poly(x**2 + y**2)
        # "+ 0.0" turns a signed zero into +0, and subtracting +0 is the
        # identity on every float, -0 included: with zero coefficients the
        # normal is the conic kind's to the bit.
        return n_conic - be.stack(
            [x * sigma_p + 0.0, y * sigma_p + 0.0, be.zeros_like(x)], axis=1
        )

    # -- the refinement ------------------------------------------------------

    def _code(self, code: int, like):
        return resident_scalar(self, "status", float(code), like)

    def _refine(self, o, d, t0, seed_ok, conic_seed, t_neg0=None, t_pos0=None, bracket0=None):
        """Guarded, masked Newton from ``t0`` (no gradient; see module doc).

        Args:
            o, d: Ray origins and directions, (M, 3).
            t0: Seeds, (M,).
            seed_ok: Lanes with a usable seed.
            conic_seed: Lanes whose seed is a root of the base conic (their
                first residual is the polynomial alone).
            t_neg0, t_pos0, bracket0: An initial bracket (``f < 0`` and
                ``f > 0`` ends) and the lanes that have one, or ``None``.

        Returns:
            ``(t, status, steps, fp, gnorm, tol)`` per lane: the final
            iterate, the status (``HIT`` for converged, else the reason), the
            number of steps taken, and ``f'``, ``|grad G|`` and the residual
            tolerance at the final evaluation.
        """
        ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        ones = be.ones_like(t0)
        zeros = be.zeros_like(t0)
        rmin = _tol.radicand_min(t0)
        eta = self.guard_eta
        none = ~(ones > 0.0)

        t = be.where(seed_ok, t0, zeros)
        active = seed_ok
        status = be.where(
            seed_ok, self._code(NOT_CONVERGED, t0), self._code(NO_SEED, t0)
        )
        steps = zeros
        if bracket0 is None:
            t_neg, t_pos, has_neg, has_pos = zeros, zeros, none, none
        else:
            t_neg, t_pos, has_neg, has_pos = t_neg0, t_pos0, bracket0, bracket0
        t_prev = t
        has_prev = none
        fp_last = ones
        gn_last = ones
        tol_last = ones

        for it in range(self.max_iterations + 1):
            x = ox + t * dx
            y = oy + t * dy
            z = oz + t * dz
            f, fp, gn, poly, dom = self._evaluate(x, y, z, dx, dy, dz, rmin)
            if it == 0:
                # A base-conic seed solves the base conic: its residual is -P
                # alone (module docstring, item 3).
                f = be.where(conic_seed, -poly, f)
            scale = be.maximum(
                be.maximum(be.abs(x), be.abs(y)), be.maximum(be.abs(z), ones)
            )
            tol = self.residual_k * _tol.ulp(scale) * gn
            conv = be.abs(f) <= tol
            conv = conv & (dom | conic_seed) if it == 0 else conv & dom
            fp_last = be.where(active, fp, fp_last)
            gn_last = be.where(active, gn, gn_last)
            tol_last = be.where(active, tol, tol_last)
            guard = be.abs(fp) >= eta * gn
            safe_fp = be.where(guard, fp, ones)
            t_newton = t - f / safe_fp
            done = active & conv
            # Polish: a lane that has just converged takes one more Newton
            # step (quadratic convergence puts it at the rounding floor), then
            # freezes. With a residual of exactly zero -- a base-conic seed
            # and zero coefficients -- the step is zero and t is unchanged.
            polish = done & guard & be.isfinite(t_newton)
            t = be.where(polish, t_newton, t)
            status = be.where(done, self._code(HIT, t0), status)
            active = active & ~conv
            if it == self.max_iterations:
                break

            # Bracket ends from every evaluated iterate inside the domain.
            neg = active & dom & (f < 0.0)
            pos = active & dom & (f > 0.0)
            t_neg = be.where(neg, t, t_neg)
            t_pos = be.where(pos, t, t_pos)
            has_neg = has_neg | neg
            has_pos = has_pos | pos
            bracket = has_neg & has_pos
            lo = be.minimum(t_neg, t_pos)
            hi = be.maximum(t_neg, t_pos)

            # Inside a bracket a Newton step is safe whatever the slope: it is
            # taken when it lands strictly inside, and bisection replaces it
            # otherwise. Without a bracket the tangent guard decides.
            inside = (t_newton > lo) & (t_newton < hi)
            newton_ok = (
                dom
                & be.isfinite(t_newton)
                & ((bracket & inside) | (~bracket & guard))
            )
            t_mid = 0.5 * (t_neg + t_pos)
            t_back = 0.5 * (t_prev + t)

            # Lanes that can neither step nor fall back end here.
            stuck = active & ~newton_ok & ~bracket & (dom | ~has_prev)
            status = be.where(stuck & dom, self._code(GRAZING, t0), status)
            status = be.where(stuck & ~dom, self._code(DOMAIN, t0), status)
            active = active & ~stuck

            t_new = be.where(newton_ok, t_newton, be.where(bracket, t_mid, t_back))
            t_prev = be.where(active & dom, t, t_prev)
            has_prev = has_prev | (active & dom)
            t = be.where(active, t_new, t)
            steps = steps + be.where(active, ones, zeros)

        return t, status, steps, fp_last, gn_last, tol_last

    def _scan_region(self) -> tuple[float, float, float]:
        """Detached ``(a, z_lo, z_hi)`` of the scan region, cached on the
        parameter values (host bookkeeping, like the conic's curvature)."""
        key = (
            as_float(self.radius),
            as_float(self.conic),
            as_float(self.aperture_radius),
            tuple(self.coefficient_values()),
        )
        cached = getattr(self, "_region_cache", None)
        if cached is None or cached[0] != key:
            z_lo, z_hi = self.sag_range()
            cached = (key, (key[2], z_lo, z_hi))
            self._region_cache = cached
        return cached[1]

    def _scan(self, o, d, eps):
        """Bracket the first crossing along the ray inside the scan region.

        The region is the aperture cylinder ``r <= a`` cut by the slab
        ``z_lo <= z <= z_hi`` that contains the sag over the aperture
        (:meth:`sag_range`). Every crossing of the surface inside the aperture
        lies in it, and the sag is defined everywhere in it. ``f`` is sampled
        at ``scan_samples + 1`` evenly spaced points of the ray's segment in
        the region (from ``eps`` on); the first sign change brackets the
        nearest crossing that the spacing resolves. A first sample already on
        the surface (``|f| <= tol``, a ray leaving it) is not a sign change.

        Returns:
            ``(t_seed, t_neg, t_pos, found)``: a regula-falsi seed inside the
            bracket, its two ends and the lanes that have one.
        """
        a, z_lo, z_hi = self._scan_region()
        ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        ones = be.ones_like(ox)
        zeros = be.zeros_like(ox)
        inf = ones * be.inf
        tiny = _tol.tiny_for(ox)

        # Slab.
        dz_ok = be.abs(dz) > tiny
        inv_dz = 1.0 / be.where(dz_ok, dz, ones)
        ta = (z_lo - oz) * inv_dz
        tb = (z_hi - oz) * inv_dz
        in_slab = (oz >= z_lo) & (oz <= z_hi)
        tz0 = be.where(dz_ok, be.minimum(ta, tb), be.where(in_slab, -inf, inf))
        tz1 = be.where(dz_ok, be.maximum(ta, tb), be.where(in_slab, inf, -inf))
        # Cylinder.
        qa = dx * dx + dy * dy
        qb = ox * dx + oy * dy
        qc = ox * ox + oy * oy - a * a
        disc = qb * qb - qa * qc
        qa_ok = qa > tiny
        root = be.where(disc > 0.0, disc, zeros) ** 0.5
        inv_qa = 1.0 / be.where(qa_ok, qa, ones)
        in_cyl = qc <= 0.0
        tc0 = be.where(qa_ok, (-qb - root) * inv_qa, be.where(in_cyl, -inf, inf))
        tc1 = be.where(qa_ok, (-qb + root) * inv_qa, be.where(in_cyl, inf, -inf))
        tc0 = be.where(qa_ok & (disc < 0.0), inf, tc0)
        t_start = be.maximum(be.maximum(tz0, tc0), eps + zeros)
        t_end = be.minimum(tz1, tc1)
        seg_ok = (t_end > t_start) & be.isfinite(t_start) & be.isfinite(t_end)
        t_start = be.where(seg_ok, t_start, zeros)
        span = be.where(seg_ok, t_end - t_start, zeros)

        m = self.scan_samples
        n = ox.shape[0]
        ts = be.concatenate([t_start + span * (j / m) for j in range(m + 1)], axis=0)
        o_rep = be.concatenate([o] * (m + 1), axis=0)
        d_rep = be.concatenate([d] * (m + 1), axis=0)
        x = o_rep[:, 0] + ts * d_rep[:, 0]
        y = o_rep[:, 1] + ts * d_rep[:, 1]
        z = o_rep[:, 2] + ts * d_rep[:, 2]
        rmin = _tol.radicand_min(ts)
        f, _, gn, _, dom = self._evaluate(
            x, y, z, d_rep[:, 0], d_rep[:, 1], d_rep[:, 2], rmin
        )
        scale = be.maximum(
            be.maximum(be.abs(x), be.abs(y)), be.maximum(be.abs(z), be.ones_like(x))
        )
        on_surface = be.abs(f) <= self.residual_k * _tol.ulp(scale) * gn
        positive = f > 0.0

        found = ~(ones > 0.0)
        t_neg = zeros
        t_pos = zeros
        f_neg = zeros
        f_pos = zeros
        # Descending, so the earliest sign change is the one that remains.
        for j in range(m - 1, -1, -1):
            s0 = slice(j * n, (j + 1) * n)
            s1 = slice((j + 1) * n, (j + 2) * n)
            change = seg_ok & dom[s0] & dom[s1] & (positive[s0] != positive[s1])
            if j == 0:
                change = change & ~on_surface[s0]
            neg_is_0 = ~positive[s0]
            t_neg = be.where(change, be.where(neg_is_0, ts[s0], ts[s1]), t_neg)
            t_pos = be.where(change, be.where(neg_is_0, ts[s1], ts[s0]), t_pos)
            f_neg = be.where(change, be.where(neg_is_0, f[s0], f[s1]), f_neg)
            f_pos = be.where(change, be.where(neg_is_0, f[s1], f[s0]), f_pos)
            found = found | change
        # Regula falsi inside the bracket (f_pos - f_neg > 0 where found).
        den = be.where(found, f_pos - f_neg, ones)
        t_seed = t_neg + (t_pos - t_neg) * (-f_neg) / den
        t_seed = be.where(found, t_seed, zeros)
        return t_seed, t_neg, t_pos, found

    # -- the public interface ------------------------------------------------

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray, eps: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the asphere (see the module docstring).

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3).
            eps: Minimum accepted distance (scalar or per ray), or ``None``.

        Returns:
            (t, normals, hit_mask, n_geom), the contract of
            :meth:`ComponentGeometry.ray_intersect`; n_geom points toward
            local +z like the conic kind's.
        """
        base = self._sync_base()
        if eps is None:
            eps = _tol.accept_t_min(be.abs(origins).max())
        n = origins.shape[0]

        with _no_grad():
            o = origins.detach() if is_tensor(origins) else origins
            d = directions.detach() if is_tensor(directions) else directions
            eps_det = eps.detach() if is_tensor(eps) else eps
            t1, t2, solvable1, solvable2 = base._quadratic_roots(o, d)
            c = base._curvature()
            seeds = []
            for t_seed, solvable in ((t1, solvable1), (t2, solvable2)):
                pz = o[:, 2] + t_seed * d[:, 2]
                on_sheet = (1.0 - (1.0 + self.conic) * c * pz) >= 0.0
                seeds.append(solvable & be.isfinite(t_seed) & on_sheet)
            t3, t_neg3, t_pos3, found3 = self._scan(o, d, eps_det)
            yes = be.ones_like(t1) > 0.0
            no = ~yes
            zeros = be.zeros_like(t1)
            t_r, status_r, steps_r, fp_r, gn_r, tol_r = self._refine(
                be.concatenate([o, o, o], axis=0),
                be.concatenate([d, d, d], axis=0),
                be.concatenate([t1, t2, t3], axis=0),
                be.concatenate([seeds[0], seeds[1], found3], axis=0),
                be.concatenate([yes, yes, no], axis=0),
                be.concatenate([zeros, zeros, t_neg3], axis=0),
                be.concatenate([zeros, zeros, t_pos3], axis=0),
                be.concatenate([no, no, found3], axis=0),
            )
            cands = []
            for sl in (slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)):
                t_c, st_c, steps_c = t_r[sl], status_r[sl], steps_r[sl]
                px = o[:, 0] + t_c * d[:, 0]
                py = o[:, 1] + t_c * d[:, 1]
                in_aperture = (px**2 + py**2) <= self.aperture_radius**2
                converged = st_c == 0.0
                took_steps = steps_c > 0.0
                guard_root = be.abs(fp_r[sl]) >= self.guard_eta * gn_r[sl]
                grazing = converged & took_steps & ~guard_root
                ahead = be.isfinite(t_c) & (t_c > eps_det)
                valid = converged & ~grazing & ahead & in_aperture
                st = be.where(grazing, self._code(GRAZING, t_c), st_c)
                st = be.where(
                    converged & ~grazing & ~ahead, self._code(BEHIND, t_c), st
                )
                st = be.where(
                    converged & ~grazing & ahead & ~in_aperture,
                    self._code(APERTURE, t_c),
                    st,
                )
                # A refinement that ran out of steps with its iterate outside
                # the aperture was chasing a root the aperture excludes.
                st = be.where(
                    (st_c == float(NOT_CONVERGED)) & ~in_aperture,
                    self._code(APERTURE, t_c),
                    st,
                )
                # The root's own uncertainty along the ray.
                t_unc = tol_r[sl] / be.where(
                    be.abs(fp_r[sl]) > 0.0, be.abs(fp_r[sl]), be.ones_like(t_c)
                )
                cands.append((t_c, valid, st, steps_c, fp_r[sl], t_unc))

            (ta, va, sa, na, fa, ua), (tb, vb, sb, nb, fb, ub), (ts, vs, ss, ns, fs, us) = cands
            # The base-conic candidates, exactly as the conic kind picks.
            pick1 = va & (~vb | (ta <= tb))
            pick2 = vb & ~pick1
            hit_c = pick1 | pick2
            t_c = be.where(pick1, ta, be.where(pick2, tb, zeros))
            u_c = be.where(pick1, ua, be.where(pick2, ub, zeros))
            # The scanned candidate replaces them only with a distinct, nearer
            # root (or where they found none): a root the conic seeds missed.
            pick3 = vs & (~hit_c | (ts < t_c - (u_c + us)))
            pick1 = pick1 & ~pick3
            pick2 = pick2 & ~pick3
            hit_mask = pick1 | pick2 | pick3
            t_star = be.where(
                pick3, ts, be.where(pick1, ta, be.where(pick2, tb, zeros))
            )
            ones = be.ones_like(fa)
            fp_star = be.where(
                pick3, fs, be.where(pick1, fa, be.where(pick2, fb, ones))
            )
            self.last_status = be.where(
                hit_mask,
                self._code(HIT, ta),
                be.maximum(be.maximum(sa, sb), ss),
            )
            self.last_steps = be.where(pick3, ns, be.where(pick2, nb, na))

        t = t_star
        if be.get_backend() == "torch" and self._needs_adjoint(origins, directions):
            t = self._attached_step(origins, directions, t_star, fp_star, hit_mask)

        inf_arr = be.ones_like(t) * be.inf
        t_out = be.where(hit_mask, t, inf_arr)

        # Normals from the finite hit points, as the conic kind does.
        px = origins[:, 0] + t * directions[:, 0]
        py = origins[:, 1] + t * directions[:, 1]
        px = be.where(hit_mask, px, be.zeros_like(px))
        py = be.where(hit_mask, py, be.zeros_like(py))
        n_raw = self._normal_local(px, py)
        n_len = (n_raw * n_raw).sum(axis=1, keepdims=True) ** 0.5
        n_geom = n_raw / (n_len + _tol.tiny_for(n_len))

        dot = (directions * n_geom).sum(axis=1, keepdims=True)
        normals = be.where(dot > 0, -n_geom, n_geom)
        return t_out, normals, hit_mask, n_geom

    def _needs_adjoint(self, origins, directions) -> bool:
        import torch  # noqa: PLC0415

        if not torch.is_grad_enabled():
            return False
        return (
            _requires_grad(origins)
            or _requires_grad(directions)
            or self._any_param_requires_grad()
        )

    def _attached_step(self, origins, directions, t_star, fp_star, hit_mask):
        """One attached Newton step at the root, zero-valued (module doc)."""
        ts = be.where(hit_mask, t_star, be.zeros_like(t_star))
        x = origins[:, 0] + ts * directions[:, 0]
        y = origins[:, 1] + ts * directions[:, 1]
        z = origins[:, 2] + ts * directions[:, 2]
        rmin = _tol.radicand_min(ts)
        f, _, _, _, _ = self._evaluate(
            x, y, z, directions[:, 0], directions[:, 1], directions[:, 2], rmin
        )
        den = be.where(hit_mask, fp_star, be.ones_like(fp_star))
        corr = (f - f.detach()) / den
        return ts - be.where(hit_mask, corr, be.zeros_like(corr))

    # -- diagnostics and bookkeeping -------------------------------------------

    def miss_reason_counts(self) -> dict[str, int]:
        """Counts of :attr:`last_status` by reason name (reads to the host)."""
        if self.last_status is None:
            return {}
        s = self.last_status
        arr = s.detach().cpu().numpy() if is_tensor(s) else np.asarray(s)
        codes, counts = np.unique(arr.astype(int), return_counts=True)
        return {MISS_REASONS[int(k)]: int(v) for k, v in zip(codes, counts)}

    def sag_range(self, num: int = 4097) -> tuple[float, float]:
        """Detached ``(z_min, z_max)`` of the sag over ``0 <= r <= a``.

        Sampled at ``num`` radii and widened by the largest slope times the
        sample spacing, which bounds the extremum between two samples.
        """
        a = as_float(self.aperture_radius)
        c = as_float(self._curvature())
        kp = 1.0 + as_float(self.conic)
        coeffs = self.coefficient_values()
        r = np.linspace(0.0, a, num)
        r2 = r * r
        w = np.sqrt(np.maximum(1.0 - kp * c * c * r2, 0.0))
        z = c * r2 / (1.0 + w)
        slope = np.abs(c * r / np.maximum(w, 1e-300))
        for i, ai in enumerate(coeffs):
            power = (i + 1) if self._odd else 2 * (i + 1)
            z = z + ai * r**power
            slope = slope + abs(ai) * power * r ** (power - 1)
        margin = float(np.max(slope)) * (a / (num - 1)) if num > 1 else 0.0
        return float(z.min()) - margin, float(z.max()) + margin

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB in global coordinates (host bookkeeping, detached)."""
        t_vec = np.array(transform[0], dtype=float)
        R = np.array(transform[1], dtype=float)
        r = as_float(self.aperture_radius)
        z_lo, z_hi = self.sag_range()
        z_min = min(0.0, z_lo)
        z_max = max(0.0, z_hi)
        corners_local = np.array(
            [[sx * r, sy * r, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (z_min, z_max)],
            dtype=float,
        )
        corners_global = corners_local @ R.T + t_vec
        return AABB(corners_global.min(axis=0), corners_global.max(axis=0))

    def rim_sag(self) -> float:
        """Detached sag at the aperture rim [mm]."""
        a = as_float(self.aperture_radius)
        c = as_float(self._curvature())
        kp = 1.0 + as_float(self.conic)
        z = c * a * a / (1.0 + math.sqrt(max(1.0 - kp * c * c * a * a, 0.0)))
        for i, ai in enumerate(self.coefficient_values()):
            power = (i + 1) if self._odd else 2 * (i + 1)
            z += ai * a**power
        return z

    def detached_copy(self):
        """A copy with every numeric parameter a plain float."""
        return type(self)(
            as_float(self.radius),
            as_float(self.conic),
            as_float(self.aperture_radius),
            self.coefficient_values(),
            max_iterations=self.max_iterations,
            guard_eta=self.guard_eta,
            residual_k=self.residual_k,
            scan_samples=self.scan_samples,
        )


class EvenAsphereGeometry(_AsphereGeometry):
    """Even asphere: conic base plus ``sum_i A_i r^(2 (i + 1))``.

    ``coefficients[0]`` multiplies ``r^2``, as in the sequential
    ``EvenAsphere``. See the module docstring for the intersection, the miss
    reasons and the adjoint.
    """

    _odd = False


class OddAsphereGeometry(_AsphereGeometry):
    """Odd asphere: conic base plus ``sum_i A_i r^(i + 1)``.

    ``coefficients[0]`` multiplies ``r^1``, as in the sequential
    ``OddAsphere``. A non-zero ``r^1`` term makes the vertex a cone point
    whose normal is taken as the axis there.
    """

    _odd = True
