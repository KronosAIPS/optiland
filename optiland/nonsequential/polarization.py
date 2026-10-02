"""Stokes polarization for the non-sequential engine: conventions and math.

The minimal polarization of the research repository's issue 5: every ray keeps
its scalar flux ``w`` as the Stokes ``I`` and, when polarization is on, carries
the *reduced* Stokes state ``(q, u, v) = (Q, U, V) / I`` and a unit reference
axis ``e`` perpendicular to its direction ``k``. This module holds the
conventions and the arithmetic; it touches no ray bundle and no component.

Conventions (stated here once, at the API boundary; R-06-1)
-----------------------------------------------------------

* **Basis.** ``e`` is the ``x = p`` axis of the ray's Stokes frame, ``s = k x e``
  its ``y`` axis, and ``(p, s, k)`` is right-handed (``p x s = k``). At an
  interface ``p`` lies in the plane of incidence and ``s = k x n / |k x n|``.
* **Stokes vector.** ``S = (|E_p|^2 + |E_s|^2, |E_p|^2 - |E_s|^2,
  2 Re(E_p E_s*), -2 Im(E_p E_s*))``: ``Q > 0`` is ``p``-dominant, ``U > 0`` is
  linear polarization at +45 degrees from ``p`` towards ``s``, and the sign of
  ``V`` is the one the quarter-wave retarder fixes: a fast axis at +45 degrees
  takes ``(1, 1, 0, 0)`` to ``(1, 0, 0, -1)``.
* **Time-harmonic convention.** Fields vary as ``exp(-i omega t)``, a complex
  index is ``n + i k`` with ``k >= 0`` for an absorbing medium, and an
  evanescent (totally reflected) wave takes the root with
  ``cos(theta_t) = +i kappa``. The Fresnel amplitudes are
  ``r_s = (n1 cos_i - n2 cos_t) / (n1 cos_i + n2 cos_t)`` and
  ``r_p = (n2 cos_i - n1 cos_t) / (n2 cos_i + n1 cos_t)``. This is the
  convention of the catalogue's analytic reference (its
  ``fresnel_amplitude_rs_rp``): beyond the critical angle at 1.5 -> 1.0 and
  45 degrees, ``arg(r_p r_s*) = -36.8699`` degrees. The research repository's
  theory chapter 06 states it in section 6.0 since 2026-10-01 (its issues 73
  and 78; section 6.1 printed the other sign until then), and the fork's
  thin-film module is in it too.
* **Frame rotation.** Turning the reference axis from ``e`` to ``a`` (both
  perpendicular to ``k``) by the angle ``psi`` measured about ``k`` applies
  ``M_rot(psi)`` of chapter 06 section 6.5: ``q' = cos2psi q + sin2psi u``,
  ``u' = -sin2psi q + cos2psi u``, ``v' = v``. The doubled angle is formed
  algebraically from ``e . a`` and ``(e x a) . k``, normalised by their sum
  of squares (:func:`rotation_2psi`); no trigonometric function is called.
* **Interface form.** Every element this module builds is, in its own frame,
  ``[[m00, m01, 0, 0], [m01, m00, 0, 0], [0, 0, m22, m23], [0, 0, -m23, m22]]``
  (:class:`InterfaceMueller`): a Fresnel reflection or transmission, a linear
  diattenuator or polarizer with its axis on ``p``, a linear retarder with its
  fast axis on ``p``. Applied to the reduced state it multiplies the flux by
  ``g = m00 + m01 q`` and returns the normalised output state.

Why the reduced form. ``I`` stays the engine's ``flux``, so the one ledger of
R-06-2 is literal: Beer-Lambert, roulette, split weights and detectors keep
acting on ``flux`` alone, and a trace with polarization off carries no extra
field and does no extra operation. With an unpolarized state (``q = 0``) the
flux factor ``g`` is ``m00`` exactly, the scalar coefficient, bit for bit.

The run record. A Stokes trace records ``polarization: "stokes"`` in
``SimulationResult.environment``; a scalar trace (``"off"``, the default)
carries no ``polarization`` key at all: absent means off. That is the
maintainer's ruling of 2026-09-27 on issue 5, for now. The key is to be
written for every trace, ``"off"`` included, in one change with the
record's statement of the scalar-equivalence conditions (chapter 06 section
6.9, test T-06-19): a scalar run's record then says which conditions its
numbers rely on, which a bare ``"off"`` would not.

Every function works on whatever array library the arguments are (NumPy or
torch, float32 or float64) through ``optiland.backend``, on real arrays only.
Guards follow the double-``where`` rule of the theory's chapter 09 (R-09-10):
the input of a division or a square root is masked, never its output alone.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol

#: The run-level polarization modes (R-06-9). ``"off"`` is the scalar engine
#: exactly as it was; ``"stokes"`` carries the reduced Stokes state.
MODES: tuple[str, ...] = ("off", "stokes")

#: The six optional per-ray fields of :class:`~optiland.nonsequential
#: .ray_bundle.NSQRayBundle` that hold the state when polarization is on.
POL_FIELDS: tuple[str, ...] = ("pol_q", "pol_u", "pol_v", "pol_ex", "pol_ey", "pol_ez")


#: The environment variable a backend built without an explicit
#: ``polarization`` reads (``off`` or ``stokes``; unset is ``off``), so a
#: harness that builds its own backends -- a catalogue runner, a notebook that
#: calls ``scene.trace()`` -- can be run in Stokes mode unchanged, as
#: ``OPTILAND_NSQ_COMPILE_STEP`` does for the compiled step.
POLARIZATION_ENV = "OPTILAND_NSQ_POLARIZATION"


def mode_for_backend(value: object) -> str:
    """A backend's polarization mode: its explicit argument, or the environment's.

    Args:
        value: The backend's ``polarization`` argument; ``None`` reads
            :data:`POLARIZATION_ENV`.

    Returns:
        ``"off"`` or ``"stokes"``.
    """
    if value is None:
        import os  # noqa: PLC0415

        env = os.environ.get(POLARIZATION_ENV, "").strip()
        return normalise_mode(env) if env else "off"
    return normalise_mode(value)


def normalise_mode(value: object) -> str:
    """The polarization mode as one of :data:`MODES`.

    Args:
        value: ``"off"`` or ``"stokes"``; ``False``/``None`` mean ``"off"`` and
            ``True`` means ``"stokes"``.

    Returns:
        ``"off"`` or ``"stokes"``.

    Raises:
        ValueError: For anything else.
    """
    if value is None or value is False:
        return "off"
    if value is True:
        return "stokes"
    if isinstance(value, str) and value.strip().lower() in MODES:
        return value.strip().lower()
    raise ValueError(f"polarization must be one of {MODES} (or a bool), got {value!r}")


class InterfaceMueller(NamedTuple):
    """The four distinct elements of an interface-form Mueller matrix.

    ``M = [[m00, m01, 0, 0], [m01, m00, 0, 0], [0, 0, m22, m23],
    [0, 0, -m23, m22]]`` in the element's own frame (``p`` on the plane of
    incidence, the polarizer's transmission axis or the retarder's fast axis).
    """

    m00: object
    m01: object
    m22: object
    m23: object


# ---------------------------------------------------------------------------
# Small vector helpers on component arrays
# ---------------------------------------------------------------------------


def _dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def _masked_sqrt(x):
    """``sqrt(x)`` for ``x > 0``; zero, with a finite derivative, where ``x <= 0``."""
    positive = x > 0
    root = be.where(positive, x, be.ones_like(x)) ** 0.5
    return be.where(positive, root, be.zeros_like(x))


def degeneracy_tolerance(like):
    """The ``|k x n|`` below which a plane of incidence is not defined (R-06-4).

    ``sqrt(eps)`` of the working dtype (``eps = 2 u``): 1.49e-8 at float64,
    3.45e-4 at float32. Below it the two interface eigen-polarizations are
    degenerate to within the dtype's own resolution of ``R_s - R_p`` (which
    goes as ``sin^2 theta``), so any frame is right and the rotation is
    skipped rather than a vanishing vector normalised.

    Args:
        like: An array or tensor in the working dtype.

    Returns:
        The tolerance as a 0-d array or tensor in ``like``'s dtype.
    """
    return (2.0 * _tol.unit_roundoff(like)) ** 0.5


# ---------------------------------------------------------------------------
# Frame rotation
# ---------------------------------------------------------------------------


def rotation_2psi(e, a, k):
    """``(cos 2psi, sin 2psi)`` of the rotation taking reference axis ``e`` to ``a``.

    ``psi`` is measured about ``k``: ``c = e . a`` and ``s = (e x a) . k``
    are ``|e| |a|`` times ``cos psi`` and ``sin psi``; then
    ``cos 2psi = (c^2 - s^2) / (c^2 + s^2)`` and
    ``sin 2psi = 2 c s / (c^2 + s^2)``. All three vectors are given as
    ``(x, y, z)`` tuples of per-ray component arrays; ``e`` and ``a`` are
    unit vectors perpendicular to the unit ``k``.

    Why the division by ``c^2 + s^2 = |e|^2 |a|^2``: the two axes are unit
    vectors only to the rounding of their normalisation, and without it the
    doubled-angle pair has length ``|e|^2 |a|^2`` instead of one, so the
    rotation scales ``(q, u)`` by up to ``1 + 4 u`` and can take a pure
    state past a degree of polarization of one (R-06-7). With it the pair is
    a rotation to its own rounding, and a rotation between two axes that are
    the same line (``s = 0``) is exactly ``(1, 0)``: ``c^2 / c^2``. Where
    ``c^2 + s^2`` vanishes (a zero axis, which no caller passes) the pair is
    ``(1, 0)``, the identity, behind a double ``where``.

    Returns:
        ``(c2, s2)``, per-ray arrays.
    """
    c = _dot(*e, *a)
    s = _dot(*_cross(*e, *a), *k)
    cc, ss = c * c, s * s
    den = cc + ss
    nonzero = den > 0
    safe = be.where(nonzero, den, be.ones_like(den))
    c2 = be.where(nonzero, (cc - ss) / safe, be.ones_like(den))
    s2 = be.where(nonzero, (2.0 * c * s) / safe, be.zeros_like(den))
    return c2, s2


def rotate(q, u, c2, s2):
    """Apply ``M_rot(psi)`` to the reduced state; ``(q', u')``, ``v`` unchanged."""
    return c2 * q + s2 * u, c2 * u - s2 * q


# ---------------------------------------------------------------------------
# Applying an interface-form element
# ---------------------------------------------------------------------------


def apply_interface(q, u, v, m: InterfaceMueller):
    """Apply an interface-form Mueller matrix to the reduced state.

    ``S' = M S`` with ``S = I (1, q, u, v)``: the flux factor is
    ``g = m00 + m01 q`` and the output reduced state is
    ``((m01 + m00 q), (m22 u + m23 v), (m22 v - m23 u)) / g``. Where
    ``g = 0`` (a crossed ideal polarizer, a totally reflected ray's transmitted
    branch) the output state is set to zero behind a double ``where``: the
    divisor is replaced by one before dividing, so no infinity is formed on
    either pass.

    Args:
        q, u, v: The reduced input state, per ray, in the element's frame.
        m: The element, per ray or broadcastable.

    Returns:
        ``(g, q', u', v')``.
    """
    g = m.m00 + m.m01 * q
    positive = g > 0
    safe = be.where(positive, g, be.ones_like(g))
    zero = be.zeros_like(g)
    q2 = be.where(positive, (m.m01 + m.m00 * q) / safe, zero)
    u2 = be.where(positive, (m.m22 * u + m.m23 * v) / safe, zero)
    v2 = be.where(positive, (m.m22 * v - m.m23 * u) / safe, zero)
    return g, q2, u2, v2


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------


def reflection_mueller(rs, rp, tir=None, tir_phase=None) -> InterfaceMueller:
    """Fresnel reflection (chapter 06 section 6.4) from real amplitudes.

    Below the critical angle ``r_s`` and ``r_p`` are real, so
    ``m00 = (r_s^2 + r_p^2) / 2`` -- the same expression, in the same order,
    as the scalar engine's reflectance, so an unpolarized ray's flux factor
    is the scalar one bit for bit -- ``m01 = (r_p^2 - r_s^2) / 2``,
    ``m22 = r_p r_s`` and ``m23 = 0``. Where ``tir`` is set the moduli are one
    and the relative phase is kept (R-06-5): ``m00 = 1``, ``m01 = 0``,
    ``(m22, m23) = (Re, Im)(r_p r_s*)`` from ``tir_phase``.

    Args:
        rs, rp: Real Fresnel amplitudes (the engine's own), per ray.
        tir: Optional total-internal-reflection mask.
        tir_phase: ``(re, im)`` of ``r_p r_s*`` on the ``tir`` lanes
            (:func:`tir_relative_phase`); required with ``tir``.

    Returns:
        The reflection's :class:`InterfaceMueller`.
    """
    m00 = 0.5 * (rs**2 + rp**2)
    m01 = 0.5 * (rp**2 - rs**2)
    m22 = rp * rs
    m23 = be.zeros_like(m22)
    if tir is None:
        return InterfaceMueller(m00, m01, m22, m23)
    x_re, x_im = tir_phase
    return InterfaceMueller(
        be.where(tir, be.ones_like(m00), m00),
        be.where(tir, be.zeros_like(m01), m01),
        be.where(tir, x_re, m22),
        be.where(tir, x_im, m23),
    )


def tir_relative_phase(n1, n2, cos_i, sin2_t, tir):
    """``(Re, Im)`` of ``r_p r_s*`` beyond the critical angle, in real arithmetic.

    With ``cos_t = +i kappa``, ``kappa = sqrt(sin^2 theta_t - 1)``:
    ``r_s = (A - iB) / (A + iB)``, ``A = n1 cos_i``, ``B = n2 kappa``, and
    ``r_p = (C - iD) / (C + iD)``, ``C = n2 cos_i``, ``D = n1 kappa``. Each is
    ``(X^2 - Y^2 - 2iXY) / (X^2 + Y^2)``, so

    ``Re(r_p r_s*) = [(C^2 - D^2)(A^2 - B^2) + 4ABCD] / den``,
    ``Im(r_p r_s*) = [2AB(C^2 - D^2) - 2CD(A^2 - B^2)] / den``,
    ``den = (A^2 + B^2)(C^2 + D^2)``.

    At 1.5 -> 1.0 and 45 degrees this is ``exp(-i 36.8699 deg)``, the
    analytic reference's value. Lanes outside ``tir`` take ``kappa`` from a
    masked radicand and return a value the caller discards.

    Args:
        n1, n2: Indices of the incident and the far medium, per ray.
        cos_i: ``|cos theta_i|``, per ray.
        sin2_t: ``(n1 / n2)^2 sin^2 theta_i``, per ray.
        tir: The total-internal-reflection mask.

    Returns:
        ``(re, im)``, per ray.
    """
    radicand = be.where(tir, sin2_t - 1.0, be.ones_like(sin2_t))
    kappa = be.where(tir, radicand**0.5, be.zeros_like(sin2_t))
    A = n1 * cos_i
    B = n2 * kappa
    C = n2 * cos_i
    D = n1 * kappa
    a2 = A * A - B * B
    c2 = C * C - D * D
    den = (A * A + B * B) * (C * C + D * D)
    den = den + _tol.tiny_for(den)
    re = (c2 * a2 + 4.0 * A * B * C * D) / den
    im = (2.0 * A * B * c2 - 2.0 * C * D * a2) / den
    return re, im


def transmission_mueller(Ts, Tp, m00=None, phase=None) -> InterfaceMueller:
    """Fresnel (or coated) transmission from the power transmittances.

    ``m00 = (T_s + T_p) / 2`` (or the caller's own scalar ``T``, so the
    unpolarized flux factor is the scalar engine's bit for bit),
    ``m01 = (T_p - T_s) / 2``, and
    ``(m22, m23) = sqrt(T_s T_p) (cos Delta, sin Delta)`` with
    ``Delta = arg(t_p t_s*)`` -- zero for a bare dielectric. The square root
    is taken of a masked radicand, so a lane with ``T_s T_p = 0`` (total
    internal reflection) has a zero element and a finite derivative.

    Args:
        Ts, Tp: Power transmittances, per ray.
        m00: Optional ``m00`` to use instead of ``(Ts + Tp) / 2``.
        phase: Optional ``(cos Delta, sin Delta)``; default no phase.

    Returns:
        The transmission's :class:`InterfaceMueller`.
    """
    if m00 is None:
        m00 = 0.5 * (Ts + Tp)
    m01 = 0.5 * (Tp - Ts)
    amp = _masked_sqrt(Ts * Tp)
    if phase is None:
        return InterfaceMueller(m00, m01, amp, be.zeros_like(amp))
    cos_d, sin_d = phase
    return InterfaceMueller(m00, m01, amp * cos_d, amp * sin_d)


def diattenuator_mueller(tx, ty) -> InterfaceMueller:
    """Linear diattenuator: power transmittance ``tx`` on ``p``, ``ty`` on ``s``.

    Jones ``diag(sqrt(tx), sqrt(ty))``: ``m00 = (tx + ty) / 2``,
    ``m01 = (tx - ty) / 2``, ``m22 = sqrt(tx ty)``, ``m23 = 0``. An ideal
    linear polarizer is ``tx = 1``, ``ty = 0``; a finite extinction ratio is
    ``ty > 0``. ``tx`` and ``ty`` may be Python floats or arrays.
    """
    if isinstance(tx, float) and isinstance(ty, float):
        return InterfaceMueller(0.5 * (tx + ty), 0.5 * (tx - ty), (tx * ty) ** 0.5, 0.0)
    return InterfaceMueller(
        0.5 * (tx + ty), 0.5 * (tx - ty), _masked_sqrt(tx * ty), 0.0 * tx
    )


def polarizer_mueller(like, extinction=0.0) -> InterfaceMueller:
    """Linear polarizer with its transmission axis on ``p``, per ray.

    Args:
        like: A per-ray array giving the shape, library and dtype.
        extinction: Power transmittance of the blocked axis (0 is ideal).

    Returns:
        ``diattenuator_mueller(1, extinction)`` broadcast to ``like``.
    """
    one = be.ones_like(like)
    return diattenuator_mueller(one, one * extinction)


def retarder_mueller(cos_delta, sin_delta) -> InterfaceMueller:
    """Linear retarder with its fast axis on ``p`` and retardance ``delta``.

    ``m00 = 1``, ``m01 = 0``, ``m22 = cos delta``, ``m23 = -sin delta``: the
    matrix of the analytic reference's ``retarder_stokes`` at a fast axis of
    zero, so that a quarter wave with the fast axis at +45 degrees takes
    ``(1, 1, 0, 0)`` to ``(1, 0, 0, -1)``.

    Args:
        cos_delta, sin_delta: Per-ray cosine and sine of the retardance.
    """
    return InterfaceMueller(
        be.ones_like(cos_delta), be.zeros_like(cos_delta), cos_delta, -sin_delta
    )


def depolarize(q, u, v, kappa):
    """``diag(1, kappa, kappa, kappa)`` (chapter 06 section 6.7): ``I`` is unchanged."""
    return kappa * q, kappa * u, kappa * v


# ---------------------------------------------------------------------------
# State and frame helpers
# ---------------------------------------------------------------------------


def degree_of_polarization(q, u, v):
    """``sqrt(q^2 + u^2 + v^2)``, the degree of polarization of a reduced state."""
    return (q * q + u * u + v * v) ** 0.5


def realizability_excess(q, u, v):
    """``q^2 + u^2 + v^2 - 1``: positive only for an unphysical state (R-06-7)."""
    return q * q + u * u + v * v - 1.0


def birth_axis(kx, ky, kz):
    """The reference axis a newborn ray gets: lab ``x`` made perpendicular to ``k``.

    ``e = normalize(a - (a . k) k)`` with ``a = x``; where ``k`` is within
    :func:`degeneracy_tolerance` of ``x`` the lab ``y`` axis is used instead.
    For a ray along ``z`` this is ``e = x`` exactly.

    Args:
        kx, ky, kz: Unit direction components, per ray.

    Returns:
        ``(ex, ey, ez)``, per ray.
    """
    one = be.ones_like(kx)
    zero = be.zeros_like(kx)
    k = (kx, ky, kz)
    ex, sx2 = perpendicular_axis((one, zero, zero), k)
    ey, _ = perpendicular_axis((zero, one, zero), k)
    tol = degeneracy_tolerance(kx)
    use_x = sx2 > tol * tol
    return tuple(be.where(use_x, a, b) for a, b in zip(ex, ey, strict=True))


def perpendicular_axis(a, k):
    """``normalize(a - (a . k) k)``, formed as ``(k x a) x k``.

    The double cross product is perpendicular to ``k`` to the dtype's own
    rounding whatever the angle between ``a`` and ``k``; the subtraction
    ``a - (a . k) k`` loses that as ``u / |k x a|`` when ``a`` is nearly
    along ``k``.

    Args:
        a: ``(x, y, z)`` of the axis, per ray or broadcastable.
        k: ``(x, y, z)`` of the unit direction, per ray.

    Returns:
        ``((ex, ey, ez), |k x a|^2)``; where ``k x a`` vanishes the axis is
        returned unnormalised (zero) and the caller decides the fallback.
    """
    s = _cross(*k, *a)
    e = _cross(*s, *k)
    s2 = _dot(*s, *s)
    n2 = _dot(*e, *e)
    inv = 1.0 / be.where(n2 > 0, n2, be.ones_like(n2)) ** 0.5
    return (e[0] * inv, e[1] * inv, e[2] * inv), s2


def _detached_zeros_like(x):
    """Zeros of ``x``'s library, dtype, device and shape, with no gradient flag."""
    if be.is_torch_tensor(x):
        import torch  # noqa: PLC0415

        return torch.zeros_like(x, requires_grad=False)
    return np.zeros_like(x)


class SourcePolarization:
    """The polarization a source emits: a Stokes vector and the axis it is referred to.

    Attributes:
        stokes: ``(S0, S1, S2, S3)`` with ``S0 > 0`` and
            ``S1^2 + S2^2 + S3^2 <= S0^2``; only the ratios matter, since the
            source's ``total_flux`` sets ``I``. ``(1, 0, 0, 0)`` is
            unpolarized.
        reference_axis: The global direction ``S1 > 0`` is polarized along
            (projected perpendicular to each ray's direction), or ``None``
            for lab ``x`` (lab ``y`` for a ray along ``x``), the axis every
            unpolarized ray is born with. A ray whose direction lies within
            :func:`degeneracy_tolerance` of the given axis falls back to the
            default axis.
    """

    def __init__(self, stokes=(1.0, 0.0, 0.0, 0.0), reference_axis=None) -> None:
        s = [float(x) for x in stokes]
        if len(s) != 4:
            raise ValueError(f"stokes must have four entries, got {len(s)}")
        if not s[0] > 0:
            raise ValueError(f"stokes S0 must be positive, got {s[0]}")
        excess = (s[1] ** 2 + s[2] ** 2 + s[3] ** 2) / s[0] ** 2 - 1.0
        if excess > 1e-12:
            raise ValueError(
                f"stokes {tuple(s)} is not realizable: S1^2 + S2^2 + S3^2 exceeds S0^2 "
                "(R-06-7)"
            )
        self.stokes = tuple(s)
        if reference_axis is not None:
            a = np.asarray(reference_axis, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(a))
            if not norm > 0:
                raise ValueError("reference_axis must be a non-zero vector")
            reference_axis = tuple(float(x) for x in a / norm)
        self.reference_axis = reference_axis

    @property
    def reduced(self) -> tuple[float, float, float]:
        """``(q, u, v) = (S1, S2, S3) / S0``."""
        s0 = self.stokes[0]
        return self.stokes[1] / s0, self.stokes[2] / s0, self.stokes[3] / s0

    @property
    def unpolarized(self) -> bool:
        return self.reduced == (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {"stokes": list(self.stokes), "reference_axis": (
            None if self.reference_axis is None else list(self.reference_axis)
        )}

    @classmethod
    def from_dict(cls, d: dict) -> SourcePolarization:
        return cls(stokes=d["stokes"], reference_axis=d.get("reference_axis"))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SourcePolarization)
            and self.stokes == other.stokes
            and self.reference_axis == other.reference_axis
        )

    def __repr__(self) -> str:
        return (
            f"SourcePolarization(stokes={self.stokes}, "
            f"reference_axis={self.reference_axis})"
        )


def set_source_polarization(
    scene, name: str, stokes=(1.0, 0.0, 0.0, 0.0), reference_axis=None
):
    """Give the scene's source ``name`` a polarization (read only in Stokes mode).

    A scalar trace ignores it: the scalar engine traces the flux the source
    emits, which the Stokes vector does not change (R-06-9: one set of model
    inputs for both modes).

    Args:
        scene: The :class:`~optiland.nonsequential.scene.NSQScene`.
        name: The source's registry name.
        stokes: See :class:`SourcePolarization`.
        reference_axis: See :class:`SourcePolarization`.

    Returns:
        The :class:`SourcePolarization` set.
    """
    pol = SourcePolarization(stokes, reference_axis)
    scene.source_registry.get(name).polarization = pol
    return pol


def _constant_like(x, value: float):
    return _detached_zeros_like(x) + value


def prepare_bundle(rays, source=None) -> None:
    """Give a freshly generated, device-placed bundle its polarization state, in place.

    Called once per batch by the trace loop when polarization is on, after the
    backend has placed the bundle on its library and device. The state is the
    source's :class:`SourcePolarization` (its ``polarization`` attribute, set
    by :func:`set_source_polarization`), or unpolarized,
    ``(q, u, v) = (0, 0, 0)``, when the source has none. The reference axis is
    :func:`birth_axis` of the direction, or the source's own axis projected
    perpendicular to it. A state a source's ``generate`` set itself is moved
    onto the ray state's library and dtype. ``flux`` is not touched: it is
    the Stokes ``I`` as it stands.

    Args:
        rays: An :class:`~optiland.nonsequential.ray_bundle.NSQRayBundle`.
        source: The source that generated it, or ``None``.
    """
    pol = getattr(source, "polarization", None)
    if rays.pol_q is None:
        if pol is None or pol.unpolarized:
            rays.pol_q = _detached_zeros_like(rays.flux)
            rays.pol_u = _detached_zeros_like(rays.flux)
            rays.pol_v = _detached_zeros_like(rays.flux)
        else:
            q, u, v = pol.reduced
            rays.pol_q = _constant_like(rays.flux, q)
            rays.pol_u = _constant_like(rays.flux, u)
            rays.pol_v = _constant_like(rays.flux, v)
        default = birth_axis(rays.L, rays.M, rays.N)
        if pol is None or pol.reference_axis is None:
            rays.pol_ex, rays.pol_ey, rays.pol_ez = default
            return
        a = tuple(_constant_like(rays.flux, c) for c in pol.reference_axis)
        e, s2 = perpendicular_axis(a, (rays.L, rays.M, rays.N))
        tol = degeneracy_tolerance(rays.L)
        ok = s2 > tol * tol
        rays.pol_ex = be.where(ok, e[0], default[0])
        rays.pol_ey = be.where(ok, e[1], default[1])
        rays.pol_ez = be.where(ok, e[2], default[2])
        return
    for name in POL_FIELDS:
        value = getattr(rays, name)
        if be.is_torch_tensor(rays.flux) and not be.is_torch_tensor(value):
            import torch  # noqa: PLC0415

            value = torch.as_tensor(
                np.asarray(value), dtype=rays.flux.dtype, device=rays.flux.device
            )
        elif not be.is_torch_tensor(rays.flux):
            value = np.asarray(value, dtype=np.asarray(rays.flux).dtype)
        setattr(rays, name, value)


def transport_axis(e, k):
    """Re-project a reference axis perpendicular to a new direction and normalise it.

    For an element that changes ``k`` without a plane of incidence (a
    paraxial lens, a transmissive detector crossing): ``e' = normalize(e -
    (e . k) k)``. The Stokes state is unchanged.

    Args:
        e: ``(ex, ey, ez)`` per-ray components.
        k: ``(kx, ky, kz)``, the new unit direction.

    Returns:
        ``(ex', ey', ez')``.
    """
    return perpendicular_axis(e, k)[0]


# ---------------------------------------------------------------------------
# The thin-film adapter: s and p coefficients in this module's convention
# ---------------------------------------------------------------------------


class SPCoefficients(NamedTuple):
    """What a coated (or bare, or metallic) interface gives a Mueller matrix.

    Attributes:
        Rs, Rp, Ts, Tp: Power reflectances and transmittances, per ray.
        xr_re, xr_im: ``Re`` and ``Im`` of ``r_p r_s*`` in this module's
            convention (the reflection's ``m22`` and ``m23``).
        xt_cos, xt_sin: ``cos`` and ``sin`` of ``Delta = arg(t_p t_s*)``
            (``(1, 0)`` where either transmittance is zero).
        phase_valid: Per-ray mask; True everywhere since the thin-film module
            states one time convention (issue 78). Kept for its readers.
    """

    Rs: object
    Rp: object
    Ts: object
    Tp: object
    xr_re: object
    xr_im: object
    xt_cos: object
    xt_sin: object
    phase_valid: object

    def reflection(self) -> InterfaceMueller:
        """The reflection element: ``m00 = (R_s + R_p) / 2`` and the phase terms."""
        return InterfaceMueller(
            0.5 * (self.Rs + self.Rp), 0.5 * (self.Rp - self.Rs), self.xr_re, self.xr_im
        )

    def transmission(self) -> InterfaceMueller:
        """The transmission element: ``(T_s + T_p) / 2``, ``sqrt(T_s T_p) e^{iD}``."""
        return transmission_mueller(self.Ts, self.Tp, phase=(self.xt_cos, self.xt_sin))


def thin_film_sp(stack, wavelength_um, cos_theta_i, reverse=None) -> SPCoefficients:
    """The fork's thin-film module's s and p results, as Mueller terms.

    ``ThinFilmStack.compute_rtRTA_elementwise(..., "s" | "p")`` gives the powers
    and the complex amplitudes in this module's own time convention
    (``exp(-i omega t)``; the thin-film module states it, the research
    repository's issue 78). One correction remains, pinned by a test
    (``tests/nonsequential/test_nsq_polarization_thin_film.py``):

    **The sign of ``r_p``.** The module's ``p`` admittance is
    ``eta_p = n / cos theta``, which gives ``r_p`` with the opposite sign to the
    Fresnel convention of the catalogue's analytic reference; ``r_p`` is
    negated.

    The transmission terms are the module's tangential-field amplitudes; the
    p field amplitude differs from its tangential part by a real positive
    factor in lossless media, so ``arg(t_p t_s*)`` is the same.

    Until 2026-10-01 the module's layers were in the ``exp(+i omega t)``
    convention while its evanescent root was the ``exp(-i omega t)`` one, and
    this adapter conjugated the reflection phase of a stack with layers or an
    absorbing medium; a coated face met beyond its critical angle matched
    neither convention and was flagged in ``phase_valid``. With the module in
    one convention the phase is right on every lane, and ``phase_valid`` is
    True everywhere (kept so that a caller reading it still can).

    Args:
        stack: A configured ``optiland.thin_film.ThinFilmStack``.
        wavelength_um: Per-ray wavelength [um].
        cos_theta_i: Per-ray ``|cos theta_i|`` in the medium the ray arrives
            from.
        reverse: Optional per-ray mask of the rays arriving from the stack's
            substrate side (the research repository's issue 83): those see
            the reversed stack, the substrate as their incident medium.

    Returns:
        :class:`SPCoefficients`, per ray, real arrays in the working dtype.
    """
    cos_theta_i = be.clip(cos_theta_i, -1.0, 1.0)
    aoi = be.arccos(cos_theta_i)
    s = stack.compute_rtRTA_elementwise(
        wavelength_um, aoi, polarization="s", reverse=reverse
    )
    p = stack.compute_rtRTA_elementwise(
        wavelength_um, aoi, polarization="p", reverse=reverse
    )
    rs, rp = s["r"], -p["r"]
    x_r = rp * be.conj(rs)
    re = be.real(x_r)
    im = be.imag(x_r)

    x_t = p["t"] * be.conj(s["t"])
    mag = be.abs(x_t)
    nonzero = mag > 0
    safe = be.where(nonzero, mag, be.ones_like(mag))
    xt_cos = be.where(nonzero, be.real(x_t) / safe, be.ones_like(mag))
    xt_sin = be.where(nonzero, be.imag(x_t) / safe, be.zeros_like(mag))
    phase_valid = be.ones_like(mag) > 0
    return SPCoefficients(
        s["R"], p["R"], s["T"], p["T"], re, im, xt_cos, xt_sin, phase_valid
    )


def coating_sp(coating, wavelength, cos_i, from_substrate=None):
    """A coating's s and p terms for a Stokes event, or ``None`` for a coating without them.

    A thin-film coating (``coating.stack``): :func:`thin_film_sp`. A coating table
    (``optiland.coatings.TabulatedCoating``, with its phase grids; it refuses
    without them): its own ``sp``. Anything else (a ``SimpleCoating``, a
    constant) has no s and p split: ``None``.

    The scalar ``R`` and ``T`` of the same coating are ``(R_s + R_p) / 2`` and
    ``(T_s + T_p) / 2`` of these terms, bit for bit: the thin-film adapter's
    ``evaluate`` and :func:`thin_film_sp` clip the cosine, take its arccos and
    evaluate the stack for s and for p with the same operations, and the
    table's ``evaluate`` and ``sp`` read the same lookup. A Stokes event uses
    that to evaluate the coating once (the operation trim of 2026-10-02).

    Args:
        coating: The surface's coating.
        wavelength: Per-ray wavelength [um].
        cos_i: Per-ray ``|cos theta_i|``.
        from_substrate: Per-ray mask of the rays arriving from the substrate
            side, or ``None``.

    Returns:
        :class:`SPCoefficients` or ``None``.
    """
    if hasattr(coating, "stack"):
        return thin_film_sp(coating.stack, wavelength, cos_i, reverse=from_substrate)
    if callable(getattr(coating, "sp", None)):
        return coating.sp(wavelength, cos_i, from_substrate)
    return None


# ---------------------------------------------------------------------------
# The Stokes event at a refractive interface (build item 5)
# ---------------------------------------------------------------------------


def _pick(mask, a: InterfaceMueller, b: InterfaceMueller) -> InterfaceMueller:
    return InterfaceMueller(*(be.where(mask, x, y) for x, y in zip(a, b, strict=True)))


class FresnelStokes:
    """One Fresnel event in Stokes mode, split around the engine's branch draw.

    Built by :func:`fresnel_stokes` before the branch is drawn, it gives the
    engine the polarization-aware reflectance and transmittance
    ``R_eff = M_r00 + M_r01 q'`` and ``T_eff = M_t00 + M_t01 q'`` (``q'`` the
    state in the plane-of-incidence frame), which replace the scalar ``R`` and
    ``T`` in the branch probability, the weights and the ledger. With an
    unpolarized state ``q' = 0`` and they *are* the scalar values, bit for bit,
    so the branch decisions and every flux are unchanged (chapter 06 section
    6.9, conditions 1 and 2). :meth:`finish` then sets the outgoing state from
    the matrix of the branch taken and the outgoing reference axis
    ``e = s x k_out`` (R-06-3).
    """

    def __init__(self, s, q, u, v, m_r: InterfaceMueller, m_t: InterfaceMueller):
        self.s = s
        self.q, self.u, self.v = q, u, v
        self.m_r, self.m_t = m_r, m_t
        self.R_eff = m_r.m00 + m_r.m01 * q
        self.T_eff = m_t.m00 + m_t.m01 * q

    def finish(self, rays, do_reflect, hit_mask) -> None:
        """Write the outgoing state of the hit rays; the others keep theirs.

        Called after the engine has set the new direction on ``rays``.
        """
        m = _pick(do_reflect, self.m_r, self.m_t)
        _, q, u, v = apply_interface(self.q, self.u, self.v, m)
        e = _cross(*self.s, rays.L, rays.M, rays.N)
        rays.pol_q = be.where(hit_mask, q, rays.pol_q)
        rays.pol_u = be.where(hit_mask, u, rays.pol_u)
        rays.pol_v = be.where(hit_mask, v, rays.pol_v)
        rays.pol_ex = be.where(hit_mask, e[0], rays.pol_ex)
        rays.pol_ey = be.where(hit_mask, e[1], rays.pol_ey)
        rays.pol_ez = be.where(hit_mask, e[2], rays.pol_ez)

    @staticmethod
    def scatter(rays, scattered, preserves: bool = False) -> None:
        """A ray routed through a scatter lobe: see :func:`scatter_state`."""
        scatter_state(rays, scattered, preserves)


def scatter_state(rays, scattered, preserves: bool = False) -> None:
    """The state of a ray a scatter lobe sent in a new direction (R-06-8).

    The minimal polarization carries no Mueller scatter model: a lobe either
    depolarizes the rays it scatters completely (``kappa = 0``: the
    Lambertian, Harvey-Shack and tabulated lobes, whatever their angle), or,
    for the specular lobe, whose direction is the mirror direction the
    surface event already gave the state for, keeps the state. Either way the
    reference axis is carried to the lobe's direction by re-projection. An
    angle-dependent or Mueller-matrix scatter model (R-06-8's middle and full
    levels) is outside the minimal version.

    Args:
        rays: The bundle, its direction already the lobe's.
        scattered: Per-ray mask of the rays the lobe took.
        preserves: The lobe's ``preserves_polarization`` flag.
    """
    if not preserves:
        zero = _detached_zeros_like(rays.pol_q)
        rays.pol_q = be.where(scattered, zero, rays.pol_q)
        rays.pol_u = be.where(scattered, zero, rays.pol_u)
        rays.pol_v = be.where(scattered, zero, rays.pol_v)
    k = (rays.L, rays.M, rays.N)
    e, s2 = perpendicular_axis((rays.pol_ex, rays.pol_ey, rays.pol_ez), k)
    # A lobe can send a ray along its old axis, where the re-projection
    # vanishes; that ray takes the birth axis of its new direction (the
    # state of a depolarized ray does not depend on the axis).
    tol = degeneracy_tolerance(rays.L)
    ok = s2 > tol * tol
    fallback = birth_axis(*k)
    e = tuple(be.where(ok, a, b) for a, b in zip(e, fallback, strict=True))
    rays.pol_ex = be.where(scattered, e[0], rays.pol_ex)
    rays.pol_ey = be.where(scattered, e[1], rays.pol_ey)
    rays.pol_ez = be.where(scattered, e[2], rays.pol_ez)


def transport_state(rays, mask) -> None:
    """Carry the reference axis of ``mask``'s rays to their new direction.

    For an element that turns ``k`` without a plane of incidence and without
    acting on the polarization (the ideal paraxial lens): the state is kept
    and the axis re-projected perpendicular to the new direction
    (:func:`transport_axis`). The re-projection differs from a parallel
    transport of the axis by a rotation of second order in the deflection,
    which an ideal thin lens, having no polarization model of its own, does
    not define.

    Args:
        rays: The bundle, its direction already the new one.
        mask: Per-ray mask of the rays the element turned.
    """
    e = transport_axis(
        (rays.pol_ex, rays.pol_ey, rays.pol_ez), (rays.L, rays.M, rays.N)
    )
    rays.pol_ex = be.where(mask, e[0], rays.pol_ex)
    rays.pol_ey = be.where(mask, e[1], rays.pol_ey)
    rays.pol_ez = be.where(mask, e[2], rays.pol_ez)


def incidence_frame(rays, dirs, normals):
    """The ray's state in the plane-of-incidence frame of a surface, and ``s``.

    1. The incoming reference axis is re-projected perpendicular to ``k`` (a
       kind that does not transport it exactly leaves it slightly off).
    2. The plane of incidence: ``s = k x n / |k x n|``; where
       ``|k x n| < sqrt(eps)`` the interface is degenerate (normal incidence,
       R-06-4), ``s = k x e`` and the rotation is skipped exactly.
    3. The rotation from ``e`` to ``p_in = s x k`` (no trigonometric call).

    Shared by every surface event that has a plane of incidence: the
    refractive interface and the mirror.

    Args:
        rays: The bundle (reads its state and reference axis).
        dirs, normals: ``(N, 3)`` incident directions and surface normals.

    Returns:
        ``(s, q, u, v)``: ``s`` as an ``(x, y, z)`` tuple of per-ray arrays and
        the reduced state in the ``(p_in, s, k)`` frame.
    """
    k = (dirs[:, 0], dirs[:, 1], dirs[:, 2])
    nrm = (normals[:, 0], normals[:, 1], normals[:, 2])
    e = transport_axis((rays.pol_ex, rays.pol_ey, rays.pol_ez), k)

    s_raw = _cross(*k, *nrm)
    ls2 = _dot(*s_raw, *s_raw)
    tol = degeneracy_tolerance(ls2)
    deg = ls2 < tol * tol
    inv = 1.0 / be.where(deg, be.ones_like(ls2), ls2) ** 0.5
    s_frame = _cross(*k, *e)
    s = tuple(
        be.where(deg, sf, sr * inv) for sf, sr in zip(s_frame, s_raw, strict=True)
    )
    p_in = _cross(*s, *k)
    c2, s2 = rotation_2psi(e, p_in, k)
    c2 = be.where(deg, be.ones_like(c2), c2)
    s2 = be.where(deg, be.zeros_like(s2), s2)
    q, u = rotate(rays.pol_q, rays.pol_u, c2, s2)
    return s, q, u, rays.pol_v


def fresnel_stokes(
    rays, dirs, normals, n1, n2, cos_i, sin2_t, tir, rs, rp, R_used, T_used,
    coating=None, wavelength=None, from_substrate=None, coated_tir=False, sp=None,
) -> FresnelStokes:
    """The Stokes half of a refractive interface, before the branch is drawn.

    1. to 3. The state in the plane of incidence (:func:`incidence_frame`):
       the axis re-projected, ``s = k x n / |k x n|`` (``s = k x e`` at a
       degenerate interface), the rotation from ``e`` to ``p_in = s x k``.
    4. The interface elements. A bare interface: the Fresnel reflection with
       the TIR phase, and the transmission ``sqrt(T_s T_p)`` with
       ``T_s = 1 - r_s^2``, ``T_p = 1 - r_p^2`` and ``M_t01 = -M_r01``. A
       thin-film coating: :func:`thin_film_sp`; a coating table: its own
       ``sp`` (``optiland.coatings.TabulatedCoating``). A coating with scalar ``R`` and
       ``T`` only: a non-polarizing element, ``R diag(1, 1, -1, -1)`` and
       ``T diag(1, 1, 1, 1)`` (an ideal reflection flips the handedness). In
       every case ``M_00`` is the scalar ``R`` or ``T`` the engine already
       uses, exactly (R-06-6).

    Args:
        rays: The bundle (reads its state and reference axis).
        dirs, normals: ``(N, 3)`` incident directions and surface normals.
        n1, n2, cos_i, sin2_t, tir, rs, rp: The engine's own Fresnel
            quantities for these rays.
        R_used, T_used: The scalar reflectance and transmittance the engine
            uses (bare, or the coating's), after its TIR override.
        coating: The surface's coating, or ``None``.
        wavelength: Per-ray wavelength [um] (for a thin-film coating).
        from_substrate: Per-ray mask of the rays arriving from a side-aware
            coating's substrate side, or ``None`` (issue 83).
        coated_tir: Whether the coating keeps its own reflectance beyond the
            critical angle (a side-aware coating, issue 96): its ``M01`` is
            then kept on the totally reflected lanes, so a polarized ray
            meeting an absorbing coating in total internal reflection reads
            ``R_s`` or ``R_p`` rather than their mean. ``False`` (a bare face,
            a side-blind coating) keeps ``M01 = 0`` there, as before.
        sp: The coating's :class:`SPCoefficients` when the caller has already
            formed them (:func:`coating_sp`; the refractive face does, and
            takes its scalar ``R`` and ``T`` from the same terms, so the stack
            is evaluated once per event, not twice); ``None`` forms them here.

    Returns:
        A :class:`FresnelStokes`.
    """
    s, q, u, v = incidence_frame(rays, dirs, normals)

    zero = be.zeros_like(R_used)
    if coating is None:
        # Only the elements the event uses, each squared amplitude formed once
        # (the operation trim of 2026-10-02): the same operations on the same
        # operands as reflection_mueller and transmission_mueller, less the
        # reflection's m00 and the transmission's m01, which the engine's
        # R_used and -M_r01 replace; every value is bit-identical.
        phase = tir_relative_phase(n1, n2, cos_i, sin2_t, tir)
        rs2, rp2 = rs**2, rp**2
        m01 = be.where(tir, be.zeros_like(rs2), 0.5 * (rp2 - rs2))
        m22 = rp * rs
        m_r = InterfaceMueller(
            R_used, m01, be.where(tir, phase[0], m22), be.where(tir, phase[1], be.zeros_like(m22))
        )
        amp = _masked_sqrt((1.0 - rs2) * (1.0 - rp2))
        m_t = InterfaceMueller(
            T_used,
            be.where(tir, zero, -m01),
            be.where(tir, zero, amp),
            zero,
        )
    elif sp is not None or hasattr(coating, "stack") or callable(getattr(coating, "sp", None)):
        if sp is None:
            sp = coating_sp(coating, wavelength, cos_i, from_substrate)
        if coated_tir:
            # issue 96: beyond the critical angle the coating keeps its own
            # s and p reflectances (R_used there is (R_s + T_s + R_p + T_p) / 2,
            # the transmitted share returned to the reflection; T_s = T_p = 0
            # on every lane beyond the cutoff)
            m01 = be.where(
                tir,
                0.5 * ((sp.Rp + sp.Tp) - (sp.Rs + sp.Ts)),
                0.5 * (sp.Rp - sp.Rs),
            )
        else:
            m01 = be.where(tir, zero, 0.5 * (sp.Rp - sp.Rs))
        m_r = InterfaceMueller(R_used, m01, sp.xr_re, sp.xr_im)
        # m00 is the engine's T_used: the element's own (T_s + T_p) / 2 is not formed
        t = transmission_mueller(sp.Ts, sp.Tp, m00=T_used, phase=(sp.xt_cos, sp.xt_sin))
        m_t = InterfaceMueller(
            T_used, be.where(tir, zero, t.m01), be.where(tir, zero, t.m22),
            be.where(tir, zero, t.m23),
        )
    else:
        phase = tir_relative_phase(n1, n2, cos_i, sin2_t, tir)
        m_r = InterfaceMueller(
            R_used,
            zero,
            be.where(tir, phase[0], -R_used),
            be.where(tir, phase[1], zero),
        )
        m_t = InterfaceMueller(T_used, zero, T_used, zero)
    return FresnelStokes(s, q, u, v, m_r, m_t)


# ---------------------------------------------------------------------------
# The Stokes event at a mirror (build item 6)
# ---------------------------------------------------------------------------


class MirrorStokes:
    """One mirror reflection in Stokes mode.

    ``g = M00 + M01 q'`` is the flux factor (``q'`` the state in the plane of
    incidence); with an unpolarized state it is the scalar reflectance the
    mirror already applies, bit for bit. :meth:`finish` sets the outgoing
    state and the axis ``e = s x k_out`` once the new direction is on the
    bundle.
    """

    def __init__(self, s, q, u, v, m: InterfaceMueller):
        self.s = s
        self.q, self.u, self.v = q, u, v
        self.m = m
        self.R_eff = m.m00 + m.m01 * q

    def finish(self, rays, hit_mask, k_out) -> None:
        """Write the reflected state and axis of the hit rays; the others keep theirs.

        Args:
            rays: The bundle.
            hit_mask: Per-ray mask of the rays the mirror reflected.
            k_out: ``(x, y, z)`` of the specular direction, per ray (a ray a
                scatter lobe takes is then carried from it by
                :func:`scatter_state`).
        """
        _, q, u, v = apply_interface(self.q, self.u, self.v, self.m)
        e = _cross(*self.s, *k_out)
        rays.pol_q = be.where(hit_mask, q, rays.pol_q)
        rays.pol_u = be.where(hit_mask, u, rays.pol_u)
        rays.pol_v = be.where(hit_mask, v, rays.pol_v)
        rays.pol_ex = be.where(hit_mask, e[0], rays.pol_ex)
        rays.pol_ey = be.where(hit_mask, e[1], rays.pol_ey)
        rays.pol_ez = be.where(hit_mask, e[2], rays.pol_ez)


def mirror_stokes(rays, dirs, normals, R_used, stack=None, wavelength=None,
                  cos_i=None, coating=None) -> MirrorStokes:
    """The Stokes half of a mirror reflection, before the flux is weighted.

    Two kinds of mirror, each with ``M00`` the scalar reflectance ``R`` the
    mirror applies in scalar mode, exactly (R-06-6):

    * **A reflectance with no s and p split** (a constant, a
      ``callable(wavelength)``, a ``SimpleCoating``): the ideal
      non-polarizing mirror ``R diag(1, 1, -1, -1)``. It keeps the degree of
      polarization and flips the handedness: ``(q, u, v) -> (q, -u, -v)`` in
      the plane-of-incidence frame, which is the limit of the Fresnel
      reflection at normal incidence (``r_p = -r_s`` there, so
      ``m22 = r_p r_s = -R``).
    * **A thin-film stack** (``UnpolarizedThinFilmCoating`` as the mirror's
      reflectance): a dielectric or metal mirror, a layered stack or a bare
      absorbing substrate. The s and p reflectances and the relative phase
      ``r_p r_s*`` come from :func:`thin_film_sp`, in this module's
      convention: ``M01 = (R_p - R_s) / 2``, ``(M22, M23) = (Re, Im)(r_p
      r_s*)``, a coated face beyond its substrate's critical angle included.

    Args:
        rays: The bundle (reads its state and reference axis).
        dirs, normals: ``(N, 3)`` incident directions and surface normals.
        R_used: The scalar reflectance the mirror applies, per ray.
        stack: The mirror's ``ThinFilmStack``, or ``None``.
        wavelength: Per-ray wavelength [um] (for a stack or a table).
        cos_i: Per-ray ``|cos theta_i|`` (for a stack or a table).
        coating: A coating table (``optiland.coatings.TabulatedCoating``) as
            the reflectance, or ``None``: its own ``sp`` gives the s and p
            terms, and it refuses a Stokes trace without its phase grids.

    Returns:
        A :class:`MirrorStokes`.
    """
    s, q, u, v = incidence_frame(rays, dirs, normals)
    if stack is None and coating is not None and callable(getattr(coating, "sp", None)):
        sp = coating.sp(wavelength, cos_i)
        m = InterfaceMueller(R_used, 0.5 * (sp.Rp - sp.Rs), sp.xr_re, sp.xr_im)
    elif stack is None:
        zero = be.zeros_like(R_used)
        m = InterfaceMueller(R_used, zero, -R_used, zero)
    else:
        sp = thin_film_sp(stack, wavelength, cos_i)
        m = InterfaceMueller(R_used, 0.5 * (sp.Rp - sp.Rs), sp.xr_re, sp.xr_im)
    return MirrorStokes(s, q, u, v, m)


# ---------------------------------------------------------------------------
# Full 4x4 matrices, for tests and reports (NumPy, float64)
# ---------------------------------------------------------------------------


def mueller_matrix(m: InterfaceMueller) -> np.ndarray:
    """The 4x4 matrix of scalar interface-form elements, as a NumPy array."""
    m00, m01, m22, m23 = (float(np.asarray(x)) for x in m)
    return np.array(
        [
            [m00, m01, 0.0, 0.0],
            [m01, m00, 0.0, 0.0],
            [0.0, 0.0, m22, m23],
            [0.0, 0.0, -m23, m22],
        ]
    )


def rotation_matrix(psi_rad: float) -> np.ndarray:
    """``M_rot(psi)`` of chapter 06 section 6.5, as a NumPy array."""
    c, s = np.cos(2.0 * psi_rad), np.sin(2.0 * psi_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, s, 0.0],
            [0.0, -s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
