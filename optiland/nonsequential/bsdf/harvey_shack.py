"""Harvey-Shack (ABg) BSDF for Non-Sequential Raytracing.

Models micro-roughness scatter from optical surfaces.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential._utils import (
    as_attachable_param,
    clamp_int,
    host_float,
    refuse_gradients,
)
from optiland.nonsequential.bsdf.base import BaseBSDF
from optiland.nonsequential.components.base import resident_table
from optiland.nonsequential.components.sampling_support import (
    attachable_fraction,
    detached,
    lobe_branch_gate,
    onto_apple_gpu,
)
from optiland.nonsequential.ray_bundle import backend_bool_full
from optiland.nonsequential.rng import EventSlot

if TYPE_CHECKING:
    from optiland.nonsequential.rng import NSQRng

# Largest reachable direction-cosine offset: both the reference and the
# scattered direction lie in the unit disk, so |beta - beta0| <= 2. The radial
# table runs to here because an oblique reference reaches offsets up to
# 1 + |beta0|; which of them a given reference reaches is decided per sample
# (``sample``), never by the table.
_BETA_MAX = 2.0
# Nodes of the tabulated radial CDF: geometric from a small fraction of the
# break point to _BETA_MAX, so the knee of every lobe is resolved to a fixed
# relative step (0.2 percent at 8192 nodes), plus the nodes 0, 1 and 2.
_TABLE_SIZE = 8192
# Gauss-Legendre points per table cell for the radial integral.
_GAUSS_POINTS = 8
# Incidence grid of the hemispherical-TIS table: |beta0| = sin(theta_i) at
# theta_i uniform in [0, 90] degrees, which is denser in |beta0| towards
# grazing, where the table varies fastest. The linear interpolation between
# rows is the table's error; measured against a direct 40 000-point azimuth
# quadrature at 300 incidences, it is 2.4e-7 relative at worst (80 degrees,
# l0 = 0.01, s = 2) at this size, 1.5e-5 at 513 rows.
_INCIDENCE_SIZE = 4097
# Midpoints over the half period of the azimuth for that table (the
# integrand is even in the azimuth measured from beta0); raising it to
# 16 384 changes the table by less than 1e-9 relative.
_AZIMUTH_SIZE = 2048


def _lerp(x_grid, y_grid, x, size: int):
    """Piecewise-linear lookup ``y(x)`` on an increasing grid, in the backend.

    One ``searchsorted`` and one interpolation between the bracketing nodes,
    on tables uploaded beside the ray state. A query equal to a node returns
    that node's value exactly (the fraction is exactly zero there).

    Args:
        x_grid: Increasing abscissae, a resident backend array.
        y_grid: Ordinates, a resident backend array of the same size.
        x: Query points, shape (N,).
        size: Number of nodes.

    Returns:
        ``y`` at ``x``, shape (N,).
    """
    j = clamp_int(be.searchsorted(x_grid, x, side="right") - 1, 0, size - 2)
    x0 = x_grid[j]
    span = x_grid[j + 1] - x0
    y0 = y_grid[j]
    return y0 + (x - x0) / span * (y_grid[j + 1] - y0)


class HarveyShackBSDF(BaseBSDF):
    """Harvey-Shack / ABg scatter model for surface micro-roughness.

    The ABg model is a simplified form of the Harvey-Shack theory::

        BSDF(beta - beta0) = b0 / (1 + |beta - beta0| / l0)^s

    where beta and beta0 are direction cosines of the scattered and specular
    directions, b0 is the scatter level at beta=beta0, l0 is the break
    frequency, and s is the roll-off slope.

    Attributes:
        b0: Scatter amplitude at zero angle [sr^-1].
        l0: Break-point spatial frequency (dimensionless direction cosine).
        s: Power-law roll-off slope (positive).
        transmissive_fraction: Probability in [0, 1] that a given scatter
            event blurs the undeviated straight-through ray (the
            transmissive lobe, e.g. a diffuser sheet) instead of the
            specular reflection. Defaults to 0.0: a purely reflective
            blur, identical to this class's behaviour before D-5.
        weight_is_albedo: False (ruling 5 of the research repository's
            ``docs/build/R3_rulings_2026-09-27.md``). The lobe is
            normalised over the directions reachable from each incidence
            (see :meth:`sample`), so its weight has expectation one; at
            normal incidence every weight is exactly one, so ``1 - weight``
            is zero and the flag decides nothing there. Off normal
            incidence a single ray's weight is a sampling weight, not a
            physical fraction of the incident flux, so the flag is False:
            the surface books ``1 - reflectance()`` (zero, this lobe is
            lossless) as coating loss and the rest of ``1 - weight`` as the
            zero-mean sampling residual of chapter 10 section 10.2, instead
            of letting that fluctuation land in the coating bin. The
            identity (10.1) closes either way. This changes the one
            assertion of the fork's suite that asserted the base class's
            default for this lobe
            (``tests/nonsequential/test_nsq_flux_ledger.py``), under
            ruling 5.

    Polarization: in a Stokes trace (the research repository's issue 5) a ray
    this lobe scatters leaves depolarized (``preserves_polarization`` is
    False: the minimal version has no Mueller scatter model, chapter 06
    R-06-8), with its reference axis carried to the scattered direction.
    """

    weight_is_albedo = False

    def __init__(
        self, b0: float, l0: float, s: float, transmissive_fraction: float = 0.0
    ) -> None:
        """Initialize HarveyShackBSDF.

        Args:
            b0: Scatter amplitude at zero angle [sr^-1].
            l0: Break-point spatial frequency in direction-cosine space.
            s: Power-law roll-off exponent (positive).
            transmissive_fraction: Probability in [0, 1] that a scatter
                event blurs the straight-through ray instead of the
                specular reflection.

        Gradients (the research repository's chapter 09 section 9.14.3):
        ``l0``, ``s`` and ``transmissive_fraction`` may be tensors that carry
        a derivative. The direction is still drawn from their host values
        (detached); the weight carries the lobe's dependence on them at the
        drawn direction, ``w * d log f / d theta`` (the likelihood-ratio
        path), and each branch its share of the fraction. Every value is the
        same bit as with plain numbers. ``b0`` is refused: the lobe is
        normalised over the reachable directions, so ``b0`` cancels from
        the drawn direction and from the weight, and its derivative is zero
        by structure.
        """
        refuse_gradients(
            "HarveyShackBSDF",
            "zero by structure: the lobe is normalised over the reachable "
            "directions, so b0 cancels from the drawn direction and from the "
            "weight (the scatter level enters through the surface's "
            "scatter_fraction)",
            b0=b0,
        )
        self.b0 = float(b0)
        self.l0 = as_attachable_param(l0)
        self.s = as_attachable_param(s)
        self._l0 = host_float(l0)
        self._s = host_float(s)
        self.transmissive_fraction, self._tau = attachable_fraction(
            "HarveyShackBSDF", "transmissive_fraction", transmissive_fraction
        )
        # d log T / d(l0, s) per incidence row, built beside the TIS table
        # only when l0 or s carries a derivative.
        self._dlog_tis: tuple[np.ndarray, np.ndarray] | None = None
        self._beta_grid: np.ndarray | None = None
        self._cdf_grid: np.ndarray | None = None
        self._tis: float | None = None
        self._tis_disk: float | None = None
        self._incidence_grid: np.ndarray | None = None
        self._hemi_grid: np.ndarray | None = None

    def _abg(self, beta: np.ndarray) -> np.ndarray:
        """Evaluate the ABg BSDF at a direction-cosine offset.

        Args:
            beta: Magnitude of the direction-cosine offset from specular.

        Returns:
            BSDF value [sr^-1].
        """
        return self.b0 / (1.0 + (beta / self._l0) ** self._s)

    def _dlog_abg(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``d log BSDF / d l0`` and ``d log BSDF / d s`` at an offset, on the host.

        With ``x = (beta / l0)^s``: ``(s / l0) x / (1 + x)`` and
        ``-ln(beta / l0) x / (1 + x)``; both 0 at ``beta = 0``.
        """
        beta = np.asarray(beta, dtype=np.float64)
        pos = beta > 0.0
        safe = np.where(pos, beta, 1.0)
        frac = np.where(pos, 1.0 / (1.0 + (self._l0 / safe) ** self._s), 0.0)
        d_l0 = (self._s / self._l0) * frac
        d_s = np.where(pos, -np.log(safe / self._l0) * frac, 0.0)
        return d_l0, d_s

    @property
    def _score_attached(self) -> bool:
        return be.is_torch_tensor(self.l0) or be.is_torch_tensor(self.s)

    def _radial_nodes(self) -> np.ndarray:
        """Nodes of the radial table: 0, a geometric ladder to 2, and 1.

        The ladder starts at ``1e-4 * min(l0, 1)``, well inside the lobe's
        plateau, so every cell spans the same small ratio of offsets and the
        knee at ``l0`` is resolved whatever its value. The node 1 is exact,
        so the normal-incidence reach ``|beta - beta0| = 1`` is read off the
        table without interpolation.
        """
        lo = 1e-4 * min(self._l0, 1.0)
        ladder = np.geomspace(lo, _BETA_MAX, _TABLE_SIZE - 2)
        nodes = np.unique(np.concatenate([[0.0, 1.0, _BETA_MAX], ladder]))
        return nodes

    def _build_tables(self) -> None:
        """Build the radial CDF, its inverse, and the hemispherical TIS table.

        In direction-cosine space the projected solid angle is
        ``cos(theta) dOmega = d(beta_x) d(beta_y)``, so about the reference
        direction ``beta0`` the radial measure is ``2 * pi * delta d(delta)``
        for the offset ``delta = |beta - beta0|`` and

            C(delta) = integral_0^delta BSDF(t) * 2 * pi * t dt

        is the lobe's integral over the offsets up to ``delta``, tabulated to
        ``_BETA_MAX`` by Gauss-Legendre quadrature on every cell.

        A scattered direction exists only where ``|beta| < 1``: the unit disk
        about the origin, which about ``beta0`` is the offsets with
        ``delta < delta_max(psi) = -beta0.e + sqrt(1 - (beta0 x e)^2)`` along
        the azimuth ``e = (cos psi, sin psi)``. The lobe's hemispherical
        integral at that incidence -- the fraction a lossless surface with
        this BSDF scatters, chapter 04 section 4.5 -- is therefore

            TIS(beta0) = (1 / 2 pi) integral_0^{2 pi} C(delta_max(psi)) dpsi,

        which at normal incidence (``delta_max = 1`` for every azimuth) is
        ``C(1)``: chapter 11 section 11.4.24's ``pi b0 l^2 ln(1 + 1/l^2)``
        for the slope-2 lobe. It depends on ``|beta0|`` only and is
        tabulated over the incidence angle. Integrating to ``_BETA_MAX``
        instead counts offsets no propagating direction reaches (issue 18 of
        the research repository: 15 percent too much at ``l0 = 0.01``).

        ``_cdf_grid`` is ``C / C(_BETA_MAX)`` at the nodes ``_beta_grid``
        (the inverse CDF the sampler reads), ``_hemi_grid`` is
        ``TIS(beta0) / C(_BETA_MAX)`` at ``|beta0| = _incidence_grid``,
        ``_tis`` is the normal-incidence TIS and ``_tis_disk`` is
        ``C(_BETA_MAX)``.
        """
        beta = self._radial_nodes()
        x_gl, w_gl = np.polynomial.legendre.leggauss(_GAUSS_POINTS)
        a = beta[:-1, None]
        h = np.diff(beta)[:, None]
        t = a + 0.5 * h * (x_gl[None, :] + 1.0)
        cell = 0.5 * h[:, 0] * ((self._abg(t) * 2.0 * np.pi * t) @ w_gl)
        cdf = np.concatenate([[0.0], np.cumsum(cell)])

        total = float(cdf[-1])
        self._tis_disk = total
        self._beta_grid = beta
        # A degenerate (all-zero) integrand would leave the table flat, so
        # fall back to a uniform CDF in that case.
        self._cdf_grid = cdf / total if total > 0.0 else np.linspace(0, 1, cdf.size)
        i_one = int(np.flatnonzero(beta == 1.0)[0])
        self._tis = float(cdf[i_one])

        # The hemispherical table, from the same piecewise-linear C the
        # sampler reads, so the weight it divides by is the mean of the
        # numerator over the azimuth to the quadrature's accuracy.
        theta = np.linspace(0.0, 0.5 * np.pi, _INCIDENCE_SIZE)
        g = np.sin(theta)
        g[-1] = 1.0
        psi = (np.arange(_AZIMUTH_SIZE) + 0.5) * (np.pi / _AZIMUTH_SIZE)
        proj = g[:, None] * np.cos(psi)[None, :]
        perp = g[:, None] * np.sin(psi)[None, :]
        dmax = -proj + np.sqrt(np.maximum(1.0 - perp * perp, 0.0))
        hemi = np.interp(dmax, beta, self._cdf_grid).mean(axis=1)
        # Normal incidence: delta_max is 1 on every azimuth, so the mean is
        # the node value itself; set it as such rather than as a sum of
        # 2048 equal terms, so the sampler's weight there is exactly one.
        hemi[0] = self._cdf_grid[i_one]
        self._incidence_grid = g
        self._hemi_grid = hemi

        if self._score_attached:
            # d log T / d(l0, s) per incidence row (chapter 09 section
            # 9.14.3): the same Gauss-Legendre cells for d C / d theta, the
            # same azimuth midpoints and the same interpolation in the
            # offset, divided by the row's T from the same quadrature.
            d_l0_t, d_s_t = self._dlog_abg(t)
            base = self._abg(t) * 2.0 * np.pi * t
            dlog = []
            for d_t in (d_l0_t, d_s_t):
                dcell = 0.5 * h[:, 0] * ((base * d_t) @ w_gl)
                dcdf = np.concatenate([[0.0], np.cumsum(dcell)])
                num = np.interp(dmax, beta, dcdf).mean(axis=1)
                den = np.interp(dmax, beta, cdf).mean(axis=1)
                row = num / den
                row[0] = dcdf[i_one] / cdf[i_one]
                dlog.append(row)
            self._dlog_tis = (dlog[0], dlog[1])

    def _inverse_cdf(self, u):
        """Radial offset for a uniform draw, from the tabulated inverse CDF.

        The table is monotone in the CDF, so the lookup is one
        ``searchsorted`` and one linear interpolation between the bracketing
        grid points -- the same arithmetic ``numpy.interp`` performs, in the
        active backend and on the table uploaded beside the ray state rather
        than on a host copy of the draws.

        A flat stretch of the CDF (an interval the lobe puts no weight in)
        has a zero span; the fraction is taken as zero there, which returns
        the lower grid point, as an interpolation between two equal
        abscissae must.

        Args:
            u: Uniform draws in [0, 1), shape (N,).

        Returns:
            The radial direction-cosine offset for each draw, shape (N,).
        """
        cdf = resident_table(self, "cdf", self._cdf_grid)
        beta = resident_table(self, "beta", self._beta_grid)
        j = clamp_int(
            be.searchsorted(cdf, u, side="right") - 1, 0, self._cdf_grid.size - 2
        )
        c0 = cdf[j]
        c1 = cdf[j + 1]
        span = c1 - c0
        positive = span > 0
        frac = be.where(
            positive,
            (u - c0) / be.where(positive, span, be.ones_like(span)),
            be.zeros_like(span),
        )
        b0 = beta[j]
        return b0 + frac * (beta[j + 1] - b0)

    def _forward_cdf(self, delta):
        """``C(delta) / C(_BETA_MAX)`` from the table, in the backend.

        The same piecewise-linear function :meth:`_inverse_cdf` inverts, so
        an offset drawn below ``C(delta_max)`` is below ``delta_max``.
        """
        cdf = resident_table(self, "cdf", self._cdf_grid)
        beta = resident_table(self, "beta", self._beta_grid)
        return _lerp(beta, cdf, delta, self._beta_grid.size)

    def _hemispherical(self, g):
        """``TIS(|beta0|) / C(_BETA_MAX)`` from the table, in the backend."""
        grid = resident_table(self, "incidence", self._incidence_grid)
        hemi = resident_table(self, "hemi", self._hemi_grid)
        return _lerp(grid, hemi, g, self._incidence_grid.size)

    @property
    def total_integrated_scatter(self) -> float:
        """Fraction of incident power scattered at normal incidence, in [0, 1].

        The lobe's hemispherical integral, ``C(1)`` of :meth:`_build_tables`
        (for the slope-2 lobe, ``pi b0 l0^2 ln(1 + 1/l0^2)``), not its
        integral over the direction-cosine disk of radius 2.

        Returns:
            TIS, clipped to 1.0.
        """
        if self._tis is None:
            self._build_tables()
        return min(float(self._tis), 1.0)

    def total_integrated_scatter_at(self, theta_i) -> np.ndarray:
        """Fraction of incident power scattered at incidence ``theta_i``.

        The hemispherical integral of the lobe centred on the reference
        direction of that incidence (chapter 04 R-04-9 asks for it over a
        set of incidence angles): ``TIS(sin(theta_i))`` of
        :meth:`_build_tables`, read from its table.

        Args:
            theta_i: Angle of the reference direction from the normal
                [rad], scalar or array, in [0, pi/2].

        Returns:
            TIS at each angle, clipped to 1.0, as NumPy values.
        """
        if self._hemi_grid is None:
            self._build_tables()
        g = np.sin(np.clip(np.asarray(theta_i, dtype=np.float64), 0.0, 0.5 * np.pi))
        tis = np.interp(g, self._incidence_grid, self._hemi_grid) * self._tis_disk
        return np.minimum(tis, 1.0)

    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: NSQRng,
        ray_id: np.ndarray,
        bounce: np.ndarray,
        frame=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample scattered directions from the ABg lobe about a reference ray.

        The scatter offset is drawn directly from the ABg distribution in
        direction-cosine space, restricted to the directions that exist: the
        azimuth is uniform, and the radial magnitude ``|beta - beta0|`` comes
        from the tabulated inverse CDF of ``BSDF(beta) * 2 * pi * beta``
        truncated at the largest offset that azimuth reaches inside the unit
        disk. The weight ``C(delta_max(psi)) / TIS(|beta0|)`` then makes the
        samples the lobe normalised over the reachable directions, with
        expectation one: exactly one at normal incidence, a sampling weight
        off it. So the surface acts as a lossless mirror (or diffuser sheet)
        whose reflection (or straight-through transmission) is blurred by
        the ABg lobe: a polished surface (small ``l0``) stays
        near-specular/near-collimated, a rough one spreads.

        A per-ray draw against :attr:`transmissive_fraction` picks the
        reference ray the lobe is centred on: the specular reflection for a
        reflective draw, or the undeviated straight-through ray (the
        incident direction itself, unrefracted) for a transmissive one. Both
        references are expressed in the same tangent frame about ``normals``,
        so the existing ``spec_normal_sign`` (the reference ray's own sign
        against ``normals``) places the reconstructed sample on the correct
        side automatically -- no separate branch is needed downstream.

        To model the physically scaled picture instead, a bright specular beam
        plus a faint scatter halo, set the surface's ``scatter_fraction`` to
        :attr:`total_integrated_scatter`. Then a TIS fraction of the rays enter
        the halo and the rest reflect specularly.

        Sampling the lobe directly matters: drawing from a cosine-weighted
        hemisphere and correcting with a clipped ``BSDF / cos`` weight (the
        previous approach) puts essentially every sample where the ABg lobe is
        negligible, which drove surface throughput to ~1e-7 of the incident
        flux and made the model behave as a black absorber.

        Sampling is detached, and stays on whichever array library and
        device the ray state is on: the inverse-CDF table is uploaded once
        and read with a device-side ``searchsorted``.

        Args:
            num_rays: Number of rays.
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).
            rng: Keyed PCG32 RNG.
            ray_id: Per-ray identifiers, shape (N,).
            bounce: Per-ray bounce/step index, shape (N,).
            frame: The surface's rotation (local to global), or None for the
                world axes: the tangent frame the offsets are drawn in
                (``lambertian._orthonormal_basis``).

        Returns:
            (scattered_dirs, flux_weights, transmitted).
        """
        if self._beta_grid is None:
            self._build_tables()

        # The lobe is a stochastic choice, so it is detached -- but with
        # detach(), not by copying the normal and the direction to the host.
        # Every line below is elementwise arithmetic that runs wherever the
        # ray state lives.
        n_be = detached(normals)
        d_be = detached(incident_dirs)

        # Specular reflection: d - 2(d.n)n
        cos_i = (d_be * n_be).sum(axis=1, keepdims=True)
        d_spec = d_be - 2.0 * cos_i * n_be

        # Per-ray reflective-vs-transmissive lobe draw: the reference
        # ray the ABg blur is centred on. d_be itself (unrefracted) is
        # already a unit vector; only d_spec needs the below norm-guard.
        if self._tau > 0.0:
            u_lobe = rng.uniform(ray_id, bounce, EventSlot.BSDF_LOBE_BRANCH)
            transmitted = u_lobe < self._tau
            d_ref = be.where(transmitted[:, None], d_be, d_spec)
        else:
            transmitted = backend_bool_full((n_be.shape[0],), False, like=n_be)
            d_ref = d_spec

        # Rays that hit nothing carry zero direction and normal, so the
        # reference vector is zero. Guard the normalisation: a NaN here
        # propagates into the returned weights for every ray.
        d_ref_norm = (d_ref * d_ref).sum(axis=1, keepdims=True) ** 0.5
        # Zero-length-vector rejection: k ulps of 1 (a direction is O(1)),
        # not a bare 1e-12 -- docs/theory/08_precision.md sec 8.7. Taken in
        # the working dtype, so a float32 trace gets the float32 floor.
        norm_floor = 8 * _tol.ulp(be.ones_like(d_ref_norm))
        usable = d_ref_norm > norm_floor
        valid = usable[:, 0]
        safe_norm = be.where(usable, d_ref_norm, be.ones_like(d_ref_norm))
        d_ref = be.where(usable, d_ref / safe_norm, be.zeros_like(d_ref))

        from optiland.nonsequential.bsdf.lambertian import (  # noqa: PLC0415
            _orthonormal_basis,
        )

        t_vec, b_vec = _orthonormal_basis(n_be, frame)

        # Reference direction expressed in the local tangent frame.
        beta0_x = (d_ref * t_vec).sum(axis=1)
        beta0_y = (d_ref * b_vec).sum(axis=1)
        raw_sign = be.sign((d_ref * n_be).sum(axis=1))
        ref_normal_sign = be.where(
            raw_sign == 0.0, be.ones_like(raw_sign), raw_sign
        )

        # Azimuth uniform; the radial offset from the inverse CDF truncated
        # at the reach of this azimuth, delta_max(psi), so every sample is a
        # propagating direction and none is discarded (issue 18 of the
        # research repository). With e = (cos psi, sin psi) in the tangent
        # frame, |beta0 + delta e| < 1 exactly when
        # delta < -beta0.e + sqrt(1 - (beta0 x e)^2).
        u_radial = rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)
        u_azimuth = rng.uniform(ray_id, bounce, EventSlot.BSDF_U2)
        psi = 2.0 * be.pi * u_azimuth
        cos_psi = be.cos(psi)
        sin_psi = be.sin(psi)
        proj = beta0_x * cos_psi + beta0_y * sin_psi
        perp = beta0_x * sin_psi - beta0_y * cos_psi
        delta_max = -proj + be.sqrt(
            be.maximum(1.0 - perp * perp, be.zeros_like(perp))
        )
        c_max = self._forward_cdf(delta_max)
        delta = self._inverse_cdf(u_radial * c_max)

        beta_x = beta0_x + delta * cos_psi
        beta_y = beta0_y + delta * sin_psi

        # The weight that makes the truncated draw the lobe restricted to the
        # reachable directions: the draw has density BSDF / C(delta_max(psi))
        # per unit projected solid angle, the target BSDF / TIS(|beta0|), so
        # w = C(delta_max(psi)) / TIS(|beta0|), whose mean over the azimuth is
        # one (_build_tables). At normal incidence delta_max is 1 on every
        # azimuth and w is exactly one; off it, w is a sampling weight.
        g = be.sqrt(beta0_x * beta0_x + beta0_y * beta0_y)
        g = be.where(g < 1.0, g, be.ones_like(g))
        weight = c_max / self._hemispherical(g)

        # A sample can still land on or past the unit circle by rounding at
        # the truncation (measure zero); it keeps the reference direction.
        beta_sq = beta_x**2 + beta_y**2
        reachable = (beta_sq < 1.0) & valid
        normal_comp = be.sqrt(be.maximum(1.0 - beta_sq, be.zeros_like(beta_sq)))

        scattered = (
            beta_x[:, None] * t_vec
            + beta_y[:, None] * b_vec
            + (ref_normal_sign * normal_comp)[:, None] * n_be
        )
        scattered = be.where(reachable[:, None], scattered, d_ref)

        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        norms_ok = norms > 8 * _tol.ulp(be.ones_like(norms))
        scattered = be.where(
            norms_ok,
            scattered / be.where(norms_ok, norms, be.ones_like(norms)),
            be.zeros_like(scattered),
        )

        # The lobe redistributes energy rather than removing it: its weight
        # has expectation one at every incidence. The physical scatter level
        # is applied via ``scatter_fraction``, for which
        # :attr:`total_integrated_scatter` is the natural value.
        flux_weights = be.where(valid, weight, be.zeros_like(weight))

        if self._score_attached:
            flux_weights = flux_weights + flux_weights * self._score(delta, g)
        # The branch's share, attached when the fraction carries a gradient
        # (exactly 1 in value, so the weight keeps its bits).
        gate = lobe_branch_gate(
            self.transmissive_fraction, self._tau, transmitted, flux_weights
        )
        if gate is not None:
            flux_weights = flux_weights * gate

        return scattered, flux_weights, transmitted

    def _score(self, delta, g):
        """The weight's attached dependence on ``l0`` and ``s``, zero in value.

        ``sum_k (theta_k - sg(theta_k)) * D_k`` with ``D_k = d log f / d
        theta_k`` at the drawn offset ``delta`` and the incidence ``g =
        |beta0|``, evaluated detached at the host values: ``d log BSDF``
        minus ``d log TIS`` (chapter 09 section 9.14.3). The factor
        ``theta - sg(theta)`` is exactly zero, so ``w + w * score`` is ``w``
        to the bit, and its derivative is ``w * D_k`` (the likelihood-ratio
        path: the direction stays drawn at the host values).

        Args:
            delta: Drawn radial offsets, shape (N,), detached.
            g: Incidence ``|beta0|`` per ray, shape (N,), detached.

        Returns:
            The score, shape (N,).
        """
        if self._dlog_tis is None:
            self._build_tables()
        l0, s = self._l0, self._s
        pos = delta > 0.0
        safe = be.where(pos, delta, be.ones_like(delta))
        frac = be.where(
            pos, 1.0 / (1.0 + (l0 / safe) ** s), be.zeros_like(delta)
        )
        grid = resident_table(self, "incidence", self._incidence_grid)
        size = self._incidence_grid.size
        score = be.zeros_like(delta)
        if be.is_torch_tensor(self.l0):
            table = resident_table(self, "dlog_tis_l0", self._dlog_tis[0])
            d_l0 = (s / l0) * frac - _lerp(grid, table, g, size)
            score = score + onto_apple_gpu(self.l0 - detached(self.l0), d_l0) * d_l0
        if be.is_torch_tensor(self.s):
            table = resident_table(self, "dlog_tis_s", self._dlog_tis[1])
            log_ratio = be.where(pos, be.log(safe / l0), be.zeros_like(delta))
            d_s = -log_ratio * frac - _lerp(grid, table, g, size)
            score = score + onto_apple_gpu(self.s - detached(self.s), d_s) * d_s
        return score

    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Return the fraction of incident power redistributed by the lobe.

        The lobe is normalised over the directions reachable from every
        incidence, so its weight has expectation one and the surface loses
        nothing: this is 1.0, the albedo the surface books the sampling
        weight's residual against. The ABg scatter level itself is
        :attr:`total_integrated_scatter`, which is what a
        ``scatter_fraction`` should be set to for a physically scaled halo.

        Args:
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).

        Returns:
            Approximate reflectance values, shape (N,).
        """
        return be.ones(incident_dirs.shape[0])
