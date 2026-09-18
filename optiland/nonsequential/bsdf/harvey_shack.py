"""Harvey-Shack (ABg) BSDF for Non-Sequential Raytracing.

Models micro-roughness scatter from optical surfaces.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential._utils import clamp_int
from optiland.nonsequential.bsdf.base import BaseBSDF
from optiland.nonsequential.components.base import resident_table
from optiland.nonsequential.components.sampling_support import detached
from optiland.nonsequential.ray_bundle import backend_bool_full
from optiland.nonsequential.rng import EventSlot

if TYPE_CHECKING:
    from optiland.nonsequential.rng import NSQRng

# Largest reachable direction-cosine offset: both the specular and the
# scattered direction lie in the unit disk, so |beta - beta0| <= 2.
_BETA_MAX = 2.0
# Resolution of the tabulated radial inverse CDF.
_TABLE_SIZE = 4096


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
    """

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
        """
        self.b0 = float(b0)
        self.l0 = float(l0)
        self.s = float(s)
        self.transmissive_fraction = float(transmissive_fraction)
        self._beta_grid: np.ndarray | None = None
        self._cdf_grid: np.ndarray | None = None
        self._tis: float | None = None

    def _abg(self, beta: np.ndarray) -> np.ndarray:
        """Evaluate the ABg BSDF at a direction-cosine offset.

        Args:
            beta: Magnitude of the direction-cosine offset from specular.

        Returns:
            BSDF value [sr^-1].
        """
        return self.b0 / (1.0 + (beta / self.l0) ** self.s)

    def _build_tables(self) -> None:
        """Build the radial inverse-CDF table and the total integrated scatter.

        In direction-cosine space the projected solid angle is
        ``cos(theta) dOmega = d(beta_x) d(beta_y)``, so the radial measure is
        ``2 * pi * beta d(beta)`` and

            TIS = integral of BSDF(beta) * 2 * pi * beta d(beta)

        over the reachable range ``beta <= _BETA_MAX``. TIS is the fraction of
        incident power the surface scatters, and it is what each scattered ray
        carries as its weight.
        """
        beta = np.linspace(0.0, _BETA_MAX, _TABLE_SIZE)
        radial = self._abg(beta) * 2.0 * np.pi * beta

        # Cumulative trapezoidal integral.
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (radial[1:] + radial[:-1]) * np.diff(beta))]
        )
        self._tis = float(cdf[-1])
        # Normalise to a proper CDF. A degenerate (all-zero) integrand would
        # leave the table flat, so fall back to a uniform CDF in that case.
        self._cdf_grid = cdf / cdf[-1] if cdf[-1] > 0.0 else np.linspace(0, 1, cdf.size)
        self._beta_grid = beta

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

    @property
    def total_integrated_scatter(self) -> float:
        """Fraction of incident power scattered by this surface, in [0, 1].

        Returns:
            TIS, clipped to 1.0.
        """
        if self._tis is None:
            self._build_tables()
        return min(float(self._tis), 1.0)

    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: NSQRng,
        ray_id: np.ndarray,
        bounce: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample scattered directions from the ABg lobe about a reference ray.

        The scatter offset is drawn directly from the ABg distribution in
        direction-cosine space: the radial magnitude ``|beta - beta0|`` comes
        from a tabulated inverse CDF of ``BSDF(beta) * 2 * pi * beta`` and the
        azimuth is uniform. Rays keep their full flux, so the surface acts as
        a mirror (or diffuser sheet) whose reflection (or straight-through
        transmission) is blurred by the ABg lobe: a polished surface (small
        ``l0``) stays near-specular/near-collimated, a rough one spreads.

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
        if self.transmissive_fraction > 0.0:
            u_lobe = rng.uniform(ray_id, bounce, EventSlot.BSDF_LOBE_BRANCH)
            transmitted = u_lobe < self.transmissive_fraction
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

        t_vec, b_vec = _orthonormal_basis(n_be)

        # Reference direction expressed in the local tangent frame.
        beta0_x = (d_ref * t_vec).sum(axis=1)
        beta0_y = (d_ref * b_vec).sum(axis=1)
        raw_sign = be.sign((d_ref * n_be).sum(axis=1))
        ref_normal_sign = be.where(
            raw_sign == 0.0, be.ones_like(raw_sign), raw_sign
        )

        # Radial offset from the tabulated inverse CDF; azimuth uniform.
        u_radial = rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)
        u_azimuth = rng.uniform(ray_id, bounce, EventSlot.BSDF_U2)
        delta = self._inverse_cdf(u_radial)
        psi = 2.0 * be.pi * u_azimuth

        beta_x = beta0_x + delta * be.cos(psi)
        beta_y = beta0_y + delta * be.sin(psi)

        # An offset can land outside the unit disk, i.e. on the wrong side of
        # the reference ray's own hemisphere. Those samples are not
        # physically reachable: keep the reference direction and give them
        # zero weight rather than folding them back, which would distort the
        # lobe.
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

        # Full flux: the lobe redistributes energy rather than removing it.
        # The physical scatter level is applied via ``scatter_fraction``,
        # for which :attr:`total_integrated_scatter` is the natural value.
        flux_weights = be.where(
            reachable, be.ones_like(beta_sq), be.zeros_like(beta_sq)
        )

        return scattered, flux_weights, transmitted

    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Return the fraction of incident power redistributed by the lobe.

        The sampler conserves energy (rays keep full flux and are only
        redirected), so this is 1.0. The ABg scatter level itself is
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
