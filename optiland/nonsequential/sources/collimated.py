"""Collimated source for Non-Sequential Raytracing.

Parallel beam with circular aperture. Propagates along the local +z axis.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential._utils import (
    as_attachable_param,
    as_detached_param,
    host_float,
)
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.rng import EventSlot
from optiland.nonsequential.sources.base import (
    BaseNSQSource,
    Spectrum,
    medium_index_on_host,
)

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.rng import NSQRng

# Bounded rejection-sampling attempts for the truncated Gaussian disk. Each
# round accepts ~1 - exp(-2) ~ 86% of samples for the default sigma =
# radius / 2, so 32 rounds leaves an astronomically small failure
# probability; any ray still rejected after that is clamped to the
# boundary (a deterministic, keyed fallback -- never an unbounded loop).
_MAX_GAUSSIAN_ATTEMPTS = 32


class CollimatedSource(BaseNSQSource):
    """Parallel collimated beam with circular aperture.

    All rays propagate along the local +z axis (after coordinate
    transformation to global frame). Intensity profile is either
    top-hat (uniform) or truncated Gaussian.

    Attributes:
        cs: Coordinate system (local z = beam propagation axis).
        spectrum: Wavelength distribution.
        total_flux: Total beam flux [W].
        aperture_radius: Beam aperture radius [mm].
        profile: Intensity profile ('tophat' or 'gaussian').
        gaussian_sigma: Gaussian sigma [mm] (used when profile='gaussian').
        medium: Medium the source is embedded in.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        spectrum: Spectrum,
        total_flux: float = 1.0,
        aperture_radius: float = 5.0,
        profile: Literal["tophat", "gaussian"] = "tophat",
        gaussian_sigma: float | None = None,
        medium=None,
        profile_gradient: Literal["refuse", "implicit"] = "implicit",
    ) -> None:
        """Initialize CollimatedSource.

        Args:
            cs: Coordinate system.
            spectrum: Wavelength distribution.
            total_flux: Total beam flux [W].
            aperture_radius: Beam aperture radius [mm].
            profile: Intensity profile ('tophat' or 'gaussian').
            gaussian_sigma: Gaussian standard deviation [mm].
                Defaults to aperture_radius / 2 if None.
            medium: Medium the source is embedded in (default: vacuum).
            profile_gradient: ``"implicit"`` (the default since the
                maintainer's ruling of 2026-10-01): a Gaussian beam's sigma
                and radius are attached by the implicit reparameterisation
                of the truncated profile
                (:func:`~optiland.nonsequential.parameter_register
                .truncated_gaussian_tangents`; the research repository's
                chapter 09 section 9.13.1). ``"refuse"``: they raise when
                they carry a gradient. A top-hat beam's radius is attached
                either way; a sigma given to a top-hat beam reaches nothing,
                so with ``"implicit"`` the trace raises it as a dead
                parameter, and with ``"refuse"`` the constructor refuses it.
        """
        super().__init__(cs, spectrum, total_flux)
        if profile_gradient not in ("refuse", "implicit"):
            raise ValueError(
                "CollimatedSource: profile_gradient must be 'refuse' or 'implicit', "
                f"got {profile_gradient!r}."
            )
        self.profile_gradient = profile_gradient
        implicit = profile_gradient == "implicit"
        gaussian_refused = profile == "gaussian" and not implicit
        # The aperture radius of a top-hat beam may carry a derivative: the
        # trace attaches the emission points to it by the change of variables
        # (chapter 09 section 9.7, R-09-4). A truncated Gaussian's radius is
        # its truncation edge. Held at a fixed value of the radial
        # distribution function, the edge does not move, and the implicit
        # reparameterisation of chapter 09 section 9.13.1 carries the
        # derivative in the radius and in sigma. That is the default
        # (profile_gradient="implicit", the maintainer's ruling of 2026-10-01);
        # a beam built with profile_gradient="refuse" refuses both.
        if gaussian_refused:
            self.aperture_radius = as_detached_param(
                aperture_radius,
                "aperture_radius",
                "CollimatedSource",
                reason=(
                    "it truncates the Gaussian profile and the beam was built "
                    "with profile_gradient='refuse' (pass "
                    "profile_gradient='implicit' to attach it by the implicit "
                    "reparameterisation)"
                ),
            )
        else:
            self.aperture_radius = as_attachable_param(aperture_radius)
        self.profile = profile
        # With the implicit reparameterisation a sigma that is not given
        # follows the radius (sigma = R / 2): its value is the float, and the
        # trace adds the radius' tangent to it (d sigma = d R / 2). A sigma
        # given to a top-hat beam is kept as given and enters nothing; the
        # trace's dead-parameter check raises on it (R-09-5).
        self._sigma_follows_radius = gaussian_sigma is None
        if gaussian_sigma is None:
            self.gaussian_sigma = host_float(self.aperture_radius) / 2.0
        elif implicit:
            self.gaussian_sigma = as_attachable_param(gaussian_sigma)
        else:
            self.gaussian_sigma = as_detached_param(
                gaussian_sigma,
                "gaussian_sigma",
                "CollimatedSource",
                reason=(
                    "the beam was built with profile_gradient='refuse' (pass "
                    "profile_gradient='implicit' to attach it)"
                ),
            )
        self.medium = medium

    def generate(self, ray_id: np.ndarray, rng: NSQRng) -> NSQRayBundle:
        """Generate collimated rays in global coordinates.

        Positions are sampled within the circular aperture. All directions
        are along the local +z axis.

        Args:
            ray_id: Unique identifiers for the rays to generate, shape (N,).
            rng: Keyed PCG32 RNG.

        Returns:
            NSQRayBundle with all rays alive and parallel directions.
        """
        num_rays = len(ray_id)
        bounce0 = np.zeros(num_rays, dtype=np.int32)
        translation, rot = _get_transform(self.cs)

        if self.profile == "gaussian":
            # Sample truncated Gaussian on disk
            lx, ly = self._sample_gaussian_disk(ray_id, bounce0, rng)
        else:
            # Uniform disk sampling
            u1 = to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1))
            u2 = to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U2))
            r = host_float(self.aperture_radius) * np.sqrt(u1)
            phi = 2.0 * np.pi * u2
            lx = r * np.cos(phi)
            ly = r * np.sin(phi)

        lz_pos = np.zeros(num_rays)

        # All rays point in local +z direction
        dirs_local = np.zeros((num_rays, 3))
        dirs_local[:, 2] = 1.0

        pos_local = np.stack([lx, ly, lz_pos], axis=1)

        # Transform to global frame
        pos_global = pos_local @ rot.T + translation
        dirs_global = dirs_local @ rot.T

        # Sample wavelengths [µm]
        wavelengths = self.spectrum.sample(ray_id, bounce0, rng)
        # Divide preserves torch tensor when total_flux is a Tensor (for autograd)
        flux_per_ray = self.total_flux / num_rays

        # Initialize n_current/k_current from medium if provided
        medium = getattr(self, "medium", None)
        if medium is not None:
            n_init, k_init = medium_index_on_host(medium, wavelengths, num_rays)
        else:
            n_init = np.ones(num_rays)
            k_init = np.zeros(num_rays)

        return NSQRayBundle(
            x=pos_global[:, 0].copy(),
            y=pos_global[:, 1].copy(),
            z=pos_global[:, 2].copy(),
            L=dirs_global[:, 0].copy(),
            M=dirs_global[:, 1].copy(),
            N=dirs_global[:, 2].copy(),
            # be.ones * flux_per_ray keeps the torch autograd graph when
            # total_flux is a Tensor; falls back to numpy when it's a float.
            flux=be.ones(num_rays) * flux_per_ray,
            wavelength=wavelengths,
            n_current=n_init,
            bounce=bounce0,
            alive=np.ones(num_rays, dtype=bool),
            ray_id=ray_id,
            k_current=k_init,
        )

    def _sample_gaussian_disk(
        self, ray_id: np.ndarray, bounce0: np.ndarray, rng: NSQRng
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample positions from a truncated Gaussian disk (rejection sampling).

        Each ray gets its own bounded rejection sequence keyed by its own
        id: attempt ``k`` draws a fresh (u1, u2) pair via
        ``offset=k`` on the same event slots, so the sequence is a pure
        function of the ray id and never depends on how many other rays
        are still rejecting in the same batch.

        Args:
            ray_id: Unique identifiers for the rays to generate, shape (N,).
            bounce0: Zero bounce array, shape (N,).
            rng: Keyed PCG32 RNG.

        Returns:
            Tuple (x, y) of position arrays, each shape (N,).
        """
        num_rays = len(ray_id)
        max_r2 = host_float(self.aperture_radius) ** 2
        lx = np.zeros(num_rays)
        ly = np.zeros(num_rays)
        pending = np.ones(num_rays, dtype=bool)

        for attempt in range(_MAX_GAUSSIAN_ATTEMPTS):
            if not pending.any():
                break
            u1 = to_numpy(
                rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1, offset=attempt)
            )
            u2 = to_numpy(
                rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U2, offset=attempt)
            )
            # Box-Muller transform: (u1, u2) -> independent standard normals.
            r_bm = np.sqrt(-2.0 * np.log(np.maximum(u1, 1e-300)))
            theta_bm = 2.0 * np.pi * u2
            sigma = host_float(self.gaussian_sigma)
            x = sigma * r_bm * np.cos(theta_bm)
            y = sigma * r_bm * np.sin(theta_bm)
            accept = pending & (x**2 + y**2 <= max_r2)
            lx = np.where(accept, x, lx)
            ly = np.where(accept, y, ly)
            pending = pending & ~accept

        if pending.any():
            # Exhausted the attempt budget (astronomically unlikely): clamp
            # to the boundary along the last-drawn direction rather than
            # looping unboundedly or silently keeping an out-of-aperture
            # sample.
            u_fallback = to_numpy(
                rng.uniform(
                    ray_id, bounce0, EventSlot.SOURCE_U2, offset=_MAX_GAUSSIAN_ATTEMPTS
                )
            )
            theta_fallback = 2.0 * np.pi * u_fallback
            r_fallback = host_float(self.aperture_radius)
            lx = np.where(pending, r_fallback * np.cos(theta_fallback), lx)
            ly = np.where(pending, r_fallback * np.sin(theta_fallback), ly)

        return lx, ly
