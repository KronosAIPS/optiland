"""Far-field detector for Non-Sequential Raytracing.

Accumulates angular flux distribution in the far field.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential._tally import Tally, accumulate, masked_count, masked_sum
from optiland.nonsequential._utils import clamp_int, to_numpy
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.detectors.base import (
    BaseDetector,
    _accumulate_into,
    _new_bin_accumulator,
    bin_values,
)
from optiland.nonsequential.results.far_field_pattern import FarFieldPattern

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class FarFieldDetector(BaseDetector):
    """Accumulates angular flux distribution in the far field.

    Records ray directions at the detector surface and bins them into a
    polar (theta, phi) histogram.

    Attributes:
        cs: Coordinate system.
        theta_max_deg: Maximum polar angle to record [deg].
        num_bins_theta: Number of polar angle bins.
        num_bins_phi: Number of azimuthal angle bins.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        theta_max_deg: float,
        num_bins_theta: int,
        num_bins_phi: int,
        aperture_radius: float = 1e6,
        name: str = "",
        absorb: bool = True,
        side: str = "both",
        reflection_bins: int = 0,
    ) -> None:
        """Initialize FarFieldDetector.

        Args:
            cs: Coordinate system for detector position/orientation.
            theta_max_deg: Maximum polar half-angle to record [deg].
            num_bins_theta: Number of polar angle bins.
            num_bins_phi: Number of azimuthal angle bins.
            aperture_radius: Detector aperture radius [mm] (default: very large).
            name: Optional label.
            absorb: Whether a hit terminates the ray (default True).
            side: Which side of the plane is live -- ``"both"`` (default),
                ``"front"``, or ``"back"``. See
                :meth:`BaseDetector.intersect`.
            reflection_bins: Also book the arriving flux by reflection count
                (0, the default, keeps no histogram). See
                :meth:`BaseDetector.record_reflections`.
        """
        geometry = FinitePlaneGeometry(aperture_radius=aperture_radius)
        super().__init__(
            cs,
            geometry,
            name=name,
            absorb=absorb,
            side=side,
            reflection_bins=reflection_bins,
        )
        self.num_bins_theta = int(num_bins_theta)
        self.num_bins_phi = int(num_bins_phi)

        # Flat accumulation buffer: shape (num_bins_theta * num_bins_phi,).
        # The widest float the device has (float64; float32 on Apple's mps),
        # on the active backend and device, mutated in place by every
        # record() call -- see accumulator_dtype in detectors/base.py.
        self._intensity = _new_bin_accumulator(num_bins_theta * num_bins_phi)
        self._num_rays_hit = 0
        self._total_flux = Tally()

        self._theta_edges = np.linspace(0.0, theta_max_deg, num_bins_theta + 1)
        self._phi_edges = np.linspace(-180.0, 180.0, num_bins_phi + 1)
        # Per-bin solid angle, sin(theta_centre) * dtheta * dphi, held as a
        # table rather than recomputed per ray: it depends only on the bin,
        # so the per-ray work is one gather.
        self._sin_theta_centres = np.sin(
            np.radians(0.5 * (self._theta_edges[:-1] + self._theta_edges[1:]))
        )
        self._d_theta = float(np.radians(self._theta_edges[1] - self._theta_edges[0]))
        self._d_phi = float(np.radians(self._phi_edges[1] - self._phi_edges[0]))

    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Accumulate angular flux from hit rays.

        Converts ray directions to (theta, phi) in the local detector frame
        and bins them, in the active backend's own operations: the rotation
        comes from the placement resolved once per trace
        (:meth:`~optiland.nonsequential.detectors.base.BaseDetector.frame`),
        the bin edges and the per-bin solid angle from tables uploaded once
        (:meth:`~optiland.nonsequential.detectors.base.BaseDetector.table`),
        and the direction of a ray that did not hit is masked to zero rather
        than gathered out -- so nothing here needs to know how many rays hit
        and the method costs no device-to-host transfer.

        Args:
            rays: Current ray bundle.
            t: Hit distances [mm], shape (N,). Unused: a far-field bin is a
                function of the direction alone.
            hit_mask: Boolean mask of rays hitting this detector, shape (N,).
        """
        _translation, rot = self.frame()

        dirs_g = be.stack([rays.L, rays.M, rays.N], axis=1)
        dirs_l = dirs_g @ rot  # global -> local

        # A ray that did not hit contributes zero flux, but its direction
        # would still go through arccos and arctan2 -- and a dead ray's
        # direction can be the zero vector. Mask the direction itself so
        # every lane evaluates on finite numbers.
        zero = be.zeros_like(dirs_l[:, 0])
        lx = be.where(hit_mask, dirs_l[:, 0], zero)
        ly = be.where(hit_mask, dirs_l[:, 1], zero)
        lz = be.where(hit_mask, dirs_l[:, 2], zero)

        # theta is the angle from the local +z axis
        cos_theta = self._cos_theta(lz)
        theta_deg = be.degrees(be.arccos(cos_theta))
        phi_deg = be.degrees(be.arctan2(ly, lx))

        theta_edges = self.table("theta_edges", self._theta_edges)
        phi_edges = self.table("phi_edges", self._phi_edges)
        i_theta = clamp_int(
            be.searchsorted(theta_edges, theta_deg, side="right") - 1,
            0,
            self.num_bins_theta - 1,
        )
        i_phi = clamp_int(
            be.searchsorted(phi_edges, phi_deg, side="right") - 1,
            0,
            self.num_bins_phi - 1,
        )

        # Solid-angle normalisation per bin (W/sr)
        sin_centres = self.table("sin_theta_centres", self._sin_theta_centres)
        solid_angle = sin_centres[i_theta] * self._d_theta * self._d_phi
        solid_angle = be.where(solid_angle > 0, solid_angle, be.ones_like(solid_angle))

        flux_masked = be.where(hit_mask, rays.flux, be.zeros_like(rays.flux))
        _accumulate_into(
            self._intensity,
            i_theta * self.num_bins_phi + i_phi,
            flux_masked / solid_angle,
            key=getattr(rays, "ray_id", None),
        )
        self._num_rays_hit = accumulate(self._num_rays_hit, masked_count(hit_mask))
        # Track the radiometric flux separately: _intensity is divided by the
        # per-bin solid angle, so summing it gives W/sr, not W.
        self._total_flux.add(masked_sum(rays.flux, hit_mask))

    def _cos_theta(self, lz):
        """Polar cosine of each hit direction from its local z component.

        A flat far-field detector is a plane and is reached from either
        side, so the two sides fold together onto one polar angle: a ray
        leaving along -z is binned at the same theta as one leaving along
        +z. A collector that is closed on one side only -- the
        hemispherical one -- overrides this, because there the sign of the
        direction is the physical hemisphere, not an ambiguity. Backend
        operations only: this runs on the device path of :meth:`record`.

        Args:
            lz: Local z direction component per ray, shape (N,), already
                masked to zero for rays that did not hit.

        Returns:
            ``cos(theta)`` in [0, 1], shape (N,).
        """
        return be.clip(be.abs(lz), 0.0, 1.0)

    def get_result(self) -> FarFieldPattern:
        """Return the accumulated far-field pattern.

        Returns:
            FarFieldPattern with intensity [W/sr].
        """
        theta_centres = 0.5 * (self._theta_edges[:-1] + self._theta_edges[1:])
        phi_centres = 0.5 * (self._phi_edges[:-1] + self._phi_edges[1:])
        intensity = to_numpy(bin_values(self._intensity)).reshape(
            self.num_bins_theta, self.num_bins_phi
        )
        return FarFieldPattern(
            intensity=intensity.copy(),
            theta=theta_centres,
            phi=phi_centres,
            total_flux=self._total_flux.value(),
            num_rays_hit=int(to_numpy(self._num_rays_hit)),
        )

    def reset(self) -> None:
        """Clear accumulated data."""
        self._intensity = _new_bin_accumulator(self.num_bins_theta * self.num_bins_phi)
        self._num_rays_hit = 0
        self._total_flux = Tally()
        self.reset_reflection_tally()
        self.invalidate_frame()
