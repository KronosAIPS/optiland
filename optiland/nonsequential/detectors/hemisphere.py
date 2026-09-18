"""Closed hemispherical collector for Non-Sequential Raytracing.

``HemisphereDetector`` is the far-field detector on a closed shell instead
of a plane: the surface is a hemisphere of radius ``R`` about the detector's
own origin, covering the half-space its local +z points into, and the
angular distribution of everything that crosses it is binned in
``(theta, phi)`` exactly as :class:`FarFieldDetector` does.

Why a shell and not a plane. A plane placed to read the light *returning*
from a surface also sits in the beam going the other way, and a plane placed
off to one side misses the grazing part of the lobe; the catalogue names
both -- "a hemisphere that does not close, so grazing rays escape and the
flux deficit looks like absorption" (``docs/theory/11_validation_catalogue.md``
11.4.18). A closed hemisphere has neither problem: every ray leaving a point
inside it with a positive local-z direction component meets it exactly once,
whatever its azimuth and however grazing it is, and a beam arriving from
below the equator never touches it.

Binning is by ray **direction**, not by landing position. For a source of
finite size the two differ by the parallax of the emission point against the
shell radius -- a 5 mm spot on a 100 mm shell spreads a single direction over
about 1.4 degrees of landing position -- and it is the direction
distribution that a scatter lobe is defined by. Landing position is
recoverable from the ray database if it is the wanted statistic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential._utils import as_param
from optiland.nonsequential.components.geometry.analytic.spherical_cavity import (
    SphericalCavityGeometry,
)
from optiland.nonsequential.detectors.far_field import FarFieldDetector

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem


class HemisphereDetector(FarFieldDetector):
    """Angular flux distribution collected on a closed hemispherical shell.

    Attributes:
        cs: Coordinate system. The shell is centred on its origin and
            covers the half-space local +z points into.
        radius: Shell radius [mm].
        num_bins_theta: Number of polar bins, spanning 0 to 90 degrees.
        num_bins_phi: Number of azimuthal bins.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        radius: float,
        num_bins_theta: int = 18,
        num_bins_phi: int = 36,
        name: str = "",
        absorb: bool = True,
    ) -> None:
        """Initialize HemisphereDetector.

        Args:
            cs: Coordinate system for the shell's centre and axis.
            radius: Shell radius [mm].
            num_bins_theta: Number of polar bins over 0-90 degrees.
            num_bins_phi: Number of azimuthal bins over -180 to 180 degrees.
            name: Optional label.
            absorb: Whether a hit terminates the ray (default True).
        """
        super().__init__(
            cs,
            theta_max_deg=90.0,
            num_bins_theta=num_bins_theta,
            num_bins_phi=num_bins_phi,
            name=name,
            absorb=absorb,
        )
        # Replaces the flat aperture the far-field detector builds: the
        # shell is closed, so no `side` flag is needed to keep the beam
        # travelling the other way out of the measurement -- that beam is
        # below the equator and never touches the surface.
        self.radius = as_param(radius)
        self.geometry = SphericalCavityGeometry.hemisphere(radius)

    def _cos_theta(self, dirs_l: np.ndarray) -> np.ndarray:
        """Polar cosine, without the flat detector's two-sided fold.

        Args:
            dirs_l: Hit ray directions in the local frame, shape (K, 3).

        Returns:
            ``cos(theta)`` in [0, 1], shape (K,). The shell only accepts
            rays crossing it outward, so the clamp at 0 is a guard on
            rounding at the equator, not a fold of one hemisphere onto the
            other.
        """
        return np.clip(dirs_l[:, 2], 0.0, 1.0)

    def _polar_solid_angle(self) -> np.ndarray:
        """Per-bin solid angle [sr] used when the pattern was accumulated.

        Returns:
            Array of shape ``(num_bins_theta,)``: the same
            ``sin(theta_c) d_theta d_phi`` that :meth:`record` divided each
            hit's flux by, so multiplying it back is exact rather than
            approximate.
        """
        d_theta = np.radians(self._theta_edges[1] - self._theta_edges[0])
        d_phi = np.radians(self._phi_edges[1] - self._phi_edges[0])
        theta_centres = np.radians(
            0.5 * (self._theta_edges[:-1] + self._theta_edges[1:])
        )
        return np.sin(theta_centres) * d_theta * d_phi

    def polar_flux(self) -> np.ndarray:
        """Collected flux per polar bin [W], summed over azimuth.

        Returns:
            Array of shape ``(num_bins_theta,)``.
        """
        pattern = self.get_result()
        return (pattern.intensity * self._polar_solid_angle()[:, None]).sum(axis=1)

    def cone_fraction(self, half_angle_deg: float) -> float:
        """Fraction of the collected flux inside a cone about the axis.

        The cone edge must fall on a bin edge, so the answer is a sum of
        whole bins with no assumption about how flux is distributed inside
        a bin.

        Args:
            half_angle_deg: Cone half-angle [deg].

        Returns:
            Collected flux inside the cone divided by the total collected
            flux.

        Raises:
            ValueError: If the cone edge is not a polar bin edge, or if
                nothing has been collected yet.
        """
        edges = self._theta_edges
        idx = int(np.argmin(np.abs(edges - float(half_angle_deg))))
        if abs(edges[idx] - float(half_angle_deg)) > 1e-9 * max(
            1.0, float(half_angle_deg)
        ):
            raise ValueError(
                f"Cone half-angle {half_angle_deg} deg is not a polar bin "
                f"edge of this detector (edges: {edges[0]} to {edges[-1]} in "
                f"{self.num_bins_theta} steps)."
            )
        flux = self.polar_flux()
        total = float(flux.sum())
        if total <= 0.0:
            raise ValueError("No flux collected: cone fraction is undefined.")
        return float(flux[:idx].sum()) / total
