"""Config dataclasses for NSQ detectors.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class IrradianceDetectorConfig:
    """Configuration for an IrradianceDetector.

    Attributes:
        width: Detector width [mm].
        height: Detector height [mm].
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
        splat: Splatting mode — 'bilinear', 'gaussian', or 'hard'.
        splat_sigma: Gaussian splat sigma in pixels (used when splat='gaussian').
        absorb: Whether a hit terminates the ray. False makes the detector
            transmissive: the hit is recorded and the ray continues on its
            unchanged direction, enabling mid-system beam sampling.
        side: Which side of the detector plane is live: 'both' (default,
            a ray crossing from either side is recorded), 'front' (only
            rays arriving on the side the placement's normal points
            toward), or 'back'.
        reflection_bins: Also book the arriving flux by the rays'
            reflection count: this many exact bins (0 .. K-1) and one
            overflow bin, read with ``SimulationResult
            .reflection_histograms[name]``. 0 (default) keeps none.
    """

    width: float
    height: float
    num_pixels_x: int = 256
    num_pixels_y: int = 256
    splat: Literal["bilinear", "gaussian", "hard"] = "bilinear"
    splat_sigma: float = 0.5
    absorb: bool = True
    side: Literal["both", "front", "back"] = "both"
    reflection_bins: int = 0


@dataclass
class SpectralDetectorConfig:
    """Configuration for a SpectralDetector.

    Attributes:
        width: Detector width [mm].
        height: Detector height [mm].
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
        wl_min: Minimum wavelength for spectral binning [µm].
        wl_max: Maximum wavelength for spectral binning [µm].
        num_bins: Number of wavelength bins.
        splat: Spatial (x, y) splatting mode — 'bilinear', 'gaussian', or
            'hard'. The wavelength bin is always hard-assigned.
        splat_sigma: Gaussian splat sigma in pixels (used when
            ``splat='gaussian'``).
        absorb: Whether a hit terminates the ray.

    Note:
        Wavelengths are in **micrometres**, matching ``Spectrum`` and every
        other wavelength in Optiland. Visible light spans 0.4-0.7 µm, so a
        detector spanning the visible is ``wl_min=0.4, wl_max=0.7``.
    """

    width: float
    height: float
    num_pixels_x: int = 256
    num_pixels_y: int = 256
    wl_min: float = 0.4
    wl_max: float = 0.7
    num_bins: int = 100
    splat: Literal["bilinear", "gaussian", "hard"] = "bilinear"
    splat_sigma: float = 0.5
    absorb: bool = True


@dataclass
class FarFieldDetectorConfig:
    """Configuration for a FarFieldDetector.

    Attributes:
        num_theta: Number of polar angle bins.
        num_phi: Number of azimuthal angle bins.
        absorb: Whether a hit terminates the ray.
        side: Which side of the detector plane is live: 'both' (default),
            'front', or 'back'. See ``IrradianceDetectorConfig.side``.
        reflection_bins: Book the arriving flux by reflection count. See
            ``IrradianceDetectorConfig.reflection_bins``.
    """

    num_theta: int = 90
    num_phi: int = 360
    absorb: bool = True
    side: Literal["both", "front", "back"] = "both"
    reflection_bins: int = 0


@dataclass
class HemisphereDetectorConfig:
    """Configuration for a HemisphereDetector.

    A closed hemispherical shell that collects the angular distribution of
    everything leaving the half-space it covers. It has no ``side`` option:
    the shell is closed on one side by construction, which is the point of
    it.

    Attributes:
        radius: Shell radius [mm].
        num_theta: Number of polar bins over 0-90 degrees.
        num_phi: Number of azimuthal bins.
        absorb: Whether a hit terminates the ray.
        reflection_bins: Book the arriving flux by reflection count. See
            ``IrradianceDetectorConfig.reflection_bins``.
    """

    radius: float
    num_theta: int = 18
    num_phi: int = 36
    absorb: bool = True
    reflection_bins: int = 0


@dataclass
class RayDatabaseConfig:
    """Configuration for a RayDatabaseDetector.

    Attributes:
        width: Detector width [mm].
        height: Detector height [mm].
        max_rays: Maximum number of rays to store (0 = unlimited).
        absorb: Whether a hit terminates the ray.
    """

    width: float
    height: float
    max_rays: int = 0
    absorb: bool = True


@dataclass
class ColorimetricDetectorConfig(IrradianceDetectorConfig):
    """Configuration for a ColorimetricDetector: an irradiance detector that
    also books the CIE 1931 tristimulus flux X, Y, Z per pixel (illuminance,
    chromaticity, correlated colour temperature). Same fields."""


@dataclass
class ColorimetricFarFieldDetectorConfig(FarFieldDetectorConfig):
    """Configuration for a ColorimetricFarFieldDetector: a far-field detector
    that also books the tristimulus intensity per bin (candela, chromaticity
    over angle). Same fields."""
