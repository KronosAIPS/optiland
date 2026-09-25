"""Detectors subpackage for Non-Sequential Raytracing."""

from __future__ import annotations

from .base import BaseDetector
from .colorimetric import ColorimetricDetector, ColorimetricFarFieldDetector
from .configs import (
    ColorimetricDetectorConfig,
    ColorimetricFarFieldDetectorConfig,
    FarFieldDetectorConfig,
    HemisphereDetectorConfig,
    IrradianceDetectorConfig,
    RayDatabaseConfig,
    SpectralDetectorConfig,
)
from .far_field import FarFieldDetector
from .hemisphere import HemisphereDetector
from .irradiance import IrradianceDetector
from .ray_database import RayDatabaseDetector
from .registry import DetectorRegistry
from .spectral import SpectralDetector

__all__ = [
    "BaseDetector",
    "ColorimetricDetector",
    "ColorimetricDetectorConfig",
    "ColorimetricFarFieldDetector",
    "ColorimetricFarFieldDetectorConfig",
    "DetectorRegistry",
    "FarFieldDetector",
    "FarFieldDetectorConfig",
    "HemisphereDetector",
    "HemisphereDetectorConfig",
    "IrradianceDetector",
    "IrradianceDetectorConfig",
    "RayDatabaseDetector",
    "RayDatabaseConfig",
    "SpectralDetector",
    "SpectralDetectorConfig",
]
