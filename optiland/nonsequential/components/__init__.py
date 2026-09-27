"""NSQ components subpackage."""

from __future__ import annotations

from .absorbing import AbsorbingComponent
from .base import BaseComponent
from .compound import CompoundComponent
from .configs import (
    DoubletConfig,
    InteractionType,
    LensConfig,
    MirrorConfig,
    ParaxialLensConfig,
    PolarizerConfig,
    PrismConfig,
    RetarderConfig,
    SurfaceConfig,
)
from .doublet import Doublet
from .lens import Lens
from .mirror import Mirror
from .paraxial import ParaxialLens, ParaxialLensComponent
from .polarizing import PolarizingComponent, Polarizer, Retarder
from .prism import Prism
from .reflective import ReflectiveComponent
from .refractive import RefractiveComponent
from .registry import ComponentRegistry
from .volume import NonWatertightVolumeError, Volume

__all__ = [
    "AbsorbingComponent",
    "BaseComponent",
    "CompoundComponent",
    "ComponentRegistry",
    "DoubletConfig",
    "Doublet",
    "InteractionType",
    "Lens",
    "LensConfig",
    "Mirror",
    "MirrorConfig",
    "NonWatertightVolumeError",
    "ParaxialLens",
    "ParaxialLensComponent",
    "ParaxialLensConfig",
    "PolarizerConfig",
    "Polarizer",
    "PolarizingComponent",
    "Prism",
    "PrismConfig",
    "RefractiveComponent",
    "ReflectiveComponent",
    "Retarder",
    "RetarderConfig",
    "SurfaceConfig",
    "Volume",
]
