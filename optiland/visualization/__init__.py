"""Visualization package for Optiland.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import SurfaceSagViewer
    from .info import LensInfoViewer
    from .system import OpticViewer, OpticViewer3D

__all__ = [
    "LensInfoViewer",
    "OpticViewer",
    "OpticViewer3D",
    "SurfaceSagViewer",
]

# name -> submodule that defines it. Each viewer pulls in a heavy, optional
# drawing dependency (VTK, Matplotlib, pandas) at its own module level; a
# plain ``import optiland.optic`` must not pay for any of that, so the
# submodule is only imported the first time one of these names is actually
# used (PEP 562 lazy module attributes).
_LAZY_ATTRS = {
    "SurfaceSagViewer": "analysis",
    "LensInfoViewer": "info",
    "OpticViewer": "system",
    "OpticViewer3D": "system",
}


def __getattr__(name: str):
    submodule_name = _LAZY_ATTRS.get(name)
    if submodule_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    submodule = importlib.import_module(f".{submodule_name}", __name__)
    value = getattr(submodule, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
