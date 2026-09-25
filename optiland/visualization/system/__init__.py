"""System visualization package for Optiland.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optic_viewer import OpticViewer
    from .optic_viewer_3d import OpticViewer3D

__all__ = ["OpticViewer", "OpticViewer3D"]

_LAZY_ATTRS = {
    "OpticViewer": "optic_viewer",
    "OpticViewer3D": "optic_viewer_3d",
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
