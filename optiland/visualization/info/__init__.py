"""Info visualization package for Optiland.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lens_info_viewer import LensInfoViewer

__all__ = ["LensInfoViewer"]

_LAZY_ATTRS = {
    "LensInfoViewer": "lens_info_viewer",
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
