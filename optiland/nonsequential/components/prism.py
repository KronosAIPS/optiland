"""Prism compound component for Non-Sequential Raytracing.

Two plane refracting faces meeting at an apex edge, and a base: a
dispersing prism, a wedge, a deviating element. See
:class:`~optiland.nonsequential.components.configs.PrismConfig` for the
frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from optiland.nonsequential._utils import as_float
from optiland.nonsequential.components.compound import CompoundComponent
from optiland.nonsequential.components.configs import InteractionType, PrismConfig
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.components.lens import _make_surface, _resolve_material
from optiland.nonsequential.materials.nsq_material import VACUUM

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.base import BaseComponent


class Prism(CompoundComponent):
    """A prism or wedge: two plane faces at an apex angle, and a base.

    Each face is a rectangular plane whose local +z (its geometric normal)
    points into the glass, so ``material_front`` is the surrounding medium
    and ``material_back`` the glass, as for a lens's front face: the
    refraction, the Fresnel split and the medium stack follow from the
    geometry alone, whichever way a ray crosses.

    Not validated as a closed :class:`~optiland.nonsequential.components
    .volume.Volume`: the triangular end faces are open (see
    :class:`PrismConfig`), and the watertightness check samples circular and
    rectangular rims only.

    Attributes:
        _name: Registry name.
        _cs: The prism's coordinate system (apex edge on its local y axis).
        _config: The PrismConfig it was built from.
        _surfaces: Built list of sub-surfaces.
    """

    def __init__(self, name: str, cs: CoordinateSystem, config: PrismConfig) -> None:
        """Initialize Prism from a PrismConfig.

        Args:
            name: Unique identifier for this prism in the registry.
            cs: Coordinate system of the prism (see :class:`PrismConfig`).
            config: Prism geometry and material.

        Raises:
            ValueError: If the apex angle is not in (0, 180) degrees or a
                length is not positive.
        """
        apex = as_float(config.apex_angle_deg)
        if not 0.0 < apex < 180.0:
            raise ValueError(
                f"Prism {name!r}: apex_angle_deg must be in (0, 180); got {apex!r}."
            )
        if as_float(config.face_length) <= 0.0 or as_float(config.length) <= 0.0:
            raise ValueError(
                f"Prism {name!r}: face_length and length must be positive; got "
                f"{config.face_length!r} and {config.length!r}."
            )
        self._name = name
        self._cs = cs
        self._config = config
        self._surfaces: list[BaseComponent] = self._build()

    @property
    def name(self) -> str:
        """Registry name of this prism."""
        return self._name

    @property
    def surfaces(self) -> list[BaseComponent]:
        """Ordered flat list of sub-surfaces: front, back, base."""
        return self._surfaces

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """The prism's own coordinate system."""
        return self._cs

    def _build(self) -> list[BaseComponent]:
        """Construct the faces from the config.

        Each face's placement is its midpoint, turned about the local y axis
        so the face's local +z is its inward normal and its local x runs
        along the face in the principal section. With half-angle ``a = A/2``
        (``c = cos a``, ``s = sin a``):

        ========  ===================  ==================  ============
        face      midpoint (x, z)      inward normal       turn (ry)
        ========  ===================  ==================  ============
        front     (-L c / 2, -L s / 2)  (-s, 0, c)          -a
        back      (-L c / 2, +L s / 2)  (-s, 0, -c)         pi + a
        base      (-L c, 0)             (1, 0, 0)           pi / 2
        ========  ===================  ==================  ============

        Returns:
            Ordered list of sub-surfaces.
        """
        from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415

        cfg = self._config
        glass = _resolve_material(cfg.material)
        half = math.radians(as_float(cfg.apex_angle_deg)) / 2.0
        c, s = math.cos(half), math.sin(half)
        face = as_float(cfg.face_length)
        length = as_float(cfg.length)

        def placed(x: float, z: float, ry: float) -> CoordinateSystem:
            return CoordinateSystem(x=x, z=z, ry=ry, reference_cs=self._cs)

        surfaces: list[BaseComponent] = [
            _make_surface(
                placed(-0.5 * face * c, -0.5 * face * s, -half),
                FinitePlaneGeometry(width=face, height=length),
                VACUUM,
                glass,
                cfg.front,
                InteractionType.REFRACTIVE,
                f"{self._name}.front",
            ),
            _make_surface(
                placed(-0.5 * face * c, 0.5 * face * s, math.pi + half),
                FinitePlaneGeometry(width=face, height=length),
                VACUUM,
                glass,
                cfg.back,
                InteractionType.REFRACTIVE,
                f"{self._name}.back",
            ),
        ]
        if not cfg.open_base:
            surfaces.append(
                _make_surface(
                    placed(-face * c, 0.0, 0.5 * math.pi),
                    FinitePlaneGeometry(width=2.0 * face * s, height=length),
                    VACUUM,
                    glass,
                    cfg.base,
                    InteractionType.ABSORBING,
                    f"{self._name}.base",
                )
            )
        return surfaces
