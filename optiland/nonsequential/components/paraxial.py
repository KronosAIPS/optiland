"""Ideal paraxial (thin) lens for Non-Sequential Raytracing.

A plane of no thickness that deflects every ray crossing it by the
paraxial thin-lens law, and the compound that places it, optionally with an
absorbing stop around its clear aperture. See
:class:`~optiland.nonsequential.components.configs.ParaxialLensConfig`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential._utils import as_float, as_param
from optiland.nonsequential.components.absorbing import AbsorbingComponent
from optiland.nonsequential.components.base import (
    BaseComponent,
    _resident_transform,
)
from optiland.nonsequential.components.compound import CompoundComponent
from optiland.nonsequential.components.configs import ParaxialLensConfig
from optiland.nonsequential.components.geometry.analytic.annulus import (
    AnnularPlaneGeometry,
)
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.materials.nsq_material import VACUUM

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.ir.bsdf_ir import BsdfIR
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng


class ParaxialLensComponent(BaseComponent):
    """An ideal thin lens: a disc at local z = 0 that deflects by the paraxial law.

    For a ray crossing the disc at local ``(h_x, h_y)`` with local direction
    ``(L, M, N)``, the slopes along its own direction of travel are
    ``u_x = L / |N|`` and ``u_y = M / |N|``; the lens changes them to

        u_x' = u_x - h_x / f ,   u_y' = u_y - h_y / f ,

    and the ray leaves along ``(u_x', u_y', sign(N))`` normalised, on the
    same side of the plane it was heading for. That is the thin-lens law
    exactly in the slopes -- not an approximation of a real surface -- so
    every ray from an object point meets every other at the conjugate image
    point, a converging lens (``f > 0``) converges whichever way it is
    crossed, and the map from (height, slope) before to after has
    determinant one: the paraxial etendue and the Lagrange invariant are
    conserved identically (``docs/theory/11_validation_catalogue.md``
    11.4.15). The sine condition is *not* satisfied, which is what an ideal
    lens defined on slopes is: measured with true angles instead of slopes,
    a finite cone shows the residual 11.4.15 derives.

    Lossless: the weight is untouched, there is no reflected branch and no
    ledger booking, and the ray stays in the medium it was in -- the
    element has no index of its own. A ray crossing parallel to the plane
    (``N = 0``) does not hit it (the plane geometry rejects it).

    Attributes:
        focal_length: ``f`` [mm]; may be a differentiable tensor.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        focal_length: float,
        aperture_radius: float,
        name: str = "",
    ) -> None:
        """Initialize ParaxialLensComponent.

        Args:
            cs: Coordinate system; the lens plane is its local z = 0.
            focal_length: Focal length [mm]. Must be non-zero.
            aperture_radius: Clear semi-diameter [mm].
            name: Optional label.

        Raises:
            ValueError: If the focal length is zero.
        """
        if as_float(focal_length) == 0.0:
            raise ValueError(f"Paraxial lens {name!r}: focal_length must be non-zero.")
        super().__init__(
            cs,
            FinitePlaneGeometry(aperture_radius=aperture_radius),
            VACUUM,
            VACUUM,
            None,
            name,
        )
        self.focal_length = as_param(focal_length)

    def interact(
        self,
        rays: NSQRayBundle,
        t: np.ndarray,
        normals: np.ndarray,
        hit_mask: np.ndarray,
        rng: NSQRng,
        bsdf_ir: BsdfIR,
        n_geom: np.ndarray,
        sampling: SamplingPolicy | None = None,
        forced_branch: str | None = None,
    ) -> None:
        """Deflect every hit ray by the thin-lens law (in place).

        Args:
            rays: Ray bundle updated in place.
            t: Hit distances [mm], shape (N,).
            normals: Unused -- the law is written in the lens's own frame.
            hit_mask: True for rays crossing the lens, shape (N,).
            rng: Unused -- nothing here is drawn.
            bsdf_ir: Unused -- an ideal lens carries no scatter model.
            n_geom: Geometric normal, for the outgoing origin offset.
            sampling: Unused.
            forced_branch: Unused -- there is one branch.
        """
        # Onto the lens plane, rebuilt in its own frame (R-07-4).
        self.advance_to_hit(rays, t, hit_mask)

        t_be, R_be = _resident_transform(self)
        positions_l = (be.stack([rays.x, rays.y, rays.z], axis=1) - t_be) @ R_be
        directions_l = be.stack([rays.L, rays.M, rays.N], axis=1) @ R_be

        n_l = directions_l[:, 2]
        # |N| is bounded away from zero on every hit ray (the plane geometry
        # rejects a parallel one); elsewhere it is replaced by 1 so a
        # masked-out lane cannot divide by zero.
        crossing = hit_mask & (be.abs(n_l) > _tol.tiny_for(n_l))
        n_abs = be.where(crossing, be.abs(n_l), be.ones_like(n_l))
        forward = be.where(n_l < 0.0, -be.ones_like(n_l), be.ones_like(n_l))

        slope_x = directions_l[:, 0] / n_abs - positions_l[:, 0] / self.focal_length
        slope_y = directions_l[:, 1] / n_abs - positions_l[:, 1] / self.focal_length
        norm = (slope_x * slope_x + slope_y * slope_y + 1.0) ** 0.5
        out_l = be.stack([slope_x / norm, slope_y / norm, forward / norm], axis=1)
        out_g = out_l @ R_be.T

        rays.L = be.where(crossing, out_g[:, 0], rays.L)
        rays.M = be.where(crossing, out_g[:, 1], rays.M)
        rays.N = be.where(crossing, out_g[:, 2], rays.N)
        rays.bounce = be.where(hit_mask, rays.bounce + 1, rays.bounce)

        # R-07-6: leave the plane on the side the ray is heading for.
        self.offset_from_surface(rays, n_geom, hit_mask)


class ParaxialLens(CompoundComponent):
    """An ideal thin lens, optionally with an absorbing stop around it.

    Surfaces, in order: the :class:`ParaxialLensComponent` disc, and -- when
    ``stop_radius`` is set -- an absorbing annulus in the same plane from
    the clear aperture out to ``stop_radius``, so rays outside the aperture
    are stopped rather than passing by undeviated. The disc comes first: a
    ray landing exactly on the aperture edge is claimed by the lens.

    Attributes:
        _name: Registry name.
        _cs: The lens plane's coordinate system.
        _config: The ParaxialLensConfig it was built from.
        _surfaces: Built list of sub-surfaces.
    """

    def __init__(
        self, name: str, cs: CoordinateSystem, config: ParaxialLensConfig
    ) -> None:
        """Initialize ParaxialLens from a ParaxialLensConfig.

        Args:
            name: Unique identifier in the registry.
            cs: Coordinate system of the lens plane.
            config: Focal length, aperture and optional stop.

        Raises:
            ValueError: If the stop does not lie outside the aperture.
        """
        if config.stop_radius is not None and as_float(config.stop_radius) <= as_float(
            config.aperture_radius
        ):
            raise ValueError(
                f"Paraxial lens {name!r}: stop_radius ({config.stop_radius!r}) must "
                f"exceed aperture_radius ({config.aperture_radius!r})."
            )
        self._name = name
        self._cs = cs
        self._config = config
        surfaces: list[BaseComponent] = [
            ParaxialLensComponent(
                cs,
                config.focal_length,
                config.aperture_radius,
                name=f"{name}.lens",
            )
        ]
        if config.stop_radius is not None:
            surfaces.append(
                AbsorbingComponent(
                    cs,
                    AnnularPlaneGeometry(
                        inner_radius=as_float(config.aperture_radius),
                        outer_radius=as_float(config.stop_radius),
                    ),
                    name=f"{name}.stop",
                )
            )
        self._surfaces = surfaces

    @property
    def name(self) -> str:
        """Registry name of this lens."""
        return self._name

    @property
    def surfaces(self) -> list[BaseComponent]:
        """Ordered flat list of sub-surfaces."""
        return self._surfaces

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """The lens plane's coordinate system."""
        return self._cs
