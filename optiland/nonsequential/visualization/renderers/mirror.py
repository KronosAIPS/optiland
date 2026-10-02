"""2D and 3D renderers for Mirror compound components.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.visualization.renderers.base import (
    ComponentRenderer2D,
    ComponentRenderer3D,
)
from optiland.nonsequential.visualization.renderers.lens import (
    _face_sag,
    _projection_indices,
    _sag_array,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from optiland.nonsequential.components.compound import CompoundComponent


#: Abscissae of the probe that finds where a NURBS mirror's section meets its patches.
_PROBE_PTS = 513


def _drawn_range(component) -> tuple[float, float]:
    """The interval of the local ``y`` axis the mirror is drawn over.

    The config's aperture, or for a NURBS face (which does not use it) the
    part of the section that meets the patches: probed at ``_PROBE_PTS``
    abscissae across the control net's hull, from the first to the last that
    the kind's intersection hits (issue 88).
    """
    cfg = component._config
    if getattr(cfg, "nurbs", None) is None:
        return -float(cfg.aperture_radius), float(cfg.aperture_radius)
    geom = component.surfaces[0].geometry
    cp = np.asarray(geom.detached_copy().control_points, dtype=np.float64)
    hull = float(np.hypot(cp[:, 0], cp[:, 1]).max())
    y = np.linspace(-hull, hull, _PROBE_PTS)
    z = _face_sag(component.surfaces[0], y)
    on = y[np.isfinite(z)]
    if on.size == 0:
        return 0.0, 0.0
    return float(on.min()), float(on.max())


class MirrorRenderer2D(ComponentRenderer2D):
    """Renders a Mirror as a thick arc/curve in 2D."""

    def render(
        self,
        component: CompoundComponent,
        ax: Axes,
        theme=None,
        projection: str = "YZ",
    ) -> None:
        """Draw the mirror profile onto *ax*.

        Args:
            component: A Mirror CompoundComponent.
            ax: Matplotlib axes.
            theme: Optional theme.
            projection: Projection plane.
        """
        from optiland.nonsequential.components.mirror import Mirror  # noqa: PLC0415

        if not isinstance(component, Mirror):
            return

        from optiland.nonsequential.components.base import (
            _get_transform,  # noqa: PLC0415
        )

        cfg = component._config
        translation, rot = _get_transform(component._cs)
        h_idx, v_idx = _projection_indices(projection)

        n_pts = 128
        lo, hi = _drawn_range(component)
        y = np.linspace(lo, hi, n_pts)
        # The face's own sag (an asphere mirror's polynomial included,
        # KronosNSRT issue 81; a NURBS face sampled along the section by its
        # own intersection, issue 88, NaN off the patches, which the line
        # leaves as a gap); the base conic only for a kind with neither.
        z = _face_sag(component.surfaces[0], y)
        if z is None:
            z = _sag_array(cfg.radius, cfg.conic, y)

        pts_local = np.stack([np.zeros_like(y), y, z], axis=1)
        pts_global = pts_local @ rot.T + translation

        ph = pts_global[:, h_idx]
        pv = pts_global[:, v_idx]

        color = (0.5, 0.5, 0.5)
        if theme is not None:
            color = theme.parameters.get("axes.edgecolor", color)

        ax.plot(ph, pv, color=color, linewidth=2.0, zorder=3)


class MirrorRenderer3D(ComponentRenderer3D):
    """Renders a Mirror as a revolved 3D surface in VTK."""

    def render(
        self,
        component: CompoundComponent,
        renderer,
        theme=None,
    ) -> None:
        """Add VTK actors for the mirror to *renderer*.

        Args:
            component: A Mirror CompoundComponent.
            renderer: VTK renderer.
            theme: Optional theme.
        """
        from optiland.nonsequential.components.mirror import Mirror  # noqa: PLC0415

        if not isinstance(component, Mirror):
            return

        try:
            from optiland.visualization.system.utils import (
                revolve_contour,  # noqa: PLC0415
            )
        except ImportError:
            return

        from optiland.nonsequential.components.base import (
            _get_transform,  # noqa: PLC0415
        )

        cfg = component._config
        translation, rot = _get_transform(component._cs)

        n_pts = 128
        r = np.linspace(0.0, _drawn_range(component)[1], n_pts)
        z = _face_sag(component.surfaces[0], r)
        if z is None:
            z = _sag_array(cfg.radius, cfg.conic, r)
        # a NURBS face sampled along its section: only the abscissae on the patches
        # (the contour is revolved, which is exact for a surface of revolution)
        keep = np.isfinite(z)
        r, z = r[keep], z[keep]

        pts_local = np.stack([np.zeros_like(r), r, z], axis=1)
        pts_global = pts_local @ rot.T + translation

        actor = revolve_contour(pts_global[:, 0], pts_global[:, 1], pts_global[:, 2])

        # Style matching Sequential Mirror3D
        color = (0.75, 0.75, 0.75)
        if theme is not None:
            from matplotlib.colors import to_rgb  # noqa: PLC0415

            color_hex = theme.parameters.get("axes.edgecolor", "#BFBFBF")
            color = to_rgb(color_hex)

        prop = actor.GetProperty()
        prop.SetColor(color)
        prop.SetSpecular(1.0)
        prop.SetSpecularPower(100.0)  # Metallic
        prop.SetOpacity(1.0)

        renderer.AddActor(actor)
