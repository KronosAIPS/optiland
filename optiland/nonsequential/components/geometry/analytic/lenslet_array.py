"""Lenslet-array (periodic conic-cap) geometry for Non-Sequential Raytracing.

``LensletArrayGeometry`` is a rectangular lattice of identical conic caps --
one element definition (the same conic cap :class:`ConicGeometry` solves)
plus a cell lookup on the lattice -- for a microlens array, a fly's-eye
homogeniser or, with a per-cell offset table, a stepped mirror array. It is
exact for every ray at O(1) memory regardless of array size, including a
ray that grazes across several cells' footprints before it actually meets a
cap.

The surface is a height field over the local ``z = 0`` base plane: cell
``(i, j)`` is centred at ``((i - (num_x-1)/2) * pitch_x, (j - (num_y-1)/2)
* pitch_y)`` and carries ``z = sag(x - xc, y - yc) + offset(i, j)`` -- the
same conic sag :class:`ConicGeometry` uses -- clipped to its own
rectangular cell; the array's footprint is the union of the cells, and a
ray that meets no cap inside it misses.

A ray is solved by clipping it to the slab that contains every cap, then
marching its ``(x, y)`` footprint across the cells it crosses, in order,
using the 2-D digital differential analyser of

    Amanatides, J. and Woo, A. "A Fast Voxel Traversal Algorithm for Ray
    Tracing." Eurographics '87, pp. 3-10.

applied to the array's base plane instead of to a 3-D voxel volume, in the
height-field framing of

    Musgrave, F. K. "Grid tracing: fast ray tracing for height fields."
    Technical Report YALEU/DCS/RR-639, Yale University, 1988.

In each crossed cell the cell's own conic quadratic is solved with the same
stable ("citardauque") root form :class:`ConicGeometry` uses, with the
circular-aperture test it uses swapped for the cell's rectangular
membership test; the nearest valid root of the first cell that has one
wins. The number of cells marched is bounded by ``num_x + num_y + 1`` -- a
constant fixed by the lattice shape, never a per-ray or data-dependent
count -- which is more than the largest number of cells any straight line
can cross on an ``num_x`` by ``num_y`` grid (``num_x + num_y - 1``).
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential._utils import (
    as_float,
    as_param,
    clamp_int,
    floor_to_int,
    int_to_float_like,
)
from optiland.nonsequential.components.base import resident_table
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry
from optiland.nonsequential.ray_bundle import backend_bool_full


def _axis_interval(o, d, lo, hi):
    """t-interval where a ray ``o + t*d`` lies within ``[lo, hi]`` on one axis.

    Division by an exactly-zero direction component yields a genuine
    IEEE-754 signed infinity, the same identity
    :meth:`~optiland.nonsequential.components.geometry.base.AABB
    .intersects_ray` uses: from inside the slab the interval comes out
    ``(-inf, inf)`` (no constraint from this axis), and from outside it
    comes out ``(+inf, +inf)`` or ``(-inf, -inf)``, which empties the
    combined box interval once every axis is intersected -- with no branch
    needed for a ray running parallel to the axis.

    Args:
        o: Ray origin component, shape (N,).
        d: Ray direction component, shape (N,).
        lo: Lower bound on this axis (scalar or shape (N,)).
        hi: Upper bound on this axis (scalar or shape (N,)).

    Returns:
        ``(t_near, t_far)``, each shape (N,).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d = 1.0 / d
        t0 = (lo - o) * inv_d
        t1 = (hi - o) * inv_d
    return be.minimum(t0, t1), be.maximum(t0, t1)


class LensletArrayGeometry(AnalyticGeometry):
    """A rectangular lattice of identical conic caps.

    Attributes:
        pitch_x: Cell pitch along local x [mm].
        pitch_y: Cell pitch along local y [mm].
        radius: Vertex radius of curvature of every cap [mm].
        conic: Conic constant of every cap.
        num_x: Number of cells along x.
        num_y: Number of cells along y.
        sag_offsets: Per-cell sag offset table [mm], shape ``(num_y,
            num_x)``, or ``None`` for an unshifted array (every offset 0).
    """

    def __init__(
        self,
        pitch_x: float,
        pitch_y: float,
        radius: float,
        conic: float,
        num_x: int,
        num_y: int,
        sag_offsets=None,
    ) -> None:
        """Initialize LensletArrayGeometry.

        Args:
            pitch_x: Cell pitch along local x [mm].
            pitch_y: Cell pitch along local y [mm].
            radius: Vertex radius of curvature of every cap [mm].
            conic: Conic constant of every cap.
            num_x: Number of cells along x (>= 1).
            num_y: Number of cells along y (>= 1).
            sag_offsets: Per-cell sag offset [mm], shape ``(num_y, num_x)``
                (row ``j``, column ``i``), or ``None`` for all zero.

        Raises:
            ValueError: If ``num_x`` or ``num_y`` is not a positive
                integer, or ``sag_offsets`` has the wrong shape.
        """
        if int(num_x) < 1 or int(num_y) < 1:
            raise ValueError(
                "A lenslet array needs at least one cell along each axis; "
                f"got num_x={num_x}, num_y={num_y}."
            )
        self.num_x = int(num_x)
        self.num_y = int(num_y)
        self.pitch_x = as_param(pitch_x)
        self.pitch_y = as_param(pitch_y)
        self.radius = as_param(radius)
        self.conic = as_param(conic)

        # aperture_radius is never consulted: this class solves each cell's
        # cap quadratic itself (see ray_intersect) and never calls
        # ConicGeometry.ray_intersect on self._cap, so the cap's own
        # circular-aperture cutoff plays no part. self._cap exists only to
        # share ConicGeometry's sag, curvature and normal formulas.
        self._cap = ConicGeometry(
            radius=self.radius, conic=self.conic, aperture_radius=float("inf")
        )

        if sag_offsets is None:
            self.sag_offsets = None
            offsets_arr = np.zeros((self.num_y, self.num_x), dtype=np.float64)
        else:
            offsets_arr = np.asarray(sag_offsets, dtype=np.float64)
            if offsets_arr.shape != (self.num_y, self.num_x):
                raise ValueError(
                    "sag_offsets must have shape (num_y, num_x) = "
                    f"({self.num_y}, {self.num_x}); got {offsets_arr.shape}."
                )
            self.sag_offsets = offsets_arr
        self._sag_offsets_flat = np.ascontiguousarray(offsets_arr.ravel())
        self._offset_min = float(offsets_arr.min())
        self._offset_max = float(offsets_arr.max())

    def _cap_edge_sag(self) -> float:
        """Detached sag at a cell's own half-diagonal (its point farthest
        from the cell centre) -- the same edge-of-region evaluation
        :meth:`ConicGeometry.bounding_box` performs for its own aperture
        edge, reused here (a) for the z-slab every cap is clipped to in
        :meth:`ray_intersect` and (b) for this geometry's own AABB. A
        detached float, like every other AABB/search-region bound in this
        package: it only decides which ray segment to search, not any
        differentiable hit quantity.

        Returns:
            The cap's sag [mm] at its cell's half-diagonal radius.
        """
        half_px = as_float(self.pitch_x) / 2.0
        half_py = as_float(self.pitch_y) / 2.0
        r2 = half_px * half_px + half_py * half_py
        c = as_float(self._cap._curvature())
        K = as_float(self.conic)
        under_root = max(1.0 - (1.0 + K) * c * c * r2, 1e-12)
        return c * r2 / (1.0 + under_root**0.5)

    def _z_slab(self) -> tuple[float, float]:
        """The detached ``(z_min, z_max)`` [mm] slab containing every cap."""
        sag_edge = self._cap_edge_sag()
        z_lo = min(0.0, sag_edge) + self._offset_min
        z_hi = max(0.0, sag_edge) + self._offset_max
        return z_lo, z_hi

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray, eps: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the lenslet array.

        See the module docstring for the method (slab clip, then a 2-D grid
        march with a per-cell conic solve).

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3).
            eps: See :meth:`ComponentGeometry.ray_intersect`.

        Returns:
            (t, normals, hit_mask, n_geom). n_geom is the cap's own
            gradient normal, unflipped -- the same +z-is-``material_back``
            convention as :class:`ConicGeometry`.
        """
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]
        N = origins.shape[0]

        if eps is None:
            eps = _tol.accept_t_min(be.abs(origins).max())

        pitch_x, pitch_y = self.pitch_x, self.pitch_y
        half_px, half_py = pitch_x / 2.0, pitch_y / 2.0
        x_min = -(self.num_x * pitch_x) / 2.0
        x_max = (self.num_x * pitch_x) / 2.0
        y_min = -(self.num_y * pitch_y) / 2.0
        y_max = (self.num_y * pitch_y) / 2.0
        z_lo, z_hi = self._z_slab()

        c = self._cap._curvature()
        K = self.conic
        kp = 1.0 + K

        # -- Slab clip: the box [x_min,x_max] x [y_min,y_max] x [z_lo,z_hi]
        # that contains every cap. -----------------------------------------
        t_near_x, t_far_x = _axis_interval(ox, dx, x_min, x_max)
        t_near_y, t_far_y = _axis_interval(oy, dy, y_min, y_max)
        t_near_z, t_far_z = _axis_interval(oz, dz, z_lo, z_hi)
        t_box_enter = be.maximum(be.maximum(t_near_x, t_near_y), t_near_z)
        t_box_exit = be.minimum(be.minimum(t_far_x, t_far_y), t_far_z)
        t_start = be.maximum(t_box_enter, eps)
        box_valid = (t_box_exit >= t_box_enter) & (t_box_exit > eps) & (
            t_start <= t_box_exit
        )
        # A ray that never enters the box can have an infinite t_start (it
        # is parallel to, and outside, some axis's slab); box_valid already
        # excludes it from every candidate test below, but an unguarded
        # inf * 0 (a direction component that is exactly zero) would still
        # raise an "invalid value" warning while computing where it would
        # have entered. Zero it instead -- the value is never read for a
        # ray this masks out.
        t_start = be.where(box_valid, t_start, be.zeros_like(t_start))

        # -- Entry cell and DDA stepping state (Amanatides & Woo 1987). ----
        ex = ox + t_start * dx
        ey = oy + t_start * dy
        i = clamp_int(floor_to_int((ex - x_min) / pitch_x), 0, self.num_x - 1)
        j = clamp_int(floor_to_int((ey - y_min) / pitch_y), 0, self.num_y - 1)

        step_x = floor_to_int(be.sign(dx))
        step_y = floor_to_int(be.sign(dy))

        with np.errstate(divide="ignore", invalid="ignore"):
            t_delta_x = pitch_x / be.abs(dx)
            t_delta_y = pitch_y / be.abs(dy)
            i_bound_f = int_to_float_like(be.where(step_x > 0, i + 1, i), ex)
            j_bound_f = int_to_float_like(be.where(step_y > 0, j + 1, j), ey)
            x_boundary = x_min + i_bound_f * pitch_x
            y_boundary = y_min + j_bound_f * pitch_y
            t_max_x = (x_boundary - ox) / dx
            t_max_y = (y_boundary - oy) / dy
        # A ray with an exactly-zero direction component never crosses a
        # boundary on that axis; force it out of contention rather than
        # trust the sign of a 0/0 or (nonzero)/0 division (unlike
        # t_delta_*, whose numerator is always strictly positive, the
        # numerator here can be positive, negative or zero).
        inf_x = be.ones_like(t_max_x) * be.inf
        inf_y = be.ones_like(t_max_y) * be.inf
        t_max_x = be.where(step_x == 0, inf_x, t_max_x)
        t_max_y = be.where(step_y == 0, inf_y, t_max_y)

        found = backend_bool_full((N,), False, like=ox)
        active = box_valid
        t_result = be.ones_like(ox) * be.inf
        px_rel = be.zeros_like(ox)
        py_rel = be.zeros_like(oy)

        max_steps = self.num_x + self.num_y + 1
        for _ in range(max_steps):
            candidates = active & ~found

            # i, j themselves are allowed to run one step past the grid
            # (that is how a ray leaving the footprint is detected, in the
            # in-range test below); clamp only the copy used to gather this
            # iteration's cell, so an out-of-grid ray never drives an
            # out-of-bounds table read -- it is excluded from ever being
            # accepted by `candidates` regardless of what cell this reads.
            i_c = clamp_int(i, 0, self.num_x - 1)
            j_c = clamp_int(j, 0, self.num_y - 1)

            i_f = int_to_float_like(i_c, ex)
            j_f = int_to_float_like(j_c, ey)
            xc = x_min + (i_f + 0.5) * pitch_x
            yc = y_min + (j_f + 0.5) * pitch_y

            flat_idx = j_c * self.num_x + i_c
            table = resident_table(self, "sag_offsets", self._sag_offsets_flat)
            offsets = table[flat_idx]

            oxp = ox - xc
            oyp = oy - yc
            ozp = oz - offsets

            a = c * (dx * dx + dy * dy + kp * dz * dz)
            b = 2.0 * (c * (oxp * dx + oyp * dy + kp * ozp * dz) - dz)
            c0 = c * (oxp * oxp + oyp * oyp + kp * ozp * ozp) - 2.0 * ozp

            disc = b * b - 4.0 * a * c0
            disc_ok = disc >= 0.0
            disc_floor = _tol.radicand_floor(be.ones_like(disc))
            sqrt_disc = be.where(
                disc_ok, be.maximum(disc, disc_floor) ** 0.5, be.zeros_like(disc)
            )
            # Numerically stable ("citardauque") roots -- the same form
            # ConicGeometry.ray_intersect uses.
            sign_b = be.where(b >= 0.0, 1.0, -1.0)
            q = -0.5 * (b + sign_b * sqrt_disc)
            tiny = _tol.tiny_for(a)
            a_ok = be.abs(a) > tiny
            q_ok = be.abs(q) > tiny
            t1 = q / be.where(a_ok, a, be.ones_like(a))
            t2 = c0 / be.where(q_ok, q, be.ones_like(q))

            def _root_valid(t, solvable):
                px = ox + t * dx
                py = oy + t * dy
                pz = oz + t * dz
                in_cell = (be.abs(px - xc) <= half_px) & (be.abs(py - yc) <= half_py)
                # On the surface, sqrt(1-(1+K)c^2 r^2) = 1-(1+K) c z' with
                # z' measured from this cell's own offset sag sheet -- the
                # same "which branch of the quadric" test
                # ConicGeometry._root_valid uses.
                on_sag_sheet = (1.0 - kp * c * (pz - offsets)) >= 0.0
                valid = (
                    candidates
                    & solvable
                    & be.isfinite(t)
                    & (t > eps)
                    & in_cell
                    & on_sag_sheet
                )
                return valid, px, py

            valid1, px1, py1 = _root_valid(t1, disc_ok & a_ok)
            valid2, px2, py2 = _root_valid(t2, disc_ok & q_ok)

            pick1 = valid1 & (~valid2 | (t1 <= t2))
            pick2 = valid2 & ~pick1
            hit_this_cell = pick1 | pick2

            t_pick = be.where(pick1, t1, be.where(pick2, t2, be.ones_like(t1) * be.inf))
            px_pick = be.where(pick1, px1, be.where(pick2, px2, be.zeros_like(px1)))
            py_pick = be.where(pick1, py1, be.where(pick2, py2, be.zeros_like(py1)))

            t_result = be.where(hit_this_cell, t_pick, t_result)
            px_rel = be.where(hit_this_cell, px_pick - xc, px_rel)
            py_rel = be.where(hit_this_cell, py_pick - yc, py_rel)
            found = found | hit_this_cell

            # -- Advance to the next cell (Amanatides & Woo's DDA step). --
            adv = active & ~found
            step_axis_x = t_max_x <= t_max_y
            i = be.where(adv & step_axis_x, i + step_x, i)
            j = be.where(adv & ~step_axis_x, j + step_y, j)
            t_max_x = be.where(adv & step_axis_x, t_max_x + t_delta_x, t_max_x)
            t_max_y = be.where(adv & ~step_axis_x, t_max_y + t_delta_y, t_max_y)
            active = (
                active
                & (i >= 0)
                & (i < self.num_x)
                & (j >= 0)
                & (j < self.num_y)
            )

        hit_mask = found
        n_raw = self._cap._normal_local(px_rel, py_rel)
        n_len = (n_raw * n_raw).sum(axis=1, keepdims=True) ** 0.5
        n_geom = n_raw / (n_len + _tol.tiny_for(n_len))

        dot = (directions * n_geom).sum(axis=1, keepdims=True)
        normals = be.where(dot > 0, -n_geom, n_geom)

        inf_arr = be.ones_like(t_result) * be.inf
        t_out = be.where(hit_mask, t_result, inf_arr)

        return t_out, normals, hit_mask, n_geom

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame: the array's rectangular footprint,
            extruded through the z-slab every cap is contained in.
        """
        t_vec = np.array(transform[0], dtype=float)
        R = np.array(transform[1], dtype=float)

        pitch_x = as_float(self.pitch_x)
        pitch_y = as_float(self.pitch_y)
        x_ext = self.num_x * pitch_x / 2.0
        y_ext = self.num_y * pitch_y / 2.0
        z_min, z_max = self._z_slab()

        corners_local = np.array(
            [
                [-x_ext, -y_ext, z_min],
                [-x_ext, y_ext, z_min],
                [x_ext, -y_ext, z_min],
                [x_ext, y_ext, z_min],
                [-x_ext, -y_ext, z_max],
                [-x_ext, y_ext, z_max],
                [x_ext, -y_ext, z_max],
                [x_ext, y_ext, z_max],
            ],
            dtype=float,
        )
        corners_global = corners_local @ R.T + t_vec
        return AABB(corners_global.min(axis=0), corners_global.max(axis=0))
