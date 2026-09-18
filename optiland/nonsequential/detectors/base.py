"""Base detector for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import is_torch_tensor, to_numpy
from optiland.nonsequential import _tol
from optiland.nonsequential.components.base import (
    _get_transform,
    advance_to_hit_in_frame,
    coordinate_magnitude,
    offset_origin_from_surface,
    resident_table,
)

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.geometry.base import AABB, ComponentGeometry
    from optiland.nonsequential.ray_bundle import NSQRayBundle


# ---------------------------------------------------------------------------
# Shared accumulation-buffer helpers for the splatting detectors
# (IrradianceDetector, FarFieldDetector, SpectralDetector).
#
# The theory requirement (docs/theory/08_precision.md R-08-1, R-08-3;
# 12_gpu_mapping.md R-12-7) is that a detector image accumulates in the
# accumulation dtype (float64) whatever the traversal dtype T is, and that
# the accumulation is in place so a per-bounce splat does not reallocate the
# pixel buffer (the cost item NS7 names against the non-in-place
# ``be.index_add``). These three helpers implement that once, shared by all
# three splatting detectors, without changing the shared ``optiland.backend``
# abstraction itself -- the fix stays inside ``optiland/nonsequential``.
# ---------------------------------------------------------------------------


def _new_flat_accumulator(size: int):
    """Create a persistent float64 accumulation buffer.

    The buffer lives on the active backend and device: a NumPy float64
    array, or a Torch float64 tensor on the current device. It is
    guaranteed to not itself require grad -- regardless of the ambient
    grad-mode setting -- so it can be scatter-added into *in place* across
    many ``record()`` calls; PyTorch refuses an in-place write into a leaf
    tensor that requires grad (a torch.Tensor whose graph-tracked source
    is added in via :func:`_accumulate_into` still carries a gradient back
    to that source, since the cast and the scatter-add are both ordinary
    differentiable ops applied on top of this buffer).

    Args:
        size: Number of flat elements (e.g. ``ny * nx``).

    Returns:
        A zero-filled float64 array/tensor of shape ``(size,)``.
    """
    buf = be.zeros((size,), dtype=be.float64)
    if getattr(buf, "requires_grad", False):
        buf = buf.detach()
    return buf


def _flat_index_like(buffer, flat_np):
    """Convert a flat-index array to ``buffer``'s format.

    Args:
        buffer: The accumulation buffer (NumPy array or Torch tensor).
        flat_np: Flat pixel/bin indices, shape (N,), integer. Either a
            NumPy array or an integer tensor already on the device.

    Returns:
        ``flat_np`` unchanged for a NumPy buffer; a ``LongTensor`` on the
        same device as ``buffer`` for a Torch buffer.
    """
    if is_torch_tensor(buffer):
        import torch  # noqa: PLC0415

        if is_torch_tensor(flat_np):
            return flat_np.to(device=buffer.device, dtype=torch.long)
        return torch.from_numpy(flat_np).to(device=buffer.device, dtype=torch.long)
    return flat_np


def _accumulate_into(buffer, flat_np, contribution) -> None:
    """Scatter-add ``contribution`` into ``buffer`` in place, in float64.

    ``buffer`` is a persistent accumulation buffer created by
    :func:`_new_flat_accumulator` (float64, on the active backend and
    device). ``contribution`` may be a NumPy array or a backend
    array/tensor in the working (traversal) dtype; it is cast to float64
    before accumulating. On the Torch backend the cast keeps
    ``contribution``'s autograd graph attached -- a dtype cast is itself a
    differentiable op, so a gradient reaching ``contribution`` (e.g. from
    ``total_flux.backward()``) still reaches its source through the cast
    and the in-place ``index_add_``; ``buffer`` itself never needs to
    require grad on its own account (see :func:`_new_flat_accumulator`).

    ``flat_np`` is an integer index array of whichever library the detector
    computed it in: the bin index is a discrete function of the hit position
    (docs/theory/12_gpu_mapping.md S12.5) and so carries no gradient, but it
    is computed beside the ray state and stays there.

    Args:
        buffer: Persistent float64 accumulation buffer, shape (size,).
        flat_np: Flat bin indices for each contribution, shape (K,), integer.
        contribution: Values to add, shape (K,), any array-like.
    """
    if is_torch_tensor(buffer):
        import torch  # noqa: PLC0415

        idx = _flat_index_like(buffer, flat_np)
        if is_torch_tensor(contribution):
            src64 = contribution.to(dtype=torch.float64)
        else:
            src64 = torch.as_tensor(
                contribution, dtype=torch.float64, device=buffer.device
            )
        buffer.index_add_(0, idx, src64)
        return

    contrib_np = to_numpy(contribution).astype(np.float64, copy=False)
    np.add.at(buffer, flat_np, contrib_np)


class BaseDetector(ABC):
    """Abstract base class for detectors in the NSQ scene.

    Detectors record ray data at a surface. They intersect rays (via their
    geometry) and accumulate hit data across simulation batches.

    Detectors are **absorbing by default**: a ray that reaches a detector is
    recorded and then terminated. Setting ``absorb=False`` records the ray
    but lets it continue unchanged, so a detector can be tilted into a
    converging beam to sample it mid-system without terminating it.
    Several detectors may share one scene; among the detectors a ray would
    still reach, only the nearest one sees it, so stacking *absorbing*
    detectors down a beam records the beam at the nearest plane and nothing
    beyond it -- use ``absorb=False`` (or trace one scene per plane) to
    profile a beam at several planes.

    Detectors are **two-sided by default**: a planar detector is a plane, and
    a ray crossing it from either side is recorded. ``side="front"`` (or
    ``"back"``) restricts it to hits arriving on one side, which is what a
    collector placed to read light *returning* from a surface needs -- it
    then ignores the beam travelling the other way through the same plane,
    instead of absorbing it before it ever reaches the surface under test
    (``docs/theory/11_validation_catalogue.md`` 11.4.18).

    Attributes:
        cs: Coordinate system defining detector position and orientation.
        geometry: Surface geometry that defines the detector area.
        name: Optional human-readable label.
        absorb: Whether a hit terminates the ray. False => the ray is
            recorded and passes through unaffected.
        side: Which side of the surface is live -- ``"both"`` (default),
            ``"front"``, or ``"back"``. See :meth:`intersect`.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        name: str = "",
        absorb: bool = True,
        side: str = "both",
    ) -> None:
        """Initialize BaseDetector.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            name: Optional label.
            absorb: Whether a hit terminates the ray (default True).
                False makes the detector transmissive: the hit is recorded
                and the ray continues with its direction unchanged.
            side: ``"both"`` (default, unchanged behaviour), ``"front"``, or
                ``"back"``. The front is the side the surface normal points
                toward: for every flat detector here that is the placement's
                own normal, local +z.

        Raises:
            ValueError: If ``side`` is not one of the three accepted values.
        """
        if side not in ("both", "front", "back"):
            raise ValueError(
                f"side must be 'both', 'front', or 'back'; got {side!r}."
            )
        self.cs = cs
        self.geometry = geometry
        self.name = name
        self.absorb = bool(absorb)
        self.side = side
        self._frame = None
        self._be_tables: dict[str, object] = {}
        self._be_tables_key = None
        # The two parts of the last solved hit distance, for a transmissive
        # detector's hit-point rebuild -- see intersect(). Transient
        # per-bounce scratch, rewritten by every intersect() call and
        # validated per ray before use, never scene state.
        self._local_root: tuple[np.ndarray, np.ndarray] | None = None

    def table(self, name: str, values):
        """A constant lookup table of this detector's, resident on the backend.

        Bin edges, bin centres and the like are scene data, not ray data, so
        the binning arithmetic that reads them stays where the ray state is
        -- see :func:`~optiland.nonsequential.components.base
        .resident_table`. The cache is dropped by :meth:`reset`.

        Args:
            name: Key for this table on this detector.
            values: The table, as NumPy values.

        Returns:
            The table as an array of the active backend.
        """
        return resident_table(self, name, values)

    def frame(self):
        """This detector's global->local transform, as backend arrays.

        A detector's placement is scene data, not ray data: it does not
        change during a trace, so it is resolved once and uploaded once
        (``docs/theory/12_gpu_mapping.md`` R-12-8) instead of being rebuilt
        and re-uploaded on every bounce. The cache is dropped by
        :meth:`reset`, which every trace calls before its first bounce, so a
        moved detector is picked up by the next trace.

        Returns:
            ``(translation, rotation)`` as arrays of the active backend.
        """
        if self._frame is None:
            translation, rot = _get_transform(self.cs)
            self._frame = (be.array(translation), be.array(rot))
        return self._frame

    def invalidate_frame(self) -> None:
        """Drop the cached transform and tables; the next trace rebuilds them."""
        self._frame = None
        self._be_tables = {}
        self._be_tables_key = None

    def intersect(
        self, rays: NSQRayBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find ray intersections with this detector surface.

        An **absorbing** detector is terminal: the ray stops there, its
        recorded position is never fed back into another intersection, and
        the solve is the plain one it has always been
        (``docs/build/X1_threshold_arithmetic.md`` section 7, note 4).

        A **transmissive** detector is not. The ray carries on from the
        point the loop puts it at and the next bounce tests this same plane
        again, so the hit point has to be rebuildable in the detector's own
        frame -- which means solving from an origin advanced into the
        plane's neighbourhood and keeping the advance and the residual as
        two numbers, exactly as :meth:`~optiland.nonsequential.components
        .base.BaseComponent.intersect` does. That is the only difference
        between the two branches below.

        With ``side`` set, a hit is kept only when the ray arrives on the
        requested side. The test is on the *geometric* normal ``n_geom``,
        which every geometry defines independently of the ray's direction:
        a ray arriving on the front travels against it (``d . n_geom < 0``).
        For the flat detectors ``n_geom`` is the placement's local +z, so
        "front" is the side the detector faces; for a closed geometry --
        a hemispherical collector's shell -- the same test reads as
        "hit from the side the normal points toward", with no separate
        convention.

        Args:
            rays: Ray bundle in global coordinates.

        Returns:
            Tuple (t, normals, hit_mask, n_geom) in the global frame, with
            ``n_geom`` the geometric (unflipped) surface normal.
        """
        t_arr, R_arr = self.frame()

        positions_g = be.stack([rays.x, rays.y, rays.z], axis=1)
        directions_g = be.stack([rays.L, rays.M, rays.N], axis=1)

        positions_l = (positions_g - t_arr) @ R_arr
        directions_l = directions_g @ R_arr

        # Self-intersection accept threshold from the ray's *global*-frame
        # magnitude, per ray -- see BaseComponent.intersect for why global
        # rather than local, and why per ray rather than one scalar for the
        # whole bundle.
        t_min = _tol.accept_t_min(coordinate_magnitude(rays))

        if self.absorb:
            self._local_root = None
            t_local, normals_l, hit_mask, n_geom_l = self.geometry.ray_intersect(
                positions_l, directions_l, eps=t_min
            )
            t_adv = None
        else:
            t_adv = -(positions_l * directions_l).sum(axis=1)
            positions_adv = positions_l + t_adv[:, None] * directions_l
            t_local, normals_l, hit_mask, n_geom_l = self.geometry.ray_intersect(
                positions_adv, directions_l, eps=t_min - t_adv
            )

        # Geometry may return numpy arrays even in torch-backend mode (geometry
        # internals are numpy-based). Convert to the current backend format so
        # that be.where dispatches correctly in both NumPy and Torch paths.
        t_local = be.array(t_local)
        normals_l = be.array(normals_l)
        hit_mask = be.array(hit_mask)
        n_geom_l = be.array(n_geom_l)

        if t_adv is None:
            t_hit = t_local
        else:
            t_hit = t_local + t_adv
            self._local_root = (t_adv, t_local)

        if self.side != "both":
            # The live side, tested on the geometric normal (see the docstring).
            facing = (directions_l * n_geom_l).sum(axis=1)
            wanted = facing < 0.0 if self.side == "front" else facing > 0.0
            hit_mask = hit_mask & wanted
            t_hit = be.where(wanted, t_hit, be.full_like(t_hit, be.inf))

        # Note the accept/reject decision before overwriting t_hit -- see
        # BaseComponent.intersect for why checking the post-overwrite value
        # would be vacuous.
        accepted = t_hit > t_min
        t_hit = be.where(accepted, t_hit, be.full_like(t_hit, be.inf))
        hit_mask = hit_mask & accepted

        alive_be = be.array(rays.alive)
        t_hit = be.where(alive_be, t_hit, be.full_like(t_hit, be.inf))
        hit_mask = hit_mask & alive_be

        normals_g = normals_l @ R_arr.T
        n_geom_g = n_geom_l @ R_arr.T
        return t_hit, normals_g, hit_mask, n_geom_g

    def advance_to_hit(self, rays: NSQRayBundle, t, hit_mask) -> None:
        """Move every ray in ``hit_mask`` onto this detector's plane.

        Only a transmissive detector needs this -- see :meth:`intersect`.
        An absorbing detector has no ``_local_root``, and the shared helper
        then falls back to the plain global update per ray.

        Args:
            rays: Ray bundle, updated in place.
            t: Per-ray hit distance [mm], shape (N,).
            hit_mask: Rays that reached this detector, shape (N,).
        """
        advance_to_hit_in_frame(
            rays, t, hit_mask, self._local_root, self.frame()
        )

    def offset_from_surface(self, rays: NSQRayBundle, n_geom, hit_mask) -> None:
        """Push a transmitted ray's origin clear of this detector's plane.

        R-07-6, for the same reason a surface needs it: a ray crossing a
        plane at a grazing angle turns the half-ulp it lands off that plane
        by into a path-length root above the accept threshold, and the plane
        records it a second time. Only rays the detector let through are
        offset; a ray an absorbing detector stopped is terminal.

        Args:
            rays: Ray bundle, updated in place.
            n_geom: Geometric surface normal in the global frame, (N, 3).
            hit_mask: Rays that crossed this detector, shape (N,).
        """
        offset_origin_from_surface(rays, n_geom, hit_mask)

    @abstractmethod
    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Accumulate ray data for rays that hit this detector.

        Args:
            rays: Current ray bundle. Positions have NOT yet been advanced
                to the hit point; use t to compute hit positions.
            t: Hit distances [mm], shape (N,).
            hit_mask: Boolean mask of rays hitting this detector, shape (N,).
        """

    @abstractmethod
    def get_result(self):
        """Return the accumulated result object.

        Returns:
            A result object (IrradianceMap, FarFieldPattern, etc.).
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated data for reuse in a new simulation."""

    @property
    def bounding_box(self) -> AABB:
        """AABB of this detector in global coordinates."""
        transform = _get_transform(self.cs)
        return self.geometry.bounding_box(transform)
