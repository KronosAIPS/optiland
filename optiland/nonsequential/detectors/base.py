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
from optiland.nonsequential.components.base import _get_transform

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


def floor_to_int(x):
    """Floor ``x`` to an integer array of the same library and device.

    The bin a hit lands in is a discrete function of the landing position,
    so it never carries a gradient -- but it does not have to be computed
    on the host either, and computing it there is a device-to-host copy per
    detector per bounce.

    Args:
        x: Continuous pixel coordinate, shape (N,).

    Returns:
        ``floor(x)`` as an int64 array/tensor beside ``x``.
    """
    if is_torch_tensor(x):
        import torch  # noqa: PLC0415

        return torch.floor(x).to(torch.int64)
    return np.floor(x).astype(np.int64)


def int_to_float_like(idx, like):
    """Cast an integer index array back to ``like``'s float dtype.

    Args:
        idx: Integer array/tensor.
        like: Array/tensor whose dtype and device to match.

    Returns:
        ``idx`` as a float array/tensor beside ``like``.
    """
    if is_torch_tensor(idx):
        return idx.to(dtype=like.dtype)
    return idx.astype(np.float64)


def clamp_int(idx, lo: int, hi: int):
    """Clamp an integer index array into ``[lo, hi]``.

    ``be.clip``/``be.maximum`` push both operands through ``array()``, which
    carries the working float precision -- so an integer clamp written with
    them comes back as floats. ``be.where`` does not.

    Args:
        idx: Integer array/tensor.
        lo: Lower bound, inclusive.
        hi: Upper bound, inclusive.

    Returns:
        The clamped indices, still integer.
    """
    if is_torch_tensor(idx):
        import torch  # noqa: PLC0415

        return torch.clamp(idx, lo, hi)
    return np.clip(idx, lo, hi)


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

    ``flat_np`` is always a plain NumPy int64 index array: the bin index is
    a discrete function of the hit position (docs/theory/12_gpu_mapping.md
    S12.5) and is computed on the host in every splatting detector,
    gradient support or not -- this is unchanged from before this fix.

    Args:
        buffer: Persistent float64 accumulation buffer, shape (size,).
        flat_np: Flat bin indices for each contribution, shape (K,), int64.
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

    Attributes:
        cs: Coordinate system defining detector position and orientation.
        geometry: Surface geometry that defines the detector area.
        name: Optional human-readable label.
        absorb: Whether a hit terminates the ray. False => the ray is
            recorded and passes through unaffected.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        name: str = "",
        absorb: bool = True,
    ) -> None:
        """Initialize BaseDetector.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            name: Optional label.
            absorb: Whether a hit terminates the ray (default True).
                False makes the detector transmissive: the hit is recorded
                and the ray continues with its direction unchanged.
        """
        self.cs = cs
        self.geometry = geometry
        self.name = name
        self.absorb = bool(absorb)
        self._frame = None

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
        """Drop the cached transform; the next bounce resolves it again."""
        self._frame = None

    def intersect(
        self, rays: NSQRayBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find ray intersections with this detector surface.

        Args:
            rays: Ray bundle in global coordinates.

        Returns:
            Tuple (t, normals, hit_mask) in global frame.
        """
        t_arr, R_arr = self.frame()

        positions_g = be.stack([rays.x, rays.y, rays.z], axis=1)
        directions_g = be.stack([rays.L, rays.M, rays.N], axis=1)

        positions_l = (positions_g - t_arr) @ R_arr
        directions_l = directions_g @ R_arr

        # Self-intersection accept threshold from the ray's *global*-frame
        # magnitude -- see BaseComponent.intersect for why global, not local.
        t_min = _tol.accept_t_min(be.abs(positions_g).max())
        t_hit, normals_l, hit_mask, _n_geom_l = self.geometry.ray_intersect(
            positions_l, directions_l, eps=t_min
        )

        # Geometry may return numpy arrays even in torch-backend mode (geometry
        # internals are numpy-based). Convert to the current backend format so
        # that be.where dispatches correctly in both NumPy and Torch paths.
        t_hit = be.array(t_hit)
        normals_l = be.array(normals_l)
        hit_mask = be.array(hit_mask)

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
        return t_hit, normals_g, hit_mask

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
