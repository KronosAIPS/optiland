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
from optiland.nonsequential._utils import clamp_int
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


def accumulator_dtype():
    """The dtype detectors accumulate in: the widest float the active device has.

    float64 on the NumPy backend and on Torch's ``cpu`` and ``cuda`` devices.
    On Torch's ``mps`` device (Apple GPU) it is float32, because Metal has
    no double type and torch refuses a float64 tensor there. A float32
    accumulator adds a rounding error of at most about ``K * 6e-8`` relative
    to a bin that received ``K`` contributions (the usual ``n * u`` bound;
    a random-walk estimate is ``sqrt(K) * 6e-8``), which for any bin whose
    Monte Carlo error is ``1/sqrt(K)`` is negligible beside it. The choice
    is made per call so a scene built after ``be.set_device`` sees the
    device that is active then.

    Returns:
        ``be.float64`` or ``be.float32`` (NumPy dtype aliases; the Torch
        creation functions map them to the matching ``torch.dtype``).
    """
    if be.get_backend() == "torch":
        try:
            device = str(be.get_device())
        except Exception:  # noqa: BLE001 - defensive: no device query, assume float64 is fine
            device = "cpu"
        if device.startswith("mps"):
            return be.float32
    return be.float64


def _new_flat_accumulator(size: int):
    """Create a persistent accumulation buffer in :func:`accumulator_dtype`.

    The buffer lives on the active backend and device: a NumPy float64
    array, or a Torch tensor on the current device in the widest float that
    device has (float64 on ``cpu`` and ``cuda``, float32 on ``mps``). It is
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
        A zero-filled array/tensor of shape ``(size,)``.
    """
    buf = be.zeros((size,), dtype=accumulator_dtype())
    if getattr(buf, "requires_grad", False):
        buf = buf.detach()
    return buf


#: The largest number of elements a detector's row accumulator holds (2 MiB
#: of float64), and the largest number of rows it is split into.
_BIN_ROWS_MAX_ELEMENTS = 1 << 18
_BIN_ROWS_MAX = 1 << 16


def bin_rows(size: int) -> int:
    """How many rows a float64 accumulator of ``size`` bins is split into.

    The largest power of two with ``rows * size <= _BIN_ROWS_MAX_ELEMENTS``,
    at most :data:`_BIN_ROWS_MAX` and at least 1. A function of the
    detector's size alone, so the same for every batch size and device.

    Args:
        size: Number of bins.

    Returns:
        The number of rows, a power of two.
    """
    rows = min(_BIN_ROWS_MAX, max(1, _BIN_ROWS_MAX_ELEMENTS // max(int(size), 1)))
    return 1 << (rows.bit_length() - 1)


def _new_bin_accumulator(size: int):
    """Create a detector's persistent accumulation buffer, split into rows by ray.

    A plain scatter-add into a bin adds its contributions one after another,
    so a bin that receives ``K`` of them rounds as ``K u``, not as the
    ``log2(K) u`` of a pairwise sum: the catalogue's window adds 3.7 million
    equal weights into one bin and its ledger closed to 4.0e-11 against the
    1e-11 of section 10.3 of the theory (KronosNSRT issue 23). Where the
    device has a float64 accumulator the buffer is therefore ``(rows,
    size)``: a contribution of ray ``i`` is added into row ``i % rows``
    (:func:`_accumulate_into`), so a bin's ``K`` contributions are spread over
    ``rows`` running sums of about ``K / rows`` terms, and the rows are
    reduced pairwise when the detector is read (:func:`bin_values`). The
    error per bin is about ``(K / rows + log2 rows) u``.

    The row is chosen by the ray's id, not by its position in the batch, so
    a row receives the same contributions in the same order whatever the
    batch size: wherever a plain scatter-add was bit-identical across batch
    sizes this one is too. The buffer is added into in place and never
    rebound, as before. Where the widest float is float32 (Apple's ``mps``)
    the buffer is the flat one of :func:`_new_flat_accumulator`, accumulated
    by the grouped scatter-add as before.

    Args:
        size: Number of bins.

    Returns:
        A zero-filled ``(rows, size)`` float64 buffer, or a flat float32 one.
    """
    if accumulator_dtype() != be.float64:
        return _new_flat_accumulator(size)
    buf = be.zeros((bin_rows(size), size), dtype=be.float64)
    if getattr(buf, "requires_grad", False):
        buf = buf.detach()
    return buf


def bin_values(buffer):
    """A detector buffer's bins: its rows reduced pairwise, or the flat buffer itself.

    Args:
        buffer: A buffer from :func:`_new_bin_accumulator` or
            :func:`_new_flat_accumulator`, or ``None``.

    Returns:
        The flat ``(size,)`` bins on the buffer's library and device (a view
        of the buffer when it has one row), attached to any graph the
        contributions carried; ``None`` for ``None``.
    """
    if buffer is None or len(buffer.shape) == 1:
        return buffer
    rows = buffer
    while rows.shape[0] > 1:
        rows = rows[0::2] + rows[1::2]
    return rows[0]


def _row_flat_index(buffer, flat, key):
    """The flat index into a row buffer: row ``key % rows``, bin ``flat``.

    Args:
        buffer: A ``(rows, size)`` accumulation buffer.
        flat: Bin index per contribution, in the buffer's index format.
        key: Ray id per contribution (same format), or ``None`` to deal the
            contributions round-robin by position.

    Returns:
        The index into the buffer's flat storage.
    """
    rows, size = int(buffer.shape[0]), int(buffer.shape[1])
    if rows == 1:
        return flat
    if key is None:
        if is_torch_tensor(flat):
            import torch  # noqa: PLC0415

            key = torch.arange(flat.shape[0], device=flat.device, dtype=flat.dtype)
        else:
            key = np.arange(np.shape(flat)[0])
    return (key % rows) * size + flat


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


def _accumulate_into(buffer, flat_np, contribution, key=None) -> None:
    """Scatter-add ``contribution`` into ``buffer`` in place, in the buffer's dtype.

    ``buffer`` is a persistent accumulation buffer created by
    :func:`_new_flat_accumulator` (float64 wherever the device has it,
    float32 on Apple's ``mps``; on the active backend and device).
    ``contribution`` may be a NumPy array or a backend array/tensor in the
    working (traversal) dtype; it is cast to the buffer's dtype before
    accumulating. On the Torch backend the cast keeps ``contribution``'s
    autograd graph attached -- a dtype cast is itself a differentiable op,
    so a gradient reaching ``contribution`` (e.g. from
    ``total_flux.backward()``) still reaches its source through the cast
    and the in-place ``index_add_``; ``buffer`` itself never needs to
    require grad on its own account (see :func:`_new_flat_accumulator`).

    ``flat_np`` is an integer index array of whichever library the detector
    computed it in: the bin index is a discrete function of the hit position
    (docs/theory/12_gpu_mapping.md S12.5) and so carries no gradient, but it
    is computed beside the ray state and stays there.

    A ``(rows, size)`` buffer from :func:`_new_bin_accumulator` takes each
    contribution into the row of its ray (``key % rows``); see there.

    Args:
        buffer: Persistent accumulation buffer, shape (size,) or (rows, size).
        flat_np: Flat bin indices for each contribution, shape (K,), integer.
        contribution: Values to add, shape (K,), any array-like.
        key: The ray id of each contribution, shape (K,), for a row buffer:
            ``rays.ray_id``. ``None`` deals the contributions round-robin by
            position (bounded the same way, but not batch-invariant).
    """
    if is_torch_tensor(buffer):
        import torch  # noqa: PLC0415

        from optiland.backend.torch_backend.capabilities import (  # noqa: PLC0415
            to_device_dtype,
        )

        idx = _flat_index_like(buffer, flat_np)
        # Onto the buffer's device, then into its dtype: a device tensor
        # bound for a float64 host buffer is moved before it is cast (the
        # one-call copy from Apple's mps writes zeros there). On one device
        # this is the plain cast it always was.
        src = to_device_dtype(contribution, buffer.device, buffer.dtype)
        if buffer.dim() == 2:
            rows_key = None if key is None else _flat_index_like(buffer, key)
            buffer.view(-1).index_add_(0, _row_flat_index(buffer, idx, rows_key), src)
        elif buffer.dtype == torch.float32:
            _grouped_index_add_(buffer, idx, src)
        else:
            buffer.index_add_(0, idx, src)
        return

    contrib_np = to_numpy(contribution).astype(buffer.dtype, copy=False)
    if buffer.ndim == 2:
        key_np = None if key is None else np.asarray(to_numpy(key))
        flat_np = _row_flat_index(buffer, np.asarray(to_numpy(flat_np)), key_np)
        np.add.at(buffer.reshape(-1), flat_np, contrib_np)
        return
    np.add.at(buffer, flat_np, contrib_np)


#: The largest number of partial-sum rows the grouped float32 accumulation
#: uses, and the largest scratch it will allocate (in elements) for them.
_GROUPED_ACC_MAX_GROUPS = 256
_GROUPED_ACC_MAX_SCRATCH = 1 << 24
#: Below this many contributions per call a bare scatter-add is used.
_GROUPED_ACC_MIN_CONTRIBUTIONS = 64


def _grouped_index_add_(buffer, idx, src) -> None:
    """Scatter-add into a float32 buffer with a bounded rounding error.

    A bare ``index_add_`` into a float32 buffer rounds the running sum once
    per contribution, so a bin that receives ``K`` contributions carries an
    error of up to ``K * u32`` relative (``u32 = 6e-8``), and for equal
    contributions the rounding is systematic, not random: measured 1e-4
    relative for 18,000 equal hits in one bin on an Apple GPU. This is the
    device that has no float64 accumulator (see :func:`accumulator_dtype`),
    so the remedy is the one GPU Monte Carlo codes use without double
    atomics: partial sums. The contributions are dealt round-robin into
    ``G`` rows of a scratch ``(G, size)`` buffer (each bin then receives
    about ``K / G`` adds per row), the rows are reduced pairwise
    (``log2 G`` levels), and the result is added to ``buffer`` once. The
    error per bin is about ``(K / G + log2 G + 1) * u32``; measured 6e-8
    relative for 67,000 contributions per bin with ``G = 256``. Every step
    is an ordinary differentiable op, so autograd through the detector is
    unchanged. ``G`` is chosen from the number of contributions and capped
    so the scratch never exceeds :data:`_GROUPED_ACC_MAX_SCRATCH` elements;
    a call with fewer than :data:`_GROUPED_ACC_MIN_CONTRIBUTIONS`
    contributions uses the bare scatter-add, whose error is then bounded
    by that count.

    Args:
        buffer: The persistent float32 accumulation buffer, shape (size,).
        idx: Flat bin index per contribution, a LongTensor on the buffer's
            device, shape (K,).
        src: Contributions, float32 on the buffer's device, shape (K,).
    """
    import torch  # noqa: PLC0415

    k = int(src.shape[0])
    size = int(buffer.shape[0])
    groups = min(
        _GROUPED_ACC_MAX_GROUPS,
        max(1, k // _GROUPED_ACC_MIN_CONTRIBUTIONS),
        max(1, _GROUPED_ACC_MAX_SCRATCH // max(size, 1)),
    )
    if groups <= 1:
        buffer.index_add_(0, idx, src)
        return
    group = torch.arange(k, device=src.device, dtype=idx.dtype) % groups
    scratch = torch.zeros((groups, size), dtype=src.dtype, device=src.device)
    scratch.view(-1).index_add_(0, group * size + idx, src)
    while scratch.shape[0] > 1:
        if scratch.shape[0] % 2:
            scratch = torch.cat([scratch, torch.zeros_like(scratch[:1])], dim=0)
        scratch = scratch[0::2] + scratch[1::2]
    buffer.add_(scratch[0])


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

    Any detector can also **book arriving flux by ghost order**:
    ``reflection_bins=K`` keeps, beside the detector's own record, the flux
    that arrives with exactly ``k`` reflections in its history for
    ``k = 0 .. K-1`` and the flux with ``K`` or more in one overflow bin
    (:meth:`record_reflections`, :meth:`reflection_histogram`). The order is
    the ray's own ``reflections`` count, carried in the ray state, so a
    ghost is separated from the direct beam even where the two land on the
    same pixel (``docs/theory/11_validation_catalogue.md`` 11.4.6, 11.4.21).

    Attributes:
        cs: Coordinate system defining detector position and orientation.
        geometry: Surface geometry that defines the detector area.
        name: Optional human-readable label.
        absorb: Whether a hit terminates the ray. False => the ray is
            recorded and passes through unaffected.
        side: Which side of the surface is live -- ``"both"`` (default),
            ``"front"``, or ``"back"``. See :meth:`intersect`.
        reflection_bins: Number of exact reflection-count bins (0, the
            default, keeps no histogram).
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        name: str = "",
        absorb: bool = True,
        side: str = "both",
        reflection_bins: int = 0,
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
            reflection_bins: Keep a histogram of arriving flux by reflection
                count, with this many exact bins (``0 .. K-1``) and one
                overflow bin. 0 (the default) keeps none and costs nothing.

        Raises:
            ValueError: If ``side`` is not one of the three accepted values,
                or ``reflection_bins`` is negative.
        """
        if side not in ("both", "front", "back"):
            raise ValueError(
                f"side must be 'both', 'front', or 'back'; got {side!r}."
            )
        if int(reflection_bins) < 0:
            raise ValueError(
                f"reflection_bins must be 0 or positive; got {reflection_bins!r}."
            )
        self.cs = cs
        self.geometry = geometry
        self.name = name
        self.absorb = bool(absorb)
        self.side = side
        self.reflection_bins = int(reflection_bins)
        self.reset_reflection_tally()
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
            from optiland.nonsequential.parameter_register import (  # noqa: PLC0415
                attach_placement,
            )

            translation, rot = _get_transform(self.cs)
            # Attached when the placement carries a gradient: the value is
            # the host build's, the derivative flows to the tensors.
            self._frame = attach_placement(
                self.cs, be.array(translation), be.array(rot)
            )
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

    # ------------------------------------------------------------------
    # Arriving flux by reflection count (ghost order)
    # ------------------------------------------------------------------

    def reset_reflection_tally(self) -> None:
        """Clear the reflection-count histogram.

        Three accumulators of ``reflection_bins + 1`` entries -- the flux,
        the flux squared and the hit count per bin, the last entry being
        the overflow bin -- in :func:`accumulator_dtype`, on the active
        backend and device. Every subclass's :meth:`reset` calls this, so a
        trace starts from zero exactly as the detector's own record does.
        """
        if self.reflection_bins > 0:
            size = self.reflection_bins + 1
            self._refl_flux = _new_bin_accumulator(size)
            self._refl_flux_sq = _new_bin_accumulator(size)
            self._refl_hits = _new_bin_accumulator(size)
        else:
            self._refl_flux = None
            self._refl_flux_sq = None
            self._refl_hits = None

    def record_reflections(self, rays: NSQRayBundle, hit_mask) -> None:
        """Book each hit ray's flux in the bin of its reflection count.

        Called by the trace loop beside :meth:`record`, for the same rays.
        A ray with ``k < reflection_bins`` reflections lands in bin ``k``; one
        with more lands in the overflow bin, so the bins always sum to the
        flux the detector received and nothing is dropped from the count.
        Index arithmetic and one in-place scatter-add per accumulator,
        beside the ray state: no host read, and a no-op (without asking the
        mask anything) when the detector keeps no histogram.

        The flux squared per bin is kept for the standard error of a bin
        whose contributions are independent draws -- the sum of squared
        weights per cell of ``docs/theory/03_monte_carlo.md`` R-03-7 -- which
        holds when each launched ray reaches the detector at most once (see
        :meth:`ReflectionHistogram.standard_error`).

        Args:
            rays: Current ray bundle.
            hit_mask: Rays recorded on this detector this bounce, shape (N,).
        """
        if self.reflection_bins <= 0:
            return
        bins = clamp_int(rays.reflections, 0, self.reflection_bins)
        zero = be.zeros_like(rays.flux)
        weight = be.where(hit_mask, rays.flux, zero)
        hits = be.where(hit_mask, be.ones_like(rays.flux), zero)
        key = getattr(rays, "ray_id", None)
        _accumulate_into(self._refl_flux, bins, weight, key=key)
        _accumulate_into(self._refl_flux_sq, bins, weight * weight, key=key)
        _accumulate_into(self._refl_hits, bins, hits, key=key)

    def reflection_histogram(self):
        """The arriving flux by reflection count, or ``None`` without bins.

        Returns:
            A :class:`~optiland.nonsequential.results.reflection_histogram
            .ReflectionHistogram`, read back once (a host copy), or ``None``
            when the detector was built with ``reflection_bins=0``.
        """
        if self.reflection_bins <= 0:
            return None
        from optiland.nonsequential.results.reflection_histogram import (  # noqa: PLC0415
            ReflectionHistogram,
        )

        refl_flux = bin_values(self._refl_flux)
        flux = np.asarray(to_numpy(refl_flux), dtype=np.float64)
        flux_sq = np.asarray(to_numpy(bin_values(self._refl_flux_sq)), dtype=np.float64)
        hits = np.rint(np.asarray(to_numpy(bin_values(self._refl_hits)), dtype=np.float64))
        return ReflectionHistogram(
            flux=flux[:-1].copy(),
            flux_sq=flux_sq[:-1].copy(),
            num_rays_hit=hits[:-1].astype(np.int64),
            overflow_flux=float(flux[-1]),
            overflow_flux_sq=float(flux_sq[-1]),
            overflow_num_rays_hit=int(hits[-1]),
            data=refl_flux,
        )

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
