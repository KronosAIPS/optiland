"""Ray Bundle for Non-Sequential Raytracing.

Defines NSQRayBundle -- the core in-memory ray state.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import optiland.backend as be

# Maximum simultaneous medium nesting depth a ray's medium_stack can record
# (e.g. a cemented triplet in an immersion fluid inside a sealed housing is
# 4). A push past this depth raises MediumStackOverflowError rather than
# silently wrapping or dropping the entry.
MEDIUM_STACK_MAX_DEPTH = 8

# Sentinel medium id for "no medium" / unused stack slots.
MEDIUM_STACK_EMPTY = -1

# Whether a push past MEDIUM_STACK_MAX_DEPTH raises MediumStackOverflowError.
#
# True (the default) keeps the documented behaviour, at the cost of the one
# host read left in the push/pop kernel: deciding whether to raise means
# reducing a per-ray mask to a Python bool, which on a non-CPU device is a
# synchronisation. Set False for a device path with no synchronisation at
# all: the push then saturates (the ray keeps the medium it had) and the
# event is counted in ``medium_stack_underflows`` alongside the pop-on-empty
# events, i.e. reported as a stack inconsistency rather than raised.
MEDIUM_STACK_OVERFLOW_RAISES = True


class MediumStackOverflowError(Exception):
    """A ray's medium nesting exceeded ``MEDIUM_STACK_MAX_DEPTH``.

    Raised rather than silently wrapping or truncating: this indicates
    either a pathologically deep (past any realistic optical assembly)
    volume nesting, or a geometry defect that pushes without ever popping.
    """


def backend_int_full(shape, fill_value: int, like=None, bits: int = 64):
    """Return an integer array filled with ``fill_value``.

    The result is built with the same array library, and on the same device,
    as ``like`` -- a NumPy ndarray for NumPy ray state, a torch Tensor on the
    tensor's own device for torch ray state. Integer bookkeeping fields are
    built through this helper rather than ``be.zeros``/``be.full`` because
    those carry the backend's floating-point precision and grad flag, neither
    of which applies to an index table.

    Args:
        shape: Output shape.
        fill_value: Integer fill value.
        like: Array whose library and device the result follows. ``None``
            (or a NumPy array) gives a NumPy result.
        bits: 64 for the medium ids, 32 for depths and counters.

    Returns:
        An integer array/tensor of ``shape``.
    """
    if be.is_torch_tensor(like):
        import torch  # noqa: PLC0415

        dtype = torch.int64 if bits == 64 else torch.int32
        return torch.full(
            tuple(shape), int(fill_value), dtype=dtype, device=like.device
        )
    dtype = np.int64 if bits == 64 else np.int32
    return np.full(tuple(shape), int(fill_value), dtype=dtype)


def backend_bool_full(shape, fill_value: bool, like=None):
    """Return a boolean array filled with ``fill_value``.

    Built with the same array library, and on the same device, as ``like``.
    ``be.ones_like``/``be.zeros_like`` carry the backend's working *float*
    precision, which a mask must not.

    Args:
        shape: Output shape.
        fill_value: Boolean fill value.
        like: Array whose library and device the result follows.

    Returns:
        A boolean array/tensor of ``shape``.
    """
    if be.is_torch_tensor(like):
        import torch  # noqa: PLC0415

        return torch.full(
            tuple(shape), bool(fill_value), dtype=torch.bool, device=like.device
        )
    return np.full(tuple(shape), bool(fill_value), dtype=bool)


def backend_masked_fill(table, mask, value: int):
    """Write ``value`` into every entry of ``table`` where ``mask`` is True.

    In place and without allocating a second table; elementwise, so it costs
    no host read on either array library.

    Args:
        table: Integer table, modified in place.
        mask: Boolean mask broadcastable to ``table``'s shape.
        value: Integer to write.

    Returns:
        ``table``.
    """
    if be.is_torch_tensor(table):
        table.masked_fill_(mask, value)
        return table
    np.copyto(table, value, where=mask)
    return table


def backend_gather_slot(table, index):
    """Read one entry per row: ``table[row, index[row]]``.

    The per-ray gather a device kernel performs, on either array library.

    Args:
        table: Integer table, shape (N, D).
        index: Per-row column index, shape (N,), every value in [0, D).

    Returns:
        The gathered column, shape (N,).
    """
    if be.is_torch_tensor(table):
        import torch  # noqa: PLC0415

        return torch.gather(table, 1, index[:, None].to(torch.int64))[:, 0]
    return np.take_along_axis(table, index[:, None], axis=1)[:, 0]


def backend_scatter_slot(table, index, values):
    """Write one entry per row: ``table[row, index[row]] = values[row]``.

    The per-ray scatter a device kernel performs. Writes into ``table`` in
    place, so the caller owns a table no other bundle shares. A row that
    should not be written passes its current value back in ``values``; the
    masking is in the values, never in the index, so every lane writes.

    Args:
        table: Integer table, shape (N, D), modified in place.
        index: Per-row column index, shape (N,), every value in [0, D).
        values: Per-row value to write, shape (N,).

    Returns:
        ``table``.
    """
    if be.is_torch_tensor(table):
        import torch  # noqa: PLC0415

        table.scatter_(1, index[:, None].to(torch.int64), values[:, None])
        return table
    np.put_along_axis(table, index[:, None], values[:, None], axis=1)
    return table


def _rows_copy(arr, idx):
    """Return an independent copy of ``arr``'s rows at ``idx``."""
    sub = arr[idx]
    return sub.clone() if be.is_torch_tensor(sub) else sub.copy()


def _rows_concat(arrays):
    """Concatenate arrays along axis 0, NumPy or torch."""
    if be.is_torch_tensor(arrays[0]):
        import torch  # noqa: PLC0415

        return torch.cat(list(arrays))
    return np.concatenate(arrays)


@dataclass
class NSQRayBundle:
    """Central in-memory object carrying all live ray state.

    All arrays are shape (N,) or (N, 3). Arrays may be NumPy ndarray or
    torch Tensor depending on the active TracerBackend.

    Attributes:
        x: Position x-component [mm], shape (N,).
        y: Position y-component [mm], shape (N,).
        z: Position z-component [mm], shape (N,).
        L: Direction x-component (unit vector), shape (N,).
        M: Direction y-component (unit vector), shape (N,).
        N: Direction z-component (unit vector), shape (N,).
        flux: Current flux / throughput weight, shape (N,).
        wavelength: Wavelength [µm], shape (N,).
        n_current: Refractive index of current medium, shape (N,).
        bounce: Number of surface hits, shape (N,).
        alive: Boolean mask -- False for dead/terminated rays, shape (N,).
        ray_id: Unique ray identifier, shape (N,). None if not assigned.
        k_current: Extinction coefficient of the current medium at each
            ray's wavelength, shape (N,). 0 for a non-absorbing medium
            (vacuum, or any material with no measured extinction data).
            Feeds Beer-Lambert bulk absorption over the distance a
            ray travels before its next hit; updated alongside ``n_current``
            wherever a ray crosses into a new medium.
        medium_stack: Nested medium ids the ray has entered but not yet
            exited, shape (N, ``MEDIUM_STACK_MAX_DEPTH``), int64. Slots at
            or beyond ``medium_depth`` for a given row are
            ``MEDIUM_STACK_EMPTY``. An integer table of the same array
            library and device as the rest of the ray state (NumPy for the
            NumPy backend, a torch Tensor on the backend's device for the
            torch backend): pushed/popped by ``RefractiveComponent.interact``
            alongside ``n_current``, with masked integer arithmetic over the
            whole table, never per-row host indexing. Bookkeeping only,
            never differentiated.
        medium_depth: Stack pointer -- number of valid entries in
            ``medium_stack`` for each ray, shape (N,), int32, same library
            and device as ``medium_stack``.
        medium_stack_underflows: Cumulative count of stack inconsistencies
            for each ray -- a pop attempt on an empty ``medium_stack`` (a
            ray exiting a volume it never entered), and, when
            ``MEDIUM_STACK_OVERFLOW_RAISES`` is False, a push past
            ``MEDIUM_STACK_MAX_DEPTH``. Shape (N,), int32, same library and
            device as ``medium_stack``; summed across all rays into
            ``Diagnostics.medium_stack_underflows`` at the end of a trace.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    L: np.ndarray
    M: np.ndarray
    N: np.ndarray
    flux: np.ndarray
    wavelength: np.ndarray
    n_current: np.ndarray
    bounce: np.ndarray
    alive: np.ndarray
    ray_id: np.ndarray | None = None
    k_current: np.ndarray | None = None
    medium_stack: np.ndarray | None = None
    medium_depth: np.ndarray | None = None
    medium_stack_underflows: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.k_current is None:
            # n_current is plain NumPy at construction time (sources always
            # build a NumPy bundle; TorchBackend promotes every field to a
            # tensor afterward via _ensure_torch_bundle), so this stays
            # NumPy too rather than dispatching through the active backend.
            self.k_current = np.zeros_like(self.n_current)
        # The medium-stack fields follow the ray state they belong to: NumPy
        # for a NumPy bundle, tensors on the same device for a bundle already
        # promoted to torch. They are never created through be.* because the
        # backend's creation functions carry the working float precision and
        # the grad flag, and an integer index table wants neither.
        if self.medium_stack is None:
            self.medium_stack = backend_int_full(
                (self.num_rays, MEDIUM_STACK_MAX_DEPTH),
                MEDIUM_STACK_EMPTY,
                like=self.n_current,
                bits=64,
            )
        if self.medium_depth is None:
            self.medium_depth = backend_int_full(
                (self.num_rays,), 0, like=self.n_current, bits=32
            )
        if self.medium_stack_underflows is None:
            self.medium_stack_underflows = backend_int_full(
                (self.num_rays,), 0, like=self.n_current, bits=32
            )

    @property
    def num_rays(self) -> int:
        """Total number of rays (alive + dead)."""
        return int(self.x.shape[0])

    @property
    def num_rays_alive(self) -> int:
        """Number of alive rays."""
        return int(be.sum(self.alive))

    @property
    def positions(self) -> np.ndarray:
        """Ray positions as (N, 3) array [mm]."""
        return be.stack([self.x, self.y, self.z], axis=1)

    @property
    def directions(self) -> np.ndarray:
        """Ray directions as (N, 3) unit-vector array."""
        return be.stack([self.L, self.M, self.N], axis=1)

    def compact(self) -> NSQRayBundle:
        """Return a new bundle containing only alive rays.

        Note: compaction is disabled in TorchBackend (alive rays carry
        zero-weight rather than being removed, to keep the graph fixed-shape).
        This method is used only in the NumPy forward fast path.
        """
        mask = self.alive
        kwargs: dict = dict(
            x=self.x[mask],
            y=self.y[mask],
            z=self.z[mask],
            L=self.L[mask],
            M=self.M[mask],
            N=self.N[mask],
            flux=self.flux[mask],
            wavelength=self.wavelength[mask],
            n_current=self.n_current[mask],
            bounce=self.bounce[mask],
            alive=self.alive[mask],
            k_current=self.k_current[mask],
            medium_stack=self.medium_stack[mask],
            medium_depth=self.medium_depth[mask],
            medium_stack_underflows=self.medium_stack_underflows[mask],
        )
        if self.ray_id is not None:
            kwargs["ray_id"] = self.ray_id[mask]
        return NSQRayBundle(**kwargs)

    def advance(self, t: np.ndarray) -> None:
        """Advance ray positions along their directions by distance t.

        Args:
            t: Per-ray distances [mm], shape (N,).
        """
        self.x = self.x + t * self.L
        self.y = self.y + t * self.M
        self.z = self.z + t * self.N

    def select(self, idx: np.ndarray, ray_id: np.ndarray | None = None) -> NSQRayBundle:
        """Return a new, independent bundle containing rays at ``idx``.

        Row selection on whichever array library holds the ray state: used by
        the bounded-splitting orchestration (D2, PR11;
        :mod:`optiland.nonsequential.ir.interpreter`) to snapshot the
        pre-interaction state of a set of rays before mutating the original
        bundle in place, so the transmit child of a split can be built from
        the same starting point as the reflect child.

        Args:
            idx: Integer index array selecting rows to copy.
            ray_id: If given, overrides ``self.ray_id[idx]`` in the returned
                bundle -- used to assign the spawned rays fresh identities
                so their RNG stream (keyed by ``ray_id``) is independent of
                the sibling ray that stayed at the original id.

        Returns:
            A new :class:`NSQRayBundle`, alive on every row (splitting only
            ever snapshots rays that are alive and mid-interaction).
        """
        # Every field goes through the same library-neutral row copy: the
        # medium stack is a Tensor whenever the rest of the bundle is, and a
        # Tensor has no .copy().
        alive = _rows_copy(self.alive, idx)
        alive[...] = True
        kwargs: dict = dict(
            x=_rows_copy(self.x, idx),
            y=_rows_copy(self.y, idx),
            z=_rows_copy(self.z, idx),
            L=_rows_copy(self.L, idx),
            M=_rows_copy(self.M, idx),
            N=_rows_copy(self.N, idx),
            flux=_rows_copy(self.flux, idx),
            wavelength=_rows_copy(self.wavelength, idx),
            n_current=_rows_copy(self.n_current, idx),
            bounce=_rows_copy(self.bounce, idx),
            alive=alive,
            k_current=_rows_copy(self.k_current, idx),
            medium_stack=_rows_copy(self.medium_stack, idx),
            medium_depth=_rows_copy(self.medium_depth, idx),
            medium_stack_underflows=_rows_copy(self.medium_stack_underflows, idx),
        )
        if ray_id is not None:
            kwargs["ray_id"] = ray_id
        elif self.ray_id is not None:
            kwargs["ray_id"] = _rows_copy(self.ray_id, idx)
        return NSQRayBundle(**kwargs)

    @staticmethod
    def concat(bundles: list[NSQRayBundle]) -> NSQRayBundle:
        """Concatenate several bundles into one.

        Used by the bounded-splitting orchestration to merge
        spawned transmit-branch children back into the live bundle at the
        end of a bounce, and to merge per-primitive spawn batches within a
        single bounce.

        Args:
            bundles: Non-empty list of bundles to concatenate, in order.

        Returns:
            A new :class:`NSQRayBundle` with every field concatenated along
            axis 0.
        """
        kwargs: dict = dict(
            x=_rows_concat([b.x for b in bundles]),
            y=_rows_concat([b.y for b in bundles]),
            z=_rows_concat([b.z for b in bundles]),
            L=_rows_concat([b.L for b in bundles]),
            M=_rows_concat([b.M for b in bundles]),
            N=_rows_concat([b.N for b in bundles]),
            flux=_rows_concat([b.flux for b in bundles]),
            wavelength=_rows_concat([b.wavelength for b in bundles]),
            n_current=_rows_concat([b.n_current for b in bundles]),
            bounce=_rows_concat([b.bounce for b in bundles]),
            alive=_rows_concat([b.alive for b in bundles]),
            k_current=_rows_concat([b.k_current for b in bundles]),
            medium_stack=_rows_concat([b.medium_stack for b in bundles]),
            medium_depth=_rows_concat([b.medium_depth for b in bundles]),
            medium_stack_underflows=_rows_concat(
                [b.medium_stack_underflows for b in bundles]
            ),
        )
        if all(b.ray_id is not None for b in bundles):
            kwargs["ray_id"] = _rows_concat([b.ray_id for b in bundles])
        return NSQRayBundle(**kwargs)
