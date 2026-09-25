"""Per-trace accumulators that live where the ray state lives.

The bounce loop books flux and ray counts into these. On an array library
whose state is in host memory a running total is a Python scalar and adding
to it costs nothing. On a device the same addition would be a
device-to-host copy and a stream synchronisation *per bounce*, which is the
accounting half of the "zero host synchronisations per bounce" requirement
(``docs/theory/12_gpu_mapping.md`` R-12-4): the total must stay on the
device and be read back once, after the trace.

:class:`Tally` is that single object. It holds a Python scalar for host
state, a 0-dim device value for device state, and reads back exactly once
in :meth:`Tally.value`. :class:`Tally.vector` is the same thing for a small
fixed-length vector of counters -- one per scene primitive.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy


def masked_sum(values, mask):
    """Sum ``values`` over the rows ``mask`` selects.

    Two formulations of the same quantity, picked by where the data lives:

    - host arrays take the rows and sum them, which is what this engine has
      always done and what its recorded ledger digits are;
    - device arrays zero the unselected rows and sum the whole vector,
      because taking the rows means a boolean-mask gather whose *output
      shape* depends on the data -- a device-to-host synchronisation before
      any arithmetic happens.

    The two differ in summation order, so they differ in the last bits. The
    host formulation is kept exactly as it was so a host-backend trace stays
    bit-identical.

    Args:
        values: Per-ray values, shape (N,).
        mask: Per-ray boolean mask, shape (N,).

    Returns:
        A scalar of the same kind as ``values``.
    """
    if be.is_torch_tensor(values):
        return be.sum(be.where(mask, values, be.zeros_like(values)))
    return values[np.asarray(mask, dtype=bool)].sum()


def masked_count(mask):
    """Count the rows ``mask`` selects, without leaving the device.

    Args:
        mask: Per-ray boolean mask, shape (N,).

    Returns:
        A Python int for a host mask, a 0-dim integer value for a device
        mask.
    """
    if be.is_torch_tensor(mask):
        return mask.sum()
    return int(np.asarray(mask, dtype=bool).sum())


def accumulate(total, term):
    """``total + term``, added into ``total``'s own storage where it can be.

    For a running counter kept as an attribute (a detector's hit count, an
    absorber's totals). The first term starts from the attribute's Python
    zero, so ``0 + term`` builds the device total, a new tensor of its own;
    every later term of the same dtype and shape is added into it in place.
    The value is the addition it always was; the total is never rebound
    after its first term, which is what lets a recorded bounce be replayed.

    Args:
        total: The running total: a Python number, or a device tensor.
        term: The term to add.

    Returns:
        The total to store back -- the same object when added in place.
    """
    if (
        be.is_torch_tensor(total)
        and be.is_torch_tensor(term)
        and total.dtype == term.dtype
        and total.shape == term.shape
    ):
        return total.add_(term)
    return total + term


class Tally:
    """One running total for one trace.

    Attributes:
        is_int: Report the total as an ``int`` rather than a ``float``.
    """

    __slots__ = ("_dev", "_host", "is_int")

    def __init__(self, is_int: bool = False) -> None:
        """Create an empty tally.

        Args:
            is_int: True for a ray count, False for a flux total.
        """
        self.is_int = is_int
        self._host: float | int = 0 if is_int else 0.0
        self._dev = None

    def add(self, term) -> None:
        """Add one term.

        A device term is accumulated on the device; a host term is added to
        the Python running total. A tally that sees both keeps both and adds
        them in :meth:`value`.

        A device total adds in place after its first term: the tensor the
        total lives in is created once, from a copy of the first term, and
        every later term is added into that same storage. The arithmetic is
        the addition it always was; what changes is that the total is never
        rebound, which is what lets a recorded bounce be replayed (a CUDA
        graph writes to the memory it recorded). A term of another dtype or
        shape falls back to the out-of-place addition.

        Args:
            term: A Python number, a 0-dim array, or a 0-dim tensor.
        """
        if be.is_torch_tensor(term):
            if self._dev is None:
                self._dev = term.clone()
            elif term.dtype == self._dev.dtype and term.shape == self._dev.shape:
                self._dev.add_(term)
            else:
                self._dev = self._dev + term
        elif self.is_int:
            self._host += int(term)
        else:
            self._host += float(term)

    def add_masked_sum(self, values, mask) -> None:
        """Add ``sum(values[mask])``.

        Args:
            values: Per-ray values, shape (N,).
            mask: Per-ray boolean mask, shape (N,).
        """
        self.add(masked_sum(values, mask))

    def add_count(self, mask) -> None:
        """Add the number of rows ``mask`` selects.

        Args:
            mask: Per-ray boolean mask, shape (N,).
        """
        self.add(masked_count(mask))

    def value(self) -> float | int:
        """Read the total back, once.

        Returns:
            The running total as a Python ``int`` or ``float``.
        """
        total = self._host
        if self._dev is not None:
            dev = to_numpy(self._dev)
            total = total + (int(dev) if self.is_int else float(dev))
        return int(total) if self.is_int else float(total)

    @staticmethod
    def vector(size: int) -> _TallyVector:
        """Create a fixed-length vector of counters.

        Args:
            size: Number of counters.

        Returns:
            A :class:`_TallyVector`.
        """
        return _TallyVector(size)


class _TallyVector:
    """``size`` counters accumulated together, read back once.

    Used for the per-primitive nearest-hit counts behind the
    unreached-geometry diagnostic: incrementing a Python ``set`` per bounce
    means a device read per bounce, while adding a length-S integer vector
    does not.
    """

    __slots__ = ("_dev", "_host", "size")

    def __init__(self, size: int) -> None:
        """Create a zeroed counter vector.

        Args:
            size: Number of counters.
        """
        self.size = int(size)
        self._host = np.zeros(self.size, dtype=np.int64)
        self._dev = None

    def add_at(self, index: int, term) -> None:
        """Add ``term`` to counter ``index``.

        Args:
            index: Which counter.
            term: A Python int, or a 0-dim device value.
        """
        if be.is_torch_tensor(term):
            import torch  # noqa: PLC0415

            if self._dev is None:
                self._dev = torch.zeros(
                    self.size, dtype=torch.int64, device=term.device
                )
            self._dev[index] += term.to(torch.int64)
        else:
            self._host[index] += int(term)

    def values(self) -> np.ndarray:
        """Read the counters back, once.

        Returns:
            A NumPy int64 array of length ``size``.
        """
        if self._dev is None:
            return self._host
        return self._host + to_numpy(self._dev).astype(np.int64)
