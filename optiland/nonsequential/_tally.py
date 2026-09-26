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

A float total is **compensated** (Neumaier's variant of Kahan summation):
beside the running sum it keeps the rounding error of every addition, and
the value read back is the sum plus that error. A trace adds one partial per
batch, so an uncompensated total rounds once per batch and its value depends
on the batch size -- 2.6e-12 relative at batch size 1 on the catalogue's
diffuser against the 1e-12 that section 10.3 of the theory allows an
unordered reduction (KronosNSRT issue 25). Compensated, the total is within
a few units in the last place of the exactly rounded sum of its partials,
whatever their number. On a device the compensation is four elementwise
operations on 0-dim tensors, in place, with no host read, so a recorded
bounce still replays it.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math

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


def two_sum_error(a, b, s):
    """The rounding error of ``s = a + b``, exactly (Neumaier's branch).

    ``(a - s) + b`` when ``|a| >= |b|``, else ``(b - s) + a``: both are exact
    in floating point, and ``a + b == s + error`` holds exactly. Elementwise;
    works on Python floats, NumPy arrays and tensors alike. On tensors the
    inputs should be detached: the error carries no gradient (the gradient
    of a sum is carried by the sum itself).

    Args:
        a: One addend.
        b: The other addend.
        s: ``a + b`` as rounded.

    Returns:
        The error, of the same kind as the inputs.
    """
    if be.is_torch_tensor(s):
        import torch  # noqa: PLC0415

        return torch.where(a.abs() >= b.abs(), (a - s) + b, (b - s) + a)
    if isinstance(s, np.ndarray):
        return np.where(np.abs(a) >= np.abs(b), (a - s) + b, (b - s) + a)
    return (a - s) + b if abs(a) >= abs(b) else (b - s) + a


def _detached(x):
    return x.detach() if be.is_torch_tensor(x) else x


def accumulate_compensated(total, comp, term):
    """``total + term`` with its rounding error added into ``comp``.

    The compensated form of :func:`accumulate`, for a running float total
    kept as a pair of attributes (an absorber's flux). ``total`` is the plain
    running sum -- bit-identical to what :func:`accumulate` returns -- and
    ``total + comp`` is the compensated value. On a device both are added
    into in place after the first term, so a recorded bounce replays them.

    Args:
        total: The running sum: a Python float, or a device tensor.
        comp: Its compensation: a Python float, or a device tensor.
        term: The term to add.

    Returns:
        ``(total, comp)`` to store back.
    """
    if be.is_torch_tensor(total) or be.is_torch_tensor(term):
        if not be.is_torch_tensor(total):
            import torch  # noqa: PLC0415

            total = total + term  # the first term: a new tensor of its own
            comp = torch.zeros_like(_detached(total)) + comp
            return total, comp
        t_old = _detached(total)
        s = t_old + _detached(term)
        err = two_sum_error(t_old, _detached(term), s)
        if be.is_torch_tensor(comp) and comp.shape == err.shape and comp.dtype == err.dtype:
            comp.add_(err)
        else:
            comp = comp + err
        return accumulate(total, term), comp
    s = total + term
    return s, comp + two_sum_error(total, term, s)


class Tally:
    """One running total for one trace.

    Attributes:
        is_int: Report the total as an ``int`` rather than a ``float``.
    """

    __slots__ = ("_dev", "_dev_comp", "_host", "_host_comp", "is_int")

    def __init__(self, is_int: bool = False) -> None:
        """Create an empty tally.

        Args:
            is_int: True for a ray count, False for a flux total.
        """
        self.is_int = is_int
        self._host: float | int = 0 if is_int else 0.0
        self._host_comp = 0.0
        self._dev = None
        self._dev_comp = None

    def add(self, term) -> None:
        """Add one term.

        A device term is accumulated on the device; a host term is added to
        the Python running total. A tally that sees both keeps both and adds
        them in :meth:`value`.

        A device total adds in place after its first term: the tensor the
        total lives in is created once, from a copy of the first term, and
        every later term is added into that same storage. The running sum is
        the addition it always was; what changes is that the total is never
        rebound, which is what lets a recorded bounce be replayed (a CUDA
        graph writes to the memory it recorded). A term of another dtype or
        shape falls back to the out-of-place addition.

        A float tally also keeps the rounding error of each addition
        (:func:`two_sum_error`) in a compensation of its own -- a Python
        float beside the host sum, a 0-dim tensor added into in place beside
        the device sum -- and :meth:`value` returns the sum plus it. An
        integer tally adds exactly and keeps none.

        Args:
            term: A Python number, a 0-dim array, or a 0-dim tensor.
        """
        if be.is_torch_tensor(term):
            if self._dev is None:
                self._dev = term.clone()
                if term.is_floating_point():
                    import torch  # noqa: PLC0415

                    self._dev_comp = torch.zeros_like(term.detach())
                return
            if self._dev_comp is not None:
                old = self._dev.detach()
                new = old + term.detach()
                err = two_sum_error(old, term.detach(), new)
                if err.dtype == self._dev_comp.dtype and err.shape == self._dev_comp.shape:
                    self._dev_comp.add_(err)
                else:
                    self._dev_comp = self._dev_comp + err
            if term.dtype == self._dev.dtype and term.shape == self._dev.shape:
                self._dev.add_(term)
            else:
                self._dev = self._dev + term
        elif self.is_int:
            self._host += int(term)
        else:
            term = float(term)
            new = self._host + term
            self._host_comp += two_sum_error(self._host, term, new)
            self._host = new

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
        if self.is_int:
            total = self._host
            if self._dev is not None:
                total = total + int(to_numpy(self._dev))
            return int(total)
        # The host and device sums and their compensations, summed exactly
        # rounded: the value is the compensated total, read back once.
        parts = [self._host, self._host_comp]
        if self._dev is not None:
            parts.append(float(to_numpy(self._dev)))
            if self._dev_comp is not None:
                parts.append(float(to_numpy(self._dev_comp)))
        return math.fsum(parts)

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
