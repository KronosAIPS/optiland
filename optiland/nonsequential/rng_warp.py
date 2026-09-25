"""The keyed PCG32 generator as one Warp kernel (optional, CUDA).

:mod:`optiland.nonsequential.rng` defines the generator: every draw is a
pure function of ``(seed, ray_id, bounce, event_slot, offset)``. On the
torch backend it runs as int64 limb arithmetic, about 150 array operations
per draw, because torch has no unsigned 64-bit type. On a sphere-cavity
case that is three quarters of every bounce's dispatched operations, and at
the engine's default batch width a device bounce is bound by the host time
spent dispatching them, not by arithmetic.

`NVIDIA Warp <https://github.com/NVIDIA/warp>`_ has native ``uint64``
arithmetic that wraps modulo 2**64, so the same draw is one kernel launch:

    initstate = splitmix64(seed)                            (host, per trace)
    initseq   = splitmix64(ray_id * GAMMA ^ slot * SLOT)    (per ray)
    inc       = (initseq << 1) | 1
    state_0   = (inc + initstate) * MULT + inc
    state_d   = MULT**d * state_0 + inc * (MULT**d - 1) / (MULT - 1),
                d = bounce + offset                         (jump-ahead)
    bits      = xsh_rr(state_d)                             (32 bits)
    u         = float(bits) * 2**-32                        (working dtype)

The jump-ahead is the doubling identity of the host reference
(``rng._pcg32_advance``), run while the remaining step count is non-zero;
the limb path composes the same affine map from four 16-bit table lookups.
Both are exact modulo 2**64, so the state, the 32 output bits and the
uniform are the same bit for bit: at float64 the conversion is exact, and at
float32 both paths round the 32-bit integer to nearest (``cvt.rn`` on CUDA)
and then scale by a power of two, which is exact. The scale is passed to the
kernel as a value rather than written as a literal, so neither compiler can
choose a different constant.

Use it through the torch backend, ``TorchBackend(rng_kernel="warp")``. The
backend uses this kernel only when Warp imports and the device is CUDA;
otherwise it keeps the limb path and records why in
``SimulationResult.environment``. The functions :func:`draw_bits`,
:func:`draw_uniform` and :func:`draw_state` run the kernel on whatever
device their tensors are on, the CPU included, which is how the
conformance tests check it on a machine without CUDA.

The draws are sampling decisions and are detached, so no adjoint is
registered. The launches go through two torch custom operators
(``optiland_nsq::pcg32_bits`` and ``optiland_nsq::pcg32_uniform``) on
torch's current CUDA stream, so they are ordered with the surrounding torch
work and are recorded by a ``torch.cuda.graph`` capture; :func:`prepare`
loads the kernels on a device before any capture begins, because a module
load cannot happen inside one.

Install with the package extra: ``pip install optiland[warp]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import warp as wp

import optiland.backend as be
from optiland.nonsequential import rng as _rng
from optiland.nonsequential.rng import NSQRng

#: The device types on which :class:`WarpNSQRng` launches the kernel. The
#: torch backend's other devices keep the limb path: Warp has no Metal
#: backend, and on the CPU Warp launches one serial thread.
SUPPORTED_DEVICE_TYPES: tuple[str, ...] = ("cuda",)

_TWO_POW_MINUS_32 = 2.0**-32
_I64_MIN = -(1 << 63)


def _as_signed64(value: int) -> int:
    """The int64 whose bit pattern is the low 64 bits of ``value``."""
    v = int(value) & _rng._M64
    return v - (1 << 64) if v >= (1 << 63) else v


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------


@wp.struct
class _Keys:
    """Per-(seed, slot) constants, and the generator's own constants.

    The multiplier and the SplitMix64 constants travel as data rather than
    as literals, so no code generator has to spell a 64-bit unsigned
    literal.
    """

    mult: wp.uint64
    gamma: wp.uint64
    mix1: wp.uint64
    mix2: wp.uint64
    initstate: wp.uint64
    slot_mix: wp.uint64


@wp.func
def _state_after(rid: wp.uint64, delta: wp.uint64, k: _Keys) -> wp.uint64:
    """The LCG state of stream ``(rid, slot)`` after ``delta`` steps."""
    z = rid * k.gamma ^ k.slot_mix
    z = z + k.gamma
    z = (z ^ (z >> wp.uint64(30))) * k.mix1
    z = (z ^ (z >> wp.uint64(27))) * k.mix2
    z = z ^ (z >> wp.uint64(31))
    inc = (z << wp.uint64(1)) | wp.uint64(1)
    state = (inc + k.initstate) * k.mult + inc
    acc_mult = wp.uint64(1)
    acc_plus = wp.uint64(0)
    cur_mult = k.mult
    cur_plus = inc
    d = delta
    while d != wp.uint64(0):
        if (d & wp.uint64(1)) != wp.uint64(0):
            acc_mult = acc_mult * cur_mult
            acc_plus = acc_plus * cur_mult + cur_plus
        cur_plus = (cur_mult + wp.uint64(1)) * cur_plus
        cur_mult = cur_mult * cur_mult
        d = d >> wp.uint64(1)
    return acc_mult * state + acc_plus


@wp.func
def _xsh_rr(s: wp.uint64) -> wp.uint32:
    """PCG32 XSH-RR 64/32 output permutation."""
    xorshifted = wp.uint32(((s >> wp.uint64(18)) ^ s) >> wp.uint64(27))
    rot = wp.uint32(s >> wp.uint64(59))
    neg_rot = (wp.uint32(0) - rot) & wp.uint32(31)
    return (xorshifted >> rot) | (xorshifted << neg_rot)


@wp.func
def _delta(
    bounce: wp.array(dtype=wp.int64),
    bounce_scalar: wp.int64,
    use_scalar: wp.int32,
    offset: wp.int64,
    i: wp.int32,
) -> wp.uint64:
    """``bounce + offset`` for ray ``i``, as the limb path forms it in int64."""
    b = bounce_scalar
    if use_scalar == 0:
        b = bounce[i]
    return wp.uint64(b + offset)


@wp.kernel(enable_backward=False)
def _k_bits(
    ray_id: wp.array(dtype=wp.int64),
    bounce: wp.array(dtype=wp.int64),
    bounce_scalar: wp.int64,
    use_scalar: wp.int32,
    offset: wp.int64,
    k: _Keys,
    out: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    d = _delta(bounce, bounce_scalar, use_scalar, offset, i)
    out[i] = wp.int64(_xsh_rr(_state_after(wp.uint64(ray_id[i]), d, k)))


@wp.kernel(enable_backward=False)
def _k_uniform64(
    ray_id: wp.array(dtype=wp.int64),
    bounce: wp.array(dtype=wp.int64),
    bounce_scalar: wp.int64,
    use_scalar: wp.int32,
    offset: wp.int64,
    k: _Keys,
    scale: wp.float64,
    out: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    d = _delta(bounce, bounce_scalar, use_scalar, offset, i)
    out[i] = wp.float64(_xsh_rr(_state_after(wp.uint64(ray_id[i]), d, k))) * scale


@wp.kernel(enable_backward=False)
def _k_uniform32(
    ray_id: wp.array(dtype=wp.int64),
    bounce: wp.array(dtype=wp.int64),
    bounce_scalar: wp.int64,
    use_scalar: wp.int32,
    offset: wp.int64,
    k: _Keys,
    scale: wp.float32,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    d = _delta(bounce, bounce_scalar, use_scalar, offset, i)
    out[i] = wp.float32(_xsh_rr(_state_after(wp.uint64(ray_id[i]), d, k))) * scale


@wp.kernel(enable_backward=False)
def _k_state(
    ray_id: wp.array(dtype=wp.int64),
    bounce: wp.array(dtype=wp.int64),
    bounce_scalar: wp.int64,
    use_scalar: wp.int32,
    offset: wp.int64,
    k: _Keys,
    out: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    d = _delta(bounce, bounce_scalar, use_scalar, offset, i)
    out[i] = wp.int64(_state_after(wp.uint64(ray_id[i]), d, k))


# ---------------------------------------------------------------------------
# Launch plumbing
# ---------------------------------------------------------------------------

_keys_cache: dict[tuple[int, int], Any] = {}
_prepared: set[str] = set()
_initialised: list[bool] = [False]


def _init() -> None:
    """Initialise the Warp runtime once (``wp.init`` is itself idempotent)."""
    if not _initialised[0]:
        wp.init()
        _initialised[0] = True


def _keys(initstate: int, slot_mix: int) -> Any:
    """The kernel's constant struct for one (seed, slot), built once.

    A trace uses one seed and at most a dozen slots, so the cache holds a
    few entries per trace; it is emptied when a long campaign of seeds has
    filled it, rather than grown without bound.
    """
    key = (initstate, slot_mix)
    k = _keys_cache.get(key)
    if k is None:
        if len(_keys_cache) >= 4096:
            _keys_cache.clear()
        k = _Keys()
        k.mult = _rng._MULT_INT
        k.gamma = _rng._GAMMA_INT
        k.mix1 = _rng._MIX1_INT
        k.mix2 = _rng._MIX2_INT
        k.initstate = initstate & _rng._M64
        k.slot_mix = slot_mix & _rng._M64
        _keys_cache[key] = k
    return k


def prepare(device: Any) -> None:
    """Load the kernels on ``device`` (a torch device or its name).

    A Warp module is compiled, or read from Warp's kernel cache, and loaded
    the first time one of its kernels is launched on a device. That must not
    happen inside a CUDA-graph capture, so the torch backend calls this at
    the start of every trace, before its first bounce.

    Args:
        device: The torch device the ray arrays live on.
    """
    _init()
    tdev = torch.device(device)
    if tdev.type == "cuda" and tdev.index is None:
        tdev = torch.device("cuda", torch.cuda.current_device())
    name = str(tdev)
    if name in _prepared:
        return
    wp.load_module(module=__name__, device=wp.device_from_torch(tdev))
    _prepared.add(name)


def _launch(
    kernel: Any,
    ray_id: torch.Tensor,
    bounce: torch.Tensor,
    bounce_scalar: int,
    use_scalar: bool,
    offset: int,
    initstate: int,
    slot_mix: int,
    out: torch.Tensor,
    scale: float | None = None,
) -> None:
    _init()
    n = int(ray_id.shape[0])
    if n == 0:
        return
    tdev = ray_id.device
    inputs = [
        wp.from_torch(ray_id, dtype=wp.int64),
        wp.from_torch(bounce, dtype=wp.int64),
        wp.int64(bounce_scalar),
        wp.int32(1 if use_scalar else 0),
        wp.int64(offset),
        _keys(initstate, slot_mix),
    ]
    if scale is not None:
        inputs.append(
            wp.float64(scale) if out.dtype == torch.float64 else wp.float32(scale)
        )
    inputs.append(wp.from_torch(out))
    if tdev.type == "cuda":
        # Launch on torch's current stream: ordered with the torch work
        # around the draw, and recorded by a torch.cuda.graph capture.
        stream = wp.stream_from_torch(torch.cuda.current_stream(tdev))
        wp.launch(kernel, dim=n, inputs=inputs, stream=stream)
    else:
        wp.launch(kernel, dim=n, inputs=inputs, device=wp.device_from_torch(tdev))


@torch.library.custom_op("optiland_nsq::pcg32_bits", mutates_args=())
def _op_bits(
    ray_id: torch.Tensor,
    bounce: torch.Tensor,
    bounce_scalar: int,
    use_scalar: bool,
    offset: int,
    initstate: int,
    slot_mix: int,
) -> torch.Tensor:
    out = torch.empty(ray_id.shape[0], dtype=torch.int64, device=ray_id.device)
    _launch(
        _k_bits,
        ray_id,
        bounce,
        bounce_scalar,
        use_scalar,
        offset,
        initstate,
        slot_mix,
        out,
    )
    return out


@_op_bits.register_fake
def _(ray_id, bounce, bounce_scalar, use_scalar, offset, initstate, slot_mix):
    return ray_id.new_empty(ray_id.shape[0])


@torch.library.custom_op("optiland_nsq::pcg32_uniform", mutates_args=())
def _op_uniform(
    ray_id: torch.Tensor,
    bounce: torch.Tensor,
    bounce_scalar: int,
    use_scalar: bool,
    offset: int,
    initstate: int,
    slot_mix: int,
    double: bool,
) -> torch.Tensor:
    dtype = torch.float64 if double else torch.float32
    out = torch.empty(ray_id.shape[0], dtype=dtype, device=ray_id.device)
    kernel = _k_uniform64 if double else _k_uniform32
    _launch(
        kernel,
        ray_id,
        bounce,
        bounce_scalar,
        use_scalar,
        offset,
        initstate,
        slot_mix,
        out,
        _TWO_POW_MINUS_32,
    )
    return out


@_op_uniform.register_fake
def _(ray_id, bounce, bounce_scalar, use_scalar, offset, initstate, slot_mix, double):
    dtype = torch.float64 if double else torch.float32
    return torch.empty(ray_id.shape[0], dtype=dtype, device=ray_id.device)


# ---------------------------------------------------------------------------
# Public draw functions
# ---------------------------------------------------------------------------


def _operands(
    seed: int, ray_id: Any, bounce: Any, event_slot: int, offset: int
) -> tuple[torch.Tensor, torch.Tensor, int, bool, int, int, int] | None:
    """Normalise a draw's arguments for the kernel, or None if it cannot take them.

    The kernel takes a one-dimensional int64 ``ray_id`` tensor and either a
    per-ray bounce array of the same length or one integer. Anything else
    (a zero-length or multi-dimensional key, a bounce that broadcasts some
    other way) returns None, and the caller uses the limb path, which
    defines the same values. A zero-dimensional bounce tensor is expanded on
    its device rather than read to the host, so the draw never
    synchronises.
    """
    rid = (
        ray_id
        if isinstance(ray_id, torch.Tensor)
        else be.asarray(ray_id, dtype=np.int64)
    )
    if rid.dtype != torch.int64:
        rid = rid.to(torch.int64)
    if rid.dim() != 1 or rid.shape[0] == 0:
        return None
    rid = rid.contiguous()
    n = rid.shape[0]
    if isinstance(bounce, int | np.integer):
        b, b_scalar, use_scalar = rid, int(bounce), True
    else:
        b = (
            bounce
            if isinstance(bounce, torch.Tensor)
            else be.asarray(bounce, dtype=np.int64)
        )
        b = b.to(device=rid.device, dtype=torch.int64)
        if b.dim() == 0 or (b.dim() == 1 and b.shape[0] == 1 and n != 1):
            b = b.reshape(1).expand(n)
        if b.dim() != 1 or b.shape[0] != n:
            return None
        b, b_scalar, use_scalar = b.contiguous(), 0, False
    initstate = _as_signed64(_rng._splitmix64_int(int(seed) & _rng._M64))
    slot_mix = _as_signed64(int(event_slot) * _rng._SLOT_CONST)
    return rid, b, b_scalar, use_scalar, int(offset), initstate, slot_mix


def draw_bits(
    seed: int, ray_id: Any, bounce: Any, event_slot: int, offset: int = 0
) -> torch.Tensor:
    """:func:`~optiland.nonsequential.rng.pcg32_uint32` as one kernel.

    Runs on the device of ``ray_id`` (CUDA or CPU). Falls back to the limb
    path for arguments the kernel does not take (see :func:`_operands`).

    Args:
        seed: Trace-level RNG seed.
        ray_id: Per-ray identifiers, shape (N,).
        bounce: Per-ray bounce index, shape (N,), or one integer.
        event_slot: :class:`~optiland.nonsequential.rng.EventSlot` or int.
        offset: Extra step count added to ``bounce``.

    Returns:
        int64 tensor of shape (N,), values in [0, 2**32).
    """
    ops = _operands(seed, ray_id, bounce, event_slot, offset)
    if ops is None:
        return _rng.pcg32_uint32(seed, ray_id, bounce, event_slot, offset)
    return torch.ops.optiland_nsq.pcg32_bits(*ops)


def draw_uniform(
    seed: int,
    ray_id: Any,
    bounce: Any,
    event_slot: int,
    offset: int = 0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """:func:`~optiland.nonsequential.rng.pcg32_uniform` as one kernel.

    Args:
        seed: Trace-level RNG seed.
        ray_id: Per-ray identifiers, shape (N,).
        bounce: Per-ray bounce index, shape (N,), or one integer.
        event_slot: :class:`~optiland.nonsequential.rng.EventSlot` or int.
        offset: Extra step count added to ``bounce``.
        dtype: ``torch.float64`` or ``torch.float32``; the backend's
            working precision when omitted.

    Returns:
        Float tensor of shape (N,) in the requested dtype, on the device of
        ``ray_id``.
    """
    if dtype is None:
        dtype = torch.float64 if be.get_precision() == 64 else torch.float32
    ops = _operands(seed, ray_id, bounce, event_slot, offset)
    if ops is None:
        return _rng.pcg32_uniform(seed, ray_id, bounce, event_slot, offset)
    return torch.ops.optiland_nsq.pcg32_uniform(*ops, dtype == torch.float64)


def draw_state(
    seed: int, ray_id: Any, bounce: Any, event_slot: int, offset: int = 0
) -> torch.Tensor:
    """The LCG state after the jump-ahead, as int64 bit patterns.

    For the conformance tests: it lets the jump-ahead be compared with the
    host reference ``rng._pcg32_advance`` directly, not only through the
    output permutation, which discards half of the state's bits.

    Args:
        seed: Trace-level RNG seed.
        ray_id: Per-ray identifiers, shape (N,).
        bounce: Per-ray bounce index, shape (N,), or one integer.
        event_slot: :class:`~optiland.nonsequential.rng.EventSlot` or int.
        offset: Extra step count added to ``bounce``.

    Returns:
        int64 tensor of shape (N,): the uint64 state's bit pattern.
    """
    ops = _operands(seed, ray_id, bounce, event_slot, offset)
    if ops is None:
        raise ValueError("draw_state takes a one-dimensional, non-empty ray_id")
    rid, b, b_scalar, use_scalar, off, initstate, slot_mix = ops
    out = torch.empty(rid.shape[0], dtype=torch.int64, device=rid.device)
    _launch(_k_state, rid, b, b_scalar, use_scalar, off, initstate, slot_mix, out)
    return out


class WarpNSQRng(NSQRng):
    """:class:`~optiland.nonsequential.rng.NSQRng` drawing through the kernel.

    Same seed, same keys, same values: only where the arithmetic runs
    changes. A draw whose ray ids are not a torch tensor on a device in
    :data:`SUPPORTED_DEVICE_TYPES` (after the backend's own ``asarray``)
    takes the limb path instead, so a mixed call site can never see two
    generators.
    """

    kernel = "warp"

    def uniform(
        self,
        ray_id: Any,
        bounce: Any,
        event_slot: int,
        offset: int = 0,
    ) -> Any:
        """Draw one uniform float per ray, in the backend's working dtype.

        Args:
            ray_id: Per-ray identifiers, shape (N,).
            bounce: Per-ray bounce/step index, shape (N,) or scalar.
            event_slot: :class:`~optiland.nonsequential.rng.EventSlot` or int.
            offset: See :func:`~optiland.nonsequential.rng.pcg32_uint32`.

        Returns:
            Backend float array in [0, 1), shape (N,).
        """
        if be.get_backend() == "torch":
            rid = ray_id
            if not isinstance(rid, torch.Tensor):
                rid = be.asarray(ray_id, dtype=np.int64)
            if rid.device.type in SUPPORTED_DEVICE_TYPES:
                return draw_uniform(self.seed, rid, bounce, event_slot, offset)
        return _rng.pcg32_uniform(self.seed, ray_id, bounce, event_slot, offset)


def availability(device: Any) -> str | None:
    """Why the kernel cannot serve ``device``, or None when it can.

    Args:
        device: The torch backend's device (a name or a torch device).

    Returns:
        None when the kernel is used there; otherwise one sentence saying
        why the limb path is used instead.
    """
    tdev = torch.device(device)
    if tdev.type not in SUPPORTED_DEVICE_TYPES:
        return (
            f"the Warp generator runs on {', '.join(SUPPORTED_DEVICE_TYPES)} "
            f"only; the device is {tdev.type}"
        )
    try:
        _init()
        if tdev.type == "cuda" and not wp.is_cuda_available():
            return "Warp sees no CUDA device"
        prepare(tdev)
    except Exception as exc:  # noqa: BLE001 - any failure means "use the limb path"
        return f"Warp could not load its kernels ({type(exc).__name__})"
    return None


__all__ = [
    "SUPPORTED_DEVICE_TYPES",
    "WarpNSQRng",
    "availability",
    "draw_bits",
    "draw_state",
    "draw_uniform",
    "prepare",
]
