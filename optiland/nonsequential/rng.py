"""Counter-based PCG32 RNG for Non-Sequential Raytracing.

Every stochastic decision in the NSQ engine is a pure function of a key
``(seed, ray_id, bounce, event_slot[, offset])`` -- there is no shared
mutable stream. This is what makes results bit-identical across
``batch_size``, across NumPy compaction vs. Torch's fixed-shape bundles, and
across any two conforming backends: a ray's random numbers depend only on
its own identity, never on which other rays happen to be alive in the same
batch or in what order components were visited.

Algorithm
----------------
This is the standard O'Neill PCG32 (XSH-RR 64/32), the same generator used
by Mitsuba 3 (``pcg32.h``) and satisfied by the per-launch-index
counter-based PRNGs conventional in OptiX kernels:

    state_{k+1} = state_k * MULT + inc      (mod 2**64)
    output_k    = xsh_rr(state_k)           (32-bit)

``inc`` (the odd-valued stream selector) is derived from ``(seed, ray_id,
event_slot)`` via SplitMix64, so every ray gets its own independent stream
per event slot. ``bounce`` (plus an optional ``offset`` for multi-draw
slots such as rejection sampling) selects *which* output in that stream via
PCG32's jump-ahead identity -- a closed-form function of the LCG step count.
This is what "counter derived arithmetically rather than by stateful
advance" means: computing output number ``k`` never requires having computed
outputs ``0..k-1`` first, and no RNG object needs to persist state between
calls.

Backend-generic integer path
----------------------------
The draw runs entirely in the active array backend, on whatever device that
backend holds its arrays on: there is no host copy of ``ray_id`` or
``bounce`` and no copy of the result back. Torch has no unsigned 64-bit
type and its ``int64`` right shift is arithmetic, so a 64-bit value is
carried as two ``int64`` limbs of 32 bits each, ``(hi, lo)``, both
non-negative. In that form every operation the generator needs is available
on both backends through plain Python operators (``+ * & | ^ << >>``):

* a 64x64 multiply mod 2**64 is six 32x16 limb products, none of which can
  exceed 2**49, so no partial product overflows a signed 64-bit integer;
* a logical right shift is an arithmetic right shift of a non-negative limb
  plus a mask, so the sign bit can never leak in;
* the jump-ahead is a table walk over four 16-bit chunks of the step count
  rather than a 64-step doubling loop, which is both fewer array operations
  and free of any data-dependent control flow.

The uniform is produced in the backend's working float dtype. At float64 it
is the same double as the host implementation returns, bit for bit; at
float32 it is that same value rounded to float32, which is the price of
running the conversion where the rays live.

On CUDA the same draw can run as one Warp kernel instead of the limb
arithmetic (:mod:`optiland.nonsequential.rng_warp`, opted into with
``TorchBackend(rng_kernel="warp")``). It is an implementation of this
module's definition, not a second generator: same state, same bits, same
uniform at either precision.

Honest scope of the guarantee: the *random-number stream* per
``(ray_id, bounce, event_slot)`` is bit-identical everywhere this module is
used. Final float *results* are not guaranteed bit-identical across
NumPy/Torch/CPU/GPU, because floating-point summation order and
transcendental implementations differ -- only the random decisions and the
code path they select are guaranteed identical.

Kramer Harrison, 2026
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

import optiland.backend as be
from optiland.backend.base import BackendCapabilityError
from optiland.nonsequential._compile import compiling

# PCG32 default multiplier (O'Neill, "PCG: A Family of Simple Fast
# Space-Efficient Statistically Good Algorithms for Random Number
# Generation", 2014).
_PCG_MULT = np.uint64(6364136223846793005)

# SplitMix64 (Steele, Lea, Flood 2014) constants, used only to derive
# well-mixed PCG32 seed/stream values from our integer keys -- not part of
# the PCG32 output path itself.
_SM64_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SM64_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_MIX2 = np.uint64(0x94D049BB133111EB)

_U64_0 = np.uint64(0)
_U64_1 = np.uint64(1)
_U32_31 = np.uint32(31)

_TWO_POW_32 = 4294967296.0

# Plain-int mirrors of the constants above, for the limb path.
_MULT_INT = 6364136223846793005
_GAMMA_INT = 0x9E3779B97F4A7C15
_MIX1_INT = 0xBF58476D1CE4E5B9
_MIX2_INT = 0x94D049BB133111EB
_SLOT_CONST = 0xC2B2AE3D27D4EB4F

_M16 = 0xFFFF
_M32 = 0xFFFFFFFF
_M64 = 0xFFFFFFFFFFFFFFFF


class EventSlot(IntEnum):
    """Discriminates independent PCG32 streams within one (ray, bounce).

    Every stochastic decision draws from its own slot, so adding, removing,
    or reordering an unrelated decision can never perturb another
    decision's stream (defect D-8: a shared, position-dependent stream).
    """

    SOURCE_U1 = 0
    SOURCE_U2 = 1
    SOURCE_U3 = 2
    SOURCE_U4 = 3
    SOURCE_WAVELENGTH = 4
    FRESNEL_BRANCH = 5
    SCATTER_BRANCH = 6
    BSDF_U1 = 7
    BSDF_U2 = 8
    RR = 9
    BSDF_LOBE_BRANCH = 10
    PATH_SAMPLE = 11


# ---------------------------------------------------------------------------
# Host reference implementation
#
# This is the original NumPy uint64 generator. It is no longer on the trace
# path -- pcg32_uint32 runs the backend-generic limb path below -- but it is
# kept, and kept readable, as the independent reference the conformance
# suite checks the limb path against: it is a direct transcription of the
# PCG32 pseudocode, so a disagreement between the two is a bug in the limb
# arithmetic rather than a shared misreading of the algorithm.
# ---------------------------------------------------------------------------


def _splitmix64(z: np.ndarray) -> np.ndarray:
    """Well-mixed 64-bit hash (SplitMix64 finalizer), vectorized.

    Args:
        z: uint64 array of arbitrary raw key material.

    Returns:
        uint64 array of well-mixed values, same shape as ``z``.
    """
    z = z + _SM64_GAMMA
    z = (z ^ (z >> np.uint64(30))) * _SM64_MIX1
    z = (z ^ (z >> np.uint64(27))) * _SM64_MIX2
    return z ^ (z >> np.uint64(31))


def _pcg32_seed(
    initstate: np.ndarray, initseq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """PCG32 ``srandom_r``: derive (state, inc) from (initstate, initseq).

    Args:
        initstate: uint64 array.
        initseq: uint64 array, same shape as ``initstate``.

    Returns:
        ``(state, inc)``, each a uint64 array of the same shape.
    """
    inc = (initseq << _U64_1) | _U64_1
    state = _U64_0 * _PCG_MULT + inc
    state = state + initstate
    state = state * _PCG_MULT + inc
    return state, inc


def _pcg32_advance(
    state: np.ndarray, delta: np.ndarray, mult: np.ndarray, inc: np.ndarray
) -> np.ndarray:
    """Closed-form ``state`` after ``delta`` LCG steps (PCG32 jump-ahead).

    Uses the standard doubling identity for ``state_k = mult^k * state_0 +
    inc * (mult^k - 1) / (mult - 1)`` in O(64) fixed vectorized iterations,
    so the result is a pure function of ``(state, delta)`` -- no sequential
    per-step state advance is needed.

    Args:
        state: uint64 array, the step-0 state.
        delta: uint64 array, number of LCG steps to advance.
        mult: uint64 array, LCG multiplier (broadcastable).
        inc: uint64 array, LCG increment (broadcastable).

    Returns:
        uint64 array: state after ``delta`` steps.
    """
    acc_mult = np.ones_like(mult)
    acc_plus = np.zeros_like(mult)
    cur_mult = mult.copy()
    cur_plus = inc.copy()
    d = delta.copy()
    for _ in range(64):
        bit = (d & _U64_1).astype(bool)
        acc_mult = np.where(bit, acc_mult * cur_mult, acc_mult)
        acc_plus = np.where(bit, acc_plus * cur_mult + cur_plus, acc_plus)
        cur_plus = (cur_mult + _U64_1) * cur_plus
        cur_mult = cur_mult * cur_mult
        d = d >> _U64_1
    return acc_mult * state + acc_plus


def _pcg32_output(state: np.ndarray) -> np.ndarray:
    """PCG32 XSH-RR 64/32 output permutation.

    Args:
        state: uint64 array.

    Returns:
        uint32 array, same shape as ``state``.
    """
    xorshifted = (((state >> np.uint64(18)) ^ state) >> np.uint64(27)).astype(np.uint32)
    rot = (state >> np.uint64(59)).astype(np.uint32)
    neg_rot = (np.uint32(0) - rot) & _U32_31
    return (xorshifted >> rot) | (xorshifted << neg_rot)


def _pcg32_uint32_reference(
    seed: int,
    ray_id: np.ndarray,
    bounce: np.ndarray,
    event_slot: int,
    offset: int = 0,
) -> np.ndarray:
    """Host reference draw: the NumPy uint64 path, for conformance checks.

    Args:
        seed: Trace-level RNG seed.
        ray_id: Per-ray identifiers, NumPy array of shape (N,).
        bounce: Per-ray bounce index, NumPy array or scalar.
        event_slot: :class:`EventSlot` value or plain int.
        offset: Extra step count added to ``bounce``.

    Returns:
        uint32 NumPy array, shape (N,).
    """
    ray_id_u64 = np.asarray(ray_id).astype(np.uint64)
    bounce_u64, ray_id_u64 = np.broadcast_arrays(
        np.asarray(bounce).astype(np.uint64), ray_id_u64
    )
    bounce_u64 = bounce_u64.copy()
    ray_id_u64 = ray_id_u64.copy()

    seed_u64 = np.uint64(int(seed) & _M64)
    slot_u64 = np.uint64(int(event_slot))

    # Modular (mod 2**64) wraparound is the intended arithmetic throughout
    # this module -- it is how the LCG and the mixing hashes are defined --
    # so overflow is not an error condition here.
    with np.errstate(over="ignore"):
        initstate = _splitmix64(np.full_like(ray_id_u64, seed_u64))
        # Mix ray_id and event_slot into the stream selector so every ray
        # gets an independent stream per slot; the golden-ratio odd constant
        # avoids low-bit correlation between adjacent ray ids.
        initseq = _splitmix64(
            ray_id_u64 * _SM64_GAMMA ^ (slot_u64 * np.uint64(_SLOT_CONST))
        )

        state0, inc = _pcg32_seed(initstate, initseq)
        mult = np.full_like(state0, _PCG_MULT)
        delta = bounce_u64 + np.uint64(offset)
        state_k = _pcg32_advance(state0, delta, mult, inc)
        return _pcg32_output(state_k)


# ---------------------------------------------------------------------------
# Backend-generic 64-bit integer arithmetic on 32-bit limbs
#
# A uint64 value is carried as ``(hi, lo)``, two backend int64 arrays each
# holding 32 bits and each non-negative. Limbs may also be plain Python ints
# (a compile-time constant); the expressions below are the same either way,
# since every operator used is defined for an array against an int.
# ---------------------------------------------------------------------------

_Limbs = tuple[Any, Any]


def _const_limbs(value: int) -> _Limbs:
    """Split a Python int into the limbs of its low 64 bits."""
    v = value & _M64
    return v >> 32, v & _M32


def _to_limbs(x: Any) -> _Limbs:
    """Split a backend int64 array holding a uint64 bit pattern into limbs.

    The input may be negative (a uint64 value at or above 2**63 stored in a
    signed slot); masking after the arithmetic shift discards the sign
    extension, so the limbs are the true unsigned 32-bit halves.
    """
    return (x >> 32) & _M32, x & _M32


def _add64(a: _Limbs, b: _Limbs) -> _Limbs:
    """Add two 64-bit values mod 2**64."""
    ah, al = a
    bh, bl = b
    s = al + bl
    return (ah + bh + (s >> 32)) & _M32, s & _M32


def _mul64(a: _Limbs, b: _Limbs) -> _Limbs:
    """Multiply two 64-bit values mod 2**64.

    ``al * bl`` is formed exactly from two 32x16 products, whose largest
    intermediate is below 2**49; the two cross terms are needed only modulo
    2**32, so they are formed the same way and masked.
    """
    ah, al = a
    bh, bl = b
    bl_l = bl & _M16
    bl_h = bl >> 16
    t0 = al * bl_l
    t1 = al * bl_h
    m = (t0 >> 16) + t1
    lo = (t0 & _M16) | ((m & _M16) << 16)
    carry = m >> 16
    bh_l = bh & _M16
    bh_h = bh >> 16
    cross_a = (al * bh_l + ((al * bh_h & _M16) << 16)) & _M32
    cross_b = (ah * bl_l + ((ah * bl_h & _M16) << 16)) & _M32
    return (carry + cross_a + cross_b) & _M32, lo


def _xor64(a: _Limbs, b: _Limbs) -> _Limbs:
    """Bitwise exclusive-or of two 64-bit values."""
    return a[0] ^ b[0], a[1] ^ b[1]


def _shr64(a: _Limbs, k: int) -> _Limbs:
    """Logical right shift of a 64-bit value by ``k`` bits, ``0 < k < 32``."""
    ah, al = a
    return ah >> k, (al >> k) | ((ah << (32 - k)) & _M32)


def _shl64_1(a: _Limbs) -> _Limbs:
    """Left shift of a 64-bit value by one bit, mod 2**64."""
    ah, al = a
    return ((ah << 1) | (al >> 31)) & _M32, (al << 1) & _M32


def _splitmix64_limbs(z: _Limbs) -> _Limbs:
    """SplitMix64 finalizer on limbs -- the limb twin of :func:`_splitmix64`."""
    z = _add64(z, _const_limbs(_GAMMA_INT))
    z = _mul64(_xor64(z, _shr64(z, 30)), _const_limbs(_MIX1_INT))
    z = _mul64(_xor64(z, _shr64(z, 27)), _const_limbs(_MIX2_INT))
    return _xor64(z, _shr64(z, 31))


def _splitmix64_int(z: int) -> int:
    """SplitMix64 finalizer on a Python int (the scalar seed path)."""
    z = (z + _GAMMA_INT) & _M64
    z = ((z ^ (z >> 30)) * _MIX1_INT) & _M64
    z = ((z ^ (z >> 27)) * _MIX2_INT) & _M64
    return z ^ (z >> 31)


# ---------------------------------------------------------------------------
# Jump-ahead tables
#
# Advancing the LCG by d steps is the affine map
# ``s -> F(d) * s + inc * G(d)`` with ``F(d) = MULT**d`` and
# ``G(d) = sum_{i<d} MULT**i``, both mod 2**64. Affine maps compose, so the
# step count is split into four 16-bit chunks and each chunk's (F, G) pair
# is looked up rather than rebuilt: four table reads and four map
# applications replace the 64-iteration doubling loop, and nothing in the
# walk depends on the data.
#
# The tables cost 4 MB per backend and device, built once on first use and
# then reused; 16-bit chunks were measured against 8-bit ones (which would
# cost 32 KB but need eight applications) and are 1.6x faster per draw on
# both backends, which is worth the memory for a table this small.
# ---------------------------------------------------------------------------

_CHUNK_BITS = 16
_CHUNK_MASK = (1 << _CHUNK_BITS) - 1
_NUM_CHUNKS = 64 // _CHUNK_BITS

_host_tables: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
_backend_tables: dict[tuple[str, str | None], tuple[tuple[Any, Any], ...]] = {}


def _affine_compose(first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int]:
    """Compose two advance maps: apply ``first``, then ``second``."""
    f1, g1 = first
    f2, g2 = second
    return (f1 * f2) & _M64, (g1 * f2 + g2) & _M64


def _affine_pow(steps: int) -> tuple[int, int]:
    """Return ``(F, G)`` for advancing the LCG by ``steps`` steps."""
    result = (1, 0)
    base = (_MULT_INT, 1)
    while steps:
        if steps & 1:
            result = _affine_compose(result, base)
        base = _affine_compose(base, base)
        steps >>= 1
    return result


def _build_host_tables() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Build the four 65536-entry (F, G) chunk tables as int64 bit patterns.

    Each table is grown by doubling -- entry ``v + 2**k`` is entry ``v``
    composed with the map for ``2**k`` chunk steps -- so the whole table is
    16 vectorized passes rather than 65536 scalar compositions.
    """
    tables = []
    for chunk in range(_NUM_CHUNKS):
        f = np.ones(1, dtype=np.uint64)
        g = np.zeros(1, dtype=np.uint64)
        for bit in range(_CHUNK_BITS):
            f_step, g_step = _affine_pow(1 << (chunk * _CHUNK_BITS + bit))
            fs = np.uint64(f_step)
            gs = np.uint64(g_step)
            with np.errstate(over="ignore"):
                f = np.concatenate([f, f * fs])
                g = np.concatenate([g, g * fs + gs])
        tables.append((f.view(np.int64), g.view(np.int64)))
    return tuple(tables)


def _backend_device_key() -> str | None:
    """Return the active backend's device, or None if it has no concept of one."""
    try:
        return str(be.get_device())
    except (AttributeError, BackendCapabilityError):
        return None


def _chunk_tables() -> tuple[tuple[Any, Any], ...]:
    """Return the chunk tables as arrays of the active backend and device."""
    global _host_tables
    key = (be.get_backend(), _backend_device_key())
    tables = _backend_tables.get(key)
    if tables is None:
        if _host_tables is None:
            _host_tables = _build_host_tables()
        tables = tuple(
            (
                be.asarray(f, dtype=np.int64),
                be.asarray(g, dtype=np.int64),
            )
            for f, g in _host_tables
        )
        _backend_tables[key] = tables
    return tables


def _advance_limbs(state: _Limbs, inc: _Limbs, delta: Any) -> _Limbs:
    """Advance ``state`` by ``delta`` LCG steps, one 16-bit chunk at a time."""
    for chunk, (f_table, g_table) in enumerate(_chunk_tables()):
        shifted = delta if chunk == 0 else delta >> (chunk * _CHUNK_BITS)
        index = shifted & _CHUNK_MASK
        f = _to_limbs(f_table[index])
        g = _to_limbs(g_table[index])
        state = _add64(_mul64(f, state), _mul64(inc, g))
    return state


def _output_limbs(state: _Limbs) -> Any:
    """PCG32 XSH-RR 64/32 output permutation on limbs.

    Returns:
        Backend int64 array holding the 32-bit output in [0, 2**32).
    """
    hi, lo = state
    xor_hi = hi ^ (hi >> 18)
    xor_lo = lo ^ ((lo >> 18) | ((hi << 14) & _M32))
    xorshifted = (xor_lo >> 27) | ((xor_hi << 5) & _M32)
    rot = hi >> 27
    neg_rot = (32 - rot) & 31
    return (xorshifted >> rot) | ((xorshifted << neg_rot) & _M32)


# ---------------------------------------------------------------------------
# Public draw functions
# ---------------------------------------------------------------------------


def pcg32_uint32(
    seed: int,
    ray_id: Any,
    bounce: Any,
    event_slot: int,
    offset: int = 0,
) -> Any:
    """Draw one PCG32 32-bit output per key, as a pure function of the key.

    Runs entirely in the active array backend: the inputs are not copied to
    the host and neither is the result.

    Args:
        seed: Trace-level RNG seed, or the seed already mixed, as the pair
            of 32-bit limbs ``(hi, lo)`` of the initial state (0-d integer
            arrays; see :meth:`NSQRng.bind_device_state`).
        ray_id: Per-ray identifiers, shape (N,). Must be non-negative.
        bounce: Per-ray bounce/step index, shape (N,) or a scalar
            broadcastable to (N,). Must be non-negative.
        event_slot: Which independent stream within (ray_id, bounce) to
            draw from -- an :class:`EventSlot` value or plain int.
        offset: Extra step count added to ``bounce`` for multi-draw slots
            (e.g. successive attempts in a rejection sampler) that need a
            fresh, deterministic value without consuming a new event slot.

    Returns:
        Backend int64 array of shape (N,), values in [0, 2**32).
    """
    ray_id_int = be.asarray(ray_id, dtype=np.int64)
    delta = be.asarray(bounce, dtype=np.int64) + int(offset)

    # The seed mixes to one scalar shared by every ray, so it is folded on
    # the host as a Python int -- no array is built for it and no value
    # crosses the device boundary. A seed given as a pair of limbs is that
    # scalar already mixed (NSQRng.bind_device_state: the compiled bounce
    # step's form, a device value rather than a constant of the program).
    if isinstance(seed, tuple):
        initstate_limbs = seed
    else:
        initstate_limbs = _const_limbs(_splitmix64_int(int(seed) & _M64))
    slot_mix = (int(event_slot) * _SLOT_CONST) & _M64

    # Mix ray_id and event_slot into the stream selector so every ray gets
    # an independent stream per slot; the golden-ratio odd constant avoids
    # low-bit correlation between adjacent ray ids.
    key = _xor64(
        _mul64(_to_limbs(ray_id_int), _const_limbs(_GAMMA_INT)),
        _const_limbs(slot_mix),
    )
    initseq = _splitmix64_limbs(key)

    inc = _shl64_1(initseq)
    inc = (inc[0], inc[1] | 1)
    state = _add64(inc, initstate_limbs)
    state = _add64(_mul64(state, _const_limbs(_MULT_INT)), inc)

    return _output_limbs(_advance_limbs(state, inc, delta))


def pcg32_uniform(
    seed: int,
    ray_id: Any,
    bounce: Any,
    event_slot: int,
    offset: int = 0,
) -> Any:
    """Draw one PCG32-derived uniform float per key, in [0, 1).

    The conversion is exact at float64 -- 32 bits of mantissa are plenty --
    so the value matches the host reference double for double. At float32
    it is that double rounded to float32.

    Args:
        seed: Trace-level RNG seed.
        ray_id: Per-ray identifiers, shape (N,).
        bounce: Per-ray bounce/step index, shape (N,) or scalar.
        event_slot: :class:`EventSlot` value or plain int.
        offset: See :func:`pcg32_uint32`.

    Returns:
        Backend float array in [0, 1), shape (N,), in the backend's
        working precision.
    """
    bits = pcg32_uint32(seed, ray_id, bounce, event_slot, offset)
    return be.cast(bits) / _TWO_POW_32


class NSQRng:
    """Keyed PCG32 RNG for one trace.

    Unlike ``numpy.random.Generator``, this carries no advancing internal
    state: every draw is a pure function of ``(seed, ray_id, bounce,
    event_slot)``, so the result never depends on ``batch_size``, on
    whether the NumPy backend has compacted dead rays out of the bundle, or
    on the order in which scene components were visited.

    Attributes:
        seed: Trace-level RNG seed (defaults to 0 if none was given, so a
            trace is always reproducible even when the user does not pass
            one explicitly).
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize NSQRng.

        Args:
            seed: RNG seed. ``None`` is normalized to 0 -- there is no
                notion of nondeterministic entropy here, since every draw
                must be reproducible from the key alone.
        """
        self.seed = 0 if seed is None else int(seed)
        self.device_state = None

    def bind_device_state(self, like: Any) -> None:
        """Hold the mixed seed as two 0-d int64 arrays on ``like``'s device.

        For the torch backend's compiled bounce step. A seed read inside a
        ``torch.compile`` region is a constant of the compiled program, so
        every trace with a new seed would compile the bounce again; the same
        value held as device data is an input of the program instead. Inside
        a compiled region :meth:`uniform` draws from it; everywhere else it
        draws from :attr:`seed` as before. The two give the same bits: the
        limb arithmetic is exact integer arithmetic either way.

        Args:
            like: An array on the device the draws run on.
        """
        hi, lo = _const_limbs(_splitmix64_int(self.seed & _M64))
        device = getattr(like, "device", None)
        if be.is_torch_tensor(like):
            import torch  # noqa: PLC0415

            self.device_state = (
                torch.tensor(hi, dtype=torch.int64, device=device),
                torch.tensor(lo, dtype=torch.int64, device=device),
            )
        else:
            self.device_state = (np.int64(hi), np.int64(lo))

    def uniform(
        self,
        ray_id: Any,
        bounce: Any,
        event_slot: int,
        offset: int = 0,
    ) -> Any:
        """Draw one uniform float per ray, in [0, 1).

        Args:
            ray_id: Per-ray identifiers, shape (N,).
            bounce: Per-ray bounce/step index, shape (N,) or scalar.
            event_slot: :class:`EventSlot` value or plain int.
            offset: See :func:`pcg32_uint32`.

        Returns:
            Backend float array in [0, 1), shape (N,).
        """
        if self.device_state is not None and compiling():
            return pcg32_uniform(self.device_state, ray_id, bounce, event_slot, offset)
        return pcg32_uniform(self.seed, ray_id, bounce, event_slot, offset)
