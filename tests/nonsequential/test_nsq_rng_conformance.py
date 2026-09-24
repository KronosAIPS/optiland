"""PCG32 RNG conformance suite for Non-Sequential Raytracing.

Covers D11 / D-8: random numbers are a pure function of ``(seed, ray_id,
bounce, event_slot)``, with no shared mutable stream. This file is the
fixed-vector table a third-party backend (Mitsuba, OptiX) can run against to
prove conformance, plus the required invariants: batch-size independence,
order independence, and identical random decisions across the NumPy and
Torch backends.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.rng import (
    EventSlot,
    NSQRng,
    _pcg32_advance,
    _pcg32_output,
    _pcg32_seed,
    _pcg32_uint32_reference,
    pcg32_uint32,
    pcg32_uniform,
)

# ---------------------------------------------------------------------------
# Core PCG32 algorithm vs. the canonical O'Neill reference
# ---------------------------------------------------------------------------

# From the public-domain PCG reference implementation (pcg-c-basic
# pcg32-demo.c), seeded with pcg32_srandom_r(&rng, 42u, 54u): the first six
# 32-bit outputs of the *unrelated* raw generator (not our key-mixing
# scheme). This validates that _pcg32_seed / _pcg32_advance / _pcg32_output
# implement the real PCG32 XSH-RR 64/32 algorithm bit-for-bit, independent
# of anything this module adds on top.
_PCG32_REFERENCE_INITSTATE = 42
_PCG32_REFERENCE_INITSEQ = 54
_PCG32_REFERENCE_OUTPUTS = [
    0xA15C02B7,
    0x7B47F409,
    0xBA1D3330,
    0x83D2F293,
    0xBFA4784B,
    0xCBED606E,
]


def test_core_pcg32_matches_oneill_reference():
    """The state/advance/output primitives reproduce the canonical vectors."""
    initstate = np.array([_PCG32_REFERENCE_INITSTATE], dtype=np.uint64)
    initseq = np.array([_PCG32_REFERENCE_INITSEQ], dtype=np.uint64)
    state0, inc = _pcg32_seed(initstate, initseq)
    mult = np.full_like(state0, np.uint64(6364136223846793005))

    for k, expected in enumerate(_PCG32_REFERENCE_OUTPUTS):
        state_k = _pcg32_advance(state0, np.array([k], dtype=np.uint64), mult, inc)
        out_k = int(_pcg32_output(state_k)[0])
        assert out_k == expected, f"draw {k}: got {out_k:#x}, expected {expected:#x}"


# ---------------------------------------------------------------------------
# Fixed-vector table for the NSQ key layout
# ---------------------------------------------------------------------------

# Generated once from this module's own reference implementation. A
# conforming third-party backend must reproduce these bit-for-bit: same
# seed, same (ray_id, bounce, event_slot) keys, same uint32 outputs.
_SEED = 7
_RAY_ID = np.array([0, 1, 2, 5, 100, 999_999], dtype=np.int64)
_BOUNCE = np.array([0, 0, 3, 1, 0, 2], dtype=np.int32)

_FIXED_VECTORS: dict[EventSlot, list[int]] = {
    EventSlot.SOURCE_U1: [
        0x489673E5,
        0xF5314CF0,
        0x95FFCD8C,
        0x9D500941,
        0xD7ED4E42,
        0xA00A3EDD,
    ],
    EventSlot.FRESNEL_BRANCH: [
        0xEBA83967,
        0xC994A4B4,
        0x722B9BE5,
        0x339CDAD1,
        0x85EF336D,
        0x922DD0F3,
    ],
    EventSlot.BSDF_U2: [
        0xC9315134,
        0x34776105,
        0xDEB171CB,
        0x79E94B7C,
        0x5E71B7CA,
        0xDB3C6EAF,
    ],
}


@pytest.mark.parametrize("event_slot", sorted(_FIXED_VECTORS, key=lambda s: s.value))
def test_key_layout_fixed_vectors(event_slot):
    """NSQRng's key-mixing scheme reproduces its own frozen reference table."""
    bits = pcg32_uint32(_SEED, _RAY_ID, _BOUNCE, event_slot)
    expected = _FIXED_VECTORS[event_slot]
    for got, want in zip(bits.tolist(), expected, strict=True):
        assert got == want, f"{event_slot.name}: got {got:#010x}, want {want:#010x}"


# ---------------------------------------------------------------------------
# Invariants required by the PCG32 conformance suite
# ---------------------------------------------------------------------------


class TestKeyedDrawInvariants:
    """A draw depends only on its own key -- never on batching or ordering."""

    def test_deterministic_given_same_key(self):
        rng = NSQRng(seed=123)
        ray_id = np.arange(50)
        bounce = np.zeros(50, dtype=np.int32)
        a = rng.uniform(ray_id, bounce, EventSlot.SOURCE_U1)
        b = rng.uniform(ray_id, bounce, EventSlot.SOURCE_U1)
        np.testing.assert_array_equal(a, b)

    def test_independent_of_array_order(self):
        """Permuting the ray_id array permutes the results identically.

        This is the batch-size / compaction invariant in miniature: a ray's
        draw must not depend on which other rays share its batch or on its
        position within the array.
        """
        rng = NSQRng(seed=5)
        ray_id = np.arange(200)
        bounce = np.full(200, 3, dtype=np.int32)
        perm = np.random.default_rng(0).permutation(200)

        full = rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)
        permuted = rng.uniform(ray_id[perm], bounce[perm], EventSlot.BSDF_U1)
        np.testing.assert_array_equal(full[perm], permuted)

    def test_independent_of_batch_partitioning(self):
        """Splitting a ray_id array into arbitrary chunks changes nothing.

        Models what varying ``batch_size`` does to the trace loop: the same
        rays get processed in differently sized groups, but each ray's
        identity (and therefore its RNG draws) does not change.
        """
        rng = NSQRng(seed=9)
        ray_id = np.arange(97)
        bounce = np.ones(97, dtype=np.int32)

        whole = rng.uniform(ray_id, bounce, EventSlot.SCATTER_BRANCH)

        chunked = np.empty(97)
        for start, stop in [(0, 1), (1, 7), (7, 40), (40, 97)]:
            chunked[start:stop] = rng.uniform(
                ray_id[start:stop], bounce[start:stop], EventSlot.SCATTER_BRANCH
            )
        np.testing.assert_array_equal(whole, chunked)

    def test_distinct_event_slots_are_independent_streams(self):
        rng = NSQRng(seed=1)
        ray_id = np.arange(1000)
        bounce = np.zeros(1000, dtype=np.int32)
        u1 = rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)
        u2 = rng.uniform(ray_id, bounce, EventSlot.BSDF_U2)
        # Not identical and not (anti)correlated -- a real second stream,
        # not the same values relabeled.
        assert not np.allclose(u1, u2)
        assert abs(np.corrcoef(u1, u2)[0, 1]) < 0.1

    def test_distinct_bounces_are_independent_draws(self):
        rng = NSQRng(seed=2)
        ray_id = np.arange(1000)
        u_b0 = rng.uniform(ray_id, np.zeros(1000, dtype=np.int32), EventSlot.RR)
        u_b1 = rng.uniform(ray_id, np.ones(1000, dtype=np.int32), EventSlot.RR)
        assert not np.allclose(u_b0, u_b1)

    def test_uniform_in_unit_interval(self):
        rng = NSQRng(seed=3)
        ray_id = np.arange(50_000)
        bounce = np.zeros(50_000, dtype=np.int32)
        u = rng.uniform(ray_id, bounce, EventSlot.SOURCE_WAVELENGTH)
        assert u.min() >= 0.0
        assert u.max() < 1.0
        # Coarse two-sided sanity check on the mean, not a full uniformity
        # test -- this file is about the key contract, not RNG quality.
        assert abs(u.mean() - 0.5) < 0.01

    def test_offset_gives_an_independent_sub_draw(self):
        """``offset`` (used for bounded rejection-sampling attempts) must
        not collide with the plain (ray_id, bounce, event_slot) draw."""
        rng = NSQRng(seed=11)
        ray_id = np.arange(100)
        bounce = np.zeros(100, dtype=np.int32)
        base = rng.uniform(ray_id, bounce, EventSlot.SOURCE_U1)
        attempt1 = rng.uniform(ray_id, bounce, EventSlot.SOURCE_U1, offset=1)
        assert not np.allclose(base, attempt1)


# ---------------------------------------------------------------------------
# Cross-backend agreement (the honest scope of the guarantee)
# ---------------------------------------------------------------------------


class TestCrossBackendAgreement:
    """NumPy and Torch must make the same random decisions from the same seed.

    A collimated source's position/wavelength sampling (SOURCE_U1/U2/
    WAVELENGTH) is exercised directly by both backends' ray-generation step,
    so a scene as simple as "source -> detector" already checks that both
    backends draw the identical (seed, ray_id, bounce, event_slot) stream --
    the fixed-vector tests above already cover the physics-interaction
    slots (FRESNEL_BRANCH, BSDF_U1/U2, ...) in isolation.
    """

    @staticmethod
    def _scene():
        from optiland.coordinate_system import CoordinateSystem
        from optiland.nonsequential import (
            CollimatedSourceConfig,
            IrradianceDetectorConfig,
            NSQScene,
            Spectrum,
        )

        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=5.0,
            ),
        )
        scene.add_detector(
            "D",
            CoordinateSystem(z=40),
            IrradianceDetectorConfig(
                width=20,
                height=20,
                num_pixels_x=16,
                num_pixels_y=16,
                splat="bilinear",
            ),
        )
        return scene

    def test_flux_and_centroid_match_across_backends(self):
        pytest.importorskip("torch")
        import optiland.backend as be
        from optiland.backend.utils import to_numpy
        from optiland.nonsequential.backends.numpy_backend import NumpyBackend
        from optiland.nonsequential.backends.torch_backend import TorchBackend

        scene = self._scene()
        be.set_backend("numpy")
        np_result = scene.trace(num_rays=5_000, seed=42, backend=NumpyBackend(seed=42))
        try:
            be.set_backend("torch")
            be.set_precision("float64")
            torch_result = scene.trace(
                num_rays=5_000, seed=42, backend=TorchBackend(seed=42)
            )
        finally:
            be.set_backend("numpy")

        assert np_result.total_flux_detected == pytest.approx(
            torch_result.total_flux_detected, rel=1e-9
        )
        np_data = to_numpy(np_result.detectors["D"].data)
        torch_data = to_numpy(torch_result.detectors["D"].data)
        # Identical random decisions -> identical irradiance map, up to the
        # documented float-arithmetic tolerance between backends.
        np.testing.assert_allclose(np_data, torch_data, rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------------------
# The backend-generic limb path against the host reference
#
# The generator runs as int64 limb arithmetic in whichever array backend is
# active. These tests are the proof that moving it there changed no bit of
# the stream: a million draws per key, compared as 32-bit integers rather
# than as floats, on both backends, against the NumPy uint64 reference
# implementation kept in the module for exactly this purpose.
# ---------------------------------------------------------------------------

_BITWISE_NUM_RAYS = 1_000_000

# (seed, bounce, event_slot, offset): a plain first draw, a mid-trace
# branch draw, a deep-bounce draw, and a rejection-sampler sub-draw under a
# seed whose top bit is set (the wraparound the module explicitly permits).
_BITWISE_KEYS = [
    (0, 0, EventSlot.SOURCE_U1, 0),
    (7, 3, EventSlot.FRESNEL_BRANCH, 0),
    (12345, 17, EventSlot.BSDF_U2, 0),
    (2**63 + 1, 1, EventSlot.RR, 5),
]


@pytest.mark.parametrize(("seed", "bounce_value", "event_slot", "offset"), _BITWISE_KEYS)
def test_limb_path_is_bit_identical_to_host_reference(
    seed, bounce_value, event_slot, offset
):
    """Both backends reproduce the host uint32 stream with zero mismatches."""
    ray_id = np.arange(_BITWISE_NUM_RAYS, dtype=np.int64)
    bounce = np.full(_BITWISE_NUM_RAYS, bounce_value, dtype=np.int32)
    reference = _pcg32_uint32_reference(seed, ray_id, bounce, event_slot, offset).astype(
        np.int64
    )

    numpy_bits = to_numpy(pcg32_uint32(seed, ray_id, bounce, event_slot, offset))
    assert numpy_bits.shape == (_BITWISE_NUM_RAYS,)
    assert int(np.count_nonzero(numpy_bits != reference)) == 0

    torch = pytest.importorskip("torch")
    be.set_backend("torch")
    be.set_precision("float64")
    try:
        torch_bits = to_numpy(
            pcg32_uint32(
                seed,
                torch.from_numpy(ray_id),
                torch.from_numpy(bounce),
                event_slot,
                offset,
            )
        )
    finally:
        be.set_backend("numpy")
    assert torch_bits.shape == (_BITWISE_NUM_RAYS,)
    assert int(np.count_nonzero(torch_bits != reference)) == 0


def test_uniform_is_the_same_double_as_the_host_reference():
    """At float64 the uint32 -> [0, 1) conversion is exact, so the doubles match."""
    ray_id = np.arange(100_000, dtype=np.int64)
    bounce = np.full(100_000, 2, dtype=np.int32)
    reference = _pcg32_uint32_reference(
        99, ray_id, bounce, EventSlot.SCATTER_BRANCH
    ).astype(np.float64) / 2.0**32

    drawn = to_numpy(pcg32_uniform(99, ray_id, bounce, EventSlot.SCATTER_BRANCH))
    assert drawn.dtype == np.float64
    np.testing.assert_array_equal(drawn, reference)


def test_torch_draw_makes_no_host_transfer(monkeypatch):
    """Nothing in a Torch draw copies a ray array back to the host.

    Every route off the device is made to raise: the backend's own
    ``to_numpy`` boundary utility, and the tensor methods that force a
    synchronising copy.
    """
    torch = pytest.importorskip("torch")

    import optiland.backend.utils as backend_utils

    be.set_backend("torch")
    be.set_precision("float64")
    try:
        rng = NSQRng(seed=19)
        ray_id = torch.arange(4096, dtype=torch.int64)
        bounce = torch.full((4096,), 2, dtype=torch.int32)
        # Draw once unguarded so the jump-ahead tables are already built for
        # this backend: the test is about the draw, not about first use.
        expected = to_numpy(rng.uniform(ray_id, bounce, EventSlot.BSDF_U1))

        def _forbidden(*args, **kwargs):
            raise AssertionError("the draw copied data to the host")

        monkeypatch.setattr(backend_utils, "to_numpy", _forbidden)
        monkeypatch.setattr(torch.Tensor, "numpy", _forbidden, raising=False)
        monkeypatch.setattr(torch.Tensor, "cpu", _forbidden, raising=False)
        monkeypatch.setattr(torch.Tensor, "tolist", _forbidden, raising=False)
        monkeypatch.setattr(torch.Tensor, "item", _forbidden, raising=False)

        # The trap is armed: any of these routes off the device now raises.
        with pytest.raises(AssertionError):
            ray_id.cpu()
        with pytest.raises(AssertionError):
            backend_utils.to_numpy(ray_id)

        drawn = rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)
        is_tensor = isinstance(drawn, torch.Tensor)

        monkeypatch.undo()
        assert is_tensor
        np.testing.assert_array_equal(to_numpy(drawn), expected)
    finally:
        be.set_backend("numpy")
