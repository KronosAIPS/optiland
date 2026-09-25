"""The float32 uniform from the top 24 bits: never 1.0 (issue 62).

The keyed generator's 32-bit output ``k`` becomes a uniform in the backend's
working dtype. Until issue 62 of the research repository the float32 uniform
was ``float32(k) / 2**32``; every ``k`` from ``2**32 - 128`` up rounds to
``2**32`` in float32, so the draw was exactly 1.0 with probability ``2**-25``,
against the documented half-open interval [0, 1). The maintainer's ruling of
2026-09-25 moves float32 to the standard construction, ``(k >> 8) * 2**-24``:
an integer below ``2**24`` is exact in float32 and so is the power-of-two
scale, so the largest draw is ``1 - 2**-24``. Float64 already excludes 1.0
(``k * 2**-32`` is exact and at most ``1 - 2**-32``) and is left as it was,
so every float64 number stays bit-identical.

What is tested, all as equalities:

* the float32 uniform is ``(k >> 8) * 2**-24`` of the host reference's ``k``,
  bit for bit, on the NumPy and the Torch backend, on the Apple GPU where it
  is reachable, and from the Warp kernel where Warp imports (its CPU backend
  included);
* the float64 uniform is still ``k * 2**-32`` on both backends;
* ``2**26`` float32 draws hold no 1.0 and reach ``1 - 2**-24``, with the old
  construction on the same bits as the control (it finds a 1.0 there); the
  same sweep through the Warp kernel runs where CUDA and Warp exist;
* the draw the branch-clamp test was built on (``k = 2**32 - 14``) is now
  ``1 - 2**-24`` at float32, still above the clamp, so it still takes the
  transmit branch with the bounded weight;
* a float32 trace records ``uniform_bits = 24`` in its environment block; a
  float64 trace does not carry the key.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.coatings import SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    FinitePlaneGeometry,
    IrradianceDetectorConfig,
    NSQMaterial,
    NSQRng,
    NSQScene,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential import rng as limb
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.ir.bsdf_ir import BsdfIR
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.rng import EventSlot

U32 = 2.0**-24
TOP32 = 1.0 - U32  # the largest float32 uniform
TOP64 = 1.0 - 2.0**-32  # the largest float64 uniform

needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="the Apple GPU (torch mps) is not reachable on this host",
)


def _cuda_with_warp() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import warp as wp

        wp.init()
        return bool(wp.is_cuda_available())
    except Exception:  # noqa: BLE001
        return False


def _warp_module():
    pytest.importorskip("warp", reason="Warp not installed (the optiland[warp] extra)")
    from optiland.nonsequential import rng_warp

    return rng_warp


@pytest.fixture
def backend_state():
    """The test sets backend and precision; both are put back afterwards."""
    previous = be.get_backend()
    yield
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.set_backend("numpy")
    be.set_precision("float64")
    be.set_backend(previous)


def _use(lib: str, precision: str, device: str = "cpu") -> None:
    be.set_backend(lib)
    if lib == "torch":
        be.set_device("cpu")  # mps refuses float64, so the device comes last
        be.grad_mode.disable()
    be.set_precision(precision)
    if lib == "torch":
        be.set_device(device)


def _as_host(u) -> np.ndarray:
    return np.asarray(to_numpy(u))


# The keys of the vector checks: 100,000 consecutive ray ids at one bounce.
_SEED = 99
_N = 100_000
_BOUNCE = 2
_SLOT = EventSlot.SCATTER_BRANCH


def _reference_bits() -> np.ndarray:
    ray_id = np.arange(_N, dtype=np.int64)
    return _pcg_ref(_SEED, ray_id, _BOUNCE, _SLOT)


def _pcg_ref(seed, ray_id, bounce, slot) -> np.ndarray:
    return limb._pcg32_uint32_reference(
        seed, ray_id, np.full(ray_id.shape, bounce, dtype=np.int64), slot
    ).astype(np.int64)


def _top24(bits: np.ndarray) -> np.ndarray:
    """The construction of the ruling, on the host: (k >> 8) * 2**-24 in float32."""
    return (bits >> 8).astype(np.float32) * np.float32(U32)


# ---------------------------------------------------------------------------
# The construction, against the host reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_float32_uniform_is_the_top_24_bits(backend_state, lib):
    bits = _reference_bits()
    _use(lib, "float32")
    drawn = _as_host(
        limb.pcg32_uniform(_SEED, np.arange(_N, dtype=np.int64), _BOUNCE, _SLOT)
    )
    assert drawn.dtype == np.float32
    expected = _top24(bits)
    assert np.array_equal(drawn.view(np.int32), expected.view(np.int32))
    # The control: the construction before issue 62 gives other float32 values
    # on most of these draws, so the equality above is not vacuous.
    old = bits.astype(np.float32) / np.float32(2.0**32)
    assert int(np.count_nonzero(old != expected)) > _N // 2
    assert limb.uniform_bits() == 24


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_float64_uniform_is_unchanged(backend_state, lib):
    bits = _reference_bits()
    _use(lib, "float64")
    drawn = _as_host(
        limb.pcg32_uniform(_SEED, np.arange(_N, dtype=np.int64), _BOUNCE, _SLOT)
    )
    assert drawn.dtype == np.float64
    expected = bits.astype(np.float64) / 2.0**32
    assert np.array_equal(drawn.view(np.int64), expected.view(np.int64))
    assert limb.uniform_bits() == 32


def test_uniform_bits_names_each_precision():
    assert limb.uniform_bits(64) == limb.uniform_bits("float64") == 32
    assert limb.uniform_bits(32) == limb.uniform_bits("float32") == 24
    with pytest.raises(ValueError, match="precision"):
        limb.uniform_bits(16)


@needs_mps
def test_float32_uniform_is_the_same_on_the_apple_gpu(backend_state):
    """On ``mps`` the limb path gives the host construction bit for bit."""
    bits = _reference_bits()
    _use("torch", "float32", device="mps")
    ray_id = torch.arange(_N, dtype=torch.int64, device="mps")
    drawn = limb.pcg32_uniform(_SEED, ray_id, _BOUNCE, _SLOT)
    assert drawn.device.type == "mps" and drawn.dtype == torch.float32
    host = drawn.cpu().numpy()
    assert np.array_equal(host.view(np.int32), _top24(bits).view(np.int32))


def test_warp_float32_uniform_is_the_top_24_bits(backend_state):
    """The Warp kernel forms the same float32 uniform (on its CPU backend here)."""
    rng_warp = _warp_module()
    bits = _reference_bits()
    _use("torch", "float32")
    ray_id = torch.arange(_N, dtype=torch.int64)
    drawn = rng_warp.draw_uniform(_SEED, ray_id, _BOUNCE, _SLOT)
    assert drawn.dtype == torch.float32
    host = drawn.numpy()
    assert np.array_equal(host.view(np.int32), _top24(bits).view(np.int32))
    drawn64 = rng_warp.draw_uniform(_SEED, ray_id, _BOUNCE, _SLOT, dtype=torch.float64)
    expected64 = bits.astype(np.float64) / 2.0**32
    assert np.array_equal(drawn64.numpy().view(np.int64), expected64.view(np.int64))


# ---------------------------------------------------------------------------
# 2**26 draws: no 1.0, and the top is reached
# ---------------------------------------------------------------------------

#: The sweep's keys: ray ids 0 to 2**26 - 1 at seed 1 (the catalogue's seed),
#: bounce 0, the Fresnel-branch slot. Measured on these keys: the largest
#: float32 draw 1 - 2**-24 twice, and one draw that the old construction
#: rounded to 1.0 (the control). About 2**26 * 2**-24 = 4 draws are expected at
#: the top and 2**26 * 2**-25 = 2 old 1.0s, so the keys were not searched for.
_SWEEP_SEED = 1
_SWEEP_SLOT = EventSlot.FRESNEL_BRANCH
_SWEEP_N = 2**26
_SWEEP_CHUNK = 2**22


def _sweep(draw_uniform, draw_bits, device: str) -> dict:
    ones = top = old_ones = 0
    largest = -1.0
    for start in range(0, _SWEEP_N, _SWEEP_CHUNK):
        ray_id = torch.arange(
            start, start + _SWEEP_CHUNK, dtype=torch.int64, device=device
        )
        u = draw_uniform(_SWEEP_SEED, ray_id, 0, _SWEEP_SLOT)
        assert u.dtype == torch.float32
        largest = max(largest, float(u.max()))
        ones += int((u == 1.0).sum())
        top += int((u == TOP32).sum())
        bits = draw_bits(_SWEEP_SEED, ray_id, 0, _SWEEP_SLOT)
        old_ones += int(((bits.to(torch.float32) / 4294967296.0) == 1.0).sum())
    return {"ones": ones, "largest": largest, "top": top, "old_ones": old_ones}


def test_2_pow_26_float32_draws_hold_no_one(backend_state):
    """The torch limb path: 2**26 float32 draws, none 1.0, the largest 1 - 2**-24."""
    _use("torch", "float32")
    seen = _sweep(limb.pcg32_uniform, limb.pcg32_uint32, "cpu")
    assert seen["ones"] == 0
    assert seen["largest"] == TOP32
    assert seen["top"] >= 1
    # The control: the old construction, on the same 32-bit outputs, draws 1.0.
    assert seen["old_ones"] >= 1


@pytest.mark.skipif(not _cuda_with_warp(), reason="CUDA with Warp not available")
def test_2_pow_26_float32_warp_draws_hold_no_one(backend_state):
    """The Warp kernel on CUDA: the same sweep, the same answer."""
    rng_warp = _warp_module()
    _use("torch", "float32", device="cuda")
    rng_warp.prepare("cuda")
    seen = _sweep(rng_warp.draw_uniform, rng_warp.draw_bits, "cuda")
    assert seen["ones"] == 0
    assert seen["largest"] == TOP32
    assert seen["top"] >= 1
    assert seen["old_ones"] >= 1


# ---------------------------------------------------------------------------
# The draw that was 1.0, at the Fresnel clamp
# ---------------------------------------------------------------------------

#: Seed 0, bounce 0, Fresnel-branch slot: this ray's 32-bit output is
#: 2**32 - 14, the draw ``test_nsq_branch_clamp.py`` was built on (it rounded to
#: 1.0 at float32 before issue 62).
_RAY_ID_AT_THE_TOP = 193_628_428
_DELTA = 2.0**-26  # the coating's transmittance; its reflectance is 1.0 in float32


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_the_draw_that_was_one_is_the_largest_below_one(backend_state, lib):
    ray_id = np.array([_RAY_ID_AT_THE_TOP], dtype=np.int64)
    bits = _pcg_ref(0, ray_id, 0, EventSlot.FRESNEL_BRANCH)
    assert int(bits[0]) == 2**32 - 14
    assert float(np.float32(float(bits[0])) / np.float32(2.0**32)) == 1.0  # before
    _use(lib, "float32")
    u32 = _as_host(limb.pcg32_uniform(0, ray_id, 0, EventSlot.FRESNEL_BRANCH))
    assert float(u32[0]) == TOP32
    _use(lib, "float64")
    u64 = _as_host(limb.pcg32_uniform(0, ray_id, 0, EventSlot.FRESNEL_BRANCH))
    assert float(u64[0]) == 1.0 - 14 / 2.0**32


def _coated_window() -> RefractiveComponent:
    return RefractiveComponent(
        CoordinateSystem(z=10.0),
        FinitePlaneGeometry(40.0, 40.0),
        VACUUM,
        NSQMaterial.from_glass("N-BK7"),
        coating=SimpleCoating(transmittance=_DELTA, reflectance=1.0 - _DELTA),
    )


def test_at_the_clamp_the_largest_draw_transmits_with_the_bounded_weight(
    backend_state,
):
    """The branch clamp's float32 case, with the draw the construction now gives.

    The reflectance 1 - 2**-26 is 1.0 in float32, so the branch probability
    sits at the clamp's upper bound 1 - 4 u32 (issue 59). The draw
    1 - 2**-24 lies above it, so the ray transmits, with the weight
    T / (4 u32) = 2**-26 / 2**-22 and a finite gradient equal to it.
    """
    _use("torch", "float32")
    rays = NSQRayBundle(
        x=np.zeros(1),
        y=np.zeros(1),
        z=np.zeros(1),
        L=np.zeros(1),
        M=np.zeros(1),
        N=np.ones(1),
        flux=np.ones(1),
        wavelength=np.full(1, 0.55),
        n_current=np.ones(1),
        bounce=np.zeros(1, dtype=np.int32),
        alive=np.ones(1, dtype=bool),
        ray_id=np.array([_RAY_ID_AT_THE_TOP], dtype=np.int64),
    )
    rays = TorchBackend(seed=0)._prepare_bundle(rays)
    flux_in = torch.ones(1, dtype=torch.float32, requires_grad=True)
    rays.flux = flux_in * 1.0
    rng = NSQRng(0)
    u = rng.uniform(rays.ray_id, rays.bounce, EventSlot.FRESNEL_BRANCH)
    assert u.item() == TOP32
    _coated_window().interact(
        rays,
        torch.full((1,), 10.0, dtype=torch.float32),
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        torch.ones(1, dtype=torch.bool),
        rng,
        BsdfIR(kind="none"),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )
    rays.flux.sum().backward()
    assert rays.N.item() > 0  # transmitted
    weight, grad = rays.flux.item(), flux_in.grad.item()
    assert weight == pytest.approx(_DELTA / (4 * U32), rel=4 * U32)
    assert weight <= 1.0
    assert math.isfinite(grad) and grad == weight


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def _window_scene() -> NSQScene:
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0
        ),
    )
    scene.add_component("window", _coated_window())
    scene.add_detector(
        "D1",
        CoordinateSystem(z=20.0),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=8, num_pixels_y=8),
    )
    return scene


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_a_float32_trace_records_the_construction(backend_state, lib):
    _use(lib, "float32")
    backend = NumpyBackend(seed=3) if lib == "numpy" else TorchBackend(seed=3)
    result = _window_scene().trace(num_rays=500, seed=3, max_depth=4, backend=backend)
    assert result.environment["precision"] == "float32"
    assert result.environment["uniform_bits"] == 24


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_a_float64_trace_has_no_uniform_bits_key(backend_state, lib):
    """Float64 draws are unchanged, and so is the float64 record."""
    _use(lib, "float64")
    backend = NumpyBackend(seed=3) if lib == "numpy" else TorchBackend(seed=3)
    result = _window_scene().trace(num_rays=500, seed=3, max_depth=4, backend=backend)
    assert result.environment["precision"] == "float64"
    assert "uniform_bits" not in result.environment
