"""The keyed generator as one Warp kernel: the same numbers as the limb path.

``optiland.nonsequential.rng_warp`` draws the PCG32 stream of
``optiland.nonsequential.rng`` in one kernel instead of about 150 torch
operations. It is an implementation change only, so everything here is an
equality, never a tolerance:

* the 32 output bits and the uniforms at float64 and at float32 (compared as
  bit patterns), on 48 configurations of 200,000 draws (four seeds, four
  event slots, three offsets; random ray ids below 2**40 and bounces below
  1,100), against the torch limb path, and the bits against the host
  reference;
* the uniform at both precisions for every form a call site passes the
  bounce in (per-ray int32 or int64 tensor, a NumPy array, one integer, a
  zero-dimensional tensor);
* the LCG state after the jump-ahead, against the host reference's doubling
  loop, for step counts in every 16-bit chunk of the counter;
* whole traces through ``TorchBackend(rng_kernel="warp")``: every ledger
  entry and every detector pixel bit-identical to the default backend on a
  refracting, a scattering and an integrating-sphere scene, at both
  precisions.

The kernel runs on CUDA in the engine and on the CPU here; the trace tests
allow the CPU for the length of the test (``SUPPORTED_DEVICE_TYPES``), which
is the only thing that differs from a CUDA run. Where CUDA is present the
same tests run there too, plus the capture of a draw in a CUDA graph.

The fallback tests need no Warp: a request for the kernel on a device it
does not serve, or without Warp installed, keeps the limb path, warns about
nothing, and says so in ``SimulationResult.environment``.
"""

from __future__ import annotations

import math
import sys
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
import optiland.nonsequential as nsq_package
from optiland.backend.utils import to_numpy
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    HarveyShackBSDF,
    IrradianceDetectorConfig,
    LambertianBSDF,
    LensConfig,
    MirrorConfig,
    NSQScene,
    ReflectiveComponent,
    SpectralDetectorConfig,
    Spectrum,
    SphericalCavityGeometry,
    SphericalPort,
    SurfaceConfig,
)
from optiland.nonsequential import rng as limb
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.rng import EventSlot, NSQRng

_M64 = (1 << 64) - 1


def _warp_module():
    """The kernel module, or a skip where Warp is not installed."""
    pytest.importorskip("warp", reason="Warp not installed (the optiland[warp] extra)")
    from optiland.nonsequential import rng_warp

    return rng_warp


def _cuda_with_warp() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import warp as wp

        wp.init()
        return bool(wp.is_cuda_available())
    except Exception:  # noqa: BLE001
        return False


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture
def torch_backend_state():
    """Torch on the CPU at float64 for the test, and the state put back."""
    previous = be.get_backend()
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.disable()
    yield
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.set_backend(previous)


def _use_device(device: str) -> None:
    if device == "cuda" and not _cuda_with_warp():
        pytest.skip("CUDA with Warp not available")
    be.set_device(device)


# ---------------------------------------------------------------------------
# The draw, against the limb path and the host reference
# ---------------------------------------------------------------------------

_SEEDS = (0, 1, 42, 2**40 + 3)
_SLOTS = (0, 5, 9, 11)
_OFFSETS = (0, 3, 70_000)
_DRAWS = 200_000


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("slot", _SLOTS)
def test_bits_are_the_limb_path_bits(torch_backend_state, device, seed, slot):
    """48 configurations of 200,000 draws: zero differing outputs or uniforms.

    Each configuration compares the 32-bit outputs, and the uniforms at
    float64 and at float32 as bit patterns.
    """
    rng_warp = _warp_module()
    _use_device(device)
    gen = np.random.default_rng(1000 * slot + seed % 997)
    ray_id = torch.as_tensor(
        gen.integers(0, 2**40, size=_DRAWS), dtype=torch.int64, device=device
    )
    bounce = torch.as_tensor(
        gen.integers(0, 1100, size=_DRAWS), dtype=torch.int32, device=device
    )
    for offset in _OFFSETS:
        expected = limb.pcg32_uint32(seed, ray_id, bounce, slot, offset)
        got = rng_warp.draw_bits(seed, ray_id, bounce, slot, offset)
        assert got.dtype == torch.int64
        assert got.device == expected.device
        differing = int((got != expected).sum())
        assert differing == 0, f"offset {offset}: {differing} differing draws"

        # The limb path is itself checked against the host reference in
        # test_nsq_rng_conformance.py; this is the direct check, on a subset.
        host = limb._pcg32_uint32_reference(
            seed, to_numpy(ray_id[:20_000]), to_numpy(bounce[:20_000]), slot, offset
        ).astype(np.int64)
        np.testing.assert_array_equal(to_numpy(got[:20_000]), host)

        for precision, view in (("float64", torch.int64), ("float32", torch.int32)):
            be.set_precision(precision)
            u_ref = limb.pcg32_uniform(seed, ray_id, bounce, slot, offset)
            u_got = rng_warp.draw_uniform(seed, ray_id, bounce, slot, offset)
            assert u_got.dtype == u_ref.dtype
            differing = int((u_got.view(view) != u_ref.view(view)).sum())
            assert differing == 0, f"offset {offset}, {precision}: {differing} differ"
        be.set_precision("float64")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_uniform_is_the_same_bit_pattern(torch_backend_state, device, precision):
    """The uniform at both precisions, for every form of the bounce argument."""
    rng_warp = _warp_module()
    _use_device(device)
    be.set_precision(precision)
    view = torch.int64 if precision == "float64" else torch.int32
    n = 50_000
    gen = np.random.default_rng(3)
    ray_id_np = gen.integers(0, 2**40, size=n).astype(np.int64)
    bounce_np = gen.integers(0, 1100, size=n).astype(np.int64)
    ray_id = torch.as_tensor(ray_id_np, device=device)
    forms = {
        "int64 tensor": torch.as_tensor(bounce_np, device=device),
        "int32 tensor": torch.as_tensor(bounce_np.astype(np.int32), device=device),
        "numpy array": bounce_np,
        "python int": 17,
        "numpy integer": np.int64(4),
        "0-dim tensor": torch.tensor(9, dtype=torch.int64, device=device),
    }
    for form, bounce in forms.items():
        for slot, offset in ((EventSlot.FRESNEL_BRANCH, 0), (EventSlot.SOURCE_U1, 2)):
            expected = limb.pcg32_uniform(7, ray_id, bounce, slot, offset)
            got = rng_warp.draw_uniform(7, ray_id, bounce, slot, offset)
            assert got.dtype == expected.dtype, form
            assert got.shape == expected.shape, form
            differing = int((got.view(view) != expected.view(view)).sum())
            assert differing == 0, f"{form}, slot {slot}: {differing} differing values"

    # NumPy keys, as the sources pass them: the same values as the limb path.
    got = NSQRng(7).uniform(ray_id_np, bounce_np, EventSlot.SOURCE_U2)
    warp_rng = rng_warp.WarpNSQRng(7)
    allowed = rng_warp.SUPPORTED_DEVICE_TYPES
    try:
        rng_warp.SUPPORTED_DEVICE_TYPES = ("cuda", "cpu")
        drawn = warp_rng.uniform(ray_id_np, bounce_np, EventSlot.SOURCE_U2)
    finally:
        rng_warp.SUPPORTED_DEVICE_TYPES = allowed
    assert int((drawn.view(view) != got.view(view)).sum()) == 0


@pytest.mark.parametrize("device", _DEVICES)
def test_jump_ahead_state_is_the_host_reference_state(torch_backend_state, device):
    """The counter advances to the same LCG state, chunk by chunk.

    The output permutation keeps 32 of the state's 64 bits, so equal outputs
    do not by themselves prove equal states. This compares the state after
    the jump-ahead with the host reference's doubling loop, for step counts
    that exercise each 16-bit chunk the limb path's table walk uses.
    """
    rng_warp = _warp_module()
    _use_device(device)
    steps = [
        0,
        1,
        2,
        (1 << 16) - 1,
        1 << 16,
        (1 << 16) + 1,
        (1 << 32) - 1,
        (1 << 32) + 5,
        (1 << 48) + 7,
        (1 << 62) + 12345,
    ]
    ray_ids = [0, 1, 99, (1 << 40) - 1]
    rid = np.repeat(np.array(ray_ids, dtype=np.int64), len(steps))
    bounce = np.tile(np.array(steps, dtype=np.int64), len(ray_ids))
    for seed, slot, offset in ((0, 0, 0), (42, 5, 3), (2**63 + 1, 9, 70_000)):
        got = to_numpy(
            rng_warp.draw_state(
                seed,
                torch.as_tensor(rid, device=device),
                torch.as_tensor(bounce, device=device),
                slot,
                offset,
            )
        ).view(np.uint64)

        with np.errstate(over="ignore"):
            rid_u = rid.astype(np.uint64)
            initstate = limb._splitmix64(np.full_like(rid_u, np.uint64(seed & _M64)))
            initseq = limb._splitmix64(
                rid_u * limb._SM64_GAMMA
                ^ (np.uint64(slot) * np.uint64(limb._SLOT_CONST))
            )
            state0, inc = limb._pcg32_seed(initstate, initseq)
            mult = np.full_like(state0, limb._PCG_MULT)
            delta = bounce.astype(np.uint64) + np.uint64(offset)
            expected = limb._pcg32_advance(state0, delta, mult, inc)
        np.testing.assert_array_equal(got, expected)


def test_frozen_vectors_of_the_key_layout(torch_backend_state):
    """The conformance suite's fixed keys, drawn through the kernel."""
    rng_warp = _warp_module()
    ray_id = np.array([0, 1, 2, 5, 100, 999_999], dtype=np.int64)
    bounce = np.array([0, 0, 3, 1, 0, 2], dtype=np.int32)
    for slot in EventSlot:
        expected = limb._pcg32_uint32_reference(7, ray_id, bounce, slot).astype(
            np.int64
        )
        got = to_numpy(rng_warp.draw_bits(7, ray_id, bounce, slot))
        np.testing.assert_array_equal(got, expected)


def test_keys_the_kernel_does_not_take_use_the_limb_path(torch_backend_state):
    """An empty or two-dimensional key is drawn by the limb path, same values."""
    rng_warp = _warp_module()
    empty = torch.zeros(0, dtype=torch.int64)
    assert rng_warp.draw_bits(1, empty, 0, 5).shape == (0,)
    grid = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    np.testing.assert_array_equal(
        to_numpy(rng_warp.draw_bits(1, grid, 2, 5)),
        to_numpy(limb.pcg32_uint32(1, grid, 2, 5)),
    )


def test_a_draw_copies_nothing_to_the_host(torch_backend_state, monkeypatch):
    """No route off the device is taken, a zero-dimensional bounce included."""
    rng_warp = _warp_module()
    import optiland.backend.utils as backend_utils

    monkeypatch.setattr(rng_warp, "SUPPORTED_DEVICE_TYPES", ("cuda", "cpu"))
    rng = rng_warp.WarpNSQRng(19)
    ray_id = torch.arange(4096, dtype=torch.int64)
    bounce = torch.full((4096,), 2, dtype=torch.int32)
    scalar_bounce = torch.tensor(2, dtype=torch.int64)
    expected = to_numpy(limb.pcg32_uniform(19, ray_id, bounce, EventSlot.BSDF_U1))
    rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)  # warm: the module is loaded

    def _forbidden(*args, **kwargs):
        raise AssertionError("the draw copied data to the host")

    with monkeypatch.context() as trap:
        trap.setattr(backend_utils, "to_numpy", _forbidden)
        for name in ("numpy", "cpu", "tolist", "item", "__int__", "__index__"):
            trap.setattr(torch.Tensor, name, _forbidden, raising=False)
        with pytest.raises(AssertionError):
            ray_id.tolist()
        drawn = rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)
        drawn_scalar = rng.uniform(ray_id, scalar_bounce, EventSlot.BSDF_U1)
    np.testing.assert_array_equal(to_numpy(drawn), expected)
    np.testing.assert_array_equal(to_numpy(drawn_scalar), expected)


# ---------------------------------------------------------------------------
# Whole traces through the backend option
# ---------------------------------------------------------------------------


def _singlet() -> NSQScene:
    """Refraction: the Fresnel branch at every lens face, a rejection-sampled source."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0
        ),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50),
        LensConfig(
            r1=100.0,
            r2=-100.0,
            thickness=5.0,
            material="N-BK7",
            front_aperture_radius=12.5,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=150),
        IrradianceDetectorConfig(
            width=20, height=20, num_pixels_x=32, num_pixels_y=32, splat="bilinear"
        ),
    )
    return scene


def _scattering() -> NSQScene:
    """A Harvey-Shack lobe and a spectral tap: the scatter and lobe slots."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0
        ),
    )
    scene.add_detector(
        "TAP",
        CoordinateSystem(z=40),
        SpectralDetectorConfig(
            width=200,
            height=200,
            num_pixels_x=16,
            num_pixels_y=16,
            wl_min=0.4,
            wl_max=0.7,
            num_bins=4,
            splat="bilinear",
            absorb=False,
        ),
    )
    scene.add_mirror(
        "M1",
        CoordinateSystem(z=100),
        MirrorConfig(
            radius=0.0,
            reflectance=0.9,
            aperture_radius=50.0,
            surface=SurfaceConfig(
                bsdf=HarveyShackBSDF(b0=1e-3, l0=0.05, s=2.0), scatter_fraction=0.5
            ),
        ),
    )
    return scene


def _sphere() -> NSQScene:
    """A Lambertian cavity with two ports: many bounces, roulette, the BSDF slots."""
    radius = 50.0
    ports = [
        SphericalPort.from_area_fraction((0.0, 0.0, -1.0), 0.01),
        SphericalPort.from_area_fraction((1.0, 0.0, 0.0), 0.01),
    ]
    scene = NSQScene()
    scene.add_component(
        "wall",
        ReflectiveComponent(
            CoordinateSystem(),
            SphericalCavityGeometry(radius, ports),
            reflectance=0.9,
            bsdf=LambertianBSDF(reflectance_value=1.0),
        ),
    )
    scene.add_source(
        "beam",
        CoordinateSystem(z=-0.8 * radius),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.5876), total_flux=1.0, aperture_radius=2.5
        ),
    )
    scene.add_detector(
        "patch",
        CoordinateSystem(y=radius - 0.05, rx=math.radians(-90.0)),
        IrradianceDetectorConfig(
            width=5.0,
            height=5.0,
            num_pixels_x=1,
            num_pixels_y=1,
            splat="hard",
            absorb=False,
            side="front",
        ),
    )
    return scene


_SCENES = {"singlet": _singlet, "scattering": _scattering, "sphere": _sphere}

_LEDGER = (
    "num_rays_total",
    "num_rays_absorbed",
    "num_rays_escaped",
    "num_rays_flux_killed",
    "num_rays_depth_killed",
    "total_flux_in",
    "total_flux_detected",
    "total_flux_tapped",
    "total_flux_absorbed",
    "total_flux_coating",
    "total_flux_bulk_absorbed",
    "total_flux_escaped",
    "total_flux_lost",
    "total_flux_sampling_residual",
    "flux_conservation_error",
)


def _trace(
    scene_name: str, backend: TorchBackend, rays: int = 3000, max_depth: int = 40
):
    return _SCENES[scene_name]().trace(
        num_rays=rays, seed=11, max_depth=max_depth, batch_size=1024, backend=backend
    )


#: How far a float64 sum may move between two traces on CUDA, in units in the
#: last place. The detectors and the ledger add per-ray terms into float64
#: totals with a scatter-add whose order CUDA does not fix, so the last bits of a
#: float64 sum change from run to run whatever drew the random numbers.
#: Measured on an A100 (torch 2.14.0, 2026-09-25, the singlet and the scattering
#: scene below, 3,000 rays, seed 11): four traces of the default backend at
#: float64 differed from the first by up to 6 ulp in a detector pixel and 2 ulp in
#: ``total_flux_detected``; two Warp-drawn traces by up to 4 ulp; under
#: ``torch.use_deterministic_algorithms(True)`` two default traces were
#: bit-identical, and every float32 trace was, both ways. The draws themselves
#: are compared bit for bit on CUDA by the tests above.
_CUDA_FLOAT64_MAX_ULP = 8


def _same_float(va, vb, what: str, max_ulp: int, scale: float) -> None:
    """Equal, or within ``max_ulp`` ulps (with a floor of ``max_ulp`` ulps of ``scale``)."""
    a = np.asarray(va, dtype=np.float64)
    b = np.asarray(vb, dtype=np.float64)
    if max_ulp == 0 or a.shape != b.shape:
        np.testing.assert_array_equal(a, b, err_msg=what)
        return
    both_nan = np.isnan(a) & np.isnan(b)
    distance = np.abs(np.spacing(np.maximum(np.abs(a), np.abs(b))))
    floor = max_ulp * np.spacing(np.float64(scale))
    ok = both_nan | (np.abs(a - b) <= np.maximum(max_ulp * distance, floor))
    assert bool(np.all(ok)), (
        f"{what}: {int(np.count_nonzero(~ok))} values beyond {max_ulp} ulp"
    )


def _assert_same_trace(a, b, max_ulp: int = 0) -> None:
    """Every ledger entry and detector field of two traces.

    ``max_ulp=0`` (the CPU, and float32 everywhere) is bit equality. A float64
    trace on CUDA is compared within ``max_ulp`` ulps per value, because its
    sums are not reproducible there run to run (see ``_CUDA_FLOAT64_MAX_ULP``);
    a residual near zero, such as ``flux_conservation_error``, is compared
    within the same number of ulps of the launched flux. Counts are exact.
    """
    scale = float(a.total_flux_in) or 1.0
    for name in _LEDGER:
        va, vb = getattr(a, name), getattr(b, name)
        if isinstance(va, float) or isinstance(vb, float):
            _same_float(va, vb, name, max_ulp, scale)
        else:
            assert va == vb, f"{name}: {va!r} != {vb!r}"
    assert a.detectors.keys() == b.detectors.keys()
    for name in a.detectors:
        compared = 0
        for field in ("data", "irradiance", "intensity", "total_flux", "num_rays_hit"):
            if not hasattr(a.detectors[name], field):
                continue
            da = np.asarray(to_numpy(getattr(a.detectors[name], field)))
            db = np.asarray(to_numpy(getattr(b.detectors[name], field)))
            assert da.dtype == db.dtype, f"{name}.{field}"
            if da.dtype.kind == "f":
                _same_float(da, db, f"{name}.{field}", max_ulp, 0.0)
            else:
                np.testing.assert_array_equal(da, db, err_msg=f"{name}.{field}")
            compared += 1
        assert compared >= 2, f"{name}: nothing compared"


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("scene_name", sorted(_SCENES))
def test_a_trace_is_bit_identical_with_the_kernel(
    torch_backend_state, monkeypatch, device, precision, scene_name
):
    """Every ledger entry and every pixel, with the kernel drawing and without.

    Bit equality on the CPU and at float32 on any device. At float64 on CUDA,
    within ``_CUDA_FLOAT64_MAX_ULP`` ulps: there the detector and ledger sums
    move by a few ulps from one run to the next with the same generator, so
    bit equality of two traces would test CUDA's summation order, not the
    kernel (the measured control is at ``_CUDA_FLOAT64_MAX_ULP``).
    """
    rng_warp = _warp_module()
    _use_device(device)
    be.set_precision(precision)
    if device == "cpu":
        monkeypatch.setattr(rng_warp, "SUPPORTED_DEVICE_TYPES", ("cuda", "cpu"))

    reference = _trace(scene_name, TorchBackend(seed=11))

    # Count who draws during the fused trace: every draw must be the kernel's.
    calls = {"kernel": 0, "limb": 0}
    real_kernel, real_limb = rng_warp.draw_uniform, limb.pcg32_uniform

    def kernel_draw(*args, **kwargs):
        calls["kernel"] += 1
        return real_kernel(*args, **kwargs)

    def limb_draw(*args, **kwargs):
        calls["limb"] += 1
        return real_limb(*args, **kwargs)

    fused_backend = TorchBackend(seed=11, rng_kernel="warp")
    with monkeypatch.context() as spy:
        spy.setattr(rng_warp, "draw_uniform", kernel_draw)
        spy.setattr(limb, "pcg32_uniform", limb_draw)
        fused = _trace(scene_name, fused_backend)
    assert calls["kernel"] > 0
    assert calls["limb"] == 0, f"{calls['limb']} draws took the limb path"

    assert reference.environment["rng_kernel"] == "torch"
    assert fused.environment["rng_kernel"] == "warp"
    assert fused.environment["rng_kernel_requested"] == "warp"
    assert "rng_kernel_note" not in fused.environment
    assert fused_backend.rng_kernel_in_use == "warp"
    assert isinstance(fused_backend.rng, rng_warp.WarpNSQRng)
    cuda64 = device == "cuda" and precision == "float64"
    _assert_same_trace(reference, fused, _CUDA_FLOAT64_MAX_ULP if cuda64 else 0)


def test_the_kernels_are_loaded_before_the_first_bounce(
    torch_backend_state, monkeypatch
):
    """The module load happens at trace start, never inside the bounce loop.

    A CUDA-graph capture cannot contain a module load, so the backend loads
    the kernels on the device before the loop and the loop only launches.
    """
    rng_warp = _warp_module()
    monkeypatch.setattr(rng_warp, "SUPPORTED_DEVICE_TYPES", ("cuda", "cpu"))
    events: list[str] = []
    real_prepare = rng_warp.prepare
    real_intersect = TorchBackend.intersect_scene

    def prepare(device):
        events.append("prepare")
        return real_prepare(device)

    def intersect(self, rays, components):
        events.append("bounce")
        return real_intersect(self, rays, components)

    monkeypatch.setattr(rng_warp, "prepare", prepare)
    monkeypatch.setattr(TorchBackend, "intersect_scene", intersect)
    _trace("singlet", TorchBackend(seed=11, rng_kernel="warp"), rays=500, max_depth=4)
    assert events[0] == "prepare"
    assert "prepare" not in events[1:]


def test_the_backend_goes_back_to_the_limb_path_when_the_device_changes(
    torch_backend_state, monkeypatch
):
    """A backend reused on a device the kernel does not serve draws by limbs."""
    rng_warp = _warp_module()
    backend = TorchBackend(seed=11, rng_kernel="warp")
    with monkeypatch.context() as allow_cpu:
        allow_cpu.setattr(rng_warp, "SUPPORTED_DEVICE_TYPES", ("cuda", "cpu"))
        first = _trace("singlet", backend, rays=500)
    assert first.environment["rng_kernel"] == "warp"

    second = _singlet().trace(num_rays=500, max_depth=40, backend=backend)
    assert type(backend.rng) is NSQRng
    assert backend.rng.seed == 11
    assert second.environment["rng_kernel"] == "torch"
    assert "cuda" in second.environment["rng_kernel_note"]


# ---------------------------------------------------------------------------
# The option and its fallbacks (no Warp needed)
# ---------------------------------------------------------------------------


def test_the_default_backend_records_the_limb_path(torch_backend_state):
    result = _trace("singlet", TorchBackend(seed=11), rays=500)
    assert result.environment == {
        "array_backend": "torch",
        "device": "cpu",
        "precision": "float64",
        "rng_kernel": "torch",
        "rng_kernel_requested": "torch",
    }


def test_the_numpy_backend_records_its_own_generator():
    result = _singlet().trace(
        num_rays=500, seed=11, max_depth=40, backend=NumpyBackend(seed=11)
    )
    assert result.environment == {
        "array_backend": "numpy",
        "device": "cpu",
        "precision": "float64",
        "rng_kernel": "numpy",
    }


def test_an_unknown_kernel_is_refused():
    with pytest.raises(ValueError, match="rng_kernel"):
        TorchBackend(rng_kernel="triton")


def test_off_cuda_the_request_falls_back_silently(torch_backend_state):
    """On the CPU the limb path draws, nothing is warned, and the note says why."""
    reference = _trace("scattering", TorchBackend(seed=11), rays=800)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _trace(
            "scattering", TorchBackend(seed=11, rng_kernel="warp"), rays=800
        )
    assert not [w for w in caught if "warp" in str(w.message).lower()]
    assert result.environment["rng_kernel"] == "torch"
    assert result.environment["rng_kernel_requested"] == "warp"
    note = result.environment["rng_kernel_note"]
    assert "cuda" in note.lower() or "not importable" in note
    _assert_same_trace(reference, result)


def test_without_warp_the_request_falls_back_silently(torch_backend_state, monkeypatch):
    """With Warp absent the limb path draws and the note names the import."""
    monkeypatch.setitem(sys.modules, "optiland.nonsequential.rng_warp", None)
    monkeypatch.delattr(nsq_package, "rng_warp", raising=False)
    backend = TorchBackend(seed=11, rng_kernel="warp")
    result = _trace("singlet", backend, rays=500)
    assert backend.rng_kernel_in_use == "torch"
    assert result.environment["rng_kernel"] == "torch"
    assert "not importable" in result.environment["rng_kernel_note"]


# ---------------------------------------------------------------------------
# CUDA only: the draw inside a captured graph
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _cuda_with_warp(), reason="CUDA with Warp not available")
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_draw_is_recorded_by_a_cuda_graph(torch_backend_state, precision):
    """Captured once, replayed on new counters: the limb path's values."""
    rng_warp = _warp_module()
    be.set_device("cuda")
    be.set_precision(precision)
    view = torch.int64 if precision == "float64" else torch.int32
    n = 16_384
    ray_id = torch.arange(n, dtype=torch.int64, device="cuda")
    bounce = torch.zeros(n, dtype=torch.int32, device="cuda")
    rng = rng_warp.WarpNSQRng(5)
    rng_warp.prepare("cuda")

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        rng.uniform(ray_id, bounce, EventSlot.FRESNEL_BRANCH)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        drawn = rng.uniform(ray_id, bounce, EventSlot.FRESNEL_BRANCH)

    for b in (3, 399):
        bounce.fill_(b)
        graph.replay()
        torch.cuda.synchronize()
        expected = limb.pcg32_uniform(5, ray_id, bounce, EventSlot.FRESNEL_BRANCH)
        assert int((drawn.view(view) != expected.view(view)).sum()) == 0
