"""The component-intersection stage as Warp kernels: the torch stage's numbers.

``optiland.nonsequential.stage_warp`` computes ``BaseComponent.intersect``
for the ported spherical cavity and the conic in one Warp kernel per
component. It is an implementation change of one stage, so the forward checks
are equalities, never tolerances:

* the stage itself -- ``t``, both normals, the hit mask and the two halves of
  the hit distance ``advance_to_hit`` reads -- compared as bit patterns with
  the component's own ``intersect`` on 20,000 random rays, for an
  axis-aligned and a translated and rotated placement of each kind, at
  float64 and float32;
* whole traces through ``TorchBackend(intersect_kernel="warp")``: every
  ledger entry and every detector pixel bit-identical to the default backend
  on a refracting lens, a mirror with a scatter lobe and an integrating
  sphere, at both precisions, with a counter showing that every covered
  component went through its kernel.

The gradient is the Warp tape's adjoint of the same kernel, compared with
torch autograd through the component's own ``intersect``: agreement to a
few units of roundoff, not bits (the adjoint's sums are ordered differently).

The kernel runs on CUDA in the engine and on the CPU here: the tests allow
the CPU for their length (``SUPPORTED_DEVICE_TYPES``), which is the only
difference from a CUDA run. Where CUDA is present the stage tests run there
too, within the float64 summation window of the Warp generator's tests, and
a CUDA-graph replay with the kernels is compared with one without.
"""

from __future__ import annotations

import math
import sys

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
    Spectrum,
    SphericalCavityGeometry,
    SphericalPort,
    SurfaceConfig,
)
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.ray_bundle import NSQRayBundle


def _stage():
    """The kernel module, or a skip where Warp is not installed."""
    pytest.importorskip("warp", reason="Warp not installed (the optiland[warp] extra)")
    from optiland.nonsequential import stage_warp

    return stage_warp


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

#: The float64 summation window of a whole trace on CUDA (the Warp generator's
#: tests, ``test_nsq_rng_warp._CUDA_FLOAT64_MAX_ULP``): the detectors' and the
#: ledger's scatter-adds are not ordered on CUDA, so two traces of one backend
#: differ by up to 6 ulp there.
_CUDA_FLOAT64_MAX_ULP = 8


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
    be.grad_mode.disable()
    be.set_backend(previous)


def _use_device(device: str) -> None:
    if device == "cuda" and not _cuda_with_warp():
        pytest.skip("CUDA with Warp not available")
    be.set_device(device)


# ---------------------------------------------------------------------------
# The stage, against BaseComponent.intersect
# ---------------------------------------------------------------------------


def _bundle(p: np.ndarray, d: np.ndarray, alive: np.ndarray) -> NSQRayBundle:
    n = p.shape[0]
    rays = NSQRayBundle(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], L=d[:, 0], M=d[:, 1], N=d[:, 2],
        wavelength=np.full(n, 0.55), flux=np.ones(n), n_current=np.ones(n),
        bounce=np.zeros(n, dtype=np.int32), alive=alive,
    )
    rays.x, rays.y, rays.z, rays.L, rays.M, rays.N = (
        be.array(v) for v in (p[:, 0], p[:, 1], p[:, 2], d[:, 0], d[:, 1], d[:, 2])
    )
    rays.alive = torch.as_tensor(alive, device=rays.x.device)
    return rays


def _cavity(cs: CoordinateSystem, radius=50.0) -> ReflectiveComponent:
    ports = [
        SphericalPort.from_area_fraction((0.0, 0.0, -1.0), 0.01),
        SphericalPort.from_area_fraction((1.0, 0.0, 0.0), 0.01),
    ]
    comp = ReflectiveComponent(
        cs,
        SphericalCavityGeometry(radius, ports),
        reflectance=0.9,
        bsdf=LambertianBSDF(reflectance_value=1.0),
    )
    comp.refresh_backend_transform()
    return comp


def _lens_front(cs: CoordinateSystem, r1=30.0, conic=None):
    scene = NSQScene()
    scene.add_lens(
        "L", cs, LensConfig(r1=r1, r2=-40.0, thickness=4.0, material=1.5, front_aperture_radius=12.0)
    )
    front = scene.surfaces[0]
    if conic is not None:
        front.geometry.conic = conic
    front.refresh_backend_transform()
    return front


_N_STAGE = 20_000


def _cavity_rays(center, seed=1):
    g = np.random.default_rng(seed)
    p = g.uniform(-40.0, 40.0, (_N_STAGE, 3)) + np.asarray(center, dtype=float)
    d = g.normal(size=(_N_STAGE, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return p, d, g.uniform(size=_N_STAGE) > 0.1


def _lens_rays(seed=3):
    g = np.random.default_rng(seed)
    p = np.stack(
        [g.uniform(-15, 15, _N_STAGE), g.uniform(-15, 15, _N_STAGE), np.full(_N_STAGE, -30.0)], 1
    )
    d = np.stack(
        [g.normal(0, 0.2, _N_STAGE), g.normal(0, 0.2, _N_STAGE), np.ones(_N_STAGE)], 1
    )
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return p, d, g.uniform(size=_N_STAGE) > 0.05


_CASES = {
    "cavity-aligned": lambda: (_cavity(CoordinateSystem()), _cavity_rays((0, 0, 0))),
    "cavity-rotated": lambda: (
        _cavity(CoordinateSystem(x=3.0, y=-7.0, z=11.0, rx=0.3, ry=-0.7, rz=1.1)),
        _cavity_rays((3.0, -7.0, 11.0)),
    ),
    "conic-sphere": lambda: (_lens_front(CoordinateSystem(z=5.0)), _lens_rays()),
    "conic-flat": lambda: (_lens_front(CoordinateSystem(), r1=1.0e9), _lens_rays()),
    "conic-hyperboloid-rotated": lambda: (
        _lens_front(CoordinateSystem(x=1.0, z=2.0, rx=0.2, ry=0.4), r1=-25.0, conic=-2.3),
        _lens_rays(),
    ),
}


def _bits(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu()
    if a.dtype == torch.bool:
        return a.numpy()
    return a.view(torch.int64 if a.dtype == torch.float64 else torch.int32).numpy()


def _ulps(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().cpu().double().numpy()
    y = b.detach().cpu().double().numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    assert np.array_equal(np.isfinite(x), np.isfinite(y))
    spacing = np.spacing(np.maximum(np.abs(x[finite]), np.abs(y[finite])).astype(
        np.float64 if a.dtype == torch.float64 else np.float32))
    return float(np.max(np.abs(x[finite] - y[finite]) / spacing, initial=0.0))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_stage_is_the_components_own_intersect(torch_backend_state, device, precision, case):
    """``t``, both normals, the hit mask and ``_local_root``, bit for bit on the CPU.

    On CUDA the kernel is compared within the float64 summation window: torch's
    own CUDA matrix product and reductions are not ordered as on the CPU.
    """
    stage = _stage()
    _use_device(device)
    be.set_precision(precision)
    comp, (p, d, alive) = _CASES[case]()
    comp.refresh_backend_transform()
    rays = _bundle(p, d, alive)
    kind = stage.kind_of(comp)
    assert kind == ("cavity" if case.startswith("cavity") else "conic")

    ref = comp.intersect(rays)
    ref_root = comp._local_root
    got = stage.intersect_component(comp, kind, rays, stage.accept_threshold(rays))
    got_root = comp._local_root
    assert int(ref[2].sum()) > _N_STAGE // 4, "the configuration hits too little to test"
    names = ("t", "normals", "hit_mask", "n_geom", "t_adv", "t_local")
    for name, a, b in zip(names, (*ref, *ref_root), (*got, *got_root), strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        if device == "cpu" or a.dtype == torch.bool:
            np.testing.assert_array_equal(_bits(a), _bits(b), err_msg=name)
        else:
            assert _ulps(a, b) <= _CUDA_FLOAT64_MAX_ULP, name


def test_components_the_kernels_do_not_cover_keep_their_own_intersect(torch_backend_state):
    stage = _stage()
    scene = NSQScene()
    scene.add_lens(
        "L", CoordinateSystem(), LensConfig(r1=30.0, r2=-40.0, thickness=4.0, material=1.5,
                                            front_aperture_radius=12.0)
    )
    kinds = [stage.kind_of(c) for c in scene.surfaces]
    assert kinds.count("conic") == 2
    assert None in kinds  # the lens edge, a frustum


# ---------------------------------------------------------------------------
# The gradient: the tape's adjoint against torch autograd
# ---------------------------------------------------------------------------


def _grad_bundle(p, d):
    rays = _bundle(p, d, np.ones(p.shape[0], dtype=bool))
    leaves = [
        torch.tensor(v, dtype=torch.float64, requires_grad=True)
        for v in (p[:, 0], p[:, 1], p[:, 2], d[:, 0], d[:, 1], d[:, 2])
    ]
    rays.x, rays.y, rays.z, rays.L, rays.M, rays.N = leaves
    return rays, leaves


def _loss(out, root, weights):
    t, normals, hit, n_geom = out
    t_adv, t_local = root
    t_hit = torch.where(hit, t, torch.zeros_like(t))
    t_local = torch.where(torch.isfinite(t_local), t_local, torch.zeros_like(t_local))
    return (
        (weights[0] * t_hit).sum()
        + (weights[1] * normals).sum()
        + (weights[2] * n_geom).sum()
        + (weights[3] * t_adv).sum()
        + (weights[4] * t_local).sum()
    )


def _cavity_with_parameter():
    radius = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    return _cavity(CoordinateSystem(x=1.0, rx=0.2), radius=radius), [radius]


def _conic_with_parameters():
    radius = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    conic = torch.tensor(-0.6, dtype=torch.float64, requires_grad=True)
    return _lens_front(CoordinateSystem(z=2.0, ry=0.1), r1=radius, conic=conic), [radius, conic]


@pytest.mark.parametrize("which", ["cavity", "conic"])
def test_the_adjoint_is_torch_autograd_of_the_stage(torch_backend_state, which):
    """Every input's gradient within 1e-13 of the largest, the loss bit-identical.

    Measured (Apple silicon CPU, float64, 20,000 rays): worst 3.5e-15 for the
    cavity (the radius), 7.6e-15 for the conic (the radius).
    """
    stage = _stage()
    g = np.random.default_rng(7)
    if which == "cavity":
        p, d, _ = _cavity_rays((0, 0, 0), seed=1)
        make = _cavity_with_parameter
    else:
        p, d, _ = _lens_rays(seed=3)
        make = _conic_with_parameters
    n = p.shape[0]
    weights = [torch.tensor(g.normal(size=s)) for s in [(n,), (n, 3), (n, 3), (n,), (n,)]]
    results = []
    for route in ("torch", "warp"):
        comp, params = make()
        rays, leaves = _grad_bundle(p, d)
        if route == "torch":
            out = comp.intersect(rays)
        else:
            out = stage.intersect_component(comp, stage.kind_of(comp), rays, stage.accept_threshold(rays))
        loss = _loss(out, comp._local_root, weights)
        grads = torch.autograd.grad(loss, leaves + params)
        results.append((loss.detach(), [gr.detach().numpy() for gr in grads]))
    (loss_t, grads_t), (loss_w, grads_w) = results
    assert torch.equal(loss_t, loss_w)
    for a, b in zip(grads_t, grads_w, strict=True):
        scale = float(np.max(np.abs(a))) or 1.0
        assert float(np.max(np.abs(a - b))) <= 1e-13 * scale


# ---------------------------------------------------------------------------
# Whole traces
# ---------------------------------------------------------------------------


def _singlet() -> NSQScene:
    scene = NSQScene()
    scene.add_source(
        "S1", CoordinateSystem(),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=10.0),
    )
    scene.add_lens(
        "L1", CoordinateSystem(z=50),
        LensConfig(r1=100.0, r2=-100.0, thickness=5.0, material="N-BK7", front_aperture_radius=12.5),
    )
    scene.add_detector(
        "D1", CoordinateSystem(z=150),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=32, num_pixels_y=32, splat="bilinear"),
    )
    return scene


def _mirror() -> NSQScene:
    scene = NSQScene()
    scene.add_source(
        "S1", CoordinateSystem(),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0),
    )
    scene.add_detector(
        "TAP", CoordinateSystem(z=40),
        IrradianceDetectorConfig(width=200, height=200, num_pixels_x=16, num_pixels_y=16,
                                 splat="bilinear", absorb=False),
    )
    scene.add_mirror(
        "M1", CoordinateSystem(z=100, rx=0.05),
        MirrorConfig(radius=-400.0, reflectance=0.9, aperture_radius=50.0,
                     surface=SurfaceConfig(bsdf=HarveyShackBSDF(b0=1e-3, l0=0.05, s=2.0),
                                           scatter_fraction=0.5)),
    )
    return scene


def _sphere() -> NSQScene:
    radius = 50.0
    scene = NSQScene()
    scene.add_component("wall", _cavity(CoordinateSystem(), radius))
    scene.add_source(
        "beam", CoordinateSystem(z=-0.8 * radius),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.5876), total_flux=1.0, aperture_radius=2.5),
    )
    scene.add_detector(
        "patch", CoordinateSystem(y=radius - 0.05, rx=math.radians(-90.0)),
        IrradianceDetectorConfig(width=5.0, height=5.0, num_pixels_x=1, num_pixels_y=1,
                                 splat="hard", absorb=False, side="front"),
    )
    return scene


_SCENES = {"singlet": _singlet, "mirror": _mirror, "sphere": _sphere}

_LEDGER = (
    "num_rays_total", "num_rays_absorbed", "num_rays_escaped", "num_rays_flux_killed",
    "num_rays_depth_killed", "total_flux_in", "total_flux_detected", "total_flux_tapped",
    "total_flux_absorbed", "total_flux_coating", "total_flux_bulk_absorbed",
    "total_flux_escaped", "total_flux_lost", "total_flux_sampling_residual",
    "flux_conservation_error",
)


def _assert_same_trace(a, b, max_ulp: int = 0) -> None:
    def same(va, vb, what):
        x = np.asarray(va, dtype=np.float64)
        y = np.asarray(vb, dtype=np.float64)
        if max_ulp == 0:
            np.testing.assert_array_equal(x, y, err_msg=what)
            return
        tol = max_ulp * np.spacing(np.maximum(np.abs(x), np.abs(y)))
        floor = max_ulp * np.spacing(np.float64(float(a.total_flux_in) or 1.0))
        assert np.all(np.abs(x - y) <= np.maximum(tol, floor)), what

    for name in _LEDGER:
        va, vb = getattr(a, name), getattr(b, name)
        if isinstance(va, float) or isinstance(vb, float):
            same(va, vb, name)
        else:
            assert va == vb, f"{name}: {va!r} != {vb!r}"
    for name in a.detectors:
        for field in ("data", "total_flux", "num_rays_hit"):
            if hasattr(a.detectors[name], field):
                same(to_numpy(getattr(a.detectors[name], field)),
                     to_numpy(getattr(b.detectors[name], field)), f"{name}.{field}")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("scene_name", sorted(_SCENES))
def test_a_trace_is_bit_identical_with_the_kernels(
    torch_backend_state, monkeypatch, device, precision, scene_name
):
    stage = _stage()
    _use_device(device)
    be.set_precision(precision)
    if device == "cpu":
        monkeypatch.setattr(stage, "SUPPORTED_DEVICE_TYPES", ("cuda", "cpu"))

    def trace(backend):
        return _SCENES[scene_name]().trace(
            num_rays=3000, seed=11, max_depth=40, batch_size=1024, backend=backend
        )

    reference = trace(TorchBackend(seed=11))
    calls = {"kernel": 0}
    real = stage.intersect_component

    def counted(*args, **kwargs):
        calls["kernel"] += 1
        return real(*args, **kwargs)

    backend = TorchBackend(seed=11, intersect_kernel="warp")
    with monkeypatch.context() as spy:
        spy.setattr(stage, "intersect_component", counted)
        fused = trace(backend)
    assert calls["kernel"] > 0
    assert fused.environment["intersect_kernel"] == "warp"
    assert fused.environment["intersect_kernel_requested"] == "warp"
    assert "intersect_kernel_note" not in fused.environment
    assert "intersect_kernel" not in reference.environment
    cuda64 = device == "cuda" and precision == "float64"
    _assert_same_trace(reference, fused, _CUDA_FLOAT64_MAX_ULP if cuda64 else 0)


# ---------------------------------------------------------------------------
# The option: default, fallback, refusal
# ---------------------------------------------------------------------------


def test_the_default_backend_records_nothing_for_the_stage(torch_backend_state):
    result = _sphere().trace(num_rays=200, seed=3, max_depth=20, backend=TorchBackend(seed=3))
    assert not any(key.startswith("intersect_kernel") for key in result.environment)


def test_an_unknown_stage_kernel_is_refused():
    with pytest.raises(ValueError, match="intersect_kernel"):
        TorchBackend(intersect_kernel="triton")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the fallback off CUDA")
def test_off_cuda_the_request_falls_back_silently(torch_backend_state):
    _stage()
    backend = TorchBackend(seed=3, intersect_kernel="warp")
    reference = _sphere().trace(num_rays=500, seed=3, max_depth=30, backend=TorchBackend(seed=3))
    result = _sphere().trace(num_rays=500, seed=3, max_depth=30, backend=backend)
    assert result.environment["intersect_kernel"] == "torch"
    assert result.environment["intersect_kernel_requested"] == "warp"
    assert "cpu" in result.environment["intersect_kernel_note"]
    _assert_same_trace(reference, result)


def test_without_warp_the_request_falls_back_silently(torch_backend_state, monkeypatch):
    monkeypatch.setitem(sys.modules, "optiland.nonsequential.stage_warp", None)
    monkeypatch.delattr(nsq_package, "stage_warp", raising=False)
    result = _sphere().trace(
        num_rays=200, seed=3, max_depth=20, backend=TorchBackend(seed=3, intersect_kernel="warp")
    )
    assert result.environment["intersect_kernel"] == "torch"
    assert "not importable" in result.environment["intersect_kernel_note"]


# ---------------------------------------------------------------------------
# CUDA: the kernels inside a recorded bounce
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _cuda_with_warp(), reason="CUDA with Warp not available")
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_replayed_bounce_records_the_kernels(torch_backend_state, precision):
    """The CUDA-graph replay with the kernels equals the replay without them."""
    stage = _stage()
    be.set_device("cuda")
    be.set_precision(precision)

    def trace(**kw):
        backend = TorchBackend(seed=5, graph_replay=True, alive_check_every=0, **kw)
        result = _sphere().trace(num_rays=4096, seed=5, max_depth=60, batch_size=4096, backend=backend)
        assert backend.graph_replay_batches == 1
        return result

    reference = trace()
    fused = trace(intersect_kernel="warp")
    assert fused.environment["intersect_kernel"] == "warp"
    assert fused.environment["graph_replay"] == "cuda"
    assert stage.availability("cuda") is None
    _assert_same_trace(reference, fused, _CUDA_FLOAT64_MAX_ULP if precision == "float64" else 0)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_the_kernels_pass_the_capture_safety_check(torch_backend_state, monkeypatch, precision):
    """``graph_replay="emulate"`` refuses any host transfer inside the recorded bounce.

    The stage's per-call inputs are cached or built without a torch operation, so
    the emulated replay runs with the kernels and gives the eager fixed-width trace.
    """
    stage = _stage()
    be.set_precision(precision)
    monkeypatch.setattr(stage, "SUPPORTED_DEVICE_TYPES", ("cuda", "cpu"))

    def trace(**kw):
        backend = TorchBackend(seed=5, graph_replay="emulate", alive_check_every=0, **kw)
        result = _sphere().trace(num_rays=2048, seed=5, max_depth=40, batch_size=2048, backend=backend)
        assert backend.graph_replay_batches == 1
        return result

    reference = trace()
    fused = trace(intersect_kernel="warp")
    assert fused.environment["intersect_kernel"] == "warp"
    assert fused.environment["graph_replay"] == "emulate"
    _assert_same_trace(reference, fused)
