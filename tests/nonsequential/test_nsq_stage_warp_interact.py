"""The reflective interaction as one Warp launch: the torch stage's numbers.

``optiland.nonsequential.stage_warp_interact`` computes
``ReflectiveComponent.interact`` (the advance to the hit point, the specular
reflection, the reflectance, the scatter branch, the Lambertian lobe, the
bounce count, the origin offset and the ledger's bookings) in one Warp kernel.
It changes how one stage is computed, not what it computes, so every check is
an equality of bit patterns:

* the stage itself against the component's own ``interact``, every field of
  the bundle and both ledger tallies, for a mirror, a Lambertian wall and a
  Lambertian lobe with a transmissive fraction and a scatter fraction below
  one, each axis-aligned and translated and rotated, at float64 and float32,
  with the hit's two halves present and absent, and at every width from 1 to
  70 and at five larger widths (the placement product's order is the width's);
* whole traces through ``TorchBackend(interact_kernel="warp")``, alone and with
  the Warp intersection stage, against the default backend;
* the routing: a lobe, a reflectance or a trace the kernel does not cover runs
  the component's own ``interact`` and is counted by reason.

The kernel runs on CUDA in the engine and on the CPU here (the tests allow the
CPU through ``SUPPORTED_DEVICE_TYPES``); where CUDA is present the stage tests
run there too.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    HarveyShackBSDF,
    IrradianceDetectorConfig,
    LambertianBSDF,
    NSQScene,
    ReflectiveComponent,
    Spectrum,
    SphericalCavityGeometry,
    SphericalPort,
)
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.ir.lower import _lower_bsdf
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.rng import NSQRng


def _stages():
    pytest.importorskip("warp", reason="Warp not installed (the optiland[warp] extra)")
    from optiland.nonsequential import stage_warp, stage_warp_interact

    return stage_warp, stage_warp_interact


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
def torch_state(monkeypatch):
    """Torch on the CPU at float64, the stages allowed on the CPU; the state put back."""
    sw, si = _stages()
    monkeypatch.setattr(sw, "SUPPORTED_DEVICE_TYPES", ("cuda", "cpu"))
    monkeypatch.setattr(si, "SUPPORTED_DEVICE_TYPES", ("cuda", "cpu"))
    previous = be.get_backend()
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.disable()
    yield sw, si
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.disable()
    be.set_backend(previous)


def _use(device: str, precision: str) -> None:
    if device == "cuda" and not _cuda_with_warp():
        pytest.skip("CUDA with Warp not available")
    be.set_device(device)
    be.set_precision(precision)


# ---------------------------------------------------------------------------
# The stage against ReflectiveComponent.interact
# ---------------------------------------------------------------------------

_PORTS = [SphericalPort.from_area_fraction((0.0, 0.0, -1.0), 0.01)]
_PLACEMENTS = {
    "identity": CoordinateSystem(),
    "moved": CoordinateSystem(x=3.0, y=-7.0, z=11.0, rx=0.3, ry=-0.7, rz=1.1),
}
_SURFACES = {
    # a mirror: no lobe
    "mirror": dict(reflectance=0.93, bsdf=None),
    # the catalogue's integrating-sphere wall
    "lambertian": dict(reflectance=0.98, bsdf=LambertianBSDF(reflectance_value=1.0)),
    # a lobe with a transmissive fraction, a lobe weight and a scatter fraction below one
    "lobe-mixed": dict(reflectance=0.85, bsdf=LambertianBSDF(0.8, transmissive_fraction=0.3),
                       scatter_fraction=0.6),
}


def _component(surface: str, placement: str, radius: float = 40.0) -> ReflectiveComponent:
    spec = dict(_SURFACES[surface])
    bsdf = spec.pop("bsdf")
    if bsdf is not None:
        bsdf = LambertianBSDF(bsdf.reflectance_value, transmissive_fraction=bsdf.transmissive_fraction)
    comp = ReflectiveComponent(_PLACEMENTS[placement], SphericalCavityGeometry(radius, _PORTS), bsdf=bsdf, **spec)
    comp.refresh_backend_transform()
    return comp


def _bundle(n: int, seed: int, placement: str) -> NSQRayBundle:
    """Rays inside the cavity (some dead, varied flux and bounce counts), on the backend."""
    g = np.random.default_rng(seed)
    centre = np.array([3.0, -7.0, 11.0]) if placement == "moved" else np.zeros(3)
    p = g.uniform(-25.0, 25.0, (n, 3)) + centre
    d = g.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dev = be.get_device()
    ft = torch.float64 if be.get_precision() in (64, "float64") else torch.float32

    def arr(v, dtype=ft):
        return torch.as_tensor(np.asarray(v), dtype=dtype, device=dev)

    return NSQRayBundle(
        x=arr(p[:, 0]), y=arr(p[:, 1]), z=arr(p[:, 2]), L=arr(d[:, 0]), M=arr(d[:, 1]), N=arr(d[:, 2]),
        wavelength=arr(np.full(n, 0.55)), flux=arr(g.uniform(0.1, 2.0, n)), n_current=arr(np.ones(n)),
        k_current=arr(np.zeros(n)),
        bounce=arr(g.integers(0, 7, n), torch.int32), alive=arr(g.uniform(size=n) > 0.1, torch.bool),
        ray_id=arr(np.arange(n) * 3 + seed, torch.int64),
    )


_FIELDS = ("x", "y", "z", "L", "M", "N", "flux", "bounce")


def _run(si, surface, placement, n, seed, with_root=True, through_stage=True):
    comp = _component(surface, placement)
    rays = _bundle(n, seed, placement)
    t, normals, hit, n_geom = comp.intersect(rays)
    if not with_root:
        comp._local_root = None
    comp.reset_ledger()
    rng = NSQRng(seed)
    bsdf_ir = _lower_bsdf(comp.bsdf)
    if through_stage:
        si.interact_component(comp, rays, t, normals, hit, rng, bsdf_ir, n_geom)
    else:
        comp.interact(rays, t, normals, hit, rng, bsdf_ir, n_geom)
    out = {f: to_numpy(getattr(rays, f)) for f in _FIELDS}
    out["coating_loss"] = comp.coating_loss
    out["sampling_residual"] = comp.sampling_residual
    out["hits"] = int(to_numpy(hit).sum())
    return out


def _assert_bits(a: dict, b: dict) -> None:
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        assert x.dtype == y.dtype, key
        if x.dtype.kind == "f":
            np.testing.assert_array_equal(x.view(np.int64 if x.dtype == np.float64 else np.int32),
                                          y.view(np.int64 if y.dtype == np.float64 else np.int32), err_msg=key)
        else:
            np.testing.assert_array_equal(x, y, err_msg=key)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("placement", sorted(_PLACEMENTS))
@pytest.mark.parametrize("surface", sorted(_SURFACES))
def test_the_stage_is_the_components_own_interact(torch_state, device, precision, placement, surface):
    _, si = torch_state
    _use(device, precision)
    si.reset_routed()
    got = _run(si, surface, placement, 20_000, 7)
    assert si.launch_counts() == {"reflective": 1} and not si.routed_counts()
    ref = _run(si, surface, placement, 20_000, 7, through_stage=False)
    assert ref["hits"] > 10_000
    _assert_bits(ref, got)


def _run_twice(si, through_stage):
    """Two interactions of one component in a row: the second adds into the device tallies (one launch)."""
    comp = _component("lobe-mixed", "moved")
    comp.reset_ledger()
    rng = NSQRng(5)
    bsdf_ir = _lower_bsdf(comp.bsdf)
    for seed in (5, 6):
        rays = _bundle(4_000, seed, "moved")
        t, normals, hit, n_geom = comp.intersect(rays)
        if through_stage:
            si.interact_component(comp, rays, t, normals, hit, rng, bsdf_ir, n_geom)
        else:
            comp.interact(rays, t, normals, hit, rng, bsdf_ir, n_geom)
    tallies = (comp._tally("_coating_loss"), comp._tally("_sampling_residual"))
    return {f"{name}_{part}": to_numpy(getattr(tally, part)).reshape(1)
            for name, tally in zip(("coat", "res"), tallies, strict=True) for part in ("_dev", "_dev_comp")}


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_the_ledger_adds_into_the_tallies_as_tally_add_does(torch_state, device, precision):
    _, si = torch_state
    _use(device, precision)
    _assert_bits(_run_twice(si, through_stage=False), _run_twice(si, through_stage=True))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_bundle_the_component_did_not_intersect_takes_the_plain_advance(torch_state, device, precision):
    _, si = torch_state
    _use(device, precision)
    got = _run(si, "lobe-mixed", "moved", 5_000, 3, with_root=False)
    ref = _run(si, "lobe-mixed", "moved", 5_000, 3, with_root=False, through_stage=False)
    _assert_bits(ref, got)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_the_stage_is_bit_identical_at_every_width(torch_state, device, precision):
    _, si = torch_state
    _use(device, precision)
    for n in [*range(1, 71), 711, 712, 2843, 2844, 4097]:
        got = _run(si, "lobe-mixed", "moved", n, 100 + n)
        ref = _run(si, "lobe-mixed", "moved", n, 100 + n, through_stage=False)
        _assert_bits(ref, got)


def test_the_probes_say_what_the_kernel_evaluates(torch_state):
    """``x ** 0.5`` is the square root on the CPU; the azimuth's sine and cosine are torch's there, not Warp's."""
    _, si = torch_state
    for dtype in (torch.float64, torch.float32):
        result = si.probes(dtype, "cpu")
        assert result["sqrt_pow"] is True
        assert result["trig_in_kernel"] in (True, False)


# ---------------------------------------------------------------------------
# Whole traces
# ---------------------------------------------------------------------------


def _sphere(placement: str = "identity") -> NSQScene:
    radius = 50.0
    cs = _PLACEMENTS[placement]
    scene = NSQScene()
    wall = ReflectiveComponent(
        cs, SphericalCavityGeometry(radius, [SphericalPort.from_area_fraction((0.0, 0.0, -1.0), 0.01),
                                             SphericalPort.from_area_fraction((1.0, 0.0, 0.0), 0.01)]),
        reflectance=0.9, bsdf=LambertianBSDF(reflectance_value=1.0), name="wall",
    )
    scene.add_component("wall", wall)
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


def _diffuser_box() -> NSQScene:
    """A tilted mirror and a tilted Lambertian lobe with a transmissive part, ahead of a detector."""
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=4.0),
    )
    mirror = ReflectiveComponent(
        CoordinateSystem(z=60.0, rx=0.4), SphericalCavityGeometry(80.0, []), reflectance=0.95, name="mirror",
    )
    lobe = ReflectiveComponent(
        CoordinateSystem(y=10.0, z=20.0, ry=0.2), SphericalCavityGeometry(120.0, []), reflectance=0.9,
        bsdf=LambertianBSDF(0.7, transmissive_fraction=0.25), scatter_fraction=0.5, name="lobe",
    )
    scene.add_component("mirror", mirror)
    scene.add_component("lobe", lobe)
    scene.add_detector(
        "D", CoordinateSystem(z=30.0),
        IrradianceDetectorConfig(width=100, height=100, num_pixels_x=8, num_pixels_y=8,
                                 splat="bilinear", absorb=False),
    )
    return scene


_SCENES = {"sphere": _sphere, "sphere-moved": lambda: _sphere("moved"), "box": _diffuser_box}

_LEDGER = (
    "num_rays_total", "num_rays_absorbed", "num_rays_escaped", "num_rays_flux_killed",
    "num_rays_depth_killed", "total_flux_in", "total_flux_detected", "total_flux_tapped",
    "total_flux_absorbed", "total_flux_coating", "total_flux_bulk_absorbed",
    "total_flux_escaped", "total_flux_lost", "total_flux_sampling_residual",
    "flux_conservation_error",
)


def _assert_same_trace(a, b) -> None:
    for name in _LEDGER:
        va, vb = getattr(a, name), getattr(b, name)
        if isinstance(va, float) or isinstance(vb, float):
            assert np.float64(va).view(np.int64) == np.float64(vb).view(np.int64), f"{name}: {va!r} != {vb!r}"
        else:
            assert va == vb, f"{name}: {va!r} != {vb!r}"
    for name in a.detectors:
        for field in ("data", "total_flux", "num_rays_hit"):
            if hasattr(a.detectors[name], field):
                np.testing.assert_array_equal(np.asarray(to_numpy(getattr(a.detectors[name], field))),
                                              np.asarray(to_numpy(getattr(b.detectors[name], field))),
                                              err_msg=f"{name}.{field}")


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("scene_name", sorted(_SCENES))
@pytest.mark.parametrize("with_intersect", [False, True])
def test_a_trace_is_bit_identical_with_the_kernel(torch_state, precision, scene_name, with_intersect):
    _, si = torch_state
    be.set_precision(precision)

    def trace(**kw):
        return _SCENES[scene_name]().trace(num_rays=3000, seed=11, max_depth=60, batch_size=1024,
                                           backend=TorchBackend(seed=11, **kw))

    reference = trace()
    kw = {"interact_kernel": "warp"}
    if with_intersect:
        kw["intersect_kernel"] = "warp"
    fused = trace(**kw)
    assert fused.environment["interact_kernel"] == "warp"
    assert fused.environment["interact_kernel_requested"] == "warp"
    assert "interact_kernel_routed" not in fused.environment
    assert si.launch_counts()["reflective"] > 0
    assert "interact_kernel" not in reference.environment
    _assert_same_trace(reference, fused)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_the_kernel_passes_the_capture_safety_check(torch_state, precision):
    """``graph_replay="emulate"`` refuses a host transfer inside the recorded bounce: none is made."""
    be.set_precision(precision)

    def trace(**kw):
        backend = TorchBackend(seed=5, graph_replay="emulate", alive_check_every=0, **kw)
        result = _sphere().trace(num_rays=2048, seed=5, max_depth=40, batch_size=2048, backend=backend)
        assert backend.graph_replay_batches == 1
        return result

    reference = trace()
    fused = trace(interact_kernel="warp", intersect_kernel="warp")
    assert fused.environment["graph_replay"] == "emulate"
    _assert_same_trace(reference, fused)


@pytest.mark.skipif(not _cuda_with_warp(), reason="CUDA with Warp not available")
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_replayed_bounce_on_cuda_records_the_kernel(torch_state, precision):
    be.set_device("cuda")
    be.set_precision(precision)

    def trace(**kw):
        backend = TorchBackend(seed=5, graph_replay=True, alive_check_every=0, rng_kernel="warp", **kw)
        result = _sphere().trace(num_rays=4096, seed=5, max_depth=60, batch_size=4096, backend=backend)
        assert backend.graph_replay_batches == 1
        return result

    reference = trace()
    fused = trace(interact_kernel="warp", intersect_kernel="warp")
    assert fused.environment["graph_replay"] == "cuda"
    assert fused.environment["interact_kernel"] == "warp"
    _assert_same_trace(reference, fused)


# ---------------------------------------------------------------------------
# Routing and the option
# ---------------------------------------------------------------------------


def test_what_the_kernel_does_not_cover_keeps_its_own_interact(torch_state):
    _, si = torch_state
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=4.0),
    )
    rough = ReflectiveComponent(
        CoordinateSystem(z=60.0, rx=0.4), SphericalCavityGeometry(80.0, []), reflectance=0.95,
        bsdf=HarveyShackBSDF(b0=1e-3, l0=0.05, s=2.0), scatter_fraction=0.5, name="rough",
    )
    tabled = ReflectiveComponent(
        CoordinateSystem(z=90.0), SphericalCavityGeometry(150.0, []), reflectance=lambda wl: 0.5 + 0.0 * wl,
        name="callable",
    )
    scene.add_component("rough", rough)
    scene.add_component("callable", tabled)
    scene.add_detector(
        "D", CoordinateSystem(z=30.0),
        IrradianceDetectorConfig(width=100, height=100, num_pixels_x=4, num_pixels_y=4, absorb=False),
    )

    def trace(**kw):
        return scene.trace(num_rays=500, seed=3, max_depth=10, backend=TorchBackend(seed=3, **kw))

    reference = trace()
    fused = trace(interact_kernel="warp")
    assert fused.environment["interact_kernel_routed"] == {si.ROUTE_KIND: sum(si.routed_counts().values())}
    assert si.launch_counts() == {}
    _assert_same_trace(reference, fused)


def test_a_gradient_through_the_interaction_is_routed(torch_state):
    _, si = torch_state
    be.grad_mode.enable()
    try:
        si.reset_routed()
        comp = _component("lambertian", "moved")
        rays = _bundle(256, 1, "moved")
        rays.flux = rays.flux.clone().requires_grad_(True)
        t, normals, hit, n_geom = comp.intersect(rays)
        si.interact_component(comp, rays, t, normals, hit, NSQRng(1), _lower_bsdf(comp.bsdf), n_geom)
        assert si.routed_counts() == {si.ROUTE_GRADIENT: 1}
        assert rays.flux.requires_grad
    finally:
        be.grad_mode.disable()


def test_the_default_backend_records_nothing_for_the_stage(torch_state):
    result = _sphere().trace(num_rays=200, seed=1, max_depth=5, backend=TorchBackend(seed=1))
    assert "interact_kernel" not in result.environment


def test_an_unknown_interaction_kernel_is_refused():
    with pytest.raises(ValueError, match="interact_kernel"):
        TorchBackend(interact_kernel="cuda")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the fallback off CUDA")
def test_off_cuda_the_request_falls_back_silently():
    _stages()
    previous = be.get_backend()
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    try:
        result = _sphere().trace(num_rays=200, seed=1, max_depth=5, backend=TorchBackend(seed=1, interact_kernel="warp"))
        assert result.environment["interact_kernel"] == "torch"
        assert "cuda" in result.environment["interact_kernel_note"]
    finally:
        be.set_backend(previous)


def test_each_kernel_is_a_module_of_its_own_with_the_stage_options():
    _, si = _stages()
    for module in si.kernel_modules():
        assert module.options["fuse_fp"] is False
        assert module.options["enable_backward"] is False
        assert module.options["cuda_output"] == "cubin"
        assert len(module.kernels) == 1
