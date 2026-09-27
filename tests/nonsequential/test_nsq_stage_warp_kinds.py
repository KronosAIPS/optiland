"""The Warp intersection stage on every analytic kind, and the fused nearest-hit select.

``optiland.nonsequential.stage_warp`` computes ``BaseComponent.intersect`` for
the plane, the finite plane (rectangular and circular), the annulus, the
sphere (with and without an aperture), the conic and the paraboloid, the
frustum (a lens edge and a straight tube), the ported cavity, the lenslet
array and the even and odd aspheres, in one kernel per component, and folds
each component's hit into the running nearest hit of ``intersect_scene`` in
the same launch. It is an implementation change of one stage, so every check
is an equality of bit patterns:

* each kind in five placements (identity, translated, tilted, tilted and
  translated about all three axes, and far from the origin), at float64 and
  float32, on a fan of rays that hits, misses and grazes it: ``t``, both
  normals, the hit mask, the two halves of the hit distance, and for the
  aspheres the per-ray status and step count;
* the fused ``intersect_scene`` on a scene holding every kind at once,
  against the torch backend's own ``intersect_scene``;
* routing: a mesh keeps its own intersect, and a placement or a parameter that
  carries a gradient sends the component to its own intersect with the reason
  counted, so the derivative is torch's.

The kernels run on CUDA in the engine and on the CPU here, which is the only
difference from a CUDA run; where CUDA is present every test runs there too.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    AnnularPlaneGeometry,
    ConicGeometry,
    CylindricalFrustumGeometry,
    EvenAsphereGeometry,
    FinitePlaneGeometry,
    LambertianBSDF,
    LensletArrayGeometry,
    OddAsphereGeometry,
    ParaboloidGeometry,
    PlaneGeometry,
    ReflectiveComponent,
    SphereGeometry,
    SphericalCavityGeometry,
    SphericalPort,
)
from optiland.nonsequential.backends.array_backend import ArrayBackend
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.ray_bundle import NSQRayBundle


def _stage():
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


@pytest.fixture
def torch_backend_state():
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


def _use(device: str, precision: str) -> None:
    if device == "cuda" and not _cuda_with_warp():
        pytest.skip("CUDA with Warp not available")
    be.set_device(device)
    be.set_precision(precision)


# ---------------------------------------------------------------------------
# The kinds, their footprints, and the ray fans
# ---------------------------------------------------------------------------


def _offsets(num_y, num_x, seed=5):
    return np.random.default_rng(seed).uniform(-0.2, 0.2, (num_y, num_x))


#: kind label -> (geometry factory, expected kernel kind, half-extent of the
#: footprint in the local x-y plane [mm], local z range of the surface [mm])
_GEOMETRIES = {
    "plane": (lambda: PlaneGeometry(), "plane", 12.0, (0.0, 0.0)),
    "finite-plane-rect": (lambda: FinitePlaneGeometry(width=20.0, height=12.0), "finite_plane", 12.0, (0.0, 0.0)),
    "finite-plane-circ": (lambda: FinitePlaneGeometry(aperture_radius=9.0), "finite_plane", 11.0, (0.0, 0.0)),
    "annulus": (lambda: AnnularPlaneGeometry(3.0, 11.0, z_offset=0.5), "annulus", 12.0, (0.5, 0.5)),
    "sphere": (lambda: SphereGeometry(15.0), "sphere", 16.0, (-15.0, 15.0)),
    "sphere-aperture": (lambda: SphereGeometry(15.0, aperture_radius=10.0), "sphere", 12.0, (-15.0, 15.0)),
    "conic": (lambda: ConicGeometry(radius=-25.0, conic=-2.3, aperture_radius=10.0), "conic", 11.0, (-3.0, 0.0)),
    "paraboloid": (lambda: ParaboloidGeometry(radius=40.0, aperture_radius=12.0), "conic", 13.0, (0.0, 2.0)),
    "frustum": (lambda: CylindricalFrustumGeometry(10.0, 12.0, -3.0, 4.0), "frustum", 13.0, (-3.0, 4.0)),
    "tube": (lambda: CylindricalFrustumGeometry(8.0, 8.0, -5.0, 5.0), "frustum", 9.0, (-5.0, 5.0)),
    "cavity": (
        lambda: SphericalCavityGeometry(
            20.0,
            [SphericalPort.from_area_fraction((0.0, 0.0, -1.0), 0.02),
             SphericalPort.from_area_fraction((1.0, 1.0, 0.0), 0.01)],
        ),
        "cavity", 21.0, (-20.0, 20.0),
    ),
    "lenslet": (
        lambda: LensletArrayGeometry(2.0, 1.5, 3.0, -0.5, 5, 4, sag_offsets=_offsets(4, 5)),
        "lenslet", 6.0, (-0.3, 0.6),
    ),
    "asphere-even": (
        lambda: EvenAsphereGeometry(30.0, -1.2, 10.0, (1e-4, -2e-7)),
        "asphere", 11.0, (0.0, 2.0),
    ),
    "asphere-odd": (
        lambda: OddAsphereGeometry(25.0, 0.0, 8.0, (1e-3, 1e-4, -1e-6)),
        "asphere", 9.0, (0.0, 2.0),
    ),
}

#: Five placements: identity, translated, tilted, tilted and translated
#: about all three axes, and far from the origin (a large coordinate).
_PLACEMENTS = {
    "identity": {},
    "translated": {"x": 3.0, "y": -7.0, "z": 11.0},
    "tilted": {"rx": 0.3},
    "rotated": {"x": 1.5, "y": 2.0, "z": -4.0, "rx": 0.3, "ry": -0.7, "rz": 1.1},
    "far": {"x": 2.0e3, "y": -1.5e3, "z": 5.0e2, "ry": 0.2},
}

_N = 6000


def _fan(label: str, cs: CoordinateSystem, seed: int = 3):
    """Rays in the component's local frame aimed at its footprint (some past it), placed by ``cs``."""
    _, _, half, (z_lo, z_hi) = _GEOMETRIES[label]
    g = np.random.default_rng(seed)
    n = _N
    side = np.where(g.uniform(size=n) < 0.7, -1.0, 1.0)
    start = np.stack(
        [g.uniform(-1.5 * half, 1.5 * half, n), g.uniform(-1.5 * half, 1.5 * half, n),
         np.where(side < 0, z_lo - g.uniform(2.0, 30.0, n), z_hi + g.uniform(2.0, 30.0, n))],
        axis=1,
    )
    # Some rays start inside the surface's slab (a cavity's interior, a tube's bore).
    inside = g.uniform(size=n) < 0.2
    start[inside] = np.stack(
        [g.uniform(-0.5 * half, 0.5 * half, inside.sum()), g.uniform(-0.5 * half, 0.5 * half, inside.sum()),
         g.uniform(z_lo, z_hi + 1e-9, inside.sum())],
        axis=1,
    )
    target = np.stack(
        [g.uniform(-1.2 * half, 1.2 * half, n), g.uniform(-1.2 * half, 1.2 * half, n),
         g.uniform(z_lo, z_hi + 1e-9, n)],
        axis=1,
    )
    d = target - start
    # A few isotropic directions, and a few exactly along an axis.
    iso = g.uniform(size=n) < 0.1
    d[iso] = g.normal(size=(iso.sum(), 3))
    axial = g.uniform(size=n) < 0.03
    d[axial] = [0.0, 0.0, 1.0]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t, R = _get_transform(cs)
    p_g = start @ R.T + t
    d_g = d @ R.T
    alive = g.uniform(size=n) > 0.05
    return p_g, d_g, alive


def _bundle(p, d, alive) -> NSQRayBundle:
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


def _component(label: str, cs: CoordinateSystem) -> ReflectiveComponent:
    comp = ReflectiveComponent(cs, _GEOMETRIES[label][0](), reflectance=0.9,
                               bsdf=LambertianBSDF(reflectance_value=1.0), name=label)
    comp.refresh_backend_transform()
    return comp


def _bits(t) -> np.ndarray:
    a = t.detach().cpu()
    if a.dtype == torch.bool:
        return a.numpy()
    if a.dtype in (torch.int32, torch.int64):
        return a.numpy()
    return a.view(torch.int64 if a.dtype == torch.float64 else torch.int32).numpy()


def _assert_bits(a, b, what):
    assert a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape), what
    np.testing.assert_array_equal(_bits(a), _bits(b), err_msg=what)


# ---------------------------------------------------------------------------
# Each kind against its own intersect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("placement", sorted(_PLACEMENTS))
@pytest.mark.parametrize("label", sorted(_GEOMETRIES))
def test_every_kind_is_its_own_intersect_bit_for_bit(torch_backend_state, device, precision, placement, label):
    stage = _stage()
    _use(device, precision)
    cs = CoordinateSystem(**_PLACEMENTS[placement])
    comp = _component(label, cs)
    kind = stage.kind_of(comp)
    assert kind == _GEOMETRIES[label][1]
    rays = _bundle(*_fan(label, cs))

    ref = comp.intersect(rays)
    ref_root = comp._local_root
    ref_side = (comp.geometry.last_status, comp.geometry.last_steps) if kind == "asphere" else ()
    stage.reset_routed()
    got = stage.intersect_component(comp, kind, rays)
    got_root = comp._local_root
    got_side = (comp.geometry.last_status, comp.geometry.last_steps) if kind == "asphere" else ()
    assert stage.routed_counts() == {}, stage.routed_counts()
    assert stage.launch_counts() == {kind: 1}
    hits = int(ref[2].sum())
    assert 0.05 * _N < hits < _N, f"the fan hits {hits} of {_N}: a poor test of {label}"
    names = ("t", "normals", "hit_mask", "n_geom", "t_adv", "t_local", "last_status", "last_steps")
    for name, a, b in zip(names, (*ref, *ref_root, *ref_side), (*got, *got_root, *got_side)):
        _assert_bits(a, b, f"{label} {placement} {precision} {device}: {name}")


# ---------------------------------------------------------------------------
# The fused select on a scene of every kind
# ---------------------------------------------------------------------------


def _all_kinds_scene():
    comps = []
    for k, label in enumerate(sorted(_GEOMETRIES)):
        if label == "plane":
            continue  # an infinite plane would take every ray; the finite ones stand in
        angle = 2.0 * math.pi * k / len(_GEOMETRIES)
        cs = CoordinateSystem(x=30.0 * math.cos(angle), y=30.0 * math.sin(angle), z=5.0 * (k % 3),
                              rx=0.1 * k, ry=-0.05 * k)
        comps.append(_component(label, cs))
    return comps


class _TorchStage(ArrayBackend):
    """The array backend's own ``intersect_scene`` (the torch stage) with no trace around it."""

    def __init__(self):  # noqa: D107 - only the method is used
        pass


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_the_fused_select_is_intersect_scene_bit_for_bit(torch_backend_state, device, precision):
    stage = _stage()
    _use(device, precision)
    comps = _all_kinds_scene()
    g = np.random.default_rng(17)
    n = 20_000
    p = np.stack([g.uniform(-60, 60, n), g.uniform(-60, 60, n), g.uniform(-40, 40, n)], axis=1)
    d = g.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    alive = g.uniform(size=n) > 0.05
    rays = _bundle(p, d, alive)

    ref = ArrayBackend.intersect_scene(_TorchStage(), rays, comps)
    ref_roots = [c._local_root for c in comps]
    stage.reset_routed()
    got = stage.intersect_scene(rays, comps)
    got_roots = [c._local_root for c in comps]
    assert stage.routed_counts() == {}
    assert sum(stage.launch_counts().values()) == len(comps)
    assert int((ref[2] >= 0).sum()) > n // 10
    for name, a, b in zip(("t_min", "hit_normals", "component_indices", "hit_n_geom"), ref, got, strict=True):
        _assert_bits(a, b, name)
    for comp, (a0, a1), (b0, b1) in zip(comps, ref_roots, got_roots, strict=True):
        _assert_bits(a0, b0, f"{comp.name} t_adv")
        _assert_bits(a1, b1, f"{comp.name} t_local")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class _OwnPlane(FinitePlaneGeometry):
    """A geometry that overrides ``ray_intersect`` (as a user's own kind would): not covered."""

    def ray_intersect(self, origins, directions, eps=None):
        return super().ray_intersect(origins, directions, eps)


def test_an_uncovered_kind_keeps_its_own_intersect_and_is_merged(torch_backend_state):
    stage = _stage()
    own = ReflectiveComponent(CoordinateSystem(z=3.0), _OwnPlane(width=8.0, height=8.0), reflectance=0.5)
    own.refresh_backend_transform()
    comps = [_component("finite-plane-circ", CoordinateSystem()), own]
    assert stage.kind_of(own) is None
    g = np.random.default_rng(2)
    n = 4000
    p = np.stack([g.uniform(-6, 6, n), g.uniform(-6, 6, n), np.full(n, 10.0)], axis=1)
    d = np.tile([0.0, 0.0, -1.0], (n, 1))
    rays = _bundle(p, d, np.ones(n, dtype=bool))
    ref = ArrayBackend.intersect_scene(_TorchStage(), rays, comps)
    stage.reset_routed()
    got = stage.intersect_scene(rays, comps)
    assert stage.routed_counts() == {stage.ROUTE_KIND: 1}
    assert stage.launch_counts() == {"finite_plane": 1}
    assert int((ref[2] == 1).sum()) > 100
    for name, a, b in zip(("t_min", "hit_normals", "component_indices", "hit_n_geom"), ref, got, strict=True):
        _assert_bits(a, b, name)


def test_an_attached_placement_is_routed_to_torch_with_its_gradient(torch_backend_state):
    """A placement carrying a gradient (issue 31): the kernels take the placement as detached numbers,
    so the component runs its own intersect, the reason is counted, and the gradient is torch's."""
    stage = _stage()
    tilt = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)

    def run(route):
        cs = CoordinateSystem(x=1.0, z=2.0, rx=tilt)
        comp = _component("conic", cs)
        p, d, alive = _fan("conic", CoordinateSystem(x=1.0, z=2.0, rx=0.2))
        rays = _bundle(p, d, alive)
        if route == "torch":
            out = ArrayBackend.intersect_scene(_TorchStage(), rays, [comp])
        else:
            out = stage.intersect_scene(rays, [comp])
        t = out[0]
        loss = torch.where(torch.isfinite(t), t, torch.zeros_like(t)).sum() + out[1].sum()
        return loss, torch.autograd.grad(loss, tilt)[0]

    loss_t, grad_t = run("torch")
    stage.reset_routed()
    loss_w, grad_w = run("warp")
    assert stage.routed_counts() == {stage.ROUTE_PLACEMENT: 1}
    assert stage.launch_counts() == {}
    assert torch.equal(loss_t, loss_w)
    assert torch.equal(grad_t, grad_w)
    assert float(grad_t) != 0.0


@pytest.mark.parametrize("label", ["sphere", "frustum", "lenslet", "asphere-even"])
def test_a_gradient_through_a_kind_without_the_tape_is_routed(torch_backend_state, label):
    stage = _stage()
    cs = CoordinateSystem(ry=0.1)
    comp = _component(label, cs)
    p, d, alive = _fan(label, cs)
    rays = _bundle(p, d, alive)
    rays.x = rays.x.clone().requires_grad_(True)
    stage.reset_routed()
    got = stage.intersect_component(comp, stage.kind_of(comp), rays)
    assert stage.routed_counts() == {stage.ROUTE_GRADIENT: 1}
    ref = comp.intersect(rays)
    for a, b in zip(ref, got, strict=True):
        _assert_bits(a, b, label)


def test_forward_mode_is_routed_to_torch(torch_backend_state):
    stage = _stage()
    from torch.autograd import forward_ad

    cs = CoordinateSystem(z=1.0)
    comp = _component("sphere", cs)
    p, d, alive = _fan("sphere", cs)
    rays = _bundle(p, d, alive)
    with forward_ad.dual_level():
        rays.z = forward_ad.make_dual(rays.z, torch.ones_like(rays.z))
        stage.reset_routed()
        got = stage.intersect_scene(rays, [comp])
        tangent = forward_ad.unpack_dual(got[0]).tangent
    assert tangent is not None
    assert stage.launch_counts() == {}
