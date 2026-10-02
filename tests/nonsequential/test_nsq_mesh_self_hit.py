"""The trimesh-backed mesh kind finds the hit beyond a ray's own face (KronosNSRT issue 29).

trimesh's ray query keeps hits down to 1e-6 behind the origin and, asked for one hit per
ray, returns the nearest of those. A ray leaving a mesh face (its origin on the face, or
offset from it by the loop) therefore got its own face back as the first hit, the kind's
accept threshold refused it, and the genuine hit on another face of the same mesh was never
returned: the ray left the mesh unseen. Measured before the fix on a closed box: 0 of 2,000
rays leaving a face toward the inside found the opposite walls. Traced, a second loss: the component advances a ray's
origin along it to its closest approach to the local origin before the query, past a box's
entry face, and trimesh dropped that face as behind the origin; every ray of a beam
through a glass box missed its entry face. The kind now queries trimesh from behind every
point of the mesh along the ray, asks for every hit, and keeps the nearest beyond its own
threshold (which may be one value per ray, as the loop passes it).

trimesh needs rtree for its broad phase (an axis-aligned box filter, conservative). Where
rtree is missing the broad phase is replaced here by every (ray, triangle) pair, which
returns a superset of the same candidates; trimesh's narrow phase, its forward filter and
its choice of the first hit run as shipped.
"""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh is not installed")

from optiland.coatings import SimpleCoating  # noqa: E402
from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.materials.ideal import IdealMaterial  # noqa: E402
from optiland.nonsequential import (  # noqa: E402
    VACUUM,
    CollimatedSourceConfig,
    NSQMaterial,
    NSQScene,
    RayDatabaseConfig,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential.components.geometry.mesh import MeshGeometry  # noqa: E402

HALF = 1.0


@pytest.fixture(autouse=True)
def _broad_phase(monkeypatch):
    try:
        import rtree  # noqa: F401, PLC0415
    except ImportError:
        import trimesh.ray.ray_triangle as rt  # noqa: PLC0415

        def every_pair(ray_origins, ray_directions, tree):
            n, m = len(ray_origins), int(tree)
            return np.tile(np.arange(m), n), np.repeat(np.arange(n), m)

        monkeypatch.setattr(rt, "ray_triangle_candidates", every_pair)
        monkeypatch.setattr(trimesh.base.Trimesh, "triangles_tree", property(lambda self: len(self.faces)))
    yield


def _box():
    return trimesh.creation.box(extents=(2 * HALF, 2 * HALF, 2 * HALF))


def _exit_distance(o, d):
    """Closed form: the distance from inside the box [-1, 1]^3 to its boundary along d."""
    with np.errstate(divide="ignore"):
        t = np.where(d > 0, (HALF - o) / d, np.where(d < 0, (-HALF - o) / d, np.inf))
    return t.min(axis=1)


def _leaving_rays(n=2000, seed=29):
    rng = np.random.default_rng(seed)
    yz = rng.uniform(-0.9, 0.9, (n, 2))
    d = np.column_stack([np.ones(n), rng.uniform(-0.3, 0.3, (n, 2))])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return yz, d


@pytest.mark.parametrize("offset", [0.0, 32 * 2.0**-53, 1e-9])
def test_a_ray_leaving_a_face_finds_the_far_wall(offset):
    """Rays from the face x = -1 (on it, or offset inward as the loop offsets a refracted ray)
    toward the inside: each meets the boundary at the closed-form exit distance, on the face
    the closed form names, with that face's outward normal as ``n_geom``."""
    g = MeshGeometry(_box())
    yz, d = _leaving_rays()
    o = np.column_stack([np.full(len(d), -HALF + offset), yz])
    t, normals, hit, n_geom = g.ray_intersect(o, d)
    assert hit.all()
    np.testing.assert_allclose(t, _exit_distance(o, d), rtol=0, atol=1e-12)
    p = o + t[:, None] * d
    axis = np.argmax(np.abs(p), axis=1)
    want = np.zeros_like(p)
    want[np.arange(len(p)), axis] = np.sign(p[np.arange(len(p)), axis])
    np.testing.assert_allclose(n_geom, want, atol=1e-12)
    # the facing normal points against the ray
    assert np.all((normals * d).sum(1) < 0)


def test_a_ray_from_outside_is_unchanged():
    g = MeshGeometry(_box())
    yz, d = _leaving_rays()
    o = np.column_stack([np.full(len(d), -5.0), yz])
    t, _, hit, _ = g.ray_intersect(o, d)
    # entering through x = -1 where the ray crosses it inside the face
    at = o[:, 1:] + ((-HALF - o[:, 0]) / d[:, 0])[:, None] * d[:, 1:]
    front = np.all(np.abs(at) < HALF - 1e-9, axis=1)
    assert hit[front].all()
    np.testing.assert_allclose(t[front], (-HALF - o[front, 0]) / d[front, 0], rtol=0, atol=1e-12)


def test_a_glass_box_mesh_passes_a_tilted_beam_parallel():
    """Traced: a glass box mesh (outward winding, so the glass is the front material) and a
    beam tilted 0.3 rad; a parallel slab returns every ray to the beam's direction. With the
    exit face lost every ray would leave in its refracted direction."""
    glass = NSQMaterial(optiland_material=IdealMaterial(n=1.5, k=0.0))
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(z=-1.5, rx=0.3),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=0.3),
    )
    scene.add_component(
        "B",
        RefractiveComponent(CoordinateSystem(), MeshGeometry(_box()), glass, VACUUM, name="B",
                            coating=SimpleCoating(transmittance=1.0, reflectance=0.0)),
    )
    scene.add_detector("D", CoordinateSystem(z=6.0), RayDatabaseConfig(width=40.0, height=40.0, absorb=True))
    res = scene.trace(num_rays=500, seed=29, max_depth=6)
    db = res.detectors["D"]
    L, M, N = (np.asarray(getattr(db, k), dtype=float) for k in ("L", "M", "N"))
    assert len(N) == 500
    beam = np.array([0.0, -np.sin(0.3), np.cos(0.3)])
    # the sine of each ray's angle to the beam (arccos near 1 resolves only sqrt(2 u))
    sin = np.linalg.norm(np.cross(np.column_stack([L, M, N]), beam), axis=1)
    assert sin.max() < 1e-12


# The torch backend (found by the Apple GPU audit of 2026-10-02): the kind handed
# torch tensors to numpy arithmetic and raised on every torch device. trimesh is host
# code, so a tensor is copied to the host, widened there, and the results are rounded
# to the rays' dtype on the host and moved back.

@pytest.mark.parametrize("offset", [0.0, 1e-9])
def test_on_torch_float64_the_kind_returns_the_numpy_bits(offset):
    torch = pytest.importorskip("torch")
    g = MeshGeometry(_box())
    yz, d = _leaving_rays()
    o = np.column_stack([np.full(len(d), -HALF + offset), yz])
    want = g.ray_intersect(o, d)
    got = g.ray_intersect(torch.as_tensor(o), torch.as_tensor(d))
    for w, x in zip(want, got):
        assert torch.is_tensor(x) and x.device.type == "cpu"
        assert np.array_equal(x.numpy(), w)
    assert got[0].dtype == torch.float64 and got[2].dtype == torch.bool


def test_on_torch_float32_the_kind_rounds_the_float64_answer_once():
    torch = pytest.importorskip("torch")
    g = MeshGeometry(_box())
    yz, d = _leaving_rays()
    o = np.column_stack([np.full(len(d), -HALF), yz]).astype(np.float32)
    d = d.astype(np.float32)
    want = g.ray_intersect(o.astype(np.float64), d.astype(np.float64))
    got = g.ray_intersect(torch.as_tensor(o), torch.as_tensor(d))
    assert got[0].dtype == torch.float32
    assert np.array_equal(got[0].numpy(), want[0].astype(np.float32))
    assert np.array_equal(got[2].numpy(), want[2])


def test_on_the_apple_gpu_the_kind_returns_the_cpus_bits():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("the Apple GPU (torch mps) is not reachable on this host")
    g = MeshGeometry(_box())
    yz, d = _leaving_rays()
    o = torch.as_tensor(np.column_stack([np.full(len(d), -HALF), yz]).astype(np.float32))
    d = torch.as_tensor(d.astype(np.float32))
    on_cpu = g.ray_intersect(o, d)
    on_mps = g.ray_intersect(o.to("mps"), d.to("mps"))
    for a, b in zip(on_cpu, on_mps):
        assert b.device.type == "mps" and torch.equal(a, b.cpu())
