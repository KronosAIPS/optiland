"""X1 -- where the self-intersection threshold's size comes from.

The threshold has to sit above the residual a ray's own stored position
leaves when it is sitting on a surface, and below the gap to the nearest
other surface. Wave 1 could only satisfy the first: advancing a ray as
``p + t*d`` writes the hit distance as one number of the size of the whole
leg, so the ray landed ``u*|t|`` off the surface -- 5.8e-11 mm in float64
after a 1e6 mm leg -- and the threshold had to be 16384 ulps to reject it.
At float32 and a 50 mm coordinate that is 0.06 mm, which skips any surface
nearer than that to the one just left.

These tests pin both ends: the residual (the hit point now lands within an
ulp of the surface, at any leg length and in either dtype) and the gap (a
0.05 mm plate and a cemented interface are both seen in float32).

Scene parameters follow the quickstart singlet used throughout
``test_nsq_precision.py``.

Kramer Harrison, 2026
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# Imports below intentionally follow importorskip.
# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import (
    CollimatedSourceConfig,
    DoubletConfig,
    LensConfig,
    NSQScene,
    RayDatabaseConfig,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential import _tol as tol
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.materials.nsq_material import VACUUM, NSQMaterial
from optiland.nonsequential.ray_bundle import NSQRayBundle

PLATE_THICKNESS = 0.05  # mm


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(x) -> np.ndarray:
    """Whatever the backend holds, as plain float64 NumPy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _conic_residual(comp, rays) -> np.ndarray:
    """Signed distance along each ray from its position to ``comp``'s surface.

    Evaluated in float64 from the ray's stored (working-dtype) position, so
    it measures how far that position actually is from the surface, not how
    well the engine's own arithmetic agreed with itself. First order in the
    residual, which is all that is meaningful at 1e-15 mm.
    """
    translation, rot = _get_transform(comp.cs)
    p = np.stack([_np(rays.x), _np(rays.y), _np(rays.z)], axis=1)
    d = np.stack([_np(rays.L), _np(rays.M), _np(rays.N)], axis=1)
    p_l = (p - translation) @ rot
    d_l = d @ rot

    radius = float(np.asarray(_np(comp.geometry.radius)))
    c = 0.0 if (radius == 0.0 or not np.isfinite(radius)) else 1.0 / radius
    kp = 1.0 + float(np.asarray(_np(comp.geometry.conic)))

    x, y, z = p_l[:, 0], p_l[:, 1], p_l[:, 2]
    # The quadric ConicGeometry solves: c*(x^2 + y^2 + kp*z^2) - 2z = 0.
    f = c * (x**2 + y**2 + kp * z**2) - 2.0 * z
    grad = np.stack([2 * c * x, 2 * c * y, 2 * c * kp * z - 2.0], axis=1)
    return -f / (grad * d_l).sum(axis=1)


def _one_surface_scene(source_z: float):
    """A single conic surface at z=50, reached from ``source_z``."""
    comp = RefractiveComponent(
        cs=CoordinateSystem(z=50.0),
        geometry=ConicGeometry(radius=100.0, conic=0.0, aperture_radius=12.5),
        material_front=VACUUM,
        material_back=NSQMaterial.from_glass("N-BK7"),
        name="S",
    )
    n = 2000
    rng = np.random.default_rng(42)
    r = 5.0 * np.sqrt(rng.random(n))
    phi = 2 * np.pi * rng.random(n)
    rays = NSQRayBundle(
        x=be.array(r * np.cos(phi)),
        y=be.array(r * np.sin(phi)),
        z=be.array(np.full(n, source_z)),
        L=be.array(np.zeros(n)),
        M=be.array(np.zeros(n)),
        N=be.array(np.ones(n)),
        flux=be.array(np.ones(n)),
        wavelength=be.array(np.full(n, 0.55)),
        n_current=be.array(np.ones(n)),
        bounce=np.zeros(n, dtype=np.int32),
        alive=be.array(np.ones(n, dtype=bool)),
        ray_id=np.arange(n, dtype=np.int64),
    )
    return comp, rays


def _residual_in_ulps(source_z: float) -> float:
    """Max |residual| of the advanced hit point, in ulps of its coordinate."""
    comp, rays = _one_surface_scene(source_z)
    t, _normals, hit_mask, _n_geom = comp.intersect(rays)
    comp.advance_to_hit(rays, t, hit_mask)
    hit = _np(hit_mask).astype(bool)
    residual = np.abs(_conic_residual(comp, rays))[hit]
    coord = np.max(
        np.abs(np.stack([_np(rays.x), _np(rays.y), _np(rays.z)], axis=1)), axis=1
    )[hit]
    # The residual is measured in ulps of the *working* dtype: that is the
    # step the ray's own position was written at.
    working = np.float32 if _is_float32(rays.x) else np.float64
    step = np.spacing(coord.astype(working)).astype(np.float64)
    return float(np.max(residual / step))


def _is_float32(array) -> bool:
    if isinstance(array, torch.Tensor):
        return array.dtype is torch.float32
    return array.dtype == np.float32


def _plate_scene(thickness: float = PLATE_THICKNESS):
    """Two plane interfaces ``thickness`` apart, 50 mm from the origin."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(z=0.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=2.0
        ),
    )
    scene.add_lens(
        "P1",
        CoordinateSystem(z=50.0),
        LensConfig(
            r1=0.0,
            r2=0.0,
            thickness=thickness,
            material="N-BK7",
            front_aperture_radius=5.0,
        ),
    )
    scene.add_detector(
        "D1", CoordinateSystem(z=60.0), RayDatabaseConfig(width=20, height=20)
    )
    return scene


def _doublet_scene():
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(z=0.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0
        ),
    )
    scene.add_doublet(
        "DB",
        CoordinateSystem(z=50.0),
        DoubletConfig(
            r1=60.0,
            r2=-40.0,
            r3=-150.0,
            thickness1=6.0,
            thickness2=3.0,
            material1="N-BK7",
            material2="SF5",
            aperture_radius=12.5,
        ),
    )
    scene.add_detector(
        "D1", CoordinateSystem(z=160.0), RayDatabaseConfig(width=40, height=40)
    )
    return scene


def _hits_per_surface(result) -> Counter:
    events = result.ray_paths["events"]
    hits = events[events["event_type"] == "hit"]
    return Counter(hits["component_name"].tolist())


def _use_precision(precision: str) -> None:
    if precision == "float64-numpy":
        be.set_backend("numpy")
        return
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision(precision)


# ---------------------------------------------------------------------------
# 1. the residual: where the threshold's lower bound comes from
# ---------------------------------------------------------------------------


class TestHitPointResidual:
    """A ray must land on the surface it just hit, whatever the leg length.

    The residual is what the accept threshold has to clear, so this is the
    measurement that sets ``_tol.DEFAULT_ACCEPT_K``. Wave 1's ``p + t*d``
    update left 5.8e-11 mm (about 8000 ulps of the ray's coordinate) after
    a 1e6 mm leg, which is why k had to be 16384.
    """

    @pytest.mark.parametrize("source_z", [0.0, -1e3, -1e5, -1e6])
    def test_within_one_ulp_float64(self, source_z):
        be.set_backend("numpy")
        assert _residual_in_ulps(source_z) < 1.0

    def test_independent_of_leg_length_float64(self):
        """The whole point: a 1e6 mm leg is no worse than a 50 mm one."""
        be.set_backend("numpy")
        near = _residual_in_ulps(0.0)
        far = _residual_in_ulps(-1e6)
        assert far == pytest.approx(near, abs=0.25)

    @pytest.mark.parametrize("source_z", [0.0, -1e3])
    def test_within_one_ulp_float32(self, source_z):
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float32")
        assert _residual_in_ulps(source_z) < 1.0

    def test_plain_global_update_is_the_one_that_drifts(self):
        """Control: ``p + t*d`` at the same geometry, so the bound above is
        known to be measuring the fix and not a slack assertion."""
        be.set_backend("numpy")
        comp, rays = _one_surface_scene(-1e6)
        t, _normals, hit_mask, _n_geom = comp.intersect(rays)
        t_safe = np.where(_np(hit_mask).astype(bool), _np(t), 0.0)
        rays.x = _np(rays.x) + t_safe * _np(rays.L)
        rays.y = _np(rays.y) + t_safe * _np(rays.M)
        rays.z = _np(rays.z) + t_safe * _np(rays.N)
        hit = _np(hit_mask).astype(bool)
        residual = np.abs(_conic_residual(comp, rays))[hit]
        coord = np.abs(_np(rays.z))[hit]
        assert np.max(residual / np.spacing(coord)) > 1000.0


# ---------------------------------------------------------------------------
# 2. thin features: where the threshold's upper bound comes from
# ---------------------------------------------------------------------------


class TestThinFeatures:
    """A surface close behind the one just left must still be found."""

    def test_default_threshold_is_far_below_a_thin_gap(self):
        """float32 at a 50 mm coordinate, the worst case in these scenes."""
        threshold = float(tol.accept_t_min(torch.tensor(50.0, dtype=torch.float32)))
        assert threshold < PLATE_THICKNESS / 100.0

    @pytest.mark.parametrize("precision", ["float64-numpy", "float32"])
    def test_thin_plate_registers_both_faces(self, precision):
        _use_precision(precision)
        result = _plate_scene().trace(
            num_rays=20_000, seed=42, max_depth=16, record_paths=True
        )
        unreached = set(result.diagnostics.unreached_geometry)
        assert "P1.front" not in unreached
        assert "P1.back" not in unreached, (
            f"{precision}: the plate's back face, {PLATE_THICKNESS} mm behind "
            "its front face, was never hit -- the self-intersection threshold "
            "is wider than the plate"
        )
        hits = _hits_per_surface(result)
        assert hits["P1.back"] > 0.8 * hits["P1.front"]

    def test_thin_plate_flux_agrees_between_precisions(self):
        """Skipping the back face loses its Fresnel reflection, which shows
        up directly as too much detected flux."""
        fluxes = {}
        for precision in ("float64-numpy", "float32"):
            _use_precision(precision)
            result = _plate_scene().trace(num_rays=20_000, seed=42, max_depth=16)
            fluxes[precision] = float(result.total_flux_detected)
        assert fluxes["float32"] == pytest.approx(fluxes["float64-numpy"], abs=1e-6)

    @pytest.mark.parametrize("precision", ["float64-numpy", "float32"])
    def test_cemented_doublet_internal_interface_is_hit(self, precision):
        _use_precision(precision)
        result = _doublet_scene().trace(
            num_rays=20_000, seed=42, max_depth=16, record_paths=True
        )
        assert "DB.cemented" not in set(result.diagnostics.unreached_geometry)
        hits = _hits_per_surface(result)
        assert hits["DB.cemented"] > 0.8 * hits["DB.front"]


# ---------------------------------------------------------------------------
# 3. the multiplier and the per-ray threshold
# ---------------------------------------------------------------------------


class TestAcceptMultiplier:
    def test_default_inside_the_documented_range(self):
        """docs/theory/07_geometry.md R-07-5: k configurable in [8, 64]."""
        assert 8 <= tol.DEFAULT_ACCEPT_K <= 64

    def test_threshold_is_per_ray(self):
        """One distant ray must not coarsen the threshold for the rest."""
        from optiland.nonsequential.components.base import coordinate_magnitude

        be.set_backend("numpy")
        n = 4
        rays = NSQRayBundle(
            x=np.array([0.0, 0.0, 0.0, 0.0]),
            y=np.zeros(n),
            z=np.array([1.0, 50.0, 1e4, 1e6]),
            L=np.zeros(n),
            M=np.zeros(n),
            N=np.ones(n),
            flux=np.ones(n),
            wavelength=np.full(n, 0.55),
            n_current=np.ones(n),
            bounce=np.zeros(n, dtype=np.int32),
            alive=np.ones(n, dtype=bool),
            ray_id=np.arange(n, dtype=np.int64),
        )
        magnitude = np.asarray(coordinate_magnitude(rays))
        np.testing.assert_array_equal(magnitude, [1.0, 50.0, 1e4, 1e6])
        threshold = np.asarray(tol.accept_t_min(magnitude))
        assert threshold.shape == (n,)
        assert threshold[0] < threshold[1] < threshold[2] < threshold[3]


# ---------------------------------------------------------------------------
# 4. the fallback
# ---------------------------------------------------------------------------


class TestAdvanceFallback:
    """``advance_to_hit`` reads the two parts of the distance from this
    component's own last ``intersect``. A bundle it did not intersect (a
    bounded-splitting snapshot, a direct call in a test) must still be
    advanced correctly, by the plain global update.
    """

    def test_no_intersect_call_still_advances(self):
        be.set_backend("numpy")
        comp, rays = _one_surface_scene(0.0)
        assert comp._local_root is None
        t = np.full(rays.num_rays, 3.0)
        hit_mask = np.ones(rays.num_rays, dtype=bool)
        z_before = np.array(rays.z, dtype=np.float64)
        comp.advance_to_hit(rays, t, hit_mask)
        np.testing.assert_allclose(np.asarray(rays.z), z_before + 3.0)

    def test_stale_cache_of_the_same_size_is_not_used(self):
        be.set_backend("numpy")
        comp, rays = _one_surface_scene(0.0)
        t, _normals, hit_mask, _n_geom = comp.intersect(rays)
        # A second, unrelated bundle of the same size: the cached parts no
        # longer compose to the t being applied, so every ray falls back.
        _comp2, other = _one_surface_scene(-10.0)
        t_other = np.full(other.num_rays, 7.0)
        mask = np.ones(other.num_rays, dtype=bool)
        z_before = np.array(other.z, dtype=np.float64)
        comp.advance_to_hit(other, t_other, mask)
        np.testing.assert_allclose(np.asarray(other.z), z_before + 7.0)

    def test_masked_rays_are_untouched(self):
        be.set_backend("numpy")
        comp, rays = _one_surface_scene(0.0)
        t, _normals, hit_mask, _n_geom = comp.intersect(rays)
        mask = np.asarray(hit_mask).copy()
        mask[::2] = False
        z_before = np.array(rays.z, dtype=np.float64)
        comp.advance_to_hit(rays, t, mask)
        np.testing.assert_array_equal(
            np.asarray(rays.z)[~mask], z_before[~mask]
        )
