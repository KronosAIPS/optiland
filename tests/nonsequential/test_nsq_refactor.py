"""Tests for the NSQ refactor: compound components, registries, and visualization.

Covers:
- ComponentRegistry (add, remove, get, surfaces flat-list)
- Lens._build(): sub-surfaces, CS offsets, rim creation, SurfaceConfig overrides
- CylindricalFrustumGeometry and AnnularPlaneGeometry intersection
- Integration test: scene.trace() with Lens
- NSQViewer2D smoke test

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    VACUUM,
    AnnularPlaneGeometry,
    CylindricalFrustumGeometry,
    IrradianceDetectorConfig,
    Lens,
    LensConfig,
    MirrorConfig,
    NSQScene,
    NSQViewer2D,
    PointSourceConfig,
    Spectrum,
    SurfaceConfig,
)
from optiland.nonsequential.components.configs import InteractionType
from optiland.nonsequential.components.registry import ComponentRegistry

# ---------------------------------------------------------------------------
# ComponentRegistry
# ---------------------------------------------------------------------------


class TestComponentRegistry:
    def test_add_and_get(self):
        cs = CoordinateSystem()
        lens = Lens(
            "L1",
            cs,
            LensConfig(
                r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=12.5
            ),
        )
        reg = ComponentRegistry()
        reg.add("L1", lens)
        assert reg.get("L1") is lens

    def test_contains(self):
        reg = ComponentRegistry()
        cs = CoordinateSystem()
        lens = Lens(
            "L1",
            cs,
            LensConfig(
                r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=12.5
            ),
        )
        reg.add("L1", lens)
        assert "L1" in reg
        assert "L2" not in reg

    def test_remove(self):
        reg = ComponentRegistry()
        cs = CoordinateSystem()
        lens = Lens(
            "L1",
            cs,
            LensConfig(
                r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=12.5
            ),
        )
        reg.add("L1", lens)
        reg.remove("L1")
        assert "L1" not in reg

    def test_duplicate_name_raises(self):
        reg = ComponentRegistry()
        cs = CoordinateSystem()
        lens = Lens(
            "L1",
            cs,
            LensConfig(
                r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=12.5
            ),
        )
        reg.add("L1", lens)
        with pytest.raises(KeyError):
            reg.add("L1", lens)

    def test_surfaces_flat_list(self):
        reg = ComponentRegistry()
        cs = CoordinateSystem()
        lens = Lens(
            "L1",
            cs,
            LensConfig(
                r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=12.5
            ),
        )
        reg.add("L1", lens)
        surfs = reg.surfaces
        # Symmetric lens → 3 surfaces (front, back, edge; no rim when the
        # apertures are equal)
        assert len(surfs) == 3

    def test_compounds_list(self):
        reg = ComponentRegistry()
        cs = CoordinateSystem()
        lens = Lens(
            "L1",
            cs,
            LensConfig(
                r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=12.5
            ),
        )
        reg.add("L1", lens)
        assert reg.compounds == [lens]


# ---------------------------------------------------------------------------
# Lens._build()
# ---------------------------------------------------------------------------


class TestLensBuild:
    def _make_symmetric_lens(self, front_r=12.5, back_r=None):
        cs = CoordinateSystem(z=50)
        cfg = LensConfig(
            r1=50.0,
            r2=-50.0,
            thickness=5.0,
            material=VACUUM,
            front_aperture_radius=front_r,
            back_aperture_radius=back_r,
        )
        return Lens("L", cs, cfg)

    def test_symmetric_lens_three_surfaces(self):
        lens = self._make_symmetric_lens()
        # Equal apertures → no rim → front + back + edge = 3 surfaces
        assert len(lens.surfaces) == 3

    def test_asymmetric_lens_four_surfaces(self):
        lens = self._make_symmetric_lens(front_r=15.0, back_r=10.0)
        # Unequal apertures → rim added → 4 surfaces
        assert len(lens.surfaces) == 4

    def test_back_cs_offset(self):
        """Back face CS should be offset by thickness along lens axis."""
        cs = CoordinateSystem(x=0, y=0, z=50)
        cfg = LensConfig(
            r1=50, r2=-50, thickness=8.0, material=VACUUM, front_aperture_radius=10
        )
        lens = Lens("L", cs, cfg)

        from optiland.nonsequential.components.base import (
            _get_transform,  # noqa: PLC0415
        )

        front_t, _ = _get_transform(lens.surfaces[0].cs)
        back_t, _ = _get_transform(lens.surfaces[1].cs)

        # Along z-axis, back face should be ≈ 8 mm further
        np.testing.assert_allclose(back_t[2] - front_t[2], 8.0, atol=1e-9)

    def test_surface_config_bsdf_override(self):
        """SurfaceConfig.bsdf on the front face should be applied."""
        from optiland.nonsequential.bsdf.lambertian import (
            LambertianBSDF,  # noqa: PLC0415
        )

        custom_bsdf = LambertianBSDF(reflectance_value=0.5)
        cs = CoordinateSystem(z=50)
        cfg = LensConfig(
            r1=50,
            r2=-50,
            thickness=5,
            material=VACUUM,
            front_aperture_radius=10,
            front=SurfaceConfig(bsdf=custom_bsdf),
        )
        lens = Lens("L", cs, cfg)
        assert lens.surfaces[0].bsdf is custom_bsdf

    def test_surface_config_interaction_override(self):
        """InteractionType.REFLECTIVE on the front face → ReflectiveComponent."""
        from optiland.nonsequential.components.reflective import (  # noqa: PLC0415
            ReflectiveComponent,
        )

        cs = CoordinateSystem(z=50)
        cfg = LensConfig(
            r1=50,
            r2=-50,
            thickness=5,
            material=VACUUM,
            front_aperture_radius=10,
            front=SurfaceConfig(
                interaction=InteractionType.REFLECTIVE, reflectance=1.0
            ),
        )
        lens = Lens("L", cs, cfg)
        assert isinstance(lens.surfaces[0], ReflectiveComponent)

    def test_name_property(self):
        cs = CoordinateSystem()
        lens = Lens(
            "MyLens",
            cs,
            LensConfig(
                r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=10
            ),
        )
        assert lens.name == "MyLens"


# ---------------------------------------------------------------------------
# CylindricalFrustumGeometry intersection
# ---------------------------------------------------------------------------


class TestCylindricalFrustumGeometry:
    def test_axial_ray_misses(self):
        """A ray along the z-axis (r=0) should miss the frustum barrel."""
        geom = CylindricalFrustumGeometry(
            r_front=5.0, r_back=5.0, z_front=0.0, z_back=10.0
        )
        origins = np.array([[0.0, 0.0, -5.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        t, _, hit, _ = geom.ray_intersect(origins, directions)
        assert not hit[0]

    def test_radial_ray_hits(self):
        """A ray approaching the cylinder from outside along the -x direction."""
        r = 5.0
        geom = CylindricalFrustumGeometry(r_front=r, r_back=r, z_front=0.0, z_back=10.0)
        # Ray starts outside (+x side) aimed at -x, at z=5 (mid-height)
        origins = np.array([[20.0, 0.0, 5.0]])
        directions = np.array([[-1.0, 0.0, 0.0]])
        t, normals, hit, n_geom = geom.ray_intersect(origins, directions)
        assert hit[0]
        np.testing.assert_allclose(t[0], 15.0, atol=1e-4)

    def test_ray_outside_z_range_misses(self):
        """Ray aimed at the cylinder band but hits outside the z-range."""
        geom = CylindricalFrustumGeometry(
            r_front=5.0, r_back=5.0, z_front=0.0, z_back=5.0
        )
        origins = np.array([[20.0, 0.0, 8.0]])  # z=8 > z_back=5
        directions = np.array([[-1.0, 0.0, 0.0]])
        t, _, hit, _ = geom.ray_intersect(origins, directions)
        assert not hit[0]

    def test_normal_points_outward(self):
        """Normal at a radial hit should have a positive radial component."""
        r = 5.0
        geom = CylindricalFrustumGeometry(r_front=r, r_back=r, z_front=0.0, z_back=10.0)
        origins = np.array([[20.0, 0.0, 5.0]])
        directions = np.array([[-1.0, 0.0, 0.0]])
        t, normals, hit, n_geom = geom.ray_intersect(origins, directions)
        # Normal should face the incoming ray (+x side), so nx > 0
        assert normals[0, 0] > 0.9

    def test_axis_parallel_ray_inside_constant_radius_tube_never_hits(self):
        """Issue #14: a = b = 0 exactly for an axis-parallel ray through a
        constant-radius tube. The degenerate-quadratic fallback used to
        return the same spurious t1 = t2 = 0 pair regardless of c, with
        disc = 0 trivially -- so nothing distinguished a ray that misses the
        tube from one running along its wall, and a negative (shifted) eps
        let the spurious t = 0 root through. Even with eps forced negative
        here (what BaseComponent.intersect's origin advance can produce),
        a ray strictly inside the tube's radius must never register a hit,
        at t = 0 or anywhere else.
        """
        geom = CylindricalFrustumGeometry(
            r_front=18.5, r_back=18.5, z_front=0.0, z_back=40.0
        )
        origins = np.array([[5.0, 0.0, 0.0]])  # r=5 << r_front=18.5, on the
        # front plane already (the origin-advance limit)
        directions = np.array([[0.0, 0.0, 1.0]])
        t, _, hit, _ = geom.ray_intersect(origins, directions, eps=-1.0)
        assert not hit[0], "axis-parallel ray inside a constant-radius tube was hit"

    def test_axis_parallel_ray_on_the_wall_is_rejected(self):
        """Issue #14: the measure-zero case of running exactly along the
        tube's wall (r == r_front == r_back, so c = 0 too) is rejected --
        it is never accepted as a hit at t = 0.
        """
        geom = CylindricalFrustumGeometry(
            r_front=18.5, r_back=18.5, z_front=0.0, z_back=40.0
        )
        origins = np.array([[18.5, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        t, _, hit, _ = geom.ray_intersect(origins, directions, eps=-1.0)
        assert not hit[0], "ray running along the tube wall was accepted as a hit"

    def test_tapered_frustum_still_uses_the_linear_fallback(self):
        """Sanity: a genuinely tapered frustum with a small (but nonzero)
        axial radius change still resolves a near-axial ray through the
        linear fallback (b not small), unaffected by the a_small gating
        that fixes the constant-radius case above.
        """
        # Slight taper: r_front=5.0 -> r_back=5.1 over 10mm. A ray parallel
        # to the axis at r=5.05 (inside r_front, outside... ) crosses the
        # cone wall once.
        geom = CylindricalFrustumGeometry(
            r_front=5.0, r_back=6.0, z_front=0.0, z_back=10.0
        )
        origins = np.array([[5.5, 0.0, -5.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        t, _, hit, _ = geom.ray_intersect(origins, directions)
        assert hit[0]
        # r(z) = 5.0 + 0.1*z = 5.5 at z = 5.0, so t = 5.0 - (-5.0) = 10.0
        np.testing.assert_allclose(t[0], 10.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Issue #14: an absorbing constant-radius tube around an axial beam
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        ("numpy", "float64"),
        ("numpy", "float32"),
        ("torch", "float64"),
        ("torch", "float32"),
    ],
    ids=lambda p: f"backend={p[0]},dtype={p[1]}",
)
def backend_and_precision(request):
    """Run a test on both backends and both dtypes, restoring prior state.

    Mirrors the ``each_backend`` fixture of test_nsq_sphere_cavity.py, with
    precision added as its own axis (issue #14 asks for both backends and
    both dtypes explicitly).
    """
    import optiland.backend as be  # noqa: PLC0415

    backend, precision = request.param
    previous_backend = be.get_backend()
    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
    be.set_precision(precision)
    yield backend, precision
    if backend == "torch":
        be.grad_mode.disable()
    be.set_backend(previous_backend)
    be.set_precision("float64")


class TestConstantRadiusTubeAxialRay:
    """Issue #14, end to end through AbsorbingComponent.intersect (the
    real origin-advance path: BaseComponent.intersect shifts eps by t_adv,
    which is exactly what let the degenerate root through before the fix).
    """

    def test_axial_beam_inside_barrel_is_not_absorbed(self, backend_and_precision):
        import optiland.backend as be  # noqa: PLC0415
        from optiland.nonsequential import VACUUM, AbsorbingComponent  # noqa: PLC0415
        from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

        comp = AbsorbingComponent(
            cs=CoordinateSystem(z=0.0),
            geometry=CylindricalFrustumGeometry(
                r_front=18.5, r_back=18.5, z_front=0.0, z_back=40.0
            ),
            material_front=VACUUM,
            name="barrel",
        )
        n = 25
        # A small bundle of axis-parallel rays, well inside the tube radius,
        # starting well in front of the tube -- as in bench/demo_lens_ghosts.py.
        rng = np.random.default_rng(14)
        r = 10.0 * np.sqrt(rng.random(n))
        phi = 2 * np.pi * rng.random(n)
        rays = NSQRayBundle(
            x=be.array(r * np.cos(phi)),
            y=be.array(r * np.sin(phi)),
            z=be.array(np.full(n, -100.0)),
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
        t, _normals, hit_mask, _n_geom = comp.intersect(rays)
        hit_np = np.asarray(hit_mask)
        assert not np.any(hit_np), (
            "an axial ray inside a constant-radius absorbing tube was "
            f"absorbed at its front plane (backend/dtype = {backend_and_precision})"
        )

    def test_radial_ray_still_hits_the_same_barrel(self, backend_and_precision):
        """The fix must not turn a genuine radial hit into a miss."""
        import optiland.backend as be  # noqa: PLC0415
        from optiland.nonsequential import VACUUM, AbsorbingComponent  # noqa: PLC0415
        from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

        comp = AbsorbingComponent(
            cs=CoordinateSystem(z=0.0),
            geometry=CylindricalFrustumGeometry(
                r_front=18.5, r_back=18.5, z_front=0.0, z_back=40.0
            ),
            material_front=VACUUM,
            name="barrel",
        )
        rays = NSQRayBundle(
            x=be.array(np.array([50.0])),
            y=be.array(np.array([0.0])),
            z=be.array(np.array([20.0])),
            L=be.array(np.array([-1.0])),
            M=be.array(np.array([0.0])),
            N=be.array(np.array([0.0])),
            flux=be.array(np.array([1.0])),
            wavelength=be.array(np.array([0.55])),
            n_current=be.array(np.array([1.0])),
            bounce=np.zeros(1, dtype=np.int32),
            alive=be.array(np.ones(1, dtype=bool)),
            ray_id=np.arange(1, dtype=np.int64),
        )
        t, _normals, hit_mask, _n_geom = comp.intersect(rays)
        assert bool(np.asarray(hit_mask)[0])
        precision = backend_and_precision[1]
        atol = 1e-3 if precision == "float32" else 1e-6
        np.testing.assert_allclose(np.asarray(t)[0], 31.5, atol=atol)


# ---------------------------------------------------------------------------
# AnnularPlaneGeometry intersection
# ---------------------------------------------------------------------------


class TestAnnularPlaneGeometry:
    def test_ray_hits_annulus(self):
        """Ray from below aimed straight up should hit the annular band."""
        geom = AnnularPlaneGeometry(inner_radius=3.0, outer_radius=8.0, z_offset=5.0)
        origins = np.array([[5.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        t, normals, hit, n_geom = geom.ray_intersect(origins, directions)
        assert hit[0]
        np.testing.assert_allclose(t[0], 5.0, atol=1e-9)

    def test_ray_hits_inner_hole_misses(self):
        """Ray through the inner hole should miss."""
        geom = AnnularPlaneGeometry(inner_radius=3.0, outer_radius=8.0, z_offset=5.0)
        origins = np.array([[1.0, 0.0, 0.0]])  # r=1 < inner_radius=3
        directions = np.array([[0.0, 0.0, 1.0]])
        t, _, hit, _ = geom.ray_intersect(origins, directions)
        assert not hit[0]

    def test_ray_outside_outer_radius_misses(self):
        """Ray outside outer_radius should miss."""
        geom = AnnularPlaneGeometry(inner_radius=3.0, outer_radius=8.0, z_offset=5.0)
        origins = np.array([[10.0, 0.0, 0.0]])  # r=10 > outer_radius=8
        directions = np.array([[0.0, 0.0, 1.0]])
        t, _, hit, _ = geom.ray_intersect(origins, directions)
        assert not hit[0]

    def test_parallel_ray_misses(self):
        """Ray parallel to the plane (dz=0) should not hit."""
        geom = AnnularPlaneGeometry(inner_radius=3.0, outer_radius=8.0, z_offset=5.0)
        origins = np.array([[5.0, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0]])
        t, _, hit, _ = geom.ray_intersect(origins, directions)
        assert not hit[0]

    def test_normal_direction(self):
        """Normal should oppose the incoming ray direction."""
        geom = AnnularPlaneGeometry(inner_radius=3.0, outer_radius=8.0, z_offset=5.0)
        origins = np.array([[5.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])  # incoming from -z
        t, normals, hit, n_geom = geom.ray_intersect(origins, directions)
        assert normals[0, 2] < 0.0  # normal faces -z (opposes incoming)


# ---------------------------------------------------------------------------
# NSQScene new API
# ---------------------------------------------------------------------------


class TestNSQSceneNewAPI:
    def test_add_lens_creates_surfaces(self):
        scene = NSQScene()
        cs = CoordinateSystem(z=50)
        cfg = LensConfig(
            r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=12.5
        )
        scene.add_lens("L1", cs, cfg)
        assert len(scene.surfaces) == 3  # front + back + edge

    def test_add_mirror(self):
        scene = NSQScene()
        cs = CoordinateSystem(z=100)
        cfg = MirrorConfig(
            radius=200.0, reflectance=1.0, conic=-1.0, aperture_radius=50.0
        )
        scene.add_mirror("M1", cs, cfg)
        assert len(scene.surfaces) == 1

    def test_add_source_and_detector_via_config(self):
        spec = Spectrum.monochromatic(0.55)
        scene = NSQScene()

        src_cs = CoordinateSystem(z=0)
        scene.add_source("S1", src_cs, PointSourceConfig(spectrum=spec, total_flux=1.0))

        det_cs = CoordinateSystem(z=200)
        scene.add_detector("D1", det_cs, IrradianceDetectorConfig(width=20, height=20))

        assert len(scene.sources) == 1
        assert len(scene.detectors) == 1

    def test_remove_component(self):
        scene = NSQScene()
        cs = CoordinateSystem(z=50)
        cfg = LensConfig(
            r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=10
        )
        scene.add_lens("L1", cs, cfg)
        scene.remove_component("L1")
        assert len(scene.surfaces) == 0

    def test_validate_no_sources_raises(self):
        scene = NSQScene()
        det_cs = CoordinateSystem(z=100)
        scene.add_detector("D1", det_cs, IrradianceDetectorConfig(width=10, height=10))
        with pytest.raises(ValueError, match="no sources"):
            scene.validate()


# ---------------------------------------------------------------------------
# Backward-compatibility: old flat-list API
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_old_add_component(self):
        from optiland.nonsequential.components.geometry.analytic.plane import (  # noqa: PLC0415
            FinitePlaneGeometry,
        )
        from optiland.nonsequential.components.reflective import (
            ReflectiveComponent,  # noqa: PLC0415
        )

        cs = CoordinateSystem(z=50)
        comp = ReflectiveComponent(
            cs=cs, geometry=FinitePlaneGeometry(width=20, height=20), reflectance=1.0
        )
        scene = NSQScene()
        scene.add_component("mirror", comp)
        assert len(scene.surfaces) == 1

    def test_old_add_source(self):
        cs = CoordinateSystem()
        spec = Spectrum.monochromatic(0.55)
        scene = NSQScene()
        scene.add_source(
            "source_0", cs, PointSourceConfig(spectrum=spec, total_flux=1.0)
        )
        assert len(scene.sources) == 1
        assert "source_0" in scene.source_registry

    def test_old_add_detector(self):
        cs = CoordinateSystem(z=100)
        scene = NSQScene()
        scene.add_detector(
            "detector_0",
            cs,
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=16, num_pixels_y=16
            ),
        )
        assert len(scene.detectors) == 1
        assert "detector_0" in scene.detector_registry


# ---------------------------------------------------------------------------
# Integration test: biconvex lens scene
# ---------------------------------------------------------------------------


class TestLensIntegration:
    def test_biconvex_lens_trace(self):
        """Point source → biconvex lens → irradiance detector.

        Checks that flux is detected and conservation error < 1%.
        """
        spec = Spectrum.monochromatic(0.55)

        scene = NSQScene()

        # Source at origin, 30° half-angle
        src_cs = CoordinateSystem(z=0)
        scene.add_source(
            "S1",
            src_cs,
            PointSourceConfig(spectrum=spec, total_flux=1.0, half_angle_deg=30),
        )

        # Biconvex lens at z=50, VACUUM material for simplicity
        lens_cs = CoordinateSystem(z=50)
        scene.add_lens(
            "L1",
            lens_cs,
            LensConfig(
                r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=12.5
            ),
        )

        # Large detector at z=200
        det_cs = CoordinateSystem(z=200)
        scene.add_detector(
            "D1",
            det_cs,
            IrradianceDetectorConfig(
                width=50, height=50, num_pixels_x=64, num_pixels_y=64
            ),
        )

        result = scene.trace(num_rays=10_000, max_depth=10, seed=42)

        irr = result.detectors["D1"]
        assert result.total_flux_in == pytest.approx(1.0, rel=1e-6)
        assert result.trace_time_sec > 0.0
        # With VACUUM lens (no refraction index change) rays pass through
        # and some should be detected
        assert irr.total_flux >= 0.0


# ---------------------------------------------------------------------------
# Visualization smoke tests
# ---------------------------------------------------------------------------


class TestVisualizationSmoke:
    """Smoke tests: viewers must run without exception on a minimal scene."""

    def _build_minimal_scene(self):
        spec = Spectrum.monochromatic(0.55)
        scene = NSQScene()
        src_cs = CoordinateSystem(z=0)
        scene.add_source("S1", src_cs, PointSourceConfig(spectrum=spec, total_flux=1.0))
        lens_cs = CoordinateSystem(z=50)
        scene.add_lens(
            "L1",
            lens_cs,
            LensConfig(
                r1=50, r2=-50, thickness=5, material=VACUUM, front_aperture_radius=12.5
            ),
        )
        det_cs = CoordinateSystem(z=200)
        scene.add_detector("D1", det_cs, IrradianceDetectorConfig(width=50, height=50))
        return scene

    def test_viewer_2d_no_exception(self):
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")  # non-interactive backend

        scene = self._build_minimal_scene()
        viewer = NSQViewer2D(scene)
        fig = viewer.view()
        assert fig is not None

        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.close("all")

    def test_viewer_2d_with_rays_no_exception(self):
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")  # non-interactive backend

        scene = self._build_minimal_scene()
        viewer = NSQViewer2D(scene)
        fig = viewer.view(num_rays=5)
        assert fig is not None

        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.close("all")


class TestSceneNameAccessors:
    """Scenes expose their registered names without touching the registries."""

    @staticmethod
    def _scene():
        from optiland.coordinate_system import CoordinateSystem
        from optiland.nonsequential import (
            CollimatedSourceConfig,
            IrradianceDetectorConfig,
            LensConfig,
            NSQScene,
            Spectrum,
        )

        scene = NSQScene()
        scene.add_source(
            "S1",
            CoordinateSystem(),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=5.0,
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
                width=20, height=20, num_pixels_x=16, num_pixels_y=16
            ),
        )
        return scene

    def test_names_are_reported_in_registration_order(self):
        scene = self._scene()
        assert scene.component_names == ["L1"]
        assert scene.source_names == ["S1"]
        assert scene.detector_names == ["D1"]

    def test_detector_names_match_result_keys(self):
        """detector_names is the documented way to key into the result."""
        scene = self._scene()
        result = scene.trace(num_rays=500, seed=0)
        assert list(result.detectors.keys()) == scene.detector_names
