"""The closed spherical cavity, the one-sided detector, the hemisphere.

Three things are pinned down here.

**The cavity is watertight.** Away from its ports, every direction from
every interior point meets the wall exactly once, at a point on the sphere
to the working precision. A cavity that leaks anywhere -- at a pole, along a
seam, at grazing incidence -- loses flux that an integrating sphere's
multiplier counts, and loses it silently.

**A port is an opening, not an aperture.** A ray entering through one keeps
going and reflects off the far wall, which needs both roots of the
ray-sphere quadratic tested rather than the nearest one taken and then
rejected. A port's area is the spherical cap's, so its area fraction is
``(1 - cos alpha) / 2`` exactly -- the ``f`` of the sphere multiplier.

**A detector can be one-sided.** The default stays two-sided; ``front``
takes only rays arriving on the side the placement's normal points toward.

The integrating-sphere multiplier itself is checked against
``M = rho / (1 - rho (1 - f))`` at a stated ray count, with the window taken
from the run's own standard error rather than from a fixed number.

Kramer Harrison, 2026
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    HemisphereDetectorConfig,
    IrradianceDetectorConfig,
    LambertianBSDF,
    NSQScene,
    ReflectiveComponent,
    Spectrum,
    SphericalCavityGeometry,
    SphericalPort,
)
from optiland.nonsequential.detectors.hemisphere import HemisphereDetector
from optiland.nonsequential.ir.lower import lower
from optiland.nonsequential.ir.scene_ir import scene_ir_from_dict, scene_ir_to_dict

RADIUS = 50.0
GREEN = 0.5876


@pytest.fixture(params=be.list_available_backends(), ids=lambda b: f"backend={b}")
def each_backend(request):
    """Run a test on every installed backend, and put the state back.

    Not the shared ``set_test_backend`` fixture: that one turns gradient
    mode on for Torch and never turns it off, which leaves every later test
    in the session tracing gradients -- ``to_numpy`` on a tensor that
    requires grad raises, and the nearest suite alphabetically
    (``test_nsq_thresholds``) fails for a reason that has nothing to do with
    it. This fixture restores the backend, the precision and the gradient
    mode it found.
    """
    previous = be.get_backend()
    be.set_backend(request.param)
    if request.param == "torch":
        be.set_device("cpu")
        be.set_precision("float64")
    yield request.param
    if request.param == "torch":
        be.grad_mode.disable()
    be.set_backend(previous)
    be.set_precision("float64")


def _isotropic(n: int, seed: int = 0) -> np.ndarray:
    """``n`` unit vectors, uniform on the sphere."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Geometry: watertightness
# ---------------------------------------------------------------------------
class TestWatertight:
    def test_ray_from_the_centre_always_hits_the_wall_once(self):
        """No direction escapes a closed cavity, and every hit is at R."""
        geometry = SphericalCavityGeometry(RADIUS)
        directions = _isotropic(20_000)
        origins = np.zeros_like(directions)

        t, normals, hit, n_geom = geometry.ray_intersect(origins, directions)

        assert bool(np.all(hit))
        np.testing.assert_allclose(t, RADIUS, rtol=0, atol=1e-12)
        # The shading normal faces the ray: from inside, that is inward.
        np.testing.assert_allclose(
            (normals * directions).sum(axis=1), -1.0, rtol=0, atol=1e-12
        )
        # n_geom is direction-independent and points at the centre.
        hit_points = origins + t[:, None] * directions
        np.testing.assert_allclose(n_geom, -hit_points / RADIUS, atol=1e-12)

    def test_an_interior_point_off_centre_still_sees_the_wall_everywhere(self):
        """Watertight is a property of the surface, not of the centre."""
        geometry = SphericalCavityGeometry(RADIUS)
        directions = _isotropic(20_000, seed=1)
        origins = np.tile(np.array([10.0, -20.0, 30.0]), (directions.shape[0], 1))

        t, _normals, hit, _n_geom = geometry.ray_intersect(origins, directions)

        assert bool(np.all(hit))
        landing = origins + t[:, None] * directions
        radius_error = np.abs(np.linalg.norm(landing, axis=1) - RADIUS)
        # The hit point is on the sphere to the working precision of a 50 mm
        # coordinate, not to a fixed absolute length.
        assert radius_error.max() < 64 * np.spacing(RADIUS)

    def test_a_ray_that_misses_the_sphere_reports_no_hit(self):
        geometry = SphericalCavityGeometry(RADIUS)
        origins = np.array([[0.0, 0.0, -200.0], [0.0, 60.0, -200.0]])
        directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        t, _normals, hit, _n_geom = geometry.ray_intersect(origins, directions)

        assert bool(hit[0]) and not bool(hit[1])
        assert t[1] == np.inf


# ---------------------------------------------------------------------------
# Geometry: ports
# ---------------------------------------------------------------------------
class TestPorts:
    def test_area_fraction_round_trips_through_the_half_angle(self):
        for fraction in (0.001, 0.01, 0.05, 0.25, 0.5):
            port = SphericalPort.from_area_fraction((0.0, 0.0, 1.0), fraction)
            assert port.area_fraction == pytest.approx(fraction, rel=1e-12)
        # Half the sphere is a 90 degree port, exactly.
        assert SphericalPort.from_area_fraction(
            (0.0, 0.0, 1.0), 0.5
        ).half_angle_deg == pytest.approx(90.0, abs=1e-9)

    def test_a_ray_aimed_at_a_port_passes_through(self):
        geometry = SphericalCavityGeometry(
            RADIUS, [SphericalPort((0.0, 0.0, 1.0), 10.0)]
        )
        origins = np.zeros((3, 3))
        directions = np.array(
            [
                [0.0, 0.0, 1.0],  # the port's own axis
                [math.sin(math.radians(9.0)), 0.0, math.cos(math.radians(9.0))],
                [math.sin(math.radians(11.0)), 0.0, math.cos(math.radians(11.0))],
            ]
        )

        t, _normals, hit, _n_geom = geometry.ray_intersect(origins, directions)

        assert not bool(hit[0])  # straight out of the port
        assert not bool(hit[1])  # inside the port's angular radius
        assert bool(hit[2])  # just outside it: wall
        assert t[2] == pytest.approx(RADIUS)

    def test_entering_through_a_port_hits_the_far_wall(self):
        """The near root is in the port; the far root is the hit.

        This is the case an aperture check gets wrong: it takes the nearest
        positive root, finds it outside the live area, and reports no hit at
        all -- so the ray leaves the cavity instead of reflecting off the
        wall opposite the port.
        """
        geometry = SphericalCavityGeometry(
            RADIUS, [SphericalPort((0.0, 0.0, -1.0), 15.0)]
        )
        origins = np.array([[0.0, 0.0, -80.0]])
        directions = np.array([[0.0, 0.0, 1.0]])

        t, _normals, hit, _n_geom = geometry.ray_intersect(origins, directions)

        assert bool(hit[0])
        # 80 mm to the centre plus 50 mm to the far wall, not 30 mm to the
        # near one.
        assert t[0] == pytest.approx(130.0)

    def test_the_escaping_fraction_is_the_port_area_fraction(self):
        """An isotropic bundle from the centre escapes with probability f."""
        fraction = 0.02
        geometry = SphericalCavityGeometry(
            RADIUS,
            [
                SphericalPort.from_area_fraction((0.0, 0.0, -1.0), fraction / 2),
                SphericalPort.from_area_fraction((1.0, 0.0, 0.0), fraction / 2),
            ],
        )
        assert geometry.port_area_fraction == pytest.approx(fraction, rel=1e-12)

        n = 400_000
        directions = _isotropic(n, seed=2)
        _t, _normals, hit, _n_geom = geometry.ray_intersect(
            np.zeros_like(directions), directions
        )
        escaped = 1.0 - float(hit.mean())
        # Binomial standard error of a fraction f over n draws.
        se = math.sqrt(fraction * (1.0 - fraction) / n)
        assert abs(escaped - fraction) < 4.0 * se

    def test_a_degenerate_port_is_refused(self):
        with pytest.raises(ValueError, match="zero vector"):
            SphericalPort((0.0, 0.0, 0.0), 10.0)
        with pytest.raises(ValueError, match=r"\(0, 180\]"):
            SphericalPort((0.0, 0.0, 1.0), 0.0)
        with pytest.raises(ValueError, match=r"\(0, 180\]"):
            SphericalPort((0.0, 0.0, 1.0), 181.0)
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            SphericalPort.from_area_fraction((0.0, 0.0, 1.0), 1.5)


class TestHemisphereGeometry:
    def test_only_the_retained_half_is_surface(self):
        geometry = SphericalCavityGeometry.hemisphere(RADIUS)
        directions = _isotropic(20_000, seed=3)
        origins = np.zeros_like(directions)

        _t, _normals, hit, _n_geom = geometry.ray_intersect(origins, directions)

        np.testing.assert_array_equal(hit, directions[:, 2] > 0.0)

    def test_it_closes_around_an_interior_point(self):
        """Every outgoing direction from inside meets the shell once.

        Including the grazing ones: a collector that only caught rays near
        its axis would book the missing grazing flux as absorption
        (11.4.18's own catch list).
        """
        geometry = SphericalCavityGeometry.hemisphere(RADIUS)
        rng = np.random.default_rng(4)
        # Points spread over the equatorial disc, directions with a positive
        # axial component down to 89.99 degrees from the axis.
        n = 20_000
        r = 20.0 * np.sqrt(rng.uniform(size=n))
        phi = rng.uniform(0.0, 2 * np.pi, size=n)
        origins = np.stack([r * np.cos(phi), r * np.sin(phi), np.zeros(n)], axis=1)
        cos_theta = rng.uniform(1e-8, 1.0, size=n)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        psi = rng.uniform(0.0, 2 * np.pi, size=n)
        directions = np.stack(
            [sin_theta * np.cos(psi), sin_theta * np.sin(psi), cos_theta], axis=1
        )

        t, _normals, hit, _n_geom = geometry.ray_intersect(origins, directions)

        assert bool(np.all(hit))
        landing = origins + t[:, None] * directions
        assert bool(np.all(landing[:, 2] >= 0.0))
        np.testing.assert_allclose(np.linalg.norm(landing, axis=1), RADIUS, atol=1e-9)


# ---------------------------------------------------------------------------
# The integrating sphere, end to end
# ---------------------------------------------------------------------------
def _sphere_scene(rho: float, port_fraction: float) -> NSQScene:
    ports = []
    if port_fraction > 0.0:
        ports = [
            SphericalPort.from_area_fraction((0.0, 0.0, -1.0), port_fraction / 2),
            SphericalPort.from_area_fraction((1.0, 0.0, 0.0), port_fraction / 2),
        ]
    scene = NSQScene()
    scene.add_component(
        "wall",
        ReflectiveComponent(
            CoordinateSystem(),
            SphericalCavityGeometry(RADIUS, ports),
            reflectance=rho,
            bsdf=LambertianBSDF(reflectance_value=1.0),
        ),
    )
    scene.add_source(
        "beam",
        CoordinateSystem(z=-0.8 * RADIUS),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN), total_flux=1.0, aperture_radius=2.5
        ),
    )
    scene.add_detector(
        "patch",
        CoordinateSystem(y=RADIUS - 0.05, rx=math.radians(-90.0)),
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


class TestSphereMultiplier:
    """M = rho / (1 - rho (1 - f)), the form in Le Ru et al. (2021)."""

    def test_multiplier_matches_the_closed_form(self):
        rho, fraction = 0.98, 0.02
        expected = rho / (1.0 - rho * (1.0 - fraction))

        values = []
        for seed in range(1, 6):
            result = _sphere_scene(rho, fraction).trace(
                num_rays=20_000, seed=seed, max_depth=400
            )
            # Every watt landing on the wall leaves (1 - rho) of itself in
            # the coating bin, so the bin summed over the whole chain gives
            # the multiplier with no per-bounce bookkeeping.
            values.append(
                rho
                * result.total_flux_coating
                / ((1.0 - rho) * result.total_flux_in)
            )
        values = np.asarray(values)
        mean = values.mean()
        # The window is the run's own standard error over the five traces
        # (100000 rays in total), not a fixed number: k = 3, the catalogue's
        # rule for a statistic that is a sum over a random chain.
        standard_error = values.std(ddof=1) / math.sqrt(values.size)
        assert abs(mean - expected) < 3.0 * standard_error
        # And the window itself is tight enough for the test to mean
        # something: the closed sphere would read 49, twice this.
        assert standard_error < 0.02 * expected

    def test_a_closed_sphere_reads_the_closed_sphere_limit(self):
        rho = 0.98
        expected = rho / (1.0 - rho)  # 49, the f = 0 limit
        result = _sphere_scene(rho, 0.0).trace(num_rays=4_000, seed=1, max_depth=1200)
        measured = (
            rho * result.total_flux_coating / ((1.0 - rho) * result.total_flux_in)
        )
        assert measured == pytest.approx(expected, rel=2e-3)

    def test_the_ports_take_the_flux_the_wall_does_not_absorb(self):
        """Escaped + absorbed = in, and the escaped share is f M."""
        rho, fraction = 0.98, 0.05
        multiplier = rho / (1.0 - rho * (1.0 - fraction))
        result = _sphere_scene(rho, fraction).trace(
            num_rays=20_000, seed=7, max_depth=400
        )
        assert result.total_flux_escaped / result.total_flux_in == pytest.approx(
            fraction * multiplier, rel=0.02
        )
        assert result.flux_conservation_error < 1e-12


# ---------------------------------------------------------------------------
# One-sided detectors
# ---------------------------------------------------------------------------
class TestDetectorSidedness:
    @staticmethod
    def _two_sided_scene(side: str) -> NSQScene:
        """1 W arriving on the detector's front, 2 W on its back."""
        scene = NSQScene()
        scene.add_source(
            "from_front",
            CoordinateSystem(z=10.0, rx=math.pi),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN),
                total_flux=1.0,
                aperture_radius=1.0,
            ),
        )
        scene.add_source(
            "from_back",
            CoordinateSystem(z=-10.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN),
                total_flux=2.0,
                aperture_radius=1.0,
            ),
        )
        scene.add_detector(
            "D",
            CoordinateSystem(),
            IrradianceDetectorConfig(
                width=10.0,
                height=10.0,
                num_pixels_x=1,
                num_pixels_y=1,
                splat="hard",
                side=side,
            ),
        )
        return scene

    @pytest.mark.parametrize(
        ("side", "expected"), [("both", 3.0), ("front", 1.0), ("back", 2.0)]
    )
    def test_only_the_live_side_records(self, side, expected):
        result = self._two_sided_scene(side).trace(num_rays=2_000, seed=3, max_depth=4)
        assert result.total_flux_detected == pytest.approx(expected, rel=1e-12)
        # What the detector refuses is not absorbed either: it leaves.
        assert result.total_flux_escaped == pytest.approx(3.0 - expected, rel=1e-12)

    def test_the_default_is_unchanged(self):
        scene = self._two_sided_scene("both")
        detector = scene.detector_registry.get("D")
        assert detector.side == "both"

    def test_an_unknown_side_is_refused(self):
        with pytest.raises(ValueError, match="'both', 'front', or 'back'"):
            IrradianceDetector = type(
                self._two_sided_scene("both").detector_registry.get("D")
            )
            IrradianceDetector(
                cs=CoordinateSystem(),
                width=1.0,
                height=1.0,
                num_pixels_x=1,
                num_pixels_y=1,
                side="middle",
            )


# ---------------------------------------------------------------------------
# The hemispherical collector
# ---------------------------------------------------------------------------
class TestHemisphereDetector:
    @staticmethod
    def _diffuser_scene(rho: float = 0.95, polar_bins: int = 18) -> NSQScene:
        from optiland.nonsequential import FinitePlaneGeometry

        scene = NSQScene()
        scene.add_component(
            "diffuser",
            ReflectiveComponent(
                CoordinateSystem(z=0.0),
                FinitePlaneGeometry(width=20.0, height=20.0),
                reflectance=rho,
                bsdf=LambertianBSDF(reflectance_value=1.0),
            ),
        )
        scene.add_source(
            "beam",
            CoordinateSystem(z=-50.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN),
                total_flux=1.0,
                aperture_radius=2.5,
            ),
        )
        scene.add_detector(
            "shell",
            CoordinateSystem(rx=math.pi),
            HemisphereDetectorConfig(
                radius=100.0, num_theta=polar_bins, num_phi=36
            ),
        )
        return scene

    def test_it_collects_every_reflected_ray(self):
        """Nothing escapes a closed collector, however grazing."""
        scene = self._diffuser_scene()
        result = scene.trace(num_rays=50_000, seed=1, max_depth=8)
        assert result.total_flux_escaped == 0.0
        assert scene.detector_registry.get("shell").get_result().num_rays_hit == 50_000

    def test_the_collected_flux_is_the_reflectance(self):
        rho = 0.95
        result = self._diffuser_scene(rho).trace(
            num_rays=50_000, seed=1, max_depth=8
        )
        ratio = result.total_flux_detected / result.total_flux_in
        # Not exactly rho in one realisation: the scatter branch is drawn
        # from a probability clamped 1e-6 away from 1 and compensated by an
        # attached weight, so a realisation with no ray on the rare branch
        # reads 1e-6 relative high. Unbiased, but not bit-exact.
        assert ratio == pytest.approx(rho, rel=1e-5)

    def test_the_angular_distribution_is_cosine_weighted(self):
        """F(theta) = sin^2(theta), not the 1 - cos(theta) of a uniform lobe."""
        n = 200_000
        scene = self._diffuser_scene()
        scene.trace(num_rays=n, seed=1, max_depth=8)
        detector = scene.detector_registry.get("shell")

        for angle in (10.0, 20.0, 30.0, 45.0, 60.0):
            measured = detector.cone_fraction(angle)
            expected = math.sin(math.radians(angle)) ** 2
            se = math.sqrt(expected * (1.0 - expected) / n)
            assert abs(measured - expected) < 4.0 * se, angle

        # The whole shell holds all of it.
        assert detector.cone_fraction(90.0) == pytest.approx(1.0, abs=1e-12)
        # A uniform-in-solid-angle sampler would put 0.134 inside 30
        # degrees; this separation is the point of the test.
        assert abs(detector.cone_fraction(30.0) - (1.0 - math.cos(math.radians(30.0)))) > 0.1

    def test_a_cone_edge_must_fall_on_a_bin_edge(self):
        scene = self._diffuser_scene()
        scene.trace(num_rays=2_000, seed=1, max_depth=8)
        detector = scene.detector_registry.get("shell")
        with pytest.raises(ValueError, match="not a polar bin edge"):
            detector.cone_fraction(7.0)

    def test_polar_flux_sums_to_the_collected_flux(self):
        scene = self._diffuser_scene()
        result = scene.trace(num_rays=20_000, seed=2, max_depth=8)
        detector = scene.detector_registry.get("shell")
        assert float(detector.polar_flux().sum()) == pytest.approx(
            result.total_flux_detected, rel=1e-12
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_the_detectors_round_trip_through_scene_json(self, tmp_path):
        scene = NSQScene()
        scene.add_source(
            "beam",
            CoordinateSystem(z=-10.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN),
                total_flux=1.0,
                aperture_radius=1.0,
            ),
        )
        scene.add_detector(
            "shell",
            CoordinateSystem(rx=math.pi),
            HemisphereDetectorConfig(radius=100.0, num_theta=18, num_phi=36),
        )
        scene.add_detector(
            "one_sided",
            CoordinateSystem(z=5.0),
            IrradianceDetectorConfig(
                width=4.0, height=4.0, num_pixels_x=8, num_pixels_y=8, side="front"
            ),
        )
        path = tmp_path / "scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        shell = loaded.detector_registry.get("shell")
        assert isinstance(shell, HemisphereDetector)
        assert float(shell.radius) == pytest.approx(100.0)
        assert shell.num_bins_theta == 18
        assert shell.num_bins_phi == 36
        assert loaded.detector_registry.get("one_sided").side == "front"

    def test_the_cavity_round_trips_through_the_scene_ir(self):
        ports = [
            SphericalPort.from_area_fraction((0.0, 0.0, -1.0), 0.01),
            SphericalPort((1.0, 0.0, 0.0), 12.0),
        ]
        scene = NSQScene()
        scene.add_component(
            "wall",
            ReflectiveComponent(
                CoordinateSystem(),
                SphericalCavityGeometry(RADIUS, ports),
                reflectance=0.98,
                bsdf=LambertianBSDF(reflectance_value=1.0),
            ),
        )
        scene.add_source(
            "beam",
            CoordinateSystem(z=-40.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN),
                total_flux=1.0,
                aperture_radius=2.5,
            ),
        )
        scene.add_detector(
            "shell",
            CoordinateSystem(rx=math.pi),
            HemisphereDetectorConfig(radius=100.0, num_theta=9, num_phi=12),
        )

        ir = lower(scene)
        assert ir.primitives[0].kind == "spherical_cavity"
        assert ir.sensors[0].kind == "hemisphere"

        # Rule 5 of the translatability checklist: through JSON and back,
        # unchanged.
        ir2 = scene_ir_from_dict(json.loads(json.dumps(scene_ir_to_dict(ir))))
        assert ir2.primitives[0].kind == ir.primitives[0].kind
        assert ir2.primitives[0].params == ir.primitives[0].params
        assert ir2.sensors[0].params == ir.sensors[0].params
        assert ir2.primitives[0].params["ports"][0]["half_angle_deg"] == pytest.approx(
            SphericalPort.from_area_fraction((0.0, 0.0, -1.0), 0.01).half_angle_deg
        )


# ---------------------------------------------------------------------------
# The other backend
# ---------------------------------------------------------------------------
class TestBackendAgreement:
    def test_the_cavity_solves_the_same_on_both_backends(self, each_backend):
        """The geometry is written in backend operations, not in NumPy."""
        geometry = SphericalCavityGeometry(
            RADIUS, [SphericalPort((0.0, 0.0, -1.0), 15.0)]
        )
        directions_np = _isotropic(2_000, seed=5)
        origins_np = np.zeros_like(directions_np)

        t, normals, hit, n_geom = geometry.ray_intersect(
            be.array(origins_np), be.array(directions_np)
        )

        t_np = np.asarray(be.to_numpy(t))
        hit_np = np.asarray(be.to_numpy(hit)).astype(bool)
        expected_hit = ~(directions_np[:, 2] < -math.cos(math.radians(15.0)))
        np.testing.assert_array_equal(hit_np, expected_hit)
        np.testing.assert_allclose(t_np[hit_np], RADIUS, atol=1e-9)
        assert np.asarray(be.to_numpy(normals)).shape == (2_000, 3)
        assert np.asarray(be.to_numpy(n_geom)).shape == (2_000, 3)
