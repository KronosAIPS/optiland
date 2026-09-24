"""Four additions that let the engine express the last closed-form cases.

**The reflection count.** Every ray carries the number of reflections in its
history, settled by one geometric rule (the ray leaves on the side it came
from), and any detector can book the arriving flux by it. Checked against an
independent enumeration of every path through a stack of Fresnel faces, on
both array backends, with the exhaustive (bounded-splitting) estimator: the
window's ghost series ``(1 - R)^2 R^(2m)`` and a four-plate stack.

**Exhaustive splitting on both backends, and its books.** A split books no
sampling residual and its one loss once; a split child is truncated at the
depth cap exactly as its sibling is; the Torch backend splits when built with
``allow_splitting=True`` in forward-only mode and still refuses otherwise.

**A prism.** Two plane faces at an apex angle: the deviation at the symmetric
incidence is ``2 arcsin(n sin(A/2)) - A`` to floating point, and every other
incidence deviates more.

**An ideal paraxial lens.** A plane that deflects by the thin-lens law in the
slopes: every ray from an object point meets the conjugate image point, so
the paraxial etendue is conserved identically.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    CollimatedSourceConfig,
    FarFieldDetectorConfig,
    HemisphereDetectorConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQMaterial,
    NSQScene,
    ParaxialLensConfig,
    PointSourceConfig,
    PrismConfig,
    RayDatabaseConfig,
    ReflectionHistogram,
    Spectrum,
    SurfaceConfig,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.components.paraxial import ParaxialLensComponent
from optiland.nonsequential.ir.lower import lower
from optiland.nonsequential.ir.scene_ir import (
    SamplingPolicy,
    scene_ir_from_dict,
    scene_ir_to_dict,
)
from optiland.nonsequential.ray_bundle import NSQRayBundle

N_GLASS = 1.5168
GREEN = 0.5876
R_NORMAL = ((N_GLASS - 1.0) / (N_GLASS + 1.0)) ** 2


@pytest.fixture(params=be.list_available_backends(), ids=lambda b: f"backend={b}")
def each_backend(request):
    """Run a test on every installed backend at float64, and put the state back."""
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


def _glass() -> NSQMaterial:
    return NSQMaterial(optiland_material=IdealMaterial(n=N_GLASS, k=0.0))


def _splitting_backend(name: str, seed: int = 1):
    """A backend that honours ``split_depth`` on either array library."""
    if name == "torch":
        return TorchBackend(seed=seed, allow_splitting=True)
    return NumpyBackend(seed=seed)


def _stack_histogram(reflectance: float, n_faces: int, max_interactions: int, kmax: int):
    """Exact flux by reflection count through a 1-D stack of faces, by enumeration.

    Independent of the engine: every path of at most ``max_interactions``
    interactions through ``n_faces`` parallel faces of reflectance ``R`` at
    normal incidence, followed by dynamic programming over (gap, direction)
    with one polynomial in the reflection count per state.

    Returns:
        ``(forward, backward, inside)``: the flux leaving past the last face
        and before the first one, per reflection count ``0 .. kmax``, and the
        flux still inside when the cap is reached.
    """
    transmit = 1.0 - reflectance
    forward = np.zeros(kmax + 1)
    backward = np.zeros(kmax + 1)
    state = {(0, 1): np.eye(1, kmax + 1, 0).ravel()}
    for _ in range(max_interactions):
        nxt: dict = {}
        for (gap, direction), w in state.items():
            face = gap if direction == 1 else gap - 1
            if not 0 <= face < n_faces:
                continue
            through = (gap + direction, direction)
            back = (gap, -direction)
            reflected = np.zeros_like(w)
            reflected[1:] = w[:-1] * reflectance
            nxt[through] = nxt.get(through, 0.0) + w * transmit
            nxt[back] = nxt.get(back, 0.0) + reflected
        state = {}
        for (gap, direction), w in nxt.items():
            if gap == n_faces and direction == 1:
                forward += w
            elif gap == 0 and direction == -1:
                backward += w
            else:
                state[(gap, direction)] = w
    inside = float(sum(w.sum() for w in state.values()))
    return forward, backward, inside


def _plate_stack_scene(n_plates: int, thickness: float, pitch: float, bins: int):
    """Plates in a collimated beam, a transmitted and a reflected collector."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-50.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN), total_flux=1.0, aperture_radius=2.5
        ),
    )
    for k in range(n_plates):
        scene.add_lens(
            f"P{k}",
            CoordinateSystem(z=pitch * k),
            LensConfig(
                r1=1.0e9,
                r2=1.0e9,
                thickness=thickness,
                material=_glass(),
                front_aperture_radius=10.0,
            ),
        )
    scene.add_detector(
        "T",
        CoordinateSystem(z=pitch * n_plates + 100.0),
        IrradianceDetectorConfig(
            width=20, height=20, num_pixels_x=1, num_pixels_y=1, splat="hard",
            reflection_bins=bins,
        ),
    )
    scene.add_detector(
        "B",
        CoordinateSystem(z=-40.0),
        IrradianceDetectorConfig(
            width=20, height=20, num_pixels_x=1, num_pixels_y=1, splat="hard",
            side="front", reflection_bins=bins,
        ),
    )
    return scene


def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = np.where(np.abs(b) > 0.0, np.abs(b), 1.0)
    return np.abs(a - b) / scale


# ---------------------------------------------------------------------------
# The reflection count, against path enumeration
# ---------------------------------------------------------------------------
class TestReflectionCount:
    def test_window_ghost_series_is_exact_under_splitting(self, each_backend):
        """11.4.6: the transmitted flux with 2m reflections is (1 - R)^2 R^(2m)."""
        scene = _plate_stack_scene(1, 10.0, 0.0, bins=9)
        scene.sampling_policy = SamplingPolicy(
            split_depth=40, split_budget=64.0, rr_start_flux=1e-16
        )
        res = scene.trace(
            num_rays=200, seed=1, max_depth=40, min_flux_fraction=1e-16,
            backend=_splitting_backend(each_backend),
        )
        hist = res.reflection_histograms["T"]
        orders = [(1.0 - R_NORMAL) ** 2 * R_NORMAL ** (2 * m) for m in range(4)]
        assert np.all(_rel(hist.flux[[0, 2, 4, 6]], orders) < 1e-12)
        # An odd count never reaches the far side of a window.
        assert np.all(hist.flux[[1, 3, 5, 7]] == 0.0)
        # Two paths come back with one reflection: off the front face, and
        # off the back face and out through the front.
        back = res.reflection_histograms["B"]
        one = R_NORMAL + (1.0 - R_NORMAL) ** 2 * R_NORMAL
        assert _rel(back.flux[1], one) < 1e-12
        assert res.flux_conservation_error < 1e-12

    @pytest.mark.parametrize("max_depth", [8, 14])
    def test_four_plate_stack_matches_path_enumeration(self, each_backend, max_depth):
        """Eight faces: every bin, both sides, and the truncated remainder.

        The depth cap is part of the comparison: a ray is truncated once it
        has made ``max_depth`` interactions, so the enumeration counts paths
        of at most ``max_depth - 1`` interactions and what it leaves inside
        is exactly the flux the engine books as depth-truncated.
        """
        scene = _plate_stack_scene(4, 2.0, 20.0, bins=9)
        scene.sampling_policy = SamplingPolicy(
            split_depth=max_depth, split_budget=64.0, rr_start_flux=1e-300
        )
        res = scene.trace(
            num_rays=5, seed=1, max_depth=max_depth, min_flux_fraction=1e-300,
            backend=_splitting_backend(each_backend),
        )
        forward, backward, inside = _stack_histogram(R_NORMAL, 8, max_depth - 1, 9)
        assert np.all(_rel(res.reflection_histograms["T"].flux, forward[:9]) < 1e-12)
        assert np.all(_rel(res.reflection_histograms["B"].flux, backward[:9]) < 1e-12)
        assert _rel(res.total_flux_lost, inside) < 1e-9
        assert res.flux_conservation_error < 1e-12

    def test_a_mirror_counts_one_reflection_and_a_transmissive_tap_none(self):
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN), total_flux=1.0,
                aperture_radius=1.0,
            ),
        )
        scene.add_detector(
            "tap",
            CoordinateSystem(z=20.0),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=1, num_pixels_y=1,
                absorb=False, side="back", reflection_bins=3,
            ),
        )
        scene.add_mirror(
            "M", CoordinateSystem(z=50.0),
            MirrorConfig(radius=1.0e12, reflectance=0.9, aperture_radius=10.0),
        )
        scene.add_detector(
            "home",
            CoordinateSystem(z=10.0),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=1, num_pixels_y=1,
                side="front", reflection_bins=3,
            ),
        )
        res = scene.trace(num_rays=500, seed=2, max_depth=8)
        tap = res.reflection_histograms["tap"]
        home = res.reflection_histograms["home"]
        assert tap.flux[0] == pytest.approx(1.0, rel=1e-12)
        assert tap.flux[1:].sum() == 0.0
        assert home.flux[1] == pytest.approx(0.9, rel=1e-12)
        assert home.flux[[0, 2]].sum() == 0.0 and home.overflow_flux == 0.0

    def test_bins_and_overflow_hold_the_whole_detected_flux(self):
        """Roulette mode, a deep window: nothing is dropped from the count."""
        scene = _plate_stack_scene(1, 10.0, 0.0, bins=2)
        scene.sampling_policy = SamplingPolicy(reflect_prob=0.5)
        res = scene.trace(num_rays=4000, seed=5, max_depth=40)
        for name in ("T", "B"):
            hist = res.reflection_histograms[name]
            total = res.detectors[name].total_flux_float
            assert hist.total_flux == pytest.approx(total, rel=1e-12)
            assert hist.num_rays_hit.sum() + hist.overflow_num_rays_hit == (
                res.detectors[name].num_rays_hit
            )
        assert res.reflection_histograms["T"].overflow_flux > 0.0

    def test_the_count_travels_with_the_ray_through_every_bundle_operation(self):
        rays = NSQRayBundle(
            x=np.zeros(4), y=np.zeros(4), z=np.zeros(4),
            L=np.zeros(4), M=np.zeros(4), N=np.ones(4),
            flux=np.ones(4), wavelength=np.full(4, GREEN), n_current=np.ones(4),
            bounce=np.zeros(4, dtype=np.int32), alive=np.array([True, False, True, True]),
            ray_id=np.arange(4), reflections=np.array([0, 1, 2, 3], dtype=np.int32),
        )
        assert rays.compact().reflections.tolist() == [0, 2, 3]
        assert rays.take(np.array([3, 1])).reflections.tolist() == [3, 1]
        assert rays.select(np.array([2])).reflections.tolist() == [2]
        both = NSQRayBundle.concat([rays, rays.select(np.array([0]))])
        assert both.reflections.tolist() == [0, 1, 2, 3, 0]
        fresh = NSQRayBundle(
            x=np.zeros(2), y=np.zeros(2), z=np.zeros(2), L=np.zeros(2),
            M=np.zeros(2), N=np.ones(2), flux=np.ones(2),
            wavelength=np.full(2, GREEN), n_current=np.ones(2),
            bounce=np.zeros(2, dtype=np.int32), alive=np.ones(2, dtype=bool),
        )
        assert fresh.reflections.tolist() == [0, 0]
        assert fresh.reflections.dtype == np.int32

    def test_torch_float32_carries_the_series_within_its_rounding(self):
        """The same split tree at float32: each order within 64 ulp of its chain."""
        previous = be.get_backend()
        be.set_backend("torch")
        try:
            be.set_device("cpu")
            be.set_precision("float32")
            scene = _plate_stack_scene(1, 10.0, 0.0, bins=9)
            scene.sampling_policy = SamplingPolicy(
                split_depth=40, split_budget=64.0, rr_start_flux=1e-16
            )
            res = scene.trace(
                num_rays=200, seed=1, max_depth=40, min_flux_fraction=1e-16,
                backend=TorchBackend(seed=1, allow_splitting=True),
            )
            hist = res.reflection_histograms["T"]
            orders = [(1.0 - R_NORMAL) ** 2 * R_NORMAL ** (2 * m) for m in range(4)]
            # Order m is a chain of 2m + 2 Fresnel factors, each carrying the
            # float32 rounding of R: the bound is (2m + 2) * 16 u32.
            u32 = 2.0**-24
            for m in range(4):
                assert _rel(hist.flux[2 * m], orders[m]) < (2 * m + 2) * 16 * u32
        finally:
            be.set_precision("float64")
            be.set_backend(previous)


class TestReflectionHistogramResult:
    def test_standard_error_is_the_binomial_one_for_equal_weights(self):
        n, k, w = 1000, 37, 0.25
        hist = ReflectionHistogram(
            flux=np.array([k * w]), flux_sq=np.array([k * w * w]),
            num_rays_hit=np.array([k]),
        )
        expected = w * math.sqrt(k * (n - k) / (n - 1.0))
        assert hist.standard_error(n)[0] == pytest.approx(expected, rel=1e-12)

    def test_negative_bins_are_refused(self):
        scene = NSQScene()
        with pytest.raises(ValueError, match="reflection_bins"):
            scene.add_detector(
                "D", CoordinateSystem(),
                IrradianceDetectorConfig(width=1, height=1, reflection_bins=-1),
            )

    def test_a_detector_without_bins_reports_no_histogram(self):
        scene = _plate_stack_scene(1, 10.0, 0.0, bins=0)
        res = scene.trace(num_rays=100, seed=1)
        assert res.reflection_histograms == {}

    def test_bins_survive_json_and_the_scene_ir(self, tmp_path):
        scene = NSQScene()
        scene.add_source(
            "S", CoordinateSystem(),
            PointSourceConfig(spectrum=Spectrum.monochromatic(GREEN), total_flux=1.0),
        )
        scene.add_detector(
            "I", CoordinateSystem(z=10.0),
            IrradianceDetectorConfig(width=5, height=5, reflection_bins=4),
        )
        scene.add_detector(
            "F", CoordinateSystem(z=20.0), FarFieldDetectorConfig(reflection_bins=3)
        )
        scene.add_detector(
            "H", CoordinateSystem(z=-5.0),
            HemisphereDetectorConfig(radius=40.0, reflection_bins=2),
        )
        path = tmp_path / "bins.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)
        assert [d.reflection_bins for d in loaded.detectors] == [4, 3, 2]
        ir = lower(scene, strict=False)
        assert [s.params["reflection_bins"] for s in ir.sensors] == [4, 3, 2]
        ir2 = scene_ir_from_dict(json.loads(json.dumps(scene_ir_to_dict(ir))))
        assert [s.params["reflection_bins"] for s in ir2.sensors] == [4, 3, 2]


# ---------------------------------------------------------------------------
# Exhaustive splitting: its books, and the Torch backend
# ---------------------------------------------------------------------------
class TestSplitting:
    def test_a_split_books_its_loss_once_and_no_residual(self, each_backend):
        """A lossy coating on both faces of a window, split to depth 40.

        Every interaction removes ``w (1 - R - T)`` once; the split's two
        children carry ``wR`` and ``wT``; nothing is a sampling residual. So
        the flux booked into the coating bin plus everything that arrived or
        was truncated is the launched watt, to rounding.
        """
        coating = SimpleCoating(transmittance=0.85, reflectance=0.1)
        scene = NSQScene()
        scene.add_source(
            "S", CoordinateSystem(z=-50.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN), total_flux=1.0,
                aperture_radius=2.5,
            ),
        )
        scene.add_lens(
            "W", CoordinateSystem(),
            LensConfig(
                r1=1.0e9, r2=1.0e9, thickness=10.0, material=_glass(),
                front_aperture_radius=10.0,
                front=SurfaceConfig(coating=coating), back=SurfaceConfig(coating=coating),
            ),
        )
        scene.add_detector(
            "T", CoordinateSystem(z=60.0),
            IrradianceDetectorConfig(width=20, height=20, num_pixels_x=1, num_pixels_y=1),
        )
        scene.sampling_policy = SamplingPolicy(
            split_depth=40, split_budget=64.0, rr_start_flux=1e-300
        )
        res = scene.trace(
            num_rays=50, seed=1, max_depth=40, min_flux_fraction=1e-300,
            backend=_splitting_backend(each_backend),
        )
        assert abs(res.total_flux_sampling_residual) < 1e-15
        assert res.flux_conservation_error < 1e-12
        # Closed form for the transmitted total: T^2 / (1 - R^2).
        t, r = 0.85, 0.1
        assert res.detectors["T"].total_flux_float == pytest.approx(
            t * t / (1.0 - r * r), rel=1e-12
        )

    def test_torch_refuses_to_split_by_default_and_in_gradient_mode(self):
        previous = be.get_backend()
        be.set_backend("torch")
        try:
            be.set_device("cpu")
            be.set_precision("float64")
            scene = _plate_stack_scene(1, 10.0, 0.0, bins=0)
            scene.sampling_policy = SamplingPolicy(split_depth=4)
            be.grad_mode.enable()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    scene.trace(
                        num_rays=50, seed=1,
                        backend=TorchBackend(seed=1, allow_splitting=True),
                    )
            finally:
                be.grad_mode.disable()
            assert any("split_depth" in str(w.message) for w in caught)
            assert not TorchBackend(seed=1)._splitting_enabled()
            assert TorchBackend(seed=1, allow_splitting=True)._splitting_enabled()
        finally:
            be.set_backend(previous)


# ---------------------------------------------------------------------------
# The prism
# ---------------------------------------------------------------------------
def _prism_deviation(apex_deg: float, incidence_deg: float, face: float = 20.0) -> float:
    """Trace one pencil ray through a prism at a given incidence; the deviation [deg].

    The ray is aimed at the entrance face's midpoint, in the principal
    section, turned from the inward normal toward the apex side by the
    incidence angle. Its exit direction is read off a ray database, and the
    Fresnel branch is steered to transmission (``reflect_prob = 0``, the
    weight compensates) so the one ray is the refracted one.
    """
    half = math.radians(apex_deg) / 2.0
    c, s = math.cos(half), math.sin(half)
    inward = np.array([-s, 0.0, c])
    i = math.radians(incidence_deg)
    d = np.array(
        [inward[0] * math.cos(i) + inward[2] * math.sin(i), 0.0,
         -inward[0] * math.sin(i) + inward[2] * math.cos(i)]
    )
    mid = np.array([-0.5 * face * c, 0.0, -0.5 * face * s])
    start = mid - 50.0 * d
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(x=start[0], z=start[2], ry=math.atan2(d[0], d[2])),
        PointSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN), total_flux=1.0, half_angle_deg=0.0
        ),
    )
    scene.add_prism(
        "P", CoordinateSystem(),
        PrismConfig(apex_angle_deg=apex_deg, face_length=face, length=20.0,
                    material=_glass()),
    )
    scene.add_detector(
        "D", CoordinateSystem(z=100.0), RayDatabaseConfig(width=1000, height=1000)
    )
    scene.sampling_policy = SamplingPolicy(reflect_prob=0.0)
    res = scene.trace(num_rays=1, seed=1, max_depth=8)
    db = res.detectors["D"]
    assert len(db.x) == 1
    out = np.array([float(db.L[0]), float(db.M[0]), float(db.N[0])])
    return math.degrees(math.atan2(np.linalg.norm(np.cross(d, out)), float(d @ out)))


class TestPrism:
    @pytest.mark.parametrize("apex", [60.0, 30.0])
    def test_minimum_deviation_matches_the_closed_form(self, each_backend, apex):
        half = math.radians(apex) / 2.0
        i_min = math.degrees(math.asin(N_GLASS * math.sin(half)))
        delta_min = 2.0 * i_min - apex
        assert _rel(_prism_deviation(apex, i_min), delta_min) < 1e-12

    def test_every_other_incidence_deviates_more(self):
        apex = 60.0
        i_min = math.degrees(math.asin(N_GLASS * math.sin(math.radians(apex) / 2.0)))
        delta_min = _prism_deviation(apex, i_min)
        for step in (-2.0, -1.0, 1.0, 2.0):
            assert _prism_deviation(apex, i_min + step) > delta_min

    def test_the_index_comes_back_from_the_deviation(self):
        apex = 60.0
        i_min = math.degrees(math.asin(N_GLASS * math.sin(math.radians(apex) / 2.0)))
        delta = math.radians(_prism_deviation(apex, i_min))
        a = math.radians(apex)
        n_back = math.sin((a + delta) / 2.0) / math.sin(a / 2.0)
        assert _rel(n_back, N_GLASS) < 1e-12

    def test_the_base_absorbs_and_an_open_base_is_left_out(self):
        scene = NSQScene()
        scene.add_prism(
            "P", CoordinateSystem(),
            PrismConfig(apex_angle_deg=60.0, face_length=20.0, length=20.0,
                        material=_glass()),
        )
        scene.add_prism(
            "Q", CoordinateSystem(y=100.0),
            PrismConfig(apex_angle_deg=60.0, face_length=20.0, length=20.0,
                        material=_glass(), open_base=True),
        )
        names = [s.name for s in scene.surfaces]
        assert names == ["P.front", "P.back", "P.base", "Q.front", "Q.back"]
        # A ray along +x at the base's height from outside is stopped there.
        scene.add_source(
            "S", CoordinateSystem(x=-60.0, z=0.0, ry=math.pi / 2.0),
            PointSourceConfig(spectrum=Spectrum.monochromatic(GREEN), total_flux=1.0,
                              half_angle_deg=0.0),
        )
        scene.add_detector(
            "D", CoordinateSystem(x=100.0, ry=math.pi / 2.0),
            IrradianceDetectorConfig(width=50, height=50),
        )
        res = scene.trace(num_rays=10, seed=1)
        assert res.total_flux_absorbed == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize("apex", [0.0, 180.0, -5.0])
    def test_an_impossible_apex_is_refused(self, apex):
        with pytest.raises(ValueError, match="apex_angle_deg"):
            NSQScene().add_prism(
                "P", CoordinateSystem(),
                PrismConfig(apex_angle_deg=apex, face_length=20.0, length=20.0,
                            material=_glass()),
            )

    def test_the_prism_lowers_to_planes_and_round_trips(self, tmp_path):
        scene = NSQScene()
        scene.add_prism(
            "P", CoordinateSystem(x=1.0, rz=0.2),
            PrismConfig(apex_angle_deg=45.0, face_length=12.0, length=8.0,
                        material="N-BK7", open_base=True),
        )
        scene.add_source(
            "S", CoordinateSystem(),
            PointSourceConfig(spectrum=Spectrum.monochromatic(GREEN), total_flux=1.0),
        )
        scene.add_detector("D", CoordinateSystem(z=50.0),
                           IrradianceDetectorConfig(width=5, height=5))
        ir = lower(scene)
        assert [p.kind for p in ir.primitives] == ["plane", "plane"]
        assert [p.component_kind for p in ir.primitives] == ["refractive"] * 2
        path = tmp_path / "prism.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)
        cfg = loaded.component_registry.get("P")._config
        assert (cfg.apex_angle_deg, cfg.face_length, cfg.length, cfg.open_base) == (
            45.0, 12.0, 8.0, True,
        )
        for a, b in zip(scene.surfaces, loaded.surfaces, strict=True):
            ta, ra = a.cs.get_effective_transform()
            tb, rb = b.cs.get_effective_transform()
            assert np.allclose(be.to_numpy(ta), be.to_numpy(tb))
            assert np.allclose(be.to_numpy(ra), be.to_numpy(rb))


# ---------------------------------------------------------------------------
# The ideal paraxial lens
# ---------------------------------------------------------------------------
def _paraxial_scene(x0: float, y0: float, stop: float | None = 50.0) -> NSQScene:
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(x=x0, y=y0, z=-150.0),
        PointSourceConfig(spectrum=Spectrum.monochromatic(GREEN), total_flux=1.0,
                          half_angle_deg=2.5),
    )
    scene.add_paraxial_lens(
        "L", CoordinateSystem(),
        ParaxialLensConfig(focal_length=100.0, aperture_radius=5.0, stop_radius=stop),
    )
    scene.add_detector(
        "D", CoordinateSystem(z=300.0), RayDatabaseConfig(width=100, height=100)
    )
    return scene


class TestParaxialLens:
    @pytest.mark.parametrize(("x0", "y0"), [(0.0, 0.0), (0.5, 0.0), (0.3, -0.4)])
    def test_every_ray_meets_the_conjugate_image_point(self, each_backend, x0, y0):
        """Object at 150 mm, f = 100 mm: image at 300 mm, magnification -2."""
        res = _paraxial_scene(x0, y0).trace(num_rays=4000, seed=3, max_depth=8)
        db = res.detectors["D"]
        assert len(db.x) > 1000
        assert np.max(np.abs(np.asarray(db.x) + 2.0 * x0)) < 1e-12
        assert np.max(np.abs(np.asarray(db.y) + 2.0 * y0)) < 1e-12
        # Lossless and never a reflection.
        assert np.all(np.asarray(db.flux) == pytest.approx(1.0 / 4000))
        assert res.flux_conservation_error < 1e-12

    def test_the_stop_takes_what_misses_the_aperture(self):
        stopped = _paraxial_scene(0.0, 0.0).trace(num_rays=4000, seed=3)
        open_ = _paraxial_scene(0.0, 0.0, stop=None).trace(num_rays=4000, seed=3)
        assert stopped.total_flux_absorbed > 0.3
        assert stopped.total_flux_absorbed + stopped.total_flux_detected == (
            pytest.approx(1.0, rel=1e-12)
        )
        assert open_.total_flux_absorbed == 0.0

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_a_parallel_ray_crosses_the_axis_at_the_focus_either_way(self, direction):
        """h = 2 mm parallel to the axis: slope -h/f, axis crossing at z = +-f."""
        scene = NSQScene()
        scene.add_source(
            "S", CoordinateSystem(x=2.0, z=-50.0 * direction,
                                  ry=0.0 if direction > 0 else math.pi),
            PointSourceConfig(spectrum=Spectrum.monochromatic(GREEN), total_flux=1.0,
                              half_angle_deg=0.0),
        )
        scene.add_paraxial_lens(
            "L", CoordinateSystem(),
            ParaxialLensConfig(focal_length=100.0, aperture_radius=5.0),
        )
        scene.add_detector(
            "D", CoordinateSystem(z=10.0 * direction), RayDatabaseConfig(width=50, height=50)
        )
        res = scene.trace(num_rays=1, seed=1)
        db = res.detectors["D"]
        x, lx, lz = float(db.x[0]), float(db.L[0]), float(db.N[0])
        z_cross = 10.0 * direction - x * lz / lx
        assert z_cross == pytest.approx(100.0 * direction, rel=1e-13)

    def test_the_lens_lowers_as_its_own_kind_and_round_trips(self, tmp_path):
        scene = _paraxial_scene(0.0, 0.0)
        ir = lower(scene)
        kinds = [(p.kind, p.component_kind) for p in ir.primitives]
        assert kinds == [("plane", "paraxial"), ("annulus", "absorbing")]
        assert ir.primitives[0].params["focal_length"] == 100.0
        ir2 = scene_ir_from_dict(json.loads(json.dumps(scene_ir_to_dict(ir))))
        assert ir2.primitives[0].params == ir.primitives[0].params
        path = tmp_path / "paraxial.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)
        cfg = loaded.component_registry.get("L")._config
        assert (cfg.focal_length, cfg.aperture_radius, cfg.stop_radius) == (
            100.0, 5.0, 50.0,
        )
        assert isinstance(loaded.surfaces[0], ParaxialLensComponent)

    def test_bad_configurations_are_refused(self):
        with pytest.raises(ValueError, match="focal_length"):
            NSQScene().add_paraxial_lens(
                "L", CoordinateSystem(),
                ParaxialLensConfig(focal_length=0.0, aperture_radius=5.0),
            )
        with pytest.raises(ValueError, match="stop_radius"):
            NSQScene().add_paraxial_lens(
                "L", CoordinateSystem(),
                ParaxialLensConfig(focal_length=50.0, aperture_radius=5.0, stop_radius=4.0),
            )
