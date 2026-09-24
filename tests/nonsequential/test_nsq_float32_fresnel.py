"""Float32 at the two Fresnel limits of the validation catalogue.

Two catalogue cases passed at float64 and failed at float32 on the same rays.
A ray-by-ray reproduction (``benchmarks/nonsequential/float32_mechanisms.py``)
found a mechanism for each, and neither was the one first attributed:

* **One millidegree inside the critical angle** (N-BK7 to vacuum), float32
  read a transmittance of 0.124 where float64 reads 0.0373. The float32
  radicand ``w = 1 - sin^2(theta_t)`` was right to 0.1 of a unit roundoff; the
  refraction then clamped it to ``_tol.radicand_floor``, a float64 budget of
  1e-12 scaled by the dtype's ulp ratio to 5.37e-4 at float32, so
  ``cos(theta_t)`` never fell below 0.0232. The cure is
  ``refractive.refraction_cosine``: a radicand below ``4 u_T`` is a domain
  error (total internal reflection), and every resolved radicand is
  square-rooted as it is. ``TestRefractionCosine`` and
  ``TestCriticalAngleFloat32``.

* **89 degrees incidence** (vacuum to N-BK7), float32 read a reflectance of
  exactly 1.0. No ray was re-captured by the interface it left (on either
  branch), so the grazing-exit mechanism was not it. The transmitted ray
  leaves at 41 degrees from the normal; the catalogue's detector sits
  0.001 mm *along z* behind the tilted interface, which is
  ``0.001 cos(89 deg) = 17 nm`` along the normal -- 18 ulps of the 10 mm
  coordinate at float32 -- and a surface blinds the next intersection test
  within ``16 ulp`` along its normal (the outgoing-origin offset, R-07-6) plus
  ``16 ulp`` of path along the ray (the accept threshold, R-07-5). Every
  transmitted ray passes the detector unseen. That is the engine's documented
  resolution at that coordinate, not a kernel defect, so nothing in the
  kernel changes for it: ``TestGrazingExitFloat32`` pins that no ray returns,
  and ``TestDetectorBehindAnInterface`` pins the width of the band and shows
  the same scene with the detector 0.001 mm along the normal is measured
  correctly at float32.

Every behavioural test has a control that puts the old rule back and asserts
the failure returns, so no assertion passes on slack.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch not available -- float32 runs on its CPU backend")

# ruff: noqa: E402
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQMaterial,
    NSQScene,
    RayDatabaseConfig,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential import _tol as tol
from optiland.nonsequential.components import refractive
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry
from optiland.nonsequential.materials.nsq_material import VACUUM

#: N-BK7 at the d-line, pinned as the catalogue pins it.
N_GLASS = 1.5168
WAVELENGTH = 0.5876
THETA_C = math.degrees(math.asin(1.0 / N_GLASS))
Z_INTERFACE = 10.0
U32 = 2.0**-24
U64 = 2.0**-53
#: The float32 step of the interface's own coordinate (10 mm lies in [8, 16)).
ULP32_AT_10 = 2.0**-20


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _configure(precision: str) -> None:
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision(precision)


def clamped_refraction_cosine(sin2_t):
    """The rule this change replaced: TIR only past 1, radicand clamped to radicand_floor."""
    tir = sin2_t > 1.0
    cos_t = be.where(
        tir,
        be.zeros_like(sin2_t),
        be.maximum(1.0 - sin2_t, tol.radicand_floor(be.ones_like(sin2_t))) ** 0.5,
    )
    return tir, cos_t


@pytest.fixture
def old_clamp(monkeypatch):
    """Control: put the clamped refraction cosine back."""
    monkeypatch.setattr(refractive, "refraction_cosine", clamped_refraction_cosine)


def _fresnel_t(theta_i_deg: float, n1: float, n2: float) -> float:
    """Unpolarized Fresnel transmittance, in float64."""
    ci = math.cos(math.radians(theta_i_deg))
    st = n1 / n2 * math.sin(math.radians(theta_i_deg))
    if st >= 1.0:
        return 0.0
    ct = math.sqrt(1.0 - st * st)
    rs = ((n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)) ** 2
    rp = ((n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)) ** 2
    return 1.0 - 0.5 * (rs + rp)


def _interface_scene(
    theta_deg: float,
    *,
    from_glass: bool,
    detector: str = "irradiance",
    z_detector: float | None = 10.001,
    gap_along_normal: float | None = None,
    reflect_prob=None,
    width: float = 500.0,
) -> NSQScene:
    """The catalogue's single-interface rig.

    A 0.02 mm pencil along +z meets a plane at z = 10 mm tilted by
    ``theta_deg`` about x; a detector parallel to it catches what is
    transmitted. The detector sits either where the catalogue runner puts it
    (``z_detector`` along the z axis) or ``gap_along_normal`` mm along the
    interface's own normal.
    """
    rx = math.radians(theta_deg)
    glass = NSQMaterial(optiland_material=IdealMaterial(n=N_GLASS, k=0.0))
    front, back = (glass, VACUUM) if from_glass else (VACUUM, glass)
    scene = NSQScene()
    if reflect_prob is not None:
        scene.sampling_policy = dataclasses.replace(scene.sampling_policy, reflect_prob=reflect_prob)
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(WAVELENGTH), total_flux=1.0, aperture_radius=0.02
        ),
    )
    scene.add_component(
        "IF",
        RefractiveComponent(
            CoordinateSystem(z=Z_INTERFACE, rx=rx), PlaneGeometry(), front, back, name="IF"
        ),
    )
    if gap_along_normal is not None:
        det_cs = CoordinateSystem(
            y=-gap_along_normal * math.sin(rx),
            z=Z_INTERFACE + gap_along_normal * math.cos(rx),
            rx=rx,
        )
    else:
        det_cs = CoordinateSystem(z=z_detector, rx=rx)
    if detector == "raydb":
        config = RayDatabaseConfig(width=width, height=width)
    else:
        config = IrradianceDetectorConfig(
            width=width, height=width, num_pixels_x=1, num_pixels_y=1, splat="hard"
        )
    scene.add_detector("T", det_cs, config)
    return scene


def _detected_fraction(scene: NSQScene, n_rays: int, seed: int, **kw) -> float:
    result = scene.trace(num_rays=n_rays, seed=seed, max_depth=2, **kw)
    return float(result.total_flux_detected / result.total_flux_in)


def _per_ray_transmittance(theta_deg: float, precision: str, n_rays: int = 2_000) -> np.ndarray:
    """Each transmitted ray's own Fresnel T, read off its weight.

    With the reflect branch drawn at a fixed probability of one half, a
    transmitted ray carries ``T / (1 - 1/2) = 2 T`` of its launch flux, so the
    per-ray flux at a ray database is a deterministic reading of the
    transmittance the engine computed for that ray.
    """
    _configure(precision)
    scene = _interface_scene(theta_deg, from_glass=True, detector="raydb", reflect_prob=0.5)
    result = scene.trace(num_rays=n_rays, seed=5, max_depth=2)
    flux = np.asarray(result.detectors["T"].flux, dtype=np.float64)
    return flux / (2.0 * (1.0 / n_rays))


# ---------------------------------------------------------------------------
# 1. the rule itself
# ---------------------------------------------------------------------------


class TestRefractionCosine:
    """``refraction_cosine``: a domain test in u_T, and no clamp on resolved values."""

    def test_domain_threshold_is_four_unit_roundoffs(self):
        assert tol.DEFAULT_RADICAND_K == 4
        assert float(tol.radicand_min(np.zeros(3, dtype=np.float32))) == 4 * U32
        assert float(tol.radicand_min(np.zeros(3, dtype=np.float64))) == 4 * U64
        assert float(tol.radicand_min(torch.zeros(3, dtype=torch.float32))) == 4 * U32
        assert float(tol.radicand_min(torch.zeros(3, dtype=torch.float64))) == 4 * U64

    def test_threshold_follows_the_dtype(self):
        """T-08-2: the guard changes by exactly u32 / u64 with the dtype."""
        r = float(tol.radicand_min(np.zeros(1, np.float32))) / float(
            tol.radicand_min(np.zeros(1, np.float64))
        )
        assert r == U32 / U64

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_resolved_radicands_are_bitwise_unchanged_at_float64(self, backend):
        """Wherever the old clamp returned w itself (w >= 1e-12), the new rule
        returns the same bits -- the float64 bit-identity the change rests on."""
        rng = np.random.default_rng(3)
        w = np.concatenate([np.logspace(-12, 0, 400), rng.uniform(1e-12, 1.0, 400)])
        sin2 = 1.0 - w
        if backend == "torch":
            _configure("float64")
        else:
            be.set_backend("numpy")
            be.set_precision("float64")
        s = be.array(sin2)
        tir_new, cos_new = refractive.refraction_cosine(s)
        tir_old, cos_old = clamped_refraction_cosine(s)
        # The float64 w actually formed from sin2 may round below 1e-12 on the
        # lowest few entries; compare only where the old clamp was inactive.
        live = np.asarray(be.to_numpy(1.0 - s)) >= 1e-12
        assert live.sum() > 790
        assert np.array_equal(
            np.asarray(be.to_numpy(tir_new))[live], np.asarray(be.to_numpy(tir_old))[live]
        )
        a = np.asarray(be.to_numpy(cos_new), dtype=np.float64)[live]
        b = np.asarray(be.to_numpy(cos_old), dtype=np.float64)[live]
        assert np.array_equal(a.view(np.int64), b.view(np.int64))

    @pytest.mark.parametrize(
        "backend,precision,u",
        [("torch", "float32", U32), ("torch", "float64", U64), ("numpy", "float64", U64)],
    )
    def test_unresolved_radicand_is_total_reflection(self, backend, precision, u):
        if backend == "torch":
            _configure(precision)
        else:
            be.set_backend("numpy")
            be.set_precision(precision)
        np_dtype = np.float32 if precision == "float32" else np.float64
        # 1 - k u is exact in the dtype for these small k, and so is w.
        k = np.array([-2, 0, 1, 2, 3, 4, 5, 8], dtype=np.float64)
        sin2 = be.array((1.0 - k * u).astype(np_dtype))
        tir, cos_t = refractive.refraction_cosine(sin2)
        tir = np.asarray(be.to_numpy(tir))
        cos_t = np.asarray(be.to_numpy(cos_t))
        assert cos_t.dtype == np_dtype
        assert tir.tolist() == [True, True, True, True, True, False, False, False]
        assert np.all(cos_t[tir] == 0.0)
        np.testing.assert_array_equal(cos_t[~tir], np.sqrt((k[~tir] * u).astype(np_dtype)))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_gradient_is_finite_across_the_boundary(self, dtype):
        """R-08-8: the masked input keeps sqrt'(0) out of the backward pass."""
        u = U32 if dtype == torch.float32 else U64
        _configure("float32" if dtype == torch.float32 else "float64")
        sin2 = torch.tensor(
            [1.0 + u, 1.0, 1.0 - u, 1.0 - 3 * u, 1.0 - 4 * u, 1.0 - 8 * u, 0.5],
            dtype=dtype,
            requires_grad=True,
        )
        _, cos_t = refractive.refraction_cosine(sin2)
        cos_t.sum().backward()
        assert torch.isfinite(sin2.grad).all()
        # Totally reflected lanes carry no gradient; transmitted ones do.
        assert sin2.grad[:4].abs().max() == 0.0
        assert (sin2.grad[4:] != 0.0).all()

    def test_the_float32_clamp_was_the_cause(self):
        """Control, on the rule alone: at the radicand of r1_09's bracket the old
        clamp returns 3.7 times the square root; the new rule returns it."""
        _configure("float32")
        sin2_np = np.array([1.0 - 3.98159e-05], dtype=np.float32)
        sin2 = be.array(sin2_np)
        _, old = clamped_refraction_cosine(sin2)
        _, new = refractive.refraction_cosine(sin2)
        old = float(be.to_numpy(old)[0])
        new = float(be.to_numpy(new)[0])
        exact = math.sqrt(float(np.float32(1.0) - sin2_np[0]))
        floor32 = float(tol.radicand_floor(np.ones(1, np.float32))[0])
        assert old == pytest.approx(math.sqrt(floor32), rel=1e-6)
        assert old > 3.5 * exact
        assert new == pytest.approx(exact, rel=2 * U32)


# ---------------------------------------------------------------------------
# 2. the critical angle (catalogue case r1_09)
# ---------------------------------------------------------------------------


class TestCriticalAngleFloat32:
    """One millidegree inside the critical angle, and exactly at it."""

    THETA = THETA_C - 1.0e-3

    def test_per_ray_transmittance_within_the_precision_budget(self):
        """Float32's Fresnel T for the same ray agrees with float64 within the
        budget of docs/theory/08_precision.md section 8.2(d): the relative error
        of sqrt(w) is dw / (2 w), and T is proportional to cos(theta_t) here to
        first order. dw is allowed 16 u32 -- the formula's own (3 + r^2) u = 5.3 u
        plus three ulps of cos(theta_i) at 3.5 u each. Measured: 7.1e-5 against a
        budget of 1.2e-2."""
        t64 = _per_ray_transmittance(self.THETA, "float64")
        t32 = _per_ray_transmittance(self.THETA, "float32")
        expected = _fresnel_t(self.THETA, N_GLASS, 1.0)
        assert t64.size > 500 and t32.size > 500
        assert np.ptp(t64) == 0.0 and np.ptp(t32) == 0.0  # one incidence, one T
        assert t64[0] == pytest.approx(expected, rel=1e-9)
        w = 1.0 - (N_GLASS * math.sin(math.radians(self.THETA))) ** 2
        budget = 16 * U32 / (2.0 * w)
        assert abs(t32[0] - t64[0]) / t64[0] < budget

    def test_per_ray_transmittance_control(self, old_clamp):
        """Control: with the clamp back, float32 T is 3.5 times float64's (measured 3.477)."""
        t64 = _per_ray_transmittance(self.THETA, "float64")
        t32 = _per_ray_transmittance(self.THETA, "float32")
        assert t32[0] / t64[0] > 3.0

    def test_transmitted_fraction_matches_float64(self):
        """The catalogue's own statistic: the same 20 000 rays, both dtypes."""
        n = 20_000
        fr = {}
        for precision in ("float64", "float32"):
            _configure(precision)
            fr[precision] = _detected_fraction(_interface_scene(self.THETA, from_glass=True), n, 7)
        expected = _fresnel_t(self.THETA, N_GLASS, 1.0)
        se = math.sqrt(expected * (1.0 - expected) / n)
        assert fr["float64"] == pytest.approx(expected, abs=4 * se)
        assert fr["float32"] == pytest.approx(fr["float64"], abs=4 * se)

    def test_transmitted_fraction_control(self, old_clamp):
        n = 20_000
        fr = {}
        for precision in ("float64", "float32"):
            _configure(precision)
            fr[precision] = _detected_fraction(_interface_scene(self.THETA, from_glass=True), n, 7)
        assert fr["float32"] > 3.0 * fr["float64"]

    #: A wave asserted at theta_c leaves at cos(theta_t) = 1e-6 and meets the
    #: detector plane 0.00075 / 1e-6 = 750 mm down the surface: the detector
    #: must reach that far or the test cannot see what it is testing for.
    WIDE = 1.0e5

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_nothing_is_transmitted_at_the_critical_angle(self, precision):
        """At theta_c the float64 radicand is +1 u, not negative. Drawing the
        reflect branch at one half makes any transmitted wave the engine asserts
        visible, however small its T."""
        _configure(precision)
        scene = _interface_scene(THETA_C, from_glass=True, reflect_prob=0.5, width=self.WIDE)
        assert _detected_fraction(scene, 20_000, 7) == 0.0

    def test_at_the_critical_angle_the_clamp_asserted_a_wave(self, old_clamp):
        """Control: the old clamp gave float64 a transmitted wave at theta_c
        (cos(theta_t) = 1e-6, T = 5.8e-6), found only by forcing the branch."""
        _configure("float64")
        scene = _interface_scene(THETA_C, from_glass=True, reflect_prob=0.5, width=self.WIDE)
        frac = _detected_fraction(scene, 20_000, 7)
        assert frac == pytest.approx(5.8e-6, rel=0.05)


# ---------------------------------------------------------------------------
# 3. the 89-degree point (catalogue case r1_07): no re-capture
# ---------------------------------------------------------------------------


def _hits_per_ray(result, surface_name: str) -> Counter:
    events = result.ray_paths["events"]
    hits = events[(events["event_type"] == "hit") & (events["component_name"] == surface_name)]
    return Counter(hits["ray_id"].tolist())


class TestGrazingExitFloat32:
    """No ray returns to the interface it left, at float32.

    Both grazing exits the two catalogue cases produce: the reflected ray of
    89-degree incidence from vacuum (it leaves 1 degree above the surface) and
    the transmitted ray one millidegree inside the critical angle (0.36
    degrees above it). The reproduction found no re-capture on either branch
    before this change as well: the origin offset of R-07-6 holds at float32.
    These are guards, not the failing mechanism.
    """

    @pytest.mark.parametrize(
        "theta_deg,from_glass", [(89.0, False), (THETA_C - 1.0e-3, True)]
    )
    def test_one_hit_per_ray(self, theta_deg, from_glass):
        _configure("float32")
        scene = _interface_scene(theta_deg, from_glass=from_glass, z_detector=60.0)
        result = scene.trace(num_rays=20_000, seed=7, max_depth=4, record_paths=True)
        counts = _hits_per_ray(result, "IF")
        assert len(counts) == 20_000
        assert max(counts.values()) == 1


# ---------------------------------------------------------------------------
# 4. the 89-degree point (catalogue case r1_07): the detector's resolution
# ---------------------------------------------------------------------------


class TestDetectorBehindAnInterface:
    """How close behind a surface a detector can be and still be seen.

    A ray leaving a surface starts ``16 ulp`` off it along the geometric
    normal (``_tol.origin_offset``), and a root nearer than ``16 ulp`` of path
    is rejected (``_tol.accept_t_min``). A parallel plane at a perpendicular
    gap ``g`` is reached after ``(g - 16 ulp) / cos(theta_t)``, so it is seen
    only when ``g > 16 ulp (1 + cos theta_t)``. At a 10 mm coordinate in
    float32 one ulp is 0.95 nm; at 60 degrees incidence into N-BK7
    ``cos theta_t = 0.821`` and the edge is 29.1 ulps.
    """

    THETA = 60.0

    def _edge_ulps(self) -> float:
        cos_t = math.sqrt(1.0 - (math.sin(math.radians(self.THETA)) / N_GLASS) ** 2)
        return 16.0 * (1.0 + cos_t)

    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_a_gap_above_the_band_is_seen(self, precision):
        _configure(precision)
        gap = (self._edge_ulps() + 6.0) * ULP32_AT_10
        scene = _interface_scene(self.THETA, from_glass=False, gap_along_normal=gap)
        frac = _detected_fraction(scene, 20_000, 7)
        expected = _fresnel_t(self.THETA, 1.0, N_GLASS)
        se = math.sqrt(expected * (1.0 - expected) / 20_000)
        assert frac == pytest.approx(expected, abs=4 * se)

    def test_a_gap_inside_the_band_is_not_seen_at_float32(self):
        """The same scene with the gap 6 ulps under the edge: float32 cannot
        resolve the detector from the surface, float64 can."""
        gap = (self._edge_ulps() - 6.0) * ULP32_AT_10
        _configure("float32")
        assert _detected_fraction(
            _interface_scene(self.THETA, from_glass=False, gap_along_normal=gap), 20_000, 7
        ) == 0.0
        _configure("float64")
        assert _detected_fraction(
            _interface_scene(self.THETA, from_glass=False, gap_along_normal=gap), 20_000, 7
        ) > 0.5

    def test_the_catalogue_detector_at_89_degrees_is_inside_the_band(self):
        """r1_07's detector: 0.001 mm along z behind the interface is
        0.001 cos(89 deg) = 17.5 nm along its normal, 18.3 float32 ulps, under
        the 28 ulps the 89-degree geometry needs. Float32 sees none of the
        transmitted light and reads reflectance 1."""
        _configure("float32")
        assert _detected_fraction(_interface_scene(89.0, from_glass=False), 20_000, 7) == 0.0

    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_the_same_detector_along_the_normal_is_measured(self, precision):
        """The same 0.001 mm, taken along the interface's normal (1049 ulps at
        float32): both dtypes read the Fresnel transmittance at 89 degrees."""
        _configure(precision)
        scene = _interface_scene(89.0, from_glass=False, gap_along_normal=0.001)
        frac = _detected_fraction(scene, 20_000, 7)
        expected = _fresnel_t(89.0, 1.0, N_GLASS)
        se = math.sqrt(expected * (1.0 - expected) / 20_000)
        assert frac == pytest.approx(expected, abs=4 * se)
