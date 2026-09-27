"""Mirrors, scatter lobes and the ideal lens in Stokes mode (the research repository's issue 5, item 6).

Every component the loop can hit now carries the polarization state through:

* **a mirror with a scalar reflectance** is the ideal non-polarizing mirror
  ``R diag(1, 1, -1, -1)`` in its plane of incidence. The expected state is taken
  from the physical reflection of the electric field, ``E' = -E + 2 (E . n) n``
  (the tangential field of a perfect conductor changes sign, the normal one does
  not), written in the lab frame and read in the engine's outgoing frame -- an
  independent route for the frame bookkeeping, which is where chapter 06 says
  polarized tracers go wrong;
* **a metal mirror** (a thin-film stack with no layers on an absorbing substrate)
  against the closed-form Fresnel amplitudes of the exp(-i omega t) convention;
* **a dielectric stack mirror** against the thin-film adapter's own s and p terms;
* **the scatter lobes**: Lambertian, Harvey-Shack and tabulated lobes depolarize
  the rays they scatter (the minimal version has no Mueller scatter model), the
  specular lobe keeps the mirror's state;
* **the ideal paraxial lens** keeps the state and carries the axis to the new
  direction;
* **Stokes I is the scalar flux** on mirror scenes with an unpolarized source,
  bit for bit (a two-mirror cavity, a metal mirror, a Lambertian mirror), and the
  emulated replay equals eager with the state on through a mirror.

A transmissive detector crossing does not change a ray's direction, so its axis
stays perpendicular to it with nothing to do; an absorber ends the ray.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    CollimatedSourceConfig,
    FinitePlaneGeometry,
    IrradianceDetectorConfig,
    MirrorConfig,
    NSQScene,
    ParaxialLensConfig,
    PointSourceConfig,
    ReflectiveComponent,
    Spectrum,
)
from optiland.nonsequential import polarization as P
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.bsdf.harvey_shack import HarveyShackBSDF
from optiland.nonsequential.bsdf.lambertian import LambertianBSDF
from optiland.nonsequential.bsdf.specular import SpecularBRDF
from optiland.nonsequential.components.coating_support import UnpolarizedThinFilmCoating
from optiland.nonsequential.components.paraxial import ParaxialLensComponent
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.thin_film import ThinFilmStack

U32 = 2.0**-24
WL = 0.55
METAL = (0.96, 6.69)

LEGS = [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")]
LEG_IDS = [f"{b}-{p}" for b, p in LEGS]


@pytest.fixture(autouse=True)
def _restore_backend(monkeypatch):
    monkeypatch.delenv(P.POLARIZATION_ENV, raising=False)
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _configure(kind: str, precision: str) -> float:
    """Set the leg; return the window of a short chain (about 40 operations) in its dtype."""
    be.set_backend(kind)
    if kind == "torch":
        be.set_device("cpu")
        be.grad_mode.disable()
    be.set_precision(precision)
    return 1e-14 if precision == "float64" else 64 * U32


def _backend(kind: str, polarization="stokes", **kw):
    if kind == "numpy":
        return NumpyBackend(seed=3, polarization=polarization)
    from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: PLC0415

    return TorchBackend(seed=3, polarization=polarization, **kw)


def _h(x):
    return np.asarray(be.to_numpy(x), dtype=np.float64)


_ORIGINAL = {cls: cls.interact for cls in (ReflectiveComponent, ParaxialLensComponent)}


class _Spy:
    """Record the hit rays' flux, state, axis and direction after each call of ``cls.interact``."""

    def __init__(self, monkeypatch, cls):
        self.rows = []
        original = _ORIGINAL[cls]
        signature = inspect.signature(original)

        def spy(comp, *args, **kwargs):
            bound = signature.bind(comp, *args, **kwargs)
            rays = bound.arguments["rays"]
            flux_in = _h(rays.flux)
            original(comp, *args, **kwargs)
            hit = np.asarray(be.to_numpy(bound.arguments["hit_mask"]), dtype=bool)
            if rays.pol_q is None or not hit.any():
                return
            self.rows.append(
                dict(
                    name=comp.name,
                    gain=_h(rays.flux)[hit] / flux_in[hit],
                    q=_h(rays.pol_q)[hit], u=_h(rays.pol_u)[hit], v=_h(rays.pol_v)[hit],
                    e=np.stack([_h(rays.pol_ex)[hit], _h(rays.pol_ey)[hit], _h(rays.pol_ez)[hit]], axis=1),
                    k=np.stack([_h(rays.L)[hit], _h(rays.M)[hit], _h(rays.N)[hit]], axis=1),
                )
            )

        monkeypatch.setattr(cls, "interact", spy)


# ---------------------------------------------------------------------------
# The reference route: Jones vectors in the lab frame
# ---------------------------------------------------------------------------


def _jones(q, u, v):
    """A pure state's (E_p, E_s) with |E|^2 = 1, in this module's Stokes convention."""
    a = math.sqrt((1.0 + q) / 2.0)
    b = math.sqrt((1.0 - q) / 2.0)
    return a, b * complex(math.cos(math.atan2(v, u)), math.sin(math.atan2(v, u)))


def _stokes_in_frame(E, e, k):
    """(I, q, u, v) of the complex lab field ``E`` in the frame (e, k x e, k)."""
    s = np.cross(k, e)
    Ep, Es = np.dot(E, e), np.dot(E, s)
    I = abs(Ep) ** 2 + abs(Es) ** 2
    X = Ep * np.conj(Es)
    return I, (abs(Ep) ** 2 - abs(Es) ** 2) / I, 2 * X.real / I, -2 * X.imag / I


def _fresnel_complex(n1, n2c, cos_i):
    """r_s, r_p of the exp(-i omega t) convention (index n + ik), principal root."""
    sin2 = 1.0 - cos_i * cos_i
    cos_t = np.sqrt(1.0 - (n1 / n2c) ** 2 * sin2 + 0j)
    rs = (n1 * cos_i - n2c * cos_t) / (n1 * cos_i + n2c * cos_t)
    rp = (n2c * cos_i - n1 * cos_t) / (n2c * cos_i + n1 * cos_t)
    return rs, rp


def _reflect_lab(E, k, n, rs, rp):
    """The reflected lab field for Fresnel amplitudes (rs, rp): E_p along s x k in and s x k' out."""
    s = np.cross(k, n)
    s = s / np.linalg.norm(s)
    k_out = k - 2 * np.dot(k, n) * n
    p_in, p_out = np.cross(s, k), np.cross(s, k_out)
    return rp * np.dot(E, p_in) * p_out + rs * np.dot(E, s) * s, k_out


# ---------------------------------------------------------------------------
# One flat mirror at incidence theta, a pencil along +z
# ---------------------------------------------------------------------------

REF_AXIS = (math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0)
STATES = [(0.5, math.sqrt(0.75), 0.0), (0.36, 0.48, 0.8), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)]


def _mirror_scene(theta_deg, reflectance, bsdf=None, scatter_fraction=1.0, stokes=None):
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(WL), total_flux=1.0, aperture_radius=0.001),
    )
    if stokes is not None:
        P.set_source_polarization(scene, "S", stokes=stokes, reference_axis=REF_AXIS)
    rx = math.radians(theta_deg)
    scene.add_component(
        "M",
        ReflectiveComponent(
            CoordinateSystem(z=10.0, rx=rx), FinitePlaneGeometry(aperture_radius=5.0),
            reflectance=reflectance, bsdf=bsdf, scatter_fraction=scatter_fraction, name="M",
        ),
    )
    scene.add_detector(
        "FAR", CoordinateSystem(x=900.0, z=900.0),
        IrradianceDetectorConfig(width=1, height=1, num_pixels_x=1, num_pixels_y=1, splat="hard"),
    )
    return scene


def _mirror_normal(theta_deg):
    """The mirror's normal facing the beam (the ray arrives along +z)."""
    rx = math.radians(theta_deg)
    n = np.array([0.0, -math.sin(rx), math.cos(rx)])  # local +z of CoordinateSystem(rx=rx)
    return -n if n[2] > 0 else n


def _birth(q, u, v):
    k = np.array([0.0, 0.0, 1.0])
    a = np.array(REF_AXIS)
    e = np.cross(np.cross(k, a), k)
    e = e / np.linalg.norm(e)
    Ep, Es = _jones(q, u, v)
    return Ep * e + Es * np.cross(k, e), k


def _metal_coating():
    stack = ThinFilmStack(
        incident_material=IdealMaterial(n=1.0), substrate_material=IdealMaterial(n=METAL[0], k=METAL[1]),
        reference_wl_um=WL,
    )
    return UnpolarizedThinFilmCoating(stack)


def _dbr_coating():
    stack = ThinFilmStack(
        incident_material=IdealMaterial(n=1.0), substrate_material=IdealMaterial(n=1.52), reference_wl_um=WL,
    )
    for _ in range(4):
        stack.add_layer_qwot(IdealMaterial(n=2.35))
        stack.add_layer_qwot(IdealMaterial(n=1.38))
    return UnpolarizedThinFilmCoating(stack)


class TestScalarMirror:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    @pytest.mark.parametrize("theta", [0.0, 30.0, 45.0, 70.0])
    def test_against_the_physical_reflection_of_the_field(self, leg, theta, monkeypatch):
        window = _configure(*leg)
        R = 0.9
        n = _mirror_normal(theta)
        for q, u, v in STATES:
            spy = _Spy(monkeypatch, ReflectiveComponent)
            _mirror_scene(theta, R, stokes=(1.0, q, u, v)).trace(
                num_rays=2, seed=1, max_depth=1, backend=_backend(leg[0])
            )
            row = spy.rows[0]
            E, k = _birth(q, u, v)
            E_r = -E + 2 * np.dot(E, n) * n  # the ideal mirror, R = 1
            k_out = k - 2 * np.dot(k, n) * n
            for i in range(len(row["q"])):
                e = row["e"][i]
                assert abs(np.dot(e, row["k"][i])) < window and abs(np.linalg.norm(e) - 1) < window
                assert np.allclose(row["k"][i], k_out, atol=window, rtol=0)
                I, q2, u2, v2 = _stokes_in_frame(E_r, e, row["k"][i])
                assert abs(I - 1.0) < window
                assert abs(row["gain"][i] - R) < window
                got = np.array([row["q"][i], row["u"][i], row["v"][i]])
                assert np.max(np.abs(got - [q2, u2, v2])) < window, (theta, (q, u, v), got, (q2, u2, v2))

    def test_the_handedness_flips(self, monkeypatch):
        """Circular in, the other circular out: V changes sign at every mirror (the ideal R diag(1,1,-1,-1))."""
        _configure("numpy", "float64")
        spy = _Spy(monkeypatch, ReflectiveComponent)
        _mirror_scene(45.0, 1.0, stokes=(1.0, 0.0, 0.0, 1.0)).trace(num_rays=1, seed=1, max_depth=1, backend=_backend("numpy"))
        assert abs(spy.rows[0]["v"][0] + 1.0) < 1e-15


class TestMetalMirror:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    @pytest.mark.parametrize("theta", [0.0, 45.0, 75.0])
    def test_against_the_closed_form(self, leg, theta, monkeypatch):
        window = _configure(*leg)
        n = _mirror_normal(theta)
        rs, rp = _fresnel_complex(1.0, complex(*METAL), math.cos(math.radians(theta)))
        for q, u, v in STATES + [(0.0, 0.0, 0.0)]:
            spy = _Spy(monkeypatch, ReflectiveComponent)
            _mirror_scene(theta, _metal_coating(), stokes=(1.0, q, u, v)).trace(
                num_rays=2, seed=1, max_depth=1, backend=_backend(leg[0])
            )
            row = spy.rows[0]
            if (q, u, v) == (0.0, 0.0, 0.0):
                # unpolarized: the flux factor is (R_s + R_p) / 2, DoP = |R_s - R_p| / (R_s + R_p)
                Rs, Rp = abs(rs) ** 2, abs(rp) ** 2
                assert np.max(np.abs(row["gain"] - 0.5 * (Rs + Rp))) < window
                assert np.max(np.abs(np.hypot(row["q"], row["u"]) - abs(Rs - Rp) / (Rs + Rp))) < window + 1e-15
                continue
            E, k = _birth(q, u, v)
            if theta == 0.0:
                # the plane of incidence is undefined; any frame will do for the reference
                s = np.cross(k, [1.0, 0.0, 0.0])
                s = s / np.linalg.norm(s)
                E_r = rp * np.dot(E, np.cross(s, k)) * np.cross(s, -k) + rs * np.dot(E, s) * s
                k_out = -k
            else:
                E_r, k_out = _reflect_lab(E, k, n, rs, rp)
            for i in range(len(row["q"])):
                I, q2, u2, v2 = _stokes_in_frame(E_r, row["e"][i], row["k"][i])
                assert np.allclose(row["k"][i], k_out, atol=window, rtol=0)
                assert abs(row["gain"][i] / I - 1.0) < window
                got = np.array([row["q"][i], row["u"][i], row["v"][i]])
                assert np.max(np.abs(got - [q2, u2, v2])) < window, (theta, (q, u, v), got, (q2, u2, v2))

    def test_the_perfect_conductor_limit_is_the_ideal_mirror(self):
        """The reference route's own consistency: r_s = -1, r_p = +1 is E' = -E + 2 (E . n) n."""
        k = np.array([0.0, 0.0, 1.0])
        n = _mirror_normal(35.0)
        E, _ = _birth(0.36, 0.48, 0.8)
        E_r, _ = _reflect_lab(E, k, n, -1.0, 1.0)
        assert np.allclose(E_r, -E + 2 * np.dot(E, n) * n, atol=1e-15)


class TestStackMirror:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_against_the_adapter(self, leg, monkeypatch):
        """A quarter-wave (HL)^4 mirror at 30 degrees: the event applies the adapter's s/p terms."""
        window = _configure(*leg)
        theta = 30.0
        coating = _dbr_coating()
        spy = _Spy(monkeypatch, ReflectiveComponent)
        _mirror_scene(theta, coating, stokes=(1.0, 0.0, 1.0, 0.0)).trace(
            num_rays=2, seed=1, max_depth=1, backend=_backend(leg[0])
        )
        be.set_backend("numpy")
        be.set_precision("float64")
        sp = P.thin_film_sp(_dbr_coating().stack, np.array([WL]), np.array([math.cos(math.radians(theta))]))
        m = sp.reflection()
        # (1, 0, 1, 0) in the frame of REF_AXIS; the plane of incidence is the y-z plane (p on y-z,
        # s = +-x), turned by psi = 20 deg + 90 deg from it about +z: rotate into it, apply, compare
        # in the engine's own output frame through the independent lab route.
        k = np.array([0.0, 0.0, 1.0])
        n = _mirror_normal(theta)
        E, _ = _birth(0.0, 1.0, 0.0)
        rs = math.sqrt(float(sp.Rs[0]))
        xr = complex(float(m.m22[0]), float(m.m23[0]))  # r_p r_s*
        rp = xr / rs
        E_r, _ = _reflect_lab(E, k, n, rs, rp)
        row = spy.rows[0]
        for i in range(len(row["q"])):
            I, q2, u2, v2 = _stokes_in_frame(E_r, row["e"][i], row["k"][i])
            assert abs(row["gain"][i] / I - 1.0) < window
            got = np.array([row["q"][i], row["u"][i], row["v"][i]])
            assert np.max(np.abs(got - [q2, u2, v2])) < window


# ---------------------------------------------------------------------------
# Scatter lobes
# ---------------------------------------------------------------------------


class TestLobes:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    @pytest.mark.parametrize(
        "bsdf", [LambertianBSDF(0.8), HarveyShackBSDF(b0=0.5, l0=0.01, s=1.5)], ids=["lambertian", "harvey_shack"]
    )
    def test_a_scattering_lobe_depolarizes(self, leg, bsdf, monkeypatch):
        window = _configure(*leg)
        spy = _Spy(monkeypatch, ReflectiveComponent)
        _mirror_scene(30.0, 0.9, bsdf=bsdf, stokes=(1.0, 0.36, 0.48, 0.8)).trace(
            num_rays=64, seed=2, max_depth=1, backend=_backend(leg[0])
        )
        row = spy.rows[0]
        assert not row["q"].any() and not row["u"].any() and not row["v"].any()
        dots = np.abs(np.einsum("ij,ij->i", row["e"], row["k"]))
        assert dots.max() < window and np.abs(np.linalg.norm(row["e"], axis=1) - 1).max() < window

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_the_specular_lobe_keeps_the_mirror_state(self, leg, monkeypatch):
        _configure(*leg)
        spy = _Spy(monkeypatch, ReflectiveComponent)
        stokes = (1.0, 0.36, 0.48, 0.8)
        _mirror_scene(30.0, 0.9, bsdf=SpecularBRDF(), stokes=stokes).trace(
            num_rays=4, seed=2, max_depth=1, backend=_backend(leg[0])
        )
        plain = _Spy(monkeypatch, ReflectiveComponent)
        _mirror_scene(30.0, 0.9, stokes=stokes).trace(num_rays=4, seed=2, max_depth=1, backend=_backend(leg[0]))
        a, b = spy.rows[0], plain.rows[0]
        for key in ("q", "u", "v", "e", "k"):
            assert np.array_equal(a[key], b[key]), key


# ---------------------------------------------------------------------------
# The ideal paraxial lens
# ---------------------------------------------------------------------------


class TestParaxialLens:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_the_state_is_kept_and_the_axis_follows(self, leg, monkeypatch):
        window = _configure(*leg)
        scene = NSQScene()
        scene.add_source(
            "S", CoordinateSystem(),
            CollimatedSourceConfig(spectrum=Spectrum.monochromatic(WL), total_flux=1.0, aperture_radius=4.0),
        )
        P.set_source_polarization(scene, "S", stokes=(1.0, 0.36, 0.48, 0.8), reference_axis=REF_AXIS)
        scene.add_paraxial_lens("L", CoordinateSystem(z=10.0), ParaxialLensConfig(focal_length=20.0, aperture_radius=5.0))
        scene.add_detector(
            "D", CoordinateSystem(z=40.0),
            IrradianceDetectorConfig(width=50, height=50, num_pixels_x=1, num_pixels_y=1, splat="hard"),
        )
        spy = _Spy(monkeypatch, ParaxialLensComponent)
        scene.trace(num_rays=64, seed=2, max_depth=2, backend=_backend(leg[0]))
        row = spy.rows[0]
        assert np.array_equal(row["q"], np.full_like(row["q"], np.float32(0.36) if leg[1] == "float32" else 0.36))
        assert np.array_equal(row["v"], np.full_like(row["v"], np.float32(0.8) if leg[1] == "float32" else 0.8))
        assert np.abs(1 - np.abs(row["k"][:, 2])).max() > 1e-3  # the rays were turned
        dots = np.abs(np.einsum("ij,ij->i", row["e"], row["k"]))
        assert dots.max() < window and np.abs(np.linalg.norm(row["e"], axis=1) - 1).max() < window


# ---------------------------------------------------------------------------
# Stokes I is the scalar flux on mirror scenes (chapter 06 section 6.9)
# ---------------------------------------------------------------------------


def _bits(value):
    value = be.to_numpy(value) if be.is_torch_tensor(value) else value
    if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0:
        arr = np.ascontiguousarray(np.asarray(value))
        return (str(arr.dtype), arr.tobytes().hex())
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    return value


def _ledger(result) -> dict:
    skip = {"trace_time_sec", "diagnostics", "detectors", "ray_paths", "reflection_histograms", "environment"}
    out = {k: _bits(v) for k, v in vars(result).items() if k not in skip}
    for name, det in result.detectors.items():
        for key, value in vars(det).items():
            out[f"{name}.{key}"] = _bits(value)
    return out


def _cavity(reflectance=0.99):
    """r1_20's rig: two flat mirrors 100 mm apart, a pencil between them."""
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(z=50.0),
        PointSourceConfig(spectrum=Spectrum.monochromatic(WL), total_flux=1.0, half_angle_deg=0.0),
    )
    scene.add_mirror("A", CoordinateSystem(z=0.0), MirrorConfig(radius=1.0e12, reflectance=reflectance, aperture_radius=5.0))
    scene.add_mirror("B", CoordinateSystem(z=100.0), MirrorConfig(radius=1.0e12, reflectance=reflectance, aperture_radius=5.0))
    scene.add_detector(
        "far", CoordinateSystem(z=1.0e9), IrradianceDetectorConfig(width=1.0, height=1.0, num_pixels_x=1, num_pixels_y=1)
    )
    scene.sampling_policy = SamplingPolicy(rr_start_flux=1.0e-30)
    return scene


def _folded(reflectance, bsdf=None):
    """A beam folded by a 45-degree mirror onto a detector: tilted incidence, a plane of incidence."""
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(WL), total_flux=1.0, aperture_radius=1.0),
    )
    scene.add_component(
        "M",
        ReflectiveComponent(
            CoordinateSystem(z=10.0, rx=math.radians(45.0)), FinitePlaneGeometry(aperture_radius=5.0),
            reflectance=reflectance, bsdf=bsdf, name="M",
        ),
    )
    scene.add_detector(
        "D", CoordinateSystem(y=30.0, z=10.0, rx=math.radians(90.0)),
        IrradianceDetectorConfig(width=40, height=40, num_pixels_x=4, num_pixels_y=4, splat="hard"),
    )
    return scene


class TestScalarEquivalence:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    @pytest.mark.parametrize(
        "build",
        [
            lambda: _cavity(),
            lambda: _folded(0.9),
            lambda: _folded(_metal_coating()),
            lambda: _folded(0.9, LambertianBSDF(0.8)),
        ],
        ids=["cavity", "scalar_mirror", "metal_mirror", "lambertian_mirror"],
    )
    def test_unpolarized_stokes_equals_scalar_bit_for_bit(self, leg, build):
        _configure(*leg)
        off = build().trace(num_rays=2000, seed=5, max_depth=300, backend=_backend(leg[0], "off"))
        on = build().trace(num_rays=2000, seed=5, max_depth=300, backend=_backend(leg[0], "stokes"))
        assert on.environment.get("polarization") == "stokes" and "polarization" not in off.environment
        a, b = _ledger(off), _ledger(on)
        assert a == b, sorted(k for k in a if a[k] != b.get(k))


def _tilted_cavity():
    """Two metal mirrors 5 mm apart and a beam 10 degrees off their normal: a plane of
    incidence at every reflection, about twenty reflections before a ray walks out."""
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(z=2.5, rx=math.radians(10.0)),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(WL), total_flux=1.0, aperture_radius=1.0),
    )
    P.set_source_polarization(scene, "S", stokes=(1.0, 0.36, 0.48, 0.8), reference_axis=REF_AXIS)
    for name, z in (("A", 0.0), ("B", 5.0)):
        scene.add_component(
            name,
            ReflectiveComponent(
                CoordinateSystem(z=z), FinitePlaneGeometry(aperture_radius=20.0),
                reflectance=_metal_coating(), name=name,
            ),
        )
    scene.add_detector(
        "D", CoordinateSystem(y=-40.0, z=2.5, rx=math.radians(90.0)),
        IrradianceDetectorConfig(width=2000, height=2000, num_pixels_x=4, num_pixels_y=4, splat="hard"),
    )
    return scene


class TestReplayThroughAMirror:
    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_emulated_replay_equals_eager(self, precision):
        """The state and axis are static buffers through mirror events: the emulated replay equals eager."""
        pytest.importorskip("torch")
        _configure("torch", precision)
        eager = _tilted_cavity().trace(
            num_rays=4096, seed=9, max_depth=24, batch_size=2048, backend=_backend("torch", "stokes", compact_every=0)
        )
        replay = _tilted_cavity().trace(
            num_rays=4096, seed=9, max_depth=24, batch_size=2048,
            backend=_backend("torch", "stokes", graph_replay="emulate"),
        )
        assert replay.environment["graph_replay"] == "emulate" and replay.environment["graph_replay_batches"] > 0
        assert float(be.to_numpy(eager.detectors["D"].total_flux)) > 0.0
        a, b = _ledger(eager), _ledger(replay)
        assert a == b, sorted(k for k in a if a[k] != b.get(k))
