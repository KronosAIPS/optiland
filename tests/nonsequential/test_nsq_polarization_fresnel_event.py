"""The Stokes event at a refractive interface (the research repository's issue 5, build item 5).

In Stokes mode ``RefractiveComponent.interact`` rotates each ray's state into the
plane of incidence, draws the branch from ``R_eff = M_r00 + M_r01 q'`` (the scalar
``R`` bit for bit when the ray is unpolarized), weights it by the matrix of the
branch taken and sets the outgoing reference axis ``e = s x k_out``. The expected
values come from closed forms and from the Jones route of chapter 06 section 6.3,
never from the engine:

* at Brewster's angle (the r1_28 and r1_08 physics), the reflected ray is
  s-polarized with degree one, a p-polarized ray carries no reflected flux and an
  s-polarized one carries ``R_s``;
* a slab tilted to 45 degrees (the research card's negative test): every ghost order
  summed by exhaustive splitting gives the s/p closed form
  ``(1/2)[(1 - R_s)/(1 + R_s) + (1 - R_p)/(1 + R_p)]``, not the scalar
  ``(1 - R)/(1 + R)``, which the same scene gives with the switch off;
* chapter 06's two-surface table (T-06-12) through the engine: the second
  surface's plane of incidence turned by psi about the beam;
* total internal reflection keeps the relative phase (T-06-9): a +45 degree
  linear state leaves with ``(u, v) = (cos D, -sin D)``, ``D = arg(r_p r_s*)``;
* the reference axis stays a unit vector perpendicular to the direction after
  every event (T-06-18), and a thin-film coating's Stokes reflectance is its
  scalar reflectance for an unpolarized ray.

Every scene is traced with exhaustive splitting, so each number is deterministic.
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
    NSQScene,
    PlaneGeometry,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential import polarization as P
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.coating_support import UnpolarizedThinFilmCoating
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.nonsequential.materials import VACUUM, NSQMaterial
from optiland.thin_film import ThinFilmStack

N_BK7 = 1.5168
U32 = 2.0**-24

LEGS = [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")]
LEG_IDS = [f"{b}-{p}" for b, p in LEGS]


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _configure(kind: str, precision: str) -> float:
    be.set_backend(kind)
    if kind == "torch":
        be.set_device("cpu")
        be.grad_mode.disable()
    be.set_precision(precision)
    return 1e-15 if precision == "float64" else 16 * U32


def _backend(kind: str, polarization="stokes"):
    if kind == "numpy":
        return NumpyBackend(seed=3, polarization=polarization)
    from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: PLC0415

    return TorchBackend(seed=3, polarization=polarization, allow_splitting=True)


def _glass(n=N_BK7):
    return NSQMaterial(optiland_material=IdealMaterial(n=n, k=0.0))


def _cs_facing(point, normal):
    """A coordinate system at ``point`` whose local +z is the unit ``normal`` (R = Rz Ry Rx, rz = 0)."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    rx = -math.asin(max(-1.0, min(1.0, n[1])))
    ry = math.atan2(n[0], n[2])
    return CoordinateSystem(x=point[0], y=point[1], z=point[2], rx=rx, ry=ry)


def _pencil(scene, medium=None, radius=0.001):
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=radius, medium=medium
        ),
    )


def _split(scene, depth):
    scene.sampling_policy = SamplingPolicy(split_depth=depth, split_budget=256.0, rr_start_flux=1e-300)


_ORIGINAL_INTERACT = RefractiveComponent.interact


class _Events:
    """Record every refractive interaction: the hit rays' flux, state, axis and direction after it."""

    def __init__(self, monkeypatch):
        self.rows = []
        original = _ORIGINAL_INTERACT
        signature = inspect.signature(original)

        def spy(comp, *args, **kwargs):
            original(comp, *args, **kwargs)
            bound = signature.bind(comp, *args, **kwargs)
            rays = bound.arguments["rays"]
            hit = np.asarray(be.to_numpy(bound.arguments["hit_mask"]), dtype=bool)
            if rays.pol_q is None or not hit.any():
                return

            def h(x):
                return np.asarray(be.to_numpy(x), dtype=np.float64)[hit]

            self.rows.append(
                dict(
                    name=comp.name,
                    branch=bound.arguments.get("forced_branch"),
                    flux=h(rays.flux),
                    q=h(rays.pol_q), u=h(rays.pol_u), v=h(rays.pol_v),
                    e=np.stack([h(rays.pol_ex), h(rays.pol_ey), h(rays.pol_ez)], axis=1),
                    k=np.stack([h(rays.L), h(rays.M), h(rays.N)], axis=1),
                )
            )

        monkeypatch.setattr(RefractiveComponent, "interact", spy)


def _fresnel(n1, n2, theta_deg):
    ci = math.cos(math.radians(theta_deg))
    st = n1 / n2 * math.sin(math.radians(theta_deg))
    ct = np.sqrt(complex(1 - st * st))
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    return rs, rp


# ---------------------------------------------------------------------------
# One interface at Brewster's angle
# ---------------------------------------------------------------------------


def _interface_scene(theta_deg, n1_medium=None, front=VACUUM, back=None, coating=None):
    scene = NSQScene()
    _pencil(scene, medium=n1_medium)
    rx = math.radians(theta_deg)
    comp = RefractiveComponent(
        CoordinateSystem(z=10.0, rx=rx), PlaneGeometry(), material_front=front,
        material_back=back if back is not None else _glass(), coating=coating, name="IF",
    )
    scene.add_component("IF", comp)
    # a detector the pencil never reaches: a scene needs one; the events are read directly
    scene.add_detector(
        "FAR",
        CoordinateSystem(x=500.0, z=500.0),
        IrradianceDetectorConfig(width=1, height=1, num_pixels_x=1, num_pixels_y=1, splat="hard"),
    )
    return scene


class TestBrewster:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_unpolarized_reflection_is_s_polarized(self, leg, monkeypatch):
        window = _configure(*leg)
        theta_b = math.degrees(math.atan(N_BK7))
        scene = _interface_scene(theta_b)
        _split(scene, 1)
        ev = _Events(monkeypatch)
        scene.trace(num_rays=4, seed=1, max_depth=1, min_flux_fraction=0.0, backend=_backend(leg[0]))
        refl = [r for r in ev.rows if r["branch"] == "reflect"][0]
        trans = [r for r in ev.rows if r["branch"] == "transmit"][0]
        rs, rp = _fresnel(1.0, N_BK7, theta_b)
        Rs, Rp = abs(rs) ** 2, abs(rp) ** 2
        tol_dop = 1e-12 if leg[1] == "float64" else 4 * window
        assert np.allclose(refl["flux"] * 4, 0.5 * (Rs + Rp), rtol=4 * window, atol=0)
        assert np.max(np.abs(refl["q"] + 1.0)) < tol_dop  # Q/I = -1: s-polarized
        assert np.max(np.abs(refl["u"])) < tol_dop and np.max(np.abs(refl["v"])) < tol_dop
        Ts, Tp = 1 - Rs, 1 - Rp
        assert np.max(np.abs(trans["q"] - (Tp - Ts) / (Tp + Ts))) < 8 * window + 1e-15

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_p_and_s_polarized_reflectance(self, leg, monkeypatch):
        """r1_08's physics: a p-polarized ray reflects nothing at Brewster; an s one reflects R_s."""
        window = _configure(*leg)
        theta_b = math.degrees(math.atan(N_BK7))
        rs, _ = _fresnel(1.0, N_BK7, theta_b)
        for q_in, want in ((1.0, 0.0), (-1.0, abs(rs) ** 2)):
            scene = _interface_scene(theta_b)
            _split(scene, 1)
            # p in the plane of incidence of a plane turned about x: the y axis
            P.set_source_polarization(scene, "S", stokes=(1.0, q_in, 0.0, 0.0), reference_axis=(0.0, 1.0, 0.0))
            ev = _Events(monkeypatch)
            scene.trace(num_rays=4, seed=1, max_depth=1, min_flux_fraction=0.0, backend=_backend(leg[0]))
            refl = [r for r in ev.rows if r["branch"] == "reflect"][0]
            got = refl["flux"] * 4
            if want == 0.0:
                assert np.max(np.abs(got)) < (1e-30 if leg[1] == "float64" else 1e-12)
            else:
                assert np.max(np.abs(got / want - 1.0)) < 8 * window


# ---------------------------------------------------------------------------
# The 45-degree slab: every ghost order, Stokes against scalar
# ---------------------------------------------------------------------------


def _slab_scene(theta_deg=45.0, thickness=2.0):
    scene = NSQScene()
    _pencil(scene)
    rx = math.radians(theta_deg)
    glass = _glass()
    scene.add_component(
        "F", RefractiveComponent(CoordinateSystem(z=10.0, rx=rx), PlaneGeometry(), VACUUM, glass, name="F")
    )
    scene.add_component(
        "B", RefractiveComponent(CoordinateSystem(z=10.0 + thickness, rx=rx), PlaneGeometry(), glass, VACUUM, name="B")
    )
    scene.add_detector(
        "T",
        CoordinateSystem(z=100.0),
        IrradianceDetectorConfig(width=400, height=400, num_pixels_x=1, num_pixels_y=1, splat="hard"),
    )
    _split(scene, 40)
    return scene


class TestTiltedSlab:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_stokes_gives_the_s_p_closed_form_and_scalar_does_not(self, leg):
        window = _configure(*leg)
        rs, rp = _fresnel(1.0, N_BK7, 45.0)
        Rs, Rp = abs(rs) ** 2, abs(rp) ** 2
        R = 0.5 * (Rs + Rp)
        closed_sp = 0.5 * ((1 - Rs) / (1 + Rs) + (1 - Rp) / (1 + Rp))
        closed_scalar = (1 - R) / (1 + R)
        stokes = _slab_scene().trace(num_rays=4, seed=1, max_depth=40, min_flux_fraction=0.0, backend=_backend(leg[0]))
        scalar = _slab_scene().trace(num_rays=4, seed=1, max_depth=40, min_flux_fraction=0.0, backend=_backend(leg[0], "off"))
        t_stokes = float(be.to_numpy(stokes.detectors["T"].total_flux))
        t_scalar = float(be.to_numpy(scalar.detectors["T"].total_flux))
        # 40 interactions of a few operations each: 64 u at float32 (3.8e-6)
        tol = 1e-14 if leg[1] == "float64" else 64 * U32
        assert abs(t_stokes / closed_sp - 1.0) < tol
        assert abs(t_scalar / closed_scalar - 1.0) < tol
        assert t_stokes - t_scalar > 0.003  # +0.36 %: Stokes mode is not scalar arithmetic


# ---------------------------------------------------------------------------
# Chapter 06's two-surface table through the engine (T-06-12)
# ---------------------------------------------------------------------------


def _two_surface_scene(psi_deg, n=1.5):
    scene = NSQScene()
    _pencil(scene)
    c = math.sqrt(0.5)
    glass = _glass(n)
    p1 = np.array([0.0, 0.0, 10.0])
    n1 = np.array([0.0, -c, c])  # 45 degrees; the beam leaves along +y
    k1 = np.array([0.0, 1.0, 0.0])
    p2 = p1 + 10.0 * k1
    ps = math.radians(psi_deg)
    n2 = c * k1 - c * np.array([math.sin(ps), 0.0, math.cos(ps)])
    k2 = np.array([math.sin(ps), 0.0, math.cos(ps)])
    scene.add_component(
        "S1", RefractiveComponent(_cs_facing(p1, n1), FinitePlaneGeometry(aperture_radius=1.0), VACUUM, glass, name="S1")
    )
    scene.add_component(
        "S2", RefractiveComponent(_cs_facing(p2, n2), FinitePlaneGeometry(aperture_radius=1.0), VACUUM, glass, name="S2")
    )
    scene.add_detector(
        "D",
        _cs_facing(p2 + 20.0 * k2, k2),
        IrradianceDetectorConfig(width=4, height=4, num_pixels_x=1, num_pixels_y=1, splat="hard"),
    )
    _split(scene, 3)
    return scene


_A = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]], dtype=complex)


def _jones_mueller(rp, rs):
    J = np.diag([rp, rs])
    return np.real(_A @ np.kron(J, J.conj()) @ np.linalg.inv(_A))


class TestTwoSurfaces:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_t06_12(self, leg):
        window = _configure(*leg)
        table = {0: 1.691358, 30: 1.345679, 45: 1.000000, 60: 0.654321, 90: 0.308642, 180: 1.691358}
        rs, rp = _fresnel(1.0, 1.5, 45.0)
        M = _jones_mueller(rp, rs)
        rbar = M[0, 0]
        for psi, printed in table.items():
            res = _two_surface_scene(float(psi)).trace(num_rays=4, seed=1, max_depth=3, min_flux_fraction=0.0, backend=_backend(leg[0]))
            got = float(be.to_numpy(res.detectors["D"].total_flux)) / res.total_flux_in
            ref = (M @ P.rotation_matrix(math.radians(psi)) @ M @ np.array([1.0, 0, 0, 0]))[0]
            tol = 1e-13 if leg[1] == "float64" else 64 * U32
            assert abs(got / ref - 1.0) < tol, (psi, got, ref)
            assert abs(got / rbar**2 - printed) < 5e-7 + 2 * tol

    def test_scalar_mode_is_psi_blind(self):
        """The same scenes with the switch off: R-bar squared at every psi (why the Stokes mode exists)."""
        _configure("numpy", "float64")
        rs, rp = _fresnel(1.0, 1.5, 45.0)
        rbar = 0.5 * (abs(rs) ** 2 + abs(rp) ** 2)
        for psi in (0.0, 45.0, 90.0):
            res = _two_surface_scene(psi).trace(num_rays=4, seed=1, max_depth=3, min_flux_fraction=0.0, backend=_backend("numpy", "off"))
            got = float(res.detectors["D"].total_flux) / res.total_flux_in
            assert abs(got / rbar**2 - 1.0) < 1e-13


# ---------------------------------------------------------------------------
# Total internal reflection keeps the relative phase (T-06-9)
# ---------------------------------------------------------------------------


class TestTotalInternalReflection:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_a_45_degree_state_leaves_elliptical(self, leg, monkeypatch):
        window = _configure(*leg)
        glass = _glass(1.5)
        scene = _interface_scene(45.0, n1_medium=glass, front=glass, back=VACUUM)
        _split(scene, 1)
        P.set_source_polarization(scene, "S", stokes=(1.0, 0.0, 1.0, 0.0), reference_axis=(0.0, 1.0, 0.0))
        ev = _Events(monkeypatch)
        scene.trace(num_rays=4, seed=1, max_depth=1, min_flux_fraction=0.0, backend=_backend(leg[0]))
        refl = [r for r in ev.rows if r["branch"] == "reflect"][0]
        rs, rp = _fresnel(1.5, 1.0, 45.0)
        x = rp * np.conj(rs)  # exp(-i 36.8699 deg)
        tol = 1e-14 if leg[1] == "float64" else 16 * window
        assert np.allclose(refl["flux"] * 4, 1.0, rtol=0, atol=tol)
        assert np.max(np.abs(refl["q"])) < tol
        assert np.max(np.abs(refl["u"] - x.real)) < tol  # 0.8
        assert np.max(np.abs(refl["v"] + x.imag)) < tol  # +0.6


# ---------------------------------------------------------------------------
# Frames, coatings, gradients
# ---------------------------------------------------------------------------


class TestFramesAndCoatings:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_t06_18_the_axis_stays_perpendicular(self, leg, monkeypatch):
        window = _configure(*leg)
        ev = _Events(monkeypatch)
        for psi in (0.0, 30.0, 90.0):
            _two_surface_scene(psi).trace(num_rays=4, seed=1, max_depth=3, min_flux_fraction=0.0, backend=_backend(leg[0]))
        _slab_scene().trace(num_rays=4, seed=1, max_depth=12, min_flux_fraction=0.0, backend=_backend(leg[0]))
        assert len(ev.rows) > 10
        tol = 1e-14 if leg[1] == "float64" else 16 * window
        for r in ev.rows:
            assert np.max(np.abs(np.sum(r["e"] * r["k"], axis=1))) < tol
            assert np.max(np.abs(np.sum(r["e"] * r["e"], axis=1) - 1.0)) < tol
            dop = np.sqrt(r["q"] ** 2 + r["u"] ** 2 + r["v"] ** 2)
            assert np.max(dop) <= 1.0 + tol  # R-06-7

    def test_coated_interface_unpolarized_is_scalar(self, monkeypatch):
        """A thin-film coating: Stokes R and T are the coating's scalar R and T for an unpolarized ray."""
        _configure("numpy", "float64")
        stack = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(1.5), reference_wl_um=0.55)
        for _ in range(5):
            stack.add_layer_qwot(IdealMaterial(2.32))
            stack.add_layer_qwot(IdealMaterial(1.38))
        coating = UnpolarizedThinFilmCoating(stack)
        fluxes = {}
        for mode in ("off", "stokes"):
            scene = _interface_scene(45.0, back=_glass(1.5), coating=coating)
            _split(scene, 1)
            ev = _Events(monkeypatch)
            scene.trace(num_rays=4, seed=1, max_depth=1, min_flux_fraction=0.0, backend=_backend("numpy", mode))
            fluxes[mode] = ev.rows
        refl = [r for r in fluxes["stokes"] if r["branch"] == "reflect"][0]
        assert abs(refl["flux"][0] * 4 - 0.9503515751153687) < 1e-15  # the r1_23 case value at 45 deg
        sp = P.thin_film_sp(stack, np.array([0.55]), np.array([math.cos(math.radians(45.0))]))
        want_q = (sp.Rp[0] - sp.Rs[0]) / (sp.Rp[0] + sp.Rs[0])
        assert abs(refl["q"][0] - want_q) < 1e-15

    def test_gradient_is_finite_at_normal_incidence(self):
        """R-09-10: the plane-of-incidence normalisation is masked; d(flux)/d(n) is finite at 0 deg.

        At 0, 1e-4 and 1 degree the Stokes-mode gradient of the transmitted flux
        with respect to the glass index is finite and equal to the scalar
        mode's (an unpolarized source, one interface: condition 1 of section
        6.9 holds for the derivative as for the value).
        """
        torch = pytest.importorskip("torch")
        from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: PLC0415

        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        be.grad_mode.enable()
        try:
            for theta in (0.0, 1e-4, 1.0):
                grads = {}
                for mode in ("off", "stokes"):
                    n = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
                    glass = NSQMaterial(optiland_material=IdealMaterial(n=n, k=0.0))
                    scene = NSQScene()
                    _pencil(scene)
                    scene.add_component(
                        "IF",
                        RefractiveComponent(
                            CoordinateSystem(z=10.0, rx=math.radians(theta)),
                            FinitePlaneGeometry(aperture_radius=5.0), VACUUM, glass, name="IF",
                        ),
                    )
                    scene.add_detector(
                        "T", CoordinateSystem(z=20.0),
                        IrradianceDetectorConfig(width=10, height=10, num_pixels_x=1, num_pixels_y=1, splat="hard"),
                    )
                    res = scene.trace(
                        num_rays=64, seed=1, max_depth=2, backend=TorchBackend(seed=1, polarization=mode)
                    )
                    (grads[mode],) = torch.autograd.grad(res.detectors["T"].total_flux, n)
                assert torch.isfinite(grads["stokes"]), theta
                assert float(grads["stokes"]) == float(grads["off"]), theta
        finally:
            be.grad_mode.disable()
