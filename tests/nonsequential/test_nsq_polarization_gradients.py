"""Gradients through the Stokes events (the research repository's issue 5, item 10).

The gradient classes of the research card's item 10: the state and its axis are
attached through directions, normals and indices; the polarizer's axis and the
retarder's retardance are attached parameters of their kinds; the branch
probability is detached and the weight attached. What is pinned here, on torch
float64 with autograd, each against a central finite difference of the same
deterministic trace or against the closed form's derivative:

* the Fresnel event with a polarized ray, d(transmitted flux)/dn at 0, 1e-4, 1
  and 45 degrees: finite at and near normal incidence, where the plane of
  incidence is normalised behind a double ``where`` (R-09-10), and equal to the
  finite difference;
* at Brewster's angle, d(reflected flux)/dn for p-polarized light is zero (``R_p``
  has a double root there) and for unpolarized light it is ``(1/2) dR_s/dn``;
* dI/d(axis) of Malus's law against ``-sin(2 theta) / 2`` per radian at seven angles,
  finite at a crossed pair, where the state is zeroed behind a double ``where``;
* dV/dW of a retarder at 45 degrees, ``-2 pi cos(2 pi W)``, through the Stokes
  detector's V tally.

Every trace is deterministic: the Fresnel branch probability is pinned to 0 or 1
(``reflect_prob``; the weight then carries ``T / (1 - p)`` or ``R / p`` with
``p`` clipped a factor 1e-12 from the end, a constant the finite difference
shares), so a finite difference and the autograd derivative see one function.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import optiland.backend as be  # noqa: E402
from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.materials import IdealMaterial  # noqa: E402
from optiland.nonsequential import (  # noqa: E402
    CollimatedSourceConfig,
    FinitePlaneGeometry,
    IrradianceDetectorConfig,
    NSQScene,
    PolarizerConfig,
    RefractiveComponent,
    RetarderConfig,
    Spectrum,
)
from optiland.nonsequential import polarization as P  # noqa: E402
from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: E402
from optiland.nonsequential.ir.scene_ir import SamplingPolicy  # noqa: E402
from optiland.nonsequential.materials import VACUUM, NSQMaterial  # noqa: E402

WL = 0.5876
STATE = (1.0, 0.36, 0.48, 0.8)
REF = (math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0)


@pytest.fixture(autouse=True)
def _torch_float64_with_grad(monkeypatch):
    monkeypatch.delenv(P.POLARIZATION_ENV, raising=False)
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.enable()
    yield
    be.grad_mode.disable()
    be.set_backend("numpy")
    be.set_precision("float64")


def _t(x):
    return torch.tensor(float(x), dtype=torch.float64, requires_grad=True)


def _value(x):
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def _fd(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def _source(scene, stokes=STATE, ref=REF):
    scene.add_source(
        "S", CoordinateSystem(),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(WL), total_flux=1.0, aperture_radius=0.01),
    )
    if stokes is not None:
        P.set_source_polarization(scene, "S", stokes=stokes, reference_axis=ref)


def _interface_flux(n, theta_deg, branch, stokes=STATE, ref=REF):
    """The flux on the chosen Fresnel branch of one interface (branch probability pinned)."""
    scene = NSQScene()
    _source(scene, stokes, ref)
    t = math.radians(theta_deg)
    glass = NSQMaterial(optiland_material=IdealMaterial(n=n, k=0.0))
    scene.add_component(
        "IF",
        RefractiveComponent(CoordinateSystem(z=10.0, ry=t), FinitePlaneGeometry(aperture_radius=5.0), VACUUM, glass, name="IF"),
    )
    if branch == "transmit":
        cs = CoordinateSystem(z=30.0)
    else:
        kx, kz = -math.sin(2 * t), -math.cos(2 * t)
        cs = CoordinateSystem(x=20.0 * kx, z=10.0 + 20.0 * kz, ry=math.atan2(kx, kz))
    scene.add_detector("D", cs, IrradianceDetectorConfig(width=40, height=40, num_pixels_x=1, num_pixels_y=1, splat="hard"))
    scene.sampling_policy = SamplingPolicy(reflect_prob=0.0 if branch == "transmit" else 1.0, rr_start_flux=1e-30)
    res = scene.trace(num_rays=4, seed=1, max_depth=2, backend=TorchBackend(seed=1, polarization="stokes"))
    return res.detectors["D"].total_flux


class TestFresnel:
    @pytest.mark.parametrize("theta", [0.0, 1e-4, 1.0, 45.0])
    def test_transmission_gradient_at_and_near_normal_incidence(self, theta):
        n = _t(1.5)
        (g,) = torch.autograd.grad(_interface_flux(n, theta, "transmit"), n)
        assert torch.isfinite(g)
        fd = _fd(lambda x: _value(_interface_flux(x, theta, "transmit")), 1.5, 1e-6)
        assert abs(float(g) - fd) < 1e-8 * max(1.0, abs(fd)), (theta, float(g), fd)

    def test_brewster_reflection(self):
        n0 = 1.5168
        theta_b = math.degrees(math.atan(n0))
        # p-polarized (lab x is p here): R_p has a double root at Brewster's angle
        n = _t(n0)
        (g_p,) = torch.autograd.grad(_interface_flux(n, theta_b, "reflect", (1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), n)
        assert torch.isfinite(g_p) and abs(float(g_p)) < 1e-12
        # unpolarized: (1/2) dR_s/dn, against the finite difference
        n = _t(n0)
        (g_u,) = torch.autograd.grad(_interface_flux(n, theta_b, "reflect", None), n)
        fd = _fd(lambda x: _value(_interface_flux(x, theta_b, "reflect", None)), n0, 1e-6)
        assert abs(float(g_u) - fd) < 1e-8, (float(g_u), fd)


def _malus(theta):
    scene = NSQScene()
    _source(scene, None)
    scene.add_polarizer("P1", CoordinateSystem(z=10.0), PolarizerConfig(axis_deg=0.0, aperture_radius=5.0))
    scene.add_polarizer("P2", CoordinateSystem(z=20.0), PolarizerConfig(axis_deg=theta, aperture_radius=5.0))
    scene.add_detector(
        "D", CoordinateSystem(z=50.0),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=1, num_pixels_y=1, splat="hard", stokes=True),
    )
    res = scene.trace(num_rays=4, seed=1, max_depth=8, backend=TorchBackend(seed=1, polarization="stokes"))
    return res.detectors["D"]


class TestPolarizer:
    @pytest.mark.parametrize("deg", [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    def test_malus_axis_gradient(self, deg):
        theta = _t(deg)
        (g,) = torch.autograd.grad(_malus(theta).total_flux, theta)
        want = -math.sin(2 * math.radians(deg)) / 2.0 * math.pi / 180.0
        assert torch.isfinite(g) and abs(float(g) - want) < 1e-14, (deg, float(g), want)

    def test_crossed_pair_state_gradient_is_finite(self):
        """g = 0 at a crossed pair: the state is zeroed behind a double where; Q and U stay differentiable."""
        theta = _t(90.0)
        st = _malus(theta).stokes
        i, q, u, v = st.totals()
        (g,) = torch.autograd.grad(i + q + u + v, theta)
        assert torch.isfinite(g) and abs(float(g)) < 1e-14


class TestRetarder:
    @pytest.mark.parametrize("w0", [0.2, 0.25, 0.3])
    def test_dv_dw(self, w0):
        w = _t(w0)
        scene = NSQScene()
        _source(scene, (1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        scene.add_retarder(
            "R", CoordinateSystem(z=10.0),
            RetarderConfig(fast_axis_deg=45.0, retardance_waves=w, aperture_radius=5.0, design_wavelength_um=WL),
        )
        scene.add_detector(
            "D", CoordinateSystem(z=50.0),
            IrradianceDetectorConfig(width=20, height=20, num_pixels_x=1, num_pixels_y=1, splat="hard", stokes=True),
        )
        res = scene.trace(num_rays=4, seed=1, max_depth=8, backend=TorchBackend(seed=1, polarization="stokes"))
        i, _, _, v = res.detectors["D"].stokes.totals()
        (g,) = torch.autograd.grad(v / i, w)
        want = -2.0 * math.pi * math.cos(2.0 * math.pi * w0)
        assert abs(float(g) - want) < 1e-12, (w0, float(g), want)
