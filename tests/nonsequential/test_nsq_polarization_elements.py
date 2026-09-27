"""The ideal polarizer and the ideal retarder as component kinds (the research repository's issue 5, item 7).

Expected values are closed forms (chapter 06 section 6.5, and the analytic
functions the catalogue's r1_27 and r1_29 are graded against):

* Malus's law, ``I = (1 + cos 2 theta) / 4`` behind two ideal polarizers at
  ``theta``, unpolarized input; a crossed pair passes nothing; a 45-degree
  polarizer between them passes 1/8; a finite extinction ``e`` lets a crossed
  pair pass ``e``;
* the quarter-wave retarder: ``(1, 1, 0, 0)`` leaves as ``(1, 0, 0, -1)`` with the
  fast axis at +45 degrees and ``(1, 0, 0, +1)`` at -45 degrees; light along the
  fast axis is left alone; a half wave at 45 degrees flips ``Q``; the retardance
  scales as ``1 / lambda`` (a quarter wave at 0.5 um is an eighth at 1 um);
* in a scalar trace a polarizer passes ``(1 + e) / 2`` and a retarder everything,
  and the ledger books what a polarizer stops as a surface loss;
* the kinds round-trip through the scene's JSON and lower to the scene IR as
  ``"polarizing"``; an axis angle given as a tensor carries a gradient.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    PolarizerConfig,
    RetarderConfig,
    Spectrum,
)
from optiland.nonsequential import polarization as P
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.polarizing import PolarizingComponent
from optiland.nonsequential.ir.lower import lower

U32 = 2.0**-24
WL = 0.5876

LEGS = [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")]
LEG_IDS = [f"{b}-{p}" for b, p in LEGS]


@pytest.fixture(autouse=True)
def _restore_backend(monkeypatch):
    monkeypatch.delenv(P.POLARIZATION_ENV, raising=False)
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _configure(kind: str, precision: str) -> float:
    """Set the leg; return the window of a short chain (64 rounding steps) in its dtype."""
    be.set_backend(kind)
    if kind == "torch":
        be.set_device("cpu")
        be.grad_mode.disable()
    be.set_precision(precision)
    return 1e-14 if precision == "float64" else 64 * U32


def _backend(kind: str, polarization="stokes"):
    if kind == "numpy":
        return NumpyBackend(seed=1, polarization=polarization)
    from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: PLC0415

    return TorchBackend(seed=1, polarization=polarization)


def _beam(scene, stokes=None, wl=WL):
    scene.add_source(
        "S", CoordinateSystem(),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(wl), total_flux=1.0, aperture_radius=2.5),
    )
    if stokes is not None:
        P.set_source_polarization(scene, "S", stokes=stokes, reference_axis=(1.0, 0.0, 0.0))


def _detector(scene, stokes=True):
    scene.add_detector(
        "D", CoordinateSystem(z=50.0),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=1, num_pixels_y=1, splat="hard", stokes=stokes),
    )


def _chain(axes, extinction=0.0):
    scene = NSQScene()
    _beam(scene)
    for i, axis in enumerate(axes):
        scene.add_polarizer(
            f"P{i}", CoordinateSystem(z=10.0 * (i + 1)),
            PolarizerConfig(axis_deg=axis, aperture_radius=5.0, extinction=extinction),
        )
    _detector(scene)
    return scene


def _passed(result):
    return float(be.to_numpy(result.detectors["D"].total_flux)) / result.total_flux_in


class TestPolarizer:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_malus_law(self, leg):
        window = _configure(*leg)
        for theta in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
            res = _chain([0.0, theta]).trace(num_rays=4, seed=1, max_depth=8, backend=_backend(leg[0]))
            want = (1.0 + math.cos(2.0 * math.radians(theta))) / 4.0
            assert abs(_passed(res) - want) < window, (theta, _passed(res), want)

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_crossed_pair_and_three_polarizers(self, leg):
        window = _configure(*leg)
        crossed = _chain([0.0, 90.0]).trace(num_rays=4, seed=1, max_depth=8, backend=_backend(leg[0]))
        assert _passed(crossed) == 0.0
        three = _chain([0.0, 45.0, 90.0]).trace(num_rays=4, seed=1, max_depth=8, backend=_backend(leg[0]))
        assert abs(_passed(three) - 0.125) < window

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_finite_extinction(self, leg):
        window = _configure(*leg)
        e = 0.01
        res = _chain([0.0, 90.0], extinction=e).trace(num_rays=4, seed=1, max_depth=8, backend=_backend(leg[0]))
        assert abs(_passed(res) - e) < window
        one = _chain([30.0], extinction=e).trace(num_rays=4, seed=1, max_depth=8, backend=_backend(leg[0]))
        stokes = one.detectors["D"].stokes
        # unpolarized in: (1 + e) / 2 passes, polarized along 30 degrees with DoP (1 - e) / (1 + e)
        assert abs(_passed(one) - 0.5 * (1 + e)) < window
        assert abs(stokes.degree_of_polarization() - (1 - e) / (1 + e)) < window
        assert abs(stokes.angle_of_polarization_deg() - 30.0) < (1e-12 if leg[1] == "float64" else 1e-4)

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_the_ledger_books_what_a_polarizer_stops(self, leg):
        _configure(*leg)
        res = _chain([0.0, 60.0]).trace(num_rays=4, seed=1, max_depth=8, backend=_backend(leg[0]))
        assert abs(res.total_flux_coating - (1.0 - _passed(res))) < 1e-6
        assert abs(res.flux_conservation_error) < 1e-6


def _retarder_scene(stokes, fast_axis_deg, waves=0.25, wl=WL, design=WL):
    scene = NSQScene()
    _beam(scene, stokes=stokes, wl=wl)
    scene.add_retarder(
        "R", CoordinateSystem(z=10.0),
        RetarderConfig(fast_axis_deg=fast_axis_deg, retardance_waves=waves, aperture_radius=5.0, design_wavelength_um=design),
    )
    _detector(scene)
    return scene


def _reduced(scene, leg):
    res = scene.trace(num_rays=4, seed=1, max_depth=8, backend=_backend(leg[0]))
    return np.array(res.detectors["D"].stokes.reduced()), _passed(res)


class TestRetarder:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_quarter_wave_and_the_sign_of_v(self, leg):
        window = _configure(*leg)
        for axis, v in ((45.0, -1.0), (-45.0, 1.0)):
            got, passed = _reduced(_retarder_scene((1, 1, 0, 0), axis), leg)
            assert np.max(np.abs(got - [0.0, 0.0, v])) < window, (axis, got)
            assert abs(passed - 1.0) < window

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_fast_axis_light_and_the_half_wave(self, leg):
        window = _configure(*leg)
        got, _ = _reduced(_retarder_scene((1, 0, 1, 0), 45.0), leg)
        assert np.max(np.abs(got - [0.0, 1.0, 0.0])) < window
        got, _ = _reduced(_retarder_scene((1, 1, 0, 0), 45.0, waves=0.5), leg)
        assert np.max(np.abs(got - [-1.0, 0.0, 0.0])) < window

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_the_retardance_scales_as_one_over_lambda(self, leg):
        window = _configure(*leg)
        got, _ = _reduced(_retarder_scene((1, 1, 0, 0), 45.0, wl=1.0, design=0.5), leg)
        d = math.pi / 4.0
        assert np.max(np.abs(got - [math.cos(d), 0.0, -math.sin(d)])) < window
        got, _ = _reduced(_retarder_scene((1, 1, 0, 0), 45.0, wl=1.0, design=None), leg)
        assert np.max(np.abs(got - [0.0, 0.0, -1.0])) < window


class TestScalarMode:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_unpolarized_transmittance(self, leg):
        window = _configure(*leg)
        for e in (0.0, 0.2):
            res = _chain([0.0, 90.0], extinction=e).trace(num_rays=4, seed=1, max_depth=8, backend=_backend(leg[0], "off"))
            # a scalar trace cannot see the crossing: each passes (1 + e) / 2 (not a model of the chain)
            assert abs(_passed(res) - (0.5 * (1 + e)) ** 2) < window
            assert res.detectors["D"].stokes is None
        res = _retarder_scene((1, 1, 0, 0), 45.0).trace(num_rays=4, seed=1, max_depth=8, backend=_backend(leg[0], "off"))
        assert _passed(res) == 1.0

    def test_unpolarized_stokes_through_one_polarizer_equals_scalar(self):
        """One polarizer and unpolarized light: the Stokes flux is the scalar flux bit for bit."""
        _configure("numpy", "float64")
        on = _chain([25.0], extinction=0.1).trace(num_rays=64, seed=1, max_depth=8, backend=_backend("numpy", "stokes"))
        off = _chain([25.0], extinction=0.1).trace(num_rays=64, seed=1, max_depth=8, backend=_backend("numpy", "off"))
        assert float(on.detectors["D"].total_flux).hex() == float(off.detectors["D"].total_flux).hex()


class TestRegistration:
    def test_json_round_trip(self, tmp_path):
        _configure("numpy", "float64")
        scene = _chain([0.0, 30.0], extinction=0.05)
        scene.add_retarder(
            "R", CoordinateSystem(z=35.0),
            RetarderConfig(fast_axis_deg=22.5, retardance_waves=0.25, aperture_radius=5.0, design_wavelength_um=WL),
        )
        path = tmp_path / "scene.json"
        scene.to_json(path)
        again = NSQScene.from_json(path)
        a = scene.trace(num_rays=4, seed=1, max_depth=8, backend=_backend("numpy"))
        b = again.trace(num_rays=4, seed=1, max_depth=8, backend=_backend("numpy"))
        assert _passed(a) == _passed(b)
        assert a.detectors["D"].stokes.totals_float() == b.detectors["D"].stokes.totals_float()

    def test_lowered_kind_and_params(self):
        _configure("numpy", "float64")
        ir = lower(_retarder_scene((1, 1, 0, 0), 45.0), strict=False)
        prim = [p for p in ir.primitives if p.component_kind == "polarizing"]
        assert len(prim) == 1
        assert prim[0].params["element"] == "retarder" and prim[0].params["retardance_waves"] == 0.25

    def test_axis_gradient(self):
        """The axis angle is attached: dI/dtheta of Malus's law, -sin(2 theta) / 2 per radian."""
        torch = pytest.importorskip("torch")
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        be.grad_mode.enable()
        try:
            theta = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
            scene = _chain([0.0, theta])
            res = scene.trace(num_rays=4, seed=1, max_depth=8, backend=_backend("torch"))
            res.detectors["D"].total_flux.backward()
            want = -math.sin(2 * math.radians(30.0)) / 2.0 * math.pi / 180.0
            assert abs(float(theta.grad) - want) < 1e-14
        finally:
            be.grad_mode.disable()

    def test_a_bad_element_is_refused(self):
        with pytest.raises(ValueError):
            PolarizingComponent(CoordinateSystem(), "waveplate", 0.0, 1.0)
        with pytest.raises(ValueError):
            PolarizingComponent(CoordinateSystem(), "polarizer", 0.0, 1.0, extinction=1.5)
