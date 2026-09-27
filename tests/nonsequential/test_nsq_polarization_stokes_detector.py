"""The polarization-resolving detector (the research repository's issue 5, item 8).

An irradiance detector built with ``stokes=True`` tallies, per pixel, ``Q = I q'``,
``U = I u'`` and ``V = I v'`` beside the flux, with ``(q', u', v')`` each ray's
state rotated into the detector's frame: the detector's local x axis projected
perpendicular to the ray. The expected values come from Jones vectors written in
the lab frame and read in that frame, never from the engine's own rotation:

* at normal incidence, for a detector turned about its normal, and for rays
  arriving 30 degrees off the normal;
* the per-pixel maps carry the state of every pixel under bilinear and Gaussian
  splats; the degree and the angle of polarization are derived on read;
* the tallies are float64 on the CPU whatever the working precision, a scalar
  trace leaves the result without Stokes maps, the flag round-trips through JSON
  and a detector without it writes the JSON it always did, and the emulated
  replay equals eager with the tallies on.
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
    IrradianceDetectorConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential import polarization as P
from optiland.nonsequential.backends.numpy_backend import NumpyBackend

U32 = 2.0**-24
WL = 0.55
REF = 20.0

LEGS = [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")]
LEG_IDS = [f"{b}-{p}" for b, p in LEGS]
STATES = [(0.5, math.sqrt(0.75), 0.0), (0.36, 0.48, 0.8), (-0.6, 0.0, -0.8)]


@pytest.fixture(autouse=True)
def _restore_backend(monkeypatch):
    monkeypatch.delenv(P.POLARIZATION_ENV, raising=False)
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _configure(kind: str, precision: str) -> float:
    be.set_backend(kind)
    if kind == "torch":
        be.set_device("cpu")
        be.grad_mode.disable()
    be.set_precision(precision)
    return 1e-14 if precision == "float64" else 64 * U32


def _backend(kind: str, polarization="stokes", **kw):
    if kind == "numpy":
        return NumpyBackend(seed=1, polarization=polarization)
    from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: PLC0415

    return TorchBackend(seed=1, polarization=polarization, **kw)


def _scene(stokes, det_cs, splat="hard", pixels=1, radius=0.5):
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(WL), total_flux=1.0, aperture_radius=radius),
    )
    if stokes is not None:
        r = math.radians(REF)
        P.set_source_polarization(scene, "S", stokes=stokes, reference_axis=(math.cos(r), math.sin(r), 0.0))
    scene.add_detector(
        "D", det_cs,
        IrradianceDetectorConfig(
            width=4, height=4, num_pixels_x=pixels, num_pixels_y=pixels, splat=splat, stokes=True
        ),
    )
    return scene


def _expected(q, u, v, x_axis, k=np.array([0.0, 0.0, 1.0])):
    """The reduced state of a pure source state, read along ``x_axis`` projected perpendicular to ``k``."""
    r = math.radians(REF)
    e = np.array([math.cos(r), math.sin(r), 0.0])
    a = math.sqrt((1 + q) / 2)
    b = math.sqrt((1 - q) / 2) * complex(math.cos(math.atan2(v, u)), math.sin(math.atan2(v, u)))
    E = a * e + b * np.cross(k, e)
    x = x_axis - np.dot(x_axis, k) * k
    x = x / np.linalg.norm(x)
    y = np.cross(k, x)
    Ex, Ey = np.dot(E, x), np.dot(E, y)
    X = Ex * np.conj(Ey)
    return np.array([abs(Ex) ** 2 - abs(Ey) ** 2, 2 * X.real, -2 * X.imag])


def _x_axis(cs):
    from optiland.nonsequential.components.base import _get_transform  # noqa: PLC0415

    _, R = _get_transform(cs)
    return np.asarray(R, dtype=float)[:, 0]


class TestFrame:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    @pytest.mark.parametrize("rz", [0.0, 35.0, -80.0])
    def test_normal_incidence(self, leg, rz):
        window = _configure(*leg)
        cs = CoordinateSystem(z=20.0, rz=math.radians(rz))
        for q, u, v in STATES:
            res = _scene((1.0, q, u, v), cs).trace(num_rays=8, seed=1, max_depth=2, backend=_backend(leg[0]))
            got = np.array(res.detectors["D"].stokes.reduced())
            want = _expected(q, u, v, _x_axis(cs))
            assert np.max(np.abs(got - want)) < window, (rz, (q, u, v), got, want)

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_oblique_arrival(self, leg):
        """Rays 30 degrees off the detector's normal: the x axis is projected perpendicular to them."""
        window = _configure(*leg)
        cs = CoordinateSystem(z=20.0, rx=math.radians(30.0), rz=math.radians(10.0))
        for q, u, v in STATES:
            res = _scene((1.0, q, u, v), cs).trace(num_rays=8, seed=1, max_depth=2, backend=_backend(leg[0]))
            got = np.array(res.detectors["D"].stokes.reduced())
            want = _expected(q, u, v, _x_axis(cs))
            assert np.max(np.abs(got - want)) < window, ((q, u, v), got, want)

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_unpolarized_light_reads_zero(self, leg):
        _configure(*leg)
        res = _scene(None, CoordinateSystem(z=20.0, rz=0.3)).trace(
            num_rays=64, seed=1, max_depth=2, backend=_backend(leg[0])
        )
        st = res.detectors["D"].stokes
        i, q, u, v = st.totals_float()
        assert i > 0 and q == 0.0 and u == 0.0 and v == 0.0
        assert st.degree_of_polarization() == 0.0


class TestMaps:
    @pytest.mark.parametrize("splat", ["bilinear", "gaussian", "hard"])
    def test_every_pixel_carries_the_state(self, splat):
        _configure("numpy", "float64")
        q, u, v = STATES[1]
        cs = CoordinateSystem(z=20.0)
        res = _scene((1.0, q, u, v), cs, splat=splat, pixels=8, radius=1.8).trace(
            num_rays=500, seed=1, max_depth=2, backend=_backend("numpy")
        )
        maps = res.detectors["D"].stokes.maps()
        lit = maps["I"] > 1e-9
        want = _expected(q, u, v, _x_axis(cs))
        for name, w in zip(("Q", "U", "V"), want, strict=True):
            assert np.max(np.abs(maps[name][lit] / maps["I"][lit] - w)) < 1e-13
        assert np.max(np.abs(maps["dop"][lit] - 1.0)) < 1e-13
        assert abs(maps["I"].sum() - float(res.detectors["D"].total_flux)) < 1e-15

    def test_degree_and_angle_on_read(self):
        _configure("numpy", "float64")
        res = _scene((1.0, 0.3, 0.4, 0.0), CoordinateSystem(z=20.0)).trace(
            num_rays=8, seed=1, max_depth=2, backend=_backend("numpy")
        )
        st = res.detectors["D"].stokes
        q, u, _ = _expected(0.6, 0.8, 0.0, np.array([1.0, 0.0, 0.0]))  # the pure state of the same direction
        assert abs(st.degree_of_polarization() - 0.5) < 1e-15
        assert abs(st.angle_of_polarization_deg() - math.degrees(0.5 * math.atan2(u, q))) < 1e-12


class TestBookkeeping:
    def test_float64_tallies_at_float32(self):
        _configure("torch", "float32")
        scene = _scene((1.0, 0.36, 0.48, 0.8), CoordinateSystem(z=20.0))
        scene.trace(num_rays=8, seed=1, max_depth=2, backend=_backend("torch"))
        det = scene.detectors[0]
        for buf in (det._stokes_q, det._stokes_u, det._stokes_v):
            assert str(buf.dtype) == "torch.float64"

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_a_scalar_trace_has_no_stokes_maps(self, leg):
        _configure(*leg)
        res = _scene((1.0, 0.36, 0.48, 0.8), CoordinateSystem(z=20.0)).trace(
            num_rays=8, seed=1, max_depth=2, backend=_backend(leg[0], "off")
        )
        assert res.detectors["D"].stokes is None
        assert res.detectors["D"].total_flux_float > 0

    def test_json(self, tmp_path):
        _configure("numpy", "float64")
        scene = _scene((1.0, 0.36, 0.48, 0.8), CoordinateSystem(z=20.0))
        scene.add_detector(
            "plain", CoordinateSystem(z=30.0), IrradianceDetectorConfig(width=4, height=4, num_pixels_x=2, num_pixels_y=2)
        )
        path = tmp_path / "scene.json"
        scene.to_json(path)
        dets = {d["name"]: d for d in json.loads(path.read_text())["detectors"]}
        assert dets["D"]["config"].get("stokes") is True if "config" in dets["D"] else dets["D"].get("stokes") is True
        plain = dets["plain"].get("config", dets["plain"])
        assert "stokes" not in plain
        again = NSQScene.from_json(path)
        assert again.detectors[0].stokes is True and again.detectors[1].stokes is False

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_emulated_replay_equals_eager(self, precision):
        """A beam walking out of a two-mirror cavity onto the detector: the tallies are
        added into in place and never rebound, so the replayed bounces fill them as eager does."""
        pytest.importorskip("torch")
        from optiland.nonsequential import FinitePlaneGeometry, ReflectiveComponent  # noqa: PLC0415

        _configure("torch", precision)

        def run(**kw):
            scene = NSQScene()
            scene.add_source(
                "S", CoordinateSystem(z=2.5, rx=math.radians(10.0)),
                CollimatedSourceConfig(spectrum=Spectrum.monochromatic(WL), total_flux=1.0, aperture_radius=1.0),
            )
            P.set_source_polarization(scene, "S", stokes=(1.0, 0.36, 0.48, 0.8), reference_axis=(1.0, 0.0, 0.0))
            for name, z in (("A", 0.0), ("B", 5.0)):
                scene.add_component(
                    name,
                    ReflectiveComponent(CoordinateSystem(z=z), FinitePlaneGeometry(aperture_radius=20.0), 0.9, name=name),
                )
            scene.add_detector(
                "D", CoordinateSystem(y=-40.0, z=2.5, rx=math.radians(90.0)),
                IrradianceDetectorConfig(
                    width=2000, height=2000, num_pixels_x=4, num_pixels_y=4, splat="bilinear", stokes=True
                ),
            )
            res = scene.trace(num_rays=4096, seed=3, max_depth=24, batch_size=2048, backend=_backend("torch", **kw))
            st = res.detectors["D"].stokes
            return res, [np.asarray(be.to_numpy(x)).tobytes() for x in (st.i, st.q, st.u, st.v)]

        eager, a = run(compact_every=0)
        replay, b = run(graph_replay="emulate")
        assert replay.environment["graph_replay"] == "emulate" and replay.environment["graph_replay_batches"] > 0
        assert eager.detectors["D"].stokes.totals_float()[0] > 0
        assert a == b
