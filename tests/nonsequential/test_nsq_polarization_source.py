"""Source polarization (the research repository's issue 5, build item 4).

A source may carry a :class:`~optiland.nonsequential.polarization.SourcePolarization`
(a Stokes vector and the global axis ``S1 > 0`` refers to), set with
:func:`~optiland.nonsequential.polarization.set_source_polarization`. It is read
only in Stokes mode, at birth; the scalar trace ignores it (one set of model
inputs for both modes, R-06-9). The linear and circular shorthands of the
research card are the cut item and are not built.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    PointSourceConfig,
    Spectrum,
)
from optiland.nonsequential import polarization as P
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.serialization import scene_from_dict, scene_to_dict


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _scene(point: bool = False) -> NSQScene:
    scene = NSQScene()
    if point:
        scene.add_source(
            "S", CoordinateSystem(), PointSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0)
        )
    else:
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=1.0),
        )
    scene.add_detector(
        "D",
        CoordinateSystem(z=10.0),
        IrradianceDetectorConfig(width=40, height=40, num_pixels_x=4, num_pixels_y=4, splat="hard"),
    )
    return scene


def _birth(scene, backend, monkeypatch) -> dict:
    captured = {}
    original = P.prepare_bundle

    def spy(rays, source=None):
        original(rays, source)
        captured.update({f: np.asarray(be.to_numpy(getattr(rays, f))) for f in P.POL_FIELDS})
        captured["k"] = np.stack([be.to_numpy(c) for c in (rays.L, rays.M, rays.N)], axis=1)

    monkeypatch.setattr(P, "prepare_bundle", spy)
    scene.trace(num_rays=300, seed=1, backend=backend, max_depth=2)
    return captured


class TestSourcePolarization:
    def test_a_stokes_vector_and_its_axis(self, monkeypatch):
        """(2, 1, 0, 1) along lab y: q = 0.5, v = 0.5, e = y for a beam along z."""
        scene = _scene()
        P.set_source_polarization(scene, "S", stokes=(2.0, 1.0, 0.0, 1.0), reference_axis=(0.0, 3.0, 0.0))
        got = _birth(scene, NumpyBackend(seed=1, polarization="stokes"), monkeypatch)
        assert np.all(got["pol_q"] == 0.5) and np.all(got["pol_u"] == 0.0) and np.all(got["pol_v"] == 0.5)
        assert np.all(got["pol_ex"] == 0.0) and np.all(got["pol_ey"] == 1.0) and np.all(got["pol_ez"] == 0.0)

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_torch_places_the_state_beside_the_rays(self, precision, monkeypatch):
        pytest.importorskip("torch")
        from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: PLC0415

        be.set_backend("torch")
        be.set_device("cpu")
        be.grad_mode.disable()
        be.set_precision(precision)
        scene = _scene()
        P.set_source_polarization(scene, "S", stokes=(1.0, 0.0, 1.0, 0.0))
        got = _birth(scene, TorchBackend(seed=1, polarization="stokes"), monkeypatch)
        assert np.all(got["pol_u"] == 1.0) and got["pol_u"].dtype == np.dtype(precision)
        assert np.all(got["pol_ex"] == 1.0)

    def test_axis_projected_perpendicular_to_each_ray(self, monkeypatch):
        """A point source: e is the axis projected perpendicular to k, a unit vector."""
        scene = _scene(point=True)
        P.set_source_polarization(scene, "S", stokes=(1.0, 1.0, 0.0, 0.0), reference_axis=(1.0, 0.0, 0.0))
        got = _birth(scene, NumpyBackend(seed=1, polarization="stokes"), monkeypatch)
        e = np.stack([got["pol_ex"], got["pol_ey"], got["pol_ez"]], axis=1)
        assert np.max(np.abs(np.sum(e * got["k"], axis=1))) < 1e-14
        assert np.max(np.abs(np.sum(e * e, axis=1) - 1.0)) < 1e-14
        # rays whose direction is within the tolerance of the axis fall back to the default
        k = got["k"]
        along = np.abs(k[:, 0]) > 1.0 - 1e-15
        assert np.all(e[~along, 0] >= 0.0)

    def test_scalar_mode_ignores_it(self):
        """R-06-9: the scalar trace of a scene with a polarized source is the scene without one."""
        a = _scene().trace(num_rays=500, seed=3, backend=NumpyBackend(seed=3), max_depth=2)
        scene = _scene()
        P.set_source_polarization(scene, "S", stokes=(1.0, -1.0, 0.0, 0.0))
        b = scene.trace(num_rays=500, seed=3, backend=NumpyBackend(seed=3), max_depth=2)
        assert a.detectors["D"].irradiance.tobytes() == b.detectors["D"].irradiance.tobytes()

    def test_realizability_and_s0_are_checked(self):
        with pytest.raises(ValueError, match="realizable"):
            P.SourcePolarization((1.0, 1.0, 0.5, 0.0))
        with pytest.raises(ValueError, match="S0"):
            P.SourcePolarization((0.0, 0.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            P.SourcePolarization((1.0, 0.0, 0.0, 0.0), reference_axis=(0.0, 0.0, 0.0))


class TestSerialization:
    def test_round_trip(self):
        scene = _scene()
        pol = P.set_source_polarization(scene, "S", stokes=(1.0, 0.0, 0.0, -1.0), reference_axis=(0.0, 1.0, 0.0))
        d = scene_to_dict(scene)
        assert d["sources"][0]["polarization"] == {"stokes": [1.0, 0.0, 0.0, -1.0], "reference_axis": [0.0, 1.0, 0.0]}
        back = scene_from_dict(d)
        assert back.source_registry.get("S").polarization == pol

    def test_a_scene_without_one_serializes_as_before(self):
        d = scene_to_dict(_scene())
        assert "polarization" not in d["sources"][0]
        assert getattr(scene_from_dict(d).source_registry.get("S"), "polarization", None) is None
