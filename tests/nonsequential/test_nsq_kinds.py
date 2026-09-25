"""The kind registries (the extension seam): built-ins, a plug-in kind end to
end on both backends, the gradient rule, refusals and entry-point loading."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    FarFieldDetector,
    FarFieldDetectorConfig,
    IrradianceDetectorConfig,
    NSQScene,
    PointSourceConfig,
    Spectrum,
    kinds,
)
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.ir import lower, scene_ir_from_dict, scene_ir_to_dict
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.rng import EventSlot
from optiland.nonsequential.serialization import scene_from_dict, scene_to_dict
from optiland.nonsequential.sources.base import BaseNSQSource

# ---------------------------------------------------------------------------
# A plug-in source kind, defined outside the engine: rays leave a ring of
# radius ``radius`` along the local +z axis.
# ---------------------------------------------------------------------------


@dataclass
class RingSourceConfig:
    spectrum: Spectrum
    total_flux: float = 1.0
    radius: float = 1.0
    medium: object = field(default=None)


class RingSource(BaseNSQSource):
    def __init__(self, cs, spectrum, total_flux=1.0, radius=1.0, medium=None):
        super().__init__(cs, spectrum, total_flux)
        self.radius = float(radius)
        self.medium = medium

    def generate(self, ray_id, rng):
        n = len(ray_id)
        bounce0 = np.zeros(n, dtype=np.int32)
        translation, rot = _get_transform(self.cs)
        phi = 2.0 * np.pi * to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1))
        pos = np.stack(
            [self.radius * np.cos(phi), self.radius * np.sin(phi), np.zeros(n)], axis=1
        ) @ rot.T + translation
        d = np.tile(rot[:, 2], (n, 1))
        return NSQRayBundle(
            x=pos[:, 0].copy(), y=pos[:, 1].copy(), z=pos[:, 2].copy(),
            L=d[:, 0].copy(), M=d[:, 1].copy(), N=d[:, 2].copy(),
            flux=be.ones(n) * (self.total_flux / n),
            wavelength=self.spectrum.sample(ray_id, bounce0, rng),
            n_current=np.ones(n), bounce=bounce0, alive=np.ones(n, dtype=bool),
            ray_id=ray_id, k_current=np.zeros(n),
        )


def _register_ring(attached=("total_flux",)):
    return kinds.register_source(
        "test_ring",
        RingSource,
        RingSourceConfig,
        build=lambda cs, c: RingSource(cs, c.spectrum, c.total_flux, c.radius, c.medium),
        to_dict=lambda s: {"radius": s.radius},
        from_dict=lambda d, spectrum, total_flux, medium: RingSourceConfig(
            spectrum=spectrum, total_flux=total_flux, radius=d["radius"], medium=medium
        ),
        lower=lambda s: {"radius": s.radius},
        attached=attached,
        overwrite=True,
    )


@pytest.fixture(autouse=True)
def _numpy_float64():
    be.set_backend("numpy")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


@pytest.fixture
def ring_kind():
    spec = _register_ring()
    yield spec
    kinds.SOURCES.unregister("test_ring")


def _ring_scene(radius=2.0, flux=3.0):
    scene = NSQScene()
    scene.add_source(
        "ring",
        CoordinateSystem(z=-10.0),
        RingSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=flux, radius=radius),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=5.0),
        IrradianceDetectorConfig(width=10.0, height=10.0, num_pixels_x=10, num_pixels_y=10, splat="hard"),
    )
    return scene


# ---------------------------------------------------------------------------
# Built-ins
# ---------------------------------------------------------------------------


def test_builtin_kinds_are_registered():
    reg = kinds.registered_kinds()
    assert reg["source"][:3] == ("point", "collimated", "extended")
    assert "tabulated" in reg["source"]
    assert set(reg["detector"]) >= {
        "irradiance", "spectral", "far_field", "hemisphere", "ray_database"
    }
    assert set(reg["geometry"]) >= {
        "conic", "paraboloid", "plane", "infinite_plane", "annulus", "frustum",
        "sphere", "spherical_cavity", "mesh",
    }
    assert set(reg["bsdf"]) >= {"lambertian", "harvey_shack", "tabulated", "specular"}
    assert set(reg["component"]) >= {"lens", "mirror", "doublet", "prism", "paraxial_lens"}
    assert set(reg["spectrum"]) >= {"lines", "piecewise_linear"}


def test_line_spectrum_json_has_no_kind_key():
    """Files written before other spectrum kinds existed must read back unchanged."""
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(), PointSourceConfig(spectrum=Spectrum.monochromatic(0.5))
    )
    d = scene_to_dict(scene)
    assert d["sources"][0]["spectrum"] == {"wavelengths": [0.5], "weights": [1.0]}
    assert list(d["sources"][0]) == [
        "type", "name", "cs", "spectrum", "total_flux", "half_angle_deg", "medium"
    ]


def test_paraboloid_and_infinite_plane_lower_to_their_ir_kinds():
    from optiland.nonsequential import ParaboloidGeometry, PlaneGeometry  # noqa: PLC0415
    from optiland.nonsequential.ir.lower import _lower_geometry  # noqa: PLC0415

    assert _lower_geometry(ParaboloidGeometry(radius=10.0, aperture_radius=3.0))[0] == "conic"
    assert _lower_geometry(PlaneGeometry()) == (
        "plane", {"width": None, "height": None, "aperture_radius": None}
    )


# ---------------------------------------------------------------------------
# A plug-in kind, end to end
# ---------------------------------------------------------------------------


def test_plugin_source_round_trips_through_json_and_ir(ring_kind):
    scene = _ring_scene()
    d = scene_to_dict(scene)
    src = d["sources"][0]
    assert src["type"] == "test_ring" and src["radius"] == 2.0
    rebuilt = scene_from_dict(json.loads(json.dumps(d)))
    assert isinstance(rebuilt.sources[0], RingSource)
    assert scene_to_dict(rebuilt) == d
    ir = lower(scene, strict=False)
    assert ir.emitters[0].kind == "test_ring"
    assert ir.emitters[0].params["radius"] == 2.0
    assert scene_ir_from_dict(scene_ir_to_dict(ir)).emitters[0].params["radius"] == 2.0


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_plugin_source_traces_on_both_backends(ring_kind, backend):
    if backend == "torch":
        pytest.importorskip("torch")
    be.set_backend(backend)
    be.set_precision("float64")
    try:
        result = _ring_scene(radius=2.0, flux=3.0).trace(num_rays=2000, seed=7)
        det = result.detectors["D"]
        assert float(to_numpy(det.total_flux)) == pytest.approx(3.0, rel=1e-12)
        # Every ray lands on the ring: the four central pixels (|x|, |y| < 1) stay dark.
        irr = np.asarray(det.irradiance)
        assert irr[4:6, 4:6].sum() == 0.0
        assert result.flux_conservation_error < 1e-12
    finally:
        be.set_backend("numpy")


def test_plugin_kind_draws_the_same_rays_on_both_backends(ring_kind):
    pytest.importorskip("torch")
    maps = []
    for backend in ("numpy", "torch"):
        be.set_backend(backend)
        be.set_precision("float64")
        try:
            maps.append(np.asarray(_ring_scene().trace(num_rays=500, seed=3).detectors["D"].irradiance))
        finally:
            be.set_backend("numpy")
    np.testing.assert_allclose(maps[0], maps[1], rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# The gradient rule (a plug-in kind can never drop a gradient silently)
# ---------------------------------------------------------------------------


def test_undeclared_gradient_raises(ring_kind):
    torch = pytest.importorskip("torch")
    config = RingSourceConfig(
        spectrum=Spectrum.monochromatic(0.55),
        radius=torch.tensor(2.0, dtype=torch.float64, requires_grad=True),
    )
    with pytest.raises(NotImplementedError, match="radius"):
        NSQScene().add_source("ring", CoordinateSystem(), config)


def test_declared_gradient_is_accepted(ring_kind):
    torch = pytest.importorskip("torch")
    config = RingSourceConfig(
        spectrum=Spectrum.monochromatic(0.55),
        total_flux=torch.tensor(2.0, dtype=torch.float64, requires_grad=True),
    )
    NSQScene().add_source("ring", CoordinateSystem(), config)


def test_builtin_detached_parameter_now_raises_instead_of_detaching():
    """IrradianceDetector used to float() its splat_sigma silently; the kind's
    gradient rule refuses a gradient-carrying value instead (R-09-5)."""
    torch = pytest.importorskip("torch")
    config = IrradianceDetectorConfig(
        width=1.0, height=1.0, splat="gaussian",
        splat_sigma=torch.tensor(0.5, dtype=torch.float64, requires_grad=True),
    )
    with pytest.raises(NotImplementedError, match="splat_sigma"):
        NSQScene().add_detector("D", CoordinateSystem(), config)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_unregistered_subclass_is_refused_not_written_as_its_parent():
    class TaggedFarField(FarFieldDetector):
        pass

    scene = NSQScene()
    scene.detector_registry.add("T", TaggedFarField(
        cs=CoordinateSystem(), theta_max_deg=90.0, num_bins_theta=4, num_bins_phi=4
    ))
    with pytest.raises(TypeError, match="TaggedFarField"):
        scene_to_dict(scene)


def test_unknown_kind_in_json_lists_the_registered_ones():
    d = {
        "nsq_schema_version": 1,
        "components": [],
        "sources": [],
        "detectors": [{"type": "irradianse", "name": "D", "cs": {}}],
    }
    with pytest.raises(ValueError, match="Did you mean: irradiance"):
        scene_from_dict(d)


def test_unregistered_config_type_is_refused():
    @dataclass
    class Unknown:
        x: float = 1.0

    with pytest.raises(TypeError, match="Unrecognised source config type"):
        NSQScene().add_source("S", CoordinateSystem(), Unknown())


def test_duplicate_registration_needs_overwrite(ring_kind):
    with pytest.raises(ValueError, match="already registered"):
        kinds.register_source(
            "test_ring", RingSource, RingSourceConfig, build=None, to_dict=None,
            from_dict=None, lower=None,
        )


def test_config_class_cannot_belong_to_two_kinds(ring_kind):
    with pytest.raises(ValueError, match="config class"):
        kinds.register_source(
            "test_ring_2", type("Other", (RingSource,), {}), RingSourceConfig,
            build=None, to_dict=None, from_dict=None, lower=None,
        )


def test_family_mismatch_is_refused():
    spec = kinds.KindSpec(family="detector", name="x", cls=int)
    with pytest.raises(ValueError, match="cannot be registered"):
        kinds.SOURCES.register(spec)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def load(self):
        return self._fn


def test_plugin_loaded_from_entry_point_on_first_miss(monkeypatch):
    calls = []

    def register():
        calls.append(1)
        _register_ring()

    def boom():
        raise RuntimeError("broken plug-in")

    monkeypatch.setattr(kinds, "_plugins_loaded", False)
    monkeypatch.setattr(
        kinds.importlib.metadata,
        "entry_points",
        lambda group: [_FakeEntryPoint("bad", boom), _FakeEntryPoint("ring", register)]
        if group == kinds.PLUGIN_GROUP
        else [],
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec = kinds.SOURCES.by_name("test_ring")
        assert spec.cls is RingSource and calls == [1]
        assert any("broken plug-in" in str(w.message) for w in caught)
        kinds.SOURCES.by_name("test_ring")  # loaded once per process
        assert calls == [1]
    finally:
        kinds.SOURCES.unregister("test_ring")
        monkeypatch.setattr(kinds, "_plugins_loaded", True)


def test_unregistered_subclass_still_traces_as_its_parent():
    """The lowering before a trace keeps the old dispatch (a test's source that
    replays given rays subclasses a built-in source); only the serializer refuses."""
    from optiland.nonsequential import CollimatedSource  # noqa: PLC0415

    class Replay(CollimatedSource):
        pass

    scene = NSQScene()
    scene.source_registry.add(
        "S", Replay(CoordinateSystem(), Spectrum.monochromatic(0.55), 1.0, aperture_radius=1.0)
    )
    scene.add_detector("D", CoordinateSystem(z=5.0), IrradianceDetectorConfig(width=4.0, height=4.0))
    assert lower(scene, strict=False).emitters[0].kind == "collimated"
    result = scene.trace(num_rays=200, seed=1)
    assert float(to_numpy(result.detectors["D"].total_flux)) == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(TypeError, match="Replay"):
        scene_to_dict(scene)
