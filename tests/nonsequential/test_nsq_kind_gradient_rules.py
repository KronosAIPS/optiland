"""T-09-2: the register is generated from the code -- every kind declares every parameter's gradient rule.

``docs/theory/09_differentiation.md`` of the research repository, section
9.13.3 (written before these tests ran), test T-09-2; the research
repository's issue 31. Each registered kind declares, for every field of its
config (sources, detectors, compound components) or every argument of its
constructor (geometries, scatter models, spectra), one rule: *attached* with
its class of chapter 09 and the stage its derivative enters, *detached* with
the reason it is not a differentiable quantity, or *refused* with the reason
its derivative is not built (``optiland/nonsequential/_builtin_gradients.py``).

The walk below goes over every registered kind of every family and fails on:
a parameter without a rule or a rule without a parameter; a kind whose
``attached`` list differs from its attached rules; a detached or refused
parameter that a build accepts as a gradient-carrying tensor (it would be a
silent detach); an attached parameter the register classifies otherwise.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import NSQScene, Spectrum, kinds
from optiland.nonsequential.components.reflective import ReflectiveComponent
from optiland.nonsequential.parameter_register import ParameterRegister

_SPECTRUM = Spectrum.monochromatic(0.55)
_CLASSES = ("interior", "interior+boundary", "boundary-only")


@pytest.fixture(autouse=True)
def _torch_float64():
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


def _g(value) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def _every_kind():
    for family in kinds.FAMILIES:
        registry = kinds.registry(family)
        for name in registry.names():
            yield family, name, registry.by_name(name)


def _parameters(spec) -> list[str]:
    """The config's fields, or the constructor's named arguments."""
    if spec.config_cls is not None:
        return [f.name for f in dataclasses.fields(spec.config_cls)]
    signature = inspect.signature(spec.cls.__init__)
    return [
        p.name
        for p in signature.parameters.values()
        if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


_ALL = list(_every_kind())
_IDS = [f"{family}:{name}" for family, name, _ in _ALL]


class TestEveryKindDeclaresEveryParameter:
    @pytest.mark.parametrize(("family", "name", "spec"), _ALL, ids=_IDS)
    def test_rules_cover_the_parameters_exactly(self, family, name, spec):
        assert spec.gradients is not None, f"{family} kind {name!r} declares no gradient rules"
        params = set(_parameters(spec))
        rules = set(spec.gradients)
        assert params - rules == set(), f"{family}:{name}: no rule for {sorted(params - rules)}"
        assert rules - params == set(), f"{family}:{name}: rules for no parameter {sorted(rules - params)}"

    @pytest.mark.parametrize(("family", "name", "spec"), _ALL, ids=_IDS)
    def test_rules_are_well_formed(self, family, name, spec):
        for param, rule in spec.gradients.items():
            assert rule.rule in ("attached", "detached", "refused"), (family, name, param)
            assert rule.text.strip(), (family, name, param)
            if rule.is_attached:
                assert rule.gradient_class in _CLASSES, (family, name, param)

    @pytest.mark.parametrize(("family", "name", "spec"), _ALL, ids=_IDS)
    def test_the_attached_list_names_the_attached_rules(self, family, name, spec):
        if spec.attached == "*" or spec.config_cls is None:
            return
        from_rules = {p for p, r in spec.gradients.items() if r.is_attached}
        assert set(spec.attached) == from_rules, f"{family}:{name}"


# -- the samples a build needs -------------------------------------------------------------

_CONFIG_SAMPLES = {
    ("source", "point"): {"spectrum": _SPECTRUM},
    ("source", "collimated"): {"spectrum": _SPECTRUM},
    ("source", "extended"): {"spectrum": _SPECTRUM},
    ("source", "tabulated"): {
        "spectrum": _SPECTRUM, "polar_angles_deg": [0.0, 3.0], "intensity": [1.0, 1.0],
    },
    ("detector", "irradiance"): {"width": 10.0, "height": 10.0},
    ("detector", "spectral"): {"width": 10.0, "height": 10.0},
    ("detector", "far_field"): {},
    ("detector", "hemisphere"): {"radius": 10.0},
    ("detector", "colorimetric"): {"width": 10.0, "height": 10.0},
    ("detector", "colorimetric_far_field"): {},
    ("detector", "ray_database"): {"width": 10.0, "height": 10.0},
    ("component", "lens"): {
        "r1": 60.0, "r2": float("inf"), "thickness": 5.0, "material": "N-BK7",
        "front_aperture_radius": 12.0,
    },
    ("component", "mirror"): {"radius": -800.0, "reflectance": 1.0},
    ("component", "doublet"): {
        "r1": 60.0, "r2": -40.0, "r3": -200.0, "thickness1": 4.0, "thickness2": 2.0,
        "material1": "N-BK7", "material2": "N-BK7", "aperture_radius": 10.0,
    },
    ("component", "prism"): {
        "apex_angle_deg": 60.0, "face_length": 20.0, "length": 20.0, "material": "N-BK7",
    },
    ("component", "paraxial_lens"): {"focal_length": 50.0, "aperture_radius": 10.0},
    ("component", "polarizer"): {"axis_deg": 0.0, "aperture_radius": 10.0},
    ("component", "retarder"): {
        "fast_axis_deg": 0.0, "retardance_waves": 0.25, "aperture_radius": 10.0,
    },
}


def _build(family: str, name: str, config):
    scene = NSQScene()
    if family == "source":
        scene.add_source("S1", CoordinateSystem(), config)
    elif family == "detector":
        scene.add_detector("D1", CoordinateSystem(z=10.0), config)
    else:
        kinds.COMPONENTS.build(scene, "C1", CoordinateSystem(z=10.0), config)
    return scene


_CONFIG_KINDS = [(f, n, s) for f, n, s in _ALL if s.config_cls is not None]


class TestConfigFields:
    """Sources, detectors and components: the scene builder applies the rules."""

    @pytest.mark.parametrize(("family", "name", "spec"), _CONFIG_KINDS, ids=[f"{f}:{n}" for f, n, _ in _CONFIG_KINDS])
    def test_a_detached_or_refused_field_raises_at_build(self, family, name, spec):
        sample = _CONFIG_SAMPLES[(family, name)]
        _build(family, name, spec.config_cls(**sample))  # the sample builds
        for param, rule in spec.gradients.items():
            if rule.is_attached:
                continue
            config = spec.config_cls(**{**sample, param: _g(1.0)})
            with pytest.raises(NotImplementedError, match=param):
                _build(family, name, config)

    _SOURCE_EXTRA = {
        ("source", "collimated", "gaussian_sigma"): {"profile": "gaussian", "profile_gradient": "implicit"},
        ("source", "tabulated", "width"): {"height": 2.0},
        ("source", "tabulated", "height"): {"width": 3.0},
    }

    @pytest.mark.parametrize(
        ("family", "name", "spec"),
        [k for k in _CONFIG_KINDS if k[0] in ("source", "detector")],
        ids=[f"{f}:{n}" for f, n, _ in _CONFIG_KINDS if f in ("source", "detector")],
    )
    def test_an_attached_field_is_registered_with_its_class(self, family, name, spec):
        sample = _CONFIG_SAMPLES[(family, name)]
        owner = "S1" if family == "source" else "D1"
        for param, rule in spec.gradients.items():
            if not rule.is_attached:
                continue
            extra = self._SOURCE_EXTRA.get((family, name, param), {})
            value = _g(float(sample.get(param) or 1.0))
            scene = _build(family, name, spec.config_cls(**{**sample, **extra, param: value}))
            # A flux given in lumens is converted on the way in and held as the
            # source's total flux (in watts), still attached to the lumens.
            held = "total_flux" if param == "total_flux_lumens" else param
            entry = ParameterRegister.from_scene(scene).find(owner, held)
            assert entry.gradient_class == spec.gradients[held].gradient_class, (family, name, param)
            assert entry.stage == spec.gradients[held].text, (family, name, param)
            assert rule.gradient_class == spec.gradients[held].gradient_class


# -- constructors: geometries, scatter models, spectra ----------------------------------------


def _nurbs_arrays():
    from tests.nonsequential.test_nsq_nurbs import _arrays, sphere_surface

    return _arrays(sphere_surface(10.0))


_CONSTRUCTOR_SAMPLES = {
    ("geometry", "conic"): lambda: {"radius": 50.0, "conic": 0.0, "aperture_radius": 10.0},
    ("geometry", "paraboloid"): lambda: {"radius": 50.0, "aperture_radius": 10.0},
    ("geometry", "plane"): lambda: {"width": 10.0, "height": 10.0},
    ("geometry", "infinite_plane"): lambda: {},
    ("geometry", "annulus"): lambda: {"inner_radius": 2.0, "outer_radius": 10.0},
    ("geometry", "frustum"): lambda: {"r_front": 5.0, "r_back": 6.0, "z_front": 0.0, "z_back": 10.0},
    ("geometry", "sphere"): lambda: {"radius": 10.0},
    ("geometry", "spherical_cavity"): lambda: {"radius": 10.0},
    ("geometry", "even_asphere"): lambda: {
        "radius": 50.0, "conic": 0.0, "aperture_radius": 10.0, "coefficients": (1e-6,),
    },
    ("geometry", "odd_asphere"): lambda: {
        "radius": 50.0, "conic": 0.0, "aperture_radius": 10.0, "coefficients": (1e-6,),
    },
    ("geometry", "nurbs"): lambda: {"arrays": _nurbs_arrays()},
    ("geometry", "lenslet_array"): lambda: {
        "pitch_x": 1.0, "pitch_y": 1.0, "radius": 5.0, "conic": 0.0, "num_x": 3, "num_y": 3,
    },
    ("bsdf", "lambertian"): lambda: {"reflectance_value": 0.5},
    ("bsdf", "harvey_shack"): lambda: {"b0": 1.0, "l0": 0.01, "s": 1.5},
    ("bsdf", "specular"): lambda: {},
    ("spectrum", "lines"): lambda: {"wavelengths": [0.55], "weights": [1.0]},
    ("spectrum", "piecewise_linear"): lambda: {"wavelengths": [0.4, 0.7], "values": [1.0, 1.0]},
}

#: Arguments that are not numbers, so a tensor is not a value they can take,
#: and kinds a sample cannot be built for here; each with the reason.
_NOT_INJECTED = {
    ("geometry", "spherical_cavity", "ports"): "port records, not a number",
    ("geometry", "nurbs", "arrays"): "the net's arrays, not a number",
    ("geometry", "mesh", "mesh"): "a mesh object, not a number",
    ("bsdf", "tabulated", "path"): "a file path",
    ("bsdf", "tabulated", "transmissive_fraction"): "the kind needs a measured table file to build",
    ("spectrum", "piecewise_linear", "label"): "a name",
}

_SAMPLE_SHAPES = {
    ("geometry", "lenslet_array", "sag_offsets"): (3, 3),
    ("spectrum", "lines", "wavelengths"): (1,),
    ("spectrum", "lines", "weights"): (1,),
    ("spectrum", "piecewise_linear", "wavelengths"): (2,),
    ("spectrum", "piecewise_linear", "values"): (2,),
}

_CONSTRUCTOR_KINDS = [(f, n, s) for f, n, s in _ALL if s.config_cls is None]


def _tensor_for(key, sample_value):
    shape = _SAMPLE_SHAPES.get(key)
    if shape is not None:
        base = np.linspace(0.5, 0.6, int(np.prod(shape))).reshape(shape)
        return torch.tensor(base, dtype=torch.float64, requires_grad=True)
    if isinstance(sample_value, (int, float)) and not isinstance(sample_value, bool):
        return _g(float(sample_value))
    return _g(1.0)


class TestConstructorArguments:
    """Geometries, scatter models and spectra: a constructor never reads a gradient as a number."""

    @pytest.mark.parametrize(
        ("family", "name", "spec"), _CONSTRUCTOR_KINDS, ids=[f"{f}:{n}" for f, n, _ in _CONSTRUCTOR_KINDS]
    )
    def test_a_detached_or_refused_argument_raises(self, family, name, spec):
        sample_of = _CONSTRUCTOR_SAMPLES.get((family, name))
        targets = [p for p, r in spec.gradients.items() if not r.is_attached]
        targets = [p for p in targets if (family, name, p) not in _NOT_INJECTED]
        if not targets:
            return
        assert sample_of is not None, f"{family}:{name} has arguments to check and no sample"
        sample = sample_of()
        spec.cls(**sample)  # the sample builds
        signature = inspect.signature(spec.cls.__init__)
        for param in targets:
            default = signature.parameters[param].default
            value = _tensor_for((family, name, param), sample.get(param, default))
            with pytest.raises((NotImplementedError, RuntimeError)):
                spec.cls(**{**sample, param: value})

    @pytest.mark.parametrize(
        ("family", "name", "spec"),
        [k for k in _CONSTRUCTOR_KINDS if k[0] == "geometry"],
        ids=[f"{f}:{n}" for f, n, _ in _CONSTRUCTOR_KINDS if f == "geometry"],
    )
    def test_an_attached_argument_is_registered_with_its_class(self, family, name, spec):
        sample_of = _CONSTRUCTOR_SAMPLES.get((family, name))
        attached = [p for p, r in spec.gradients.items() if r.is_attached]
        if not attached:
            return
        assert sample_of is not None
        for param in attached:
            sample = sample_of()
            if param in ("control_points", "weights"):
                base = spec.cls(**sample)
                value = torch.tensor(np.asarray(getattr(base, param)), dtype=torch.float64, requires_grad=True)
            elif isinstance(sample.get(param), tuple):
                value = torch.tensor(sample[param], dtype=torch.float64, requires_grad=True)
            else:
                value = _g(float(sample.get(param, 1.0)))
            geometry = spec.cls(**{**sample, param: value})
            scene = NSQScene()
            scene.add_component("G1", ReflectiveComponent(CoordinateSystem(z=10.0), geometry, reflectance=1.0))
            rows = {r["name"]: r for r in ParameterRegister.from_scene(scene).rows()}
            row = rows[f"geometry.{param}"]
            rule = spec.gradients[param]
            assert row["gradient_class"] == rule.gradient_class, (name, param)
            assert row["stage"] == rule.text, (name, param)
