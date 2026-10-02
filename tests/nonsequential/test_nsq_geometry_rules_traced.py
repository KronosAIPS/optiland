"""T-09-2's attached list, traced: a non-zero, finite gradient for every attached geometry rule.

The research repository's chapter 09 section 9.14.2 (written and committed
before this file ran) and its issue 31. The walk of
``test_nsq_kind_gradient_rules.py`` checks that every kind declares a rule for
every parameter and that the register classifies an attached parameter with
the rule's class and stage. This file traces each attached rule of the
geometry family, and each geometric attached rule of the compound components
(curvatures, conics, coefficients, thicknesses, the paraxial focal length), on
a scene that reaches it:

- an interior or interior+boundary rule: the autograd gradient of the landing
  centroid on a 2 x 2 bilinear detector is finite and non-zero and equals the
  fourth-order central difference (common random numbers) within the
  placement test's bound ``max(1e-7, 1.5 K u |f| / (h |f'|))``, ``K = 1e4``,
  ``u = 2**-53`` (``test_nsq_interior_gradients.py``; chapter 09 section 9.8).
  An array-valued parameter is checked along one fixed direction ``v``.
- a boundary-only rule (an aperture, an extent, an annulus radius): the trace
  raises naming it, "detached by contract" (R-09-5), never a bare zero.

Every scene keeps the beam away from every edge, and its detector is placed
so every landing point lies inside the square between the four pixel centres
(measured once at float64, seed 3, by a forward-only probe recorded in the
research repository's build log G4_gradients_4), so the loss is linear in
every landing point and smooth in the parameter at every stencil point. The
walk fails on an attached rule that has no scene here.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    DoubletConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQMaterial,
    NSQScene,
    ParaxialLensConfig,
    ReflectiveComponent,
    RefractiveComponent,
    Spectrum,
    kinds,
)
from optiland.nonsequential.components.geometry.analytic.annulus import AnnularPlaneGeometry
from optiland.nonsequential.components.geometry.analytic.asphere import (
    EvenAsphereGeometry,
    OddAsphereGeometry,
)
from optiland.nonsequential.components.geometry.analytic.conic import (
    ConicGeometry,
    ParaboloidGeometry,
)
from optiland.nonsequential.components.geometry.analytic.frustum import (
    CylindricalFrustumGeometry,
)
from optiland.nonsequential.components.geometry.analytic.lenslet_array import (
    LensletArrayGeometry,
)
from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry
from optiland.nonsequential.components.geometry.analytic.sphere import SphereGeometry
from optiland.nonsequential.components.geometry.analytic.spherical_cavity import (
    SphericalCavityGeometry,
    SphericalPort,
)
from optiland.nonsequential.components.geometry.nurbs.geometry import NurbsGeometry
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.nonsequential.parameter_register import BOUNDARY_ONLY, DeadParameterError

_NUM_RAYS = 2_000
_SEED = 3
_U = 2.0**-53
_K_OPS = 1e4
_SPECTRUM = Spectrum.monochromatic(0.55)


@pytest.fixture(autouse=True)
def _torch_float64():
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


def _centroid(data, width: float):
    """Flux-weighted landing ``x + y / 2`` on a 2 x 2 bilinear detector of side ``width``."""
    q = width / 4.0
    xc = torch.tensor([-q, q, -q, q], dtype=torch.float64)
    yc = torch.tensor([-q, -q, q, q], dtype=torch.float64)
    s = data.sum()
    return (data * xc).sum() / s + 0.5 * (data * yc).sum() / s


def _nurbs_arrays():
    from tests.nonsequential.test_nsq_nurbs import _arrays, sphere_surface

    return _arrays(sphere_surface(10.0))


# -- the layouts: where the beam starts, what the surface is, where the detector sits ------

_FOLD = {"z": 100.0, "ry": 0.2}

#: (source x, source y, beam radius, the component's placement, reflective?,
#: detector centre (x, y, z), detector side). Detector centres and sides are the
#: probe's landing boxes (build log G4_gradients_4): every landing within a
#: quarter of the side of the centre.
_LAYOUT = {
    "conic": (8.0, 0.0, 2.0, _FOLD, True, (-22.4, 0.0, 40.0), 8.0),
    "paraboloid": (8.0, 0.0, 2.0, _FOLD, True, (-22.4, 0.0, 40.0), 8.0),
    "even_asphere": (8.0, 0.0, 2.0, _FOLD, True, (-22.1, 0.0, 40.0), 8.0),
    "odd_asphere": (8.0, 0.0, 2.0, _FOLD, True, (-22.4, 0.0, 40.0), 8.0),
    "annulus": (8.0, 0.0, 2.0, _FOLD, True, (-16.9, 0.0, 40.0), 14.0),
    "plane": (8.0, 0.0, 2.0, _FOLD, True, (-16.7, 0.0, 40.0), 14.0),
    "lenslet_array": (2.3, 2.3, 0.2, _FOLD, True, (-25.4, 0.4, 40.0), 10.0),
    "frustum": (4.0, 0.0, 0.4, {}, True, (-8.1, 0.0, 80.0), 6.0),
    "sphere": (2.0, 0.0, 1.0, {"z": 50.0}, False, (-5.1, 0.0, 100.0), 16.0),
    "spherical_cavity": (30.0, 0.0, 2.0, {}, True, (-22.7, 0.0, 20.0), 14.0),
    "nurbs": (3.0, 0.0, 0.3, {"z": 50.0}, True, (24.5, 0.0, 10.0), 20.0),
}

#: Nominal constructor arguments of each geometry kind.
_NOMINAL = {
    "conic": {"radius": -200.0, "conic": -0.5, "aperture_radius": 25.0},
    "paraboloid": {"radius": -200.0, "aperture_radius": 25.0},
    "even_asphere": {"radius": -200.0, "conic": -0.5, "aperture_radius": 25.0, "coefficients": (1e-4, 1e-7)},
    "odd_asphere": {
        "radius": -200.0, "conic": -0.5, "aperture_radius": 25.0, "coefficients": (1e-4, 1e-5, 1e-7),
    },
    "annulus": {"inner_radius": 1.0, "outer_radius": 30.0, "z_offset": 0.5},
    "plane": {"width": 60.0, "height": 60.0},
    "lenslet_array": {"pitch_x": 2.0, "pitch_y": 2.0, "radius": -20.0, "conic": 0.0, "num_x": 5, "num_y": 5},
    "frustum": {"r_front": 5.0, "r_back": 3.0, "z_front": 10.0, "z_back": 30.0},
    "sphere": {"radius": 10.0},
    "spherical_cavity": {"radius": 100.0},
    "nurbs": {},
}

_CLASSES = {
    "conic": ConicGeometry,
    "paraboloid": ParaboloidGeometry,
    "even_asphere": EvenAsphereGeometry,
    "odd_asphere": OddAsphereGeometry,
    "annulus": AnnularPlaneGeometry,
    "plane": FinitePlaneGeometry,
    "lenslet_array": LensletArrayGeometry,
    "frustum": CylindricalFrustumGeometry,
    "sphere": SphereGeometry,
    "spherical_cavity": SphericalCavityGeometry,
    "nurbs": NurbsGeometry,
}

#: A boundary-only argument a nominal build leaves unset gets this value.
_BOUNDARY_VALUES = {
    ("plane", "aperture_radius"): 30.0,
    ("sphere", "aperture_radius"): 9.0,
}


def _geometry(kind: str, overrides: dict):
    kw = dict(_NOMINAL[kind])
    kw.update(overrides)
    if kind == "spherical_cavity":
        kw["ports"] = [SphericalPort(axis=(0.0, 0.0, -1.0), half_angle_deg=90.0)]
    if kind == "nurbs":
        return NurbsGeometry(_nurbs_arrays(), **kw)
    return _CLASSES[kind](**kw)


def _scene_for_geometry(kind: str, overrides: dict, detector=None):
    """The kind's layout with its geometry built from the nominal arguments and ``overrides``."""
    sx, sy, rb, place, reflective, centre, side = _LAYOUT[kind]
    if detector is not None:
        centre, side = detector
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(x=sx, y=sy),
        CollimatedSourceConfig(spectrum=_SPECTRUM, total_flux=1.0, aperture_radius=rb),
    )
    geometry = _geometry(kind, overrides)
    cs = CoordinateSystem(**place)
    if reflective:
        scene.add_component("G1", ReflectiveComponent(cs, geometry, reflectance=1.0))
    else:
        scene.add_component(
            "G1", RefractiveComponent(cs, geometry, VACUUM, NSQMaterial.from_glass("N-BK7"))
        )
        scene.sampling_policy = SamplingPolicy(reflect_prob=1e-6)
    _add_detector(scene, centre, side)
    return scene, side


def _add_detector(scene, centre, side, **kw):
    x, y, z = centre
    cfg = {"num_pixels_x": 2, "num_pixels_y": 2, "splat": "bilinear", **kw}
    scene.add_detector(
        "D1", CoordinateSystem(x=x, y=y, z=z), IrradianceDetectorConfig(width=side, height=side, **cfg)
    )


# -- the compound components --------------------------------------------------------------

_LENS_CS = {"x": 0.5, "y": -0.3, "z": 50.0, "rx": 0.01, "ry": -0.02}

#: (config class, nominal fields, source x, beam radius, placement, refracting?,
#: detector centre, detector side).
_COMPONENT = {
    "lens": (
        LensConfig,
        {
            "r1": 60.0, "r2": -200.0, "thickness": 5.0, "material": "N-BK7",
            "front_aperture_radius": 12.0, "back_aperture_radius": 12.0,
            "conic1": -0.5, "conic2": 2.0, "coefficients1": (1e-5,), "coefficients2": (1e-5,),
        },
        3.0, 2.0, _LENS_CS, True, (0.3, -0.3, 150.0), 4.0,
    ),
    "mirror": (
        MirrorConfig,
        {"radius": -200.0, "reflectance": 1.0, "aperture_radius": 25.0, "conic": -0.5, "coefficients": (1e-4, 1e-7)},
        8.0, 2.0, _FOLD, False, (-22.1, 0.0, 40.0), 8.0,
    ),
    "doublet": (
        DoubletConfig,
        {
            "r1": 60.0, "r2": -40.0, "r3": -200.0, "thickness1": 4.0, "thickness2": 2.0,
            "material1": "N-BK7", "material2": "N-SF5", "aperture_radius": 10.0,
            "conic1": -0.5, "conic2": 0.5, "conic3": 2.0,
            "coefficients1": (1e-5,), "coefficients2": (1e-5,), "coefficients3": (1e-5,),
        },
        3.0, 2.0, {"z": 50.0}, True, (0.7, 0.0, 150.0), 6.0,
    ),
    "paraxial_lens": (
        ParaxialLensConfig, {"focal_length": 100.0, "aperture_radius": 10.0},
        3.0, 1.0, {"z": 50.0}, False, (0.9, 0.0, 120.0), 4.0,
    ),
}

#: Attached rules of the components that are not geometry, with where they are traced.
_NOT_GEOMETRY = {
    ("mirror", "reflectance"): "a reflect weight; traced by test_nsq_interior_gradients' every-kind scene",
    ("polarizer", "axis_deg"): "a polarizing element (Stokes trace); not a geometry",
    ("polarizer", "extinction"): "a polarizing element (Stokes trace); not a geometry",
    ("polarizer", "aperture_radius"): "a polarizing element's aperture (Stokes trace)",
    ("retarder", "fast_axis_deg"): "a polarizing element (Stokes trace); not a geometry",
    ("retarder", "retardance_waves"): "a polarizing element (Stokes trace); not a geometry",
    ("retarder", "aperture_radius"): "a polarizing element's aperture (Stokes trace)",
}


def _scene_for_component(kind: str, overrides: dict, detector=None):
    cfg_cls, nominal, sx, rb, place, refracting, centre, side = _COMPONENT[kind]
    if detector is not None:
        centre, side = detector
    scene = NSQScene()
    scene.add_source(
        "S1", CoordinateSystem(x=sx), CollimatedSourceConfig(spectrum=_SPECTRUM, total_flux=1.0, aperture_radius=rb)
    )
    kinds.COMPONENTS.build(scene, "C1", CoordinateSystem(**place), cfg_cls(**{**nominal, **overrides}))
    if refracting:
        scene.sampling_policy = SamplingPolicy(reflect_prob=1e-6)
    _add_detector(scene, centre, side)
    return scene, side


# -- the walk --------------------------------------------------------------------------------


def _attached(family: str):
    registry = kinds.registry(family)
    for name in registry.names():
        for param, rule in registry.by_name(name).gradients.items():
            if rule.is_attached:
                yield name, param, rule


_GEOMETRY_ROWS = list(_attached("geometry"))
_COMPONENT_ROWS = [r for r in _attached("component") if (r[0], r[1]) not in _NOT_GEOMETRY]


def test_every_attached_rule_has_a_scene():
    """A new attached rule without a scene here fails, so the trace list stays complete."""
    missing = [(k, p) for k, p, _ in _GEOMETRY_ROWS if k not in _LAYOUT]
    missing += [(k, p) for k, p, _ in _COMPONENT_ROWS if k not in _COMPONENT]
    assert not missing, f"attached rules with no traced scene: {missing}"


def _nominal_value(kind: str, param: str, component: bool):
    if component:
        return _COMPONENT[kind][1].get(param)
    if kind == "nurbs":
        geom = _geometry("nurbs", {})
        return np.asarray(getattr(geom, param).detach().cpu().numpy() if hasattr(getattr(geom, param), "detach") else getattr(geom, param), dtype=np.float64)
    return _NOMINAL[kind].get(param, _BOUNDARY_VALUES.get((kind, param)))


def _direction(value, param: str) -> np.ndarray:
    """The fixed direction an array parameter is checked along (chapter 09 section 9.14.2).

    Each element's own magnitude times a fixed sign pattern; for NURBS
    control points, the net itself (a scaling about its centre), which keeps
    the sphere net's coincident pole and seam points together. (A pattern of
    +-1 mm pulls them apart, so the surface is not smooth along it near the
    pole the beam lands at; the chapter's amendment states the first run.)
    """
    arr = np.asarray(value, dtype=np.float64)
    if param == "control_points":
        return arr.copy()
    signs = np.where(np.arange(arr.size).reshape(arr.shape) % 2 == 0, 1.0, -1.0)
    mags = np.where(arr != 0.0, np.abs(arr), 1.0)
    return signs * mags


def _check_rule(build, kind: str, param: str, value):
    """Autograd against FD4 along the parameter (or its fixed direction); the relative gap."""
    is_array = np.ndim(value) > 0 or isinstance(value, tuple)
    base = np.asarray(value, dtype=np.float64)
    v = _direction(base, param) if is_array else None
    h = 1e-3
    if not is_array:
        h = 1e-3 * max(abs(float(value)), 1.0)

    def at(eps, tensor=None):
        if tensor is not None:
            arg = tensor
        elif is_array:
            arr = base + eps * v
            arg = tuple(arr.tolist()) if isinstance(value, tuple) else torch.tensor(arr, dtype=torch.float64)
        else:
            arg = float(value) + eps
        scene, width = build({param: arg})
        return _centroid(scene.trace(num_rays=_NUM_RAYS, seed=_SEED, max_depth=8).detectors["D1"].data, width)

    param_t = torch.tensor(base, dtype=torch.float64, requires_grad=True)
    f = at(0.0, tensor=param_t)
    assert f.requires_grad, f"{kind}.{param}: the parameter never reached the autograd graph"
    (grad,) = torch.autograd.grad(f, param_t)
    assert torch.isfinite(grad).all(), f"{kind}.{param}: {grad}"
    ad = float((grad.numpy() * v).sum()) if is_array else grad.item()
    with torch.no_grad():
        fp2, fp1, fm1, fm2 = (float(at(m * h)) for m in (2, 1, -1, -2))
    fd = (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)
    assert ad != 0.0, f"{kind}.{param}: a zero gradient"
    assert fd != 0.0, f"{kind}.{param}: the loss does not move"
    tol = max(1e-7, 1.5 * _K_OPS * _U * abs(f.detach().item()) / (h * abs(fd)))
    rel = abs(ad - fd) / abs(fd)
    assert rel < tol, f"{kind}.{param}: autograd {ad:.12e} vs FD4 {fd:.12e}: relative {rel:.2e} > {tol:.1e}"
    return ad, fd, rel, tol


def _check_boundary_raise(build, owner_prefix: str, param: str, value):
    tensor = torch.tensor(np.asarray(value, dtype=np.float64), requires_grad=True)
    scene, _ = build({param: tensor})
    with pytest.raises(DeadParameterError) as info:
        scene.trace(num_rays=_NUM_RAYS, seed=_SEED, max_depth=8)
    # A component's field is raised under the surfaces that hold it (a lens's
    # front_aperture_radius as ``C1.front: geometry.aperture_radius``), so the
    # check is on the owners, the tensor and the reason.
    raised = {(o, n): r for o, n, _s, r in info.value.dead}
    assert raised, f"{param}: nothing raised"
    for (owner, name), reason in raised.items():
        assert owner.startswith(owner_prefix), (owner, name)
        assert "detached by contract" in reason, (owner, name, reason)
    from optiland.nonsequential.parameter_register import ParameterRegister

    rows = [e for e in ParameterRegister.from_scene(scene) if e.tensor is tensor]
    assert len(rows) == 1 and rows[0].gradient_class == BOUNDARY_ONLY, rows
    assert (rows[0].owner, rows[0].name) in raised


class TestGeometryRules:
    @pytest.mark.parametrize(
        ("kind", "param", "rule"), _GEOMETRY_ROWS, ids=[f"{k}.{p}" for k, p, _ in _GEOMETRY_ROWS]
    )
    def test_traced(self, kind, param, rule):
        value = _nominal_value(kind, param, component=False)

        def build(overrides):
            return _scene_for_geometry(kind, overrides)

        if rule.gradient_class == BOUNDARY_ONLY:
            _check_boundary_raise(build, "G1", param, value)
        else:
            _check_rule(build, kind, param, value)


class TestComponentGeometryRules:
    @pytest.mark.parametrize(
        ("kind", "param", "rule"), _COMPONENT_ROWS, ids=[f"{k}.{p}" for k, p, _ in _COMPONENT_ROWS]
    )
    def test_traced(self, kind, param, rule):
        value = _nominal_value(kind, param, component=True)

        def build(overrides):
            return _scene_for_component(kind, overrides)

        if rule.gradient_class == BOUNDARY_ONLY:
            _check_boundary_raise(build, "C1", param, value)
        else:
            _check_rule(build, kind, param, value)
