"""An occluder's parameters: the interior derivative is zero by structure (T-09-4).

``docs/theory/09_differentiation.md`` of the research repository, sections 9.3
and 9.13.4 (written before these tests ran), test T-09-4's clause for the
boundary term disabled; the research repository's issue 31. An absorbing
surface ends every ray that reaches it and books the weight the ray arrives
with, so its placement and shape decide only which rays it stops: the interior
derivative of every output is zero by structure and the whole derivative is
the boundary term of its silhouette, which the engine does not compute (issue
3). The engine raises on such a parameter instead of returning that zero.

The forward answer moves when the occluder moves (the signature section 9.3
names), and the gradient the trace would hand back without the raise is
exactly zero: both are asserted, so the raise is shown to stop a bare zero, not
a live derivative.
"""

from __future__ import annotations

import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import (
    AbsorbingComponent,
    CollimatedSourceConfig,
    FinitePlaneGeometry,
    IrradianceDetectorConfig,
    MirrorConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.parameter_register import (
    BOUNDARY_ONLY,
    INTERIOR_BOUNDARY,
    DeadParameterError,
    ParameterRegister,
)


@pytest.fixture(autouse=True)
def _torch_float64():
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


def _g(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def _scene(shift, width=10.0, occluder_ref=None):
    """A wide beam, a partial occluder at z = 50 shifted in x, a detector at z = 120."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(z=0.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=10.0
        ),
    )
    occluder = AbsorbingComponent(
        CoordinateSystem(x=shift, z=50.0, reference_cs=occluder_ref),
        FinitePlaneGeometry(width=width, height=40.0),
    )
    scene.add_component("OCC", occluder)
    scene.add_detector(
        "D1",
        CoordinateSystem(z=120.0),
        IrradianceDetectorConfig(width=60, height=60, num_pixels_x=16, num_pixels_y=16),
    )
    return scene


def _trace(scene):
    return scene.trace(num_rays=4_000, seed=17, max_depth=6)


def _occluder_owner(scene) -> str:
    return next(r["owner"] for r in ParameterRegister.from_scene(scene).rows())


class TestOccluderPlacement:
    def test_the_forward_answer_moves(self):
        """The finite difference is not zero: the occluder's position matters."""
        with torch.no_grad():
            low = _trace(_scene(-0.5)).total_flux_detected
            high = _trace(_scene(0.5)).total_flux_detected
        assert high != low

    def test_raises_instead_of_a_bare_zero(self):
        shift = _g(0.0)
        scene = _scene(shift)
        with pytest.raises(DeadParameterError) as info:
            _trace(scene)
        ((owner, name, stage, reason),) = info.value.dead
        assert name == "cs.x"
        assert "silhouette" in stage
        assert "zero by structure" in reason and "boundary term" in reason
        # What the trace would have handed back: exactly zero.
        data = info.value.result.detectors["D1"].data
        (grad,) = torch.autograd.grad(data.sum(), shift, allow_unused=True)
        assert grad is None or grad.item() == 0.0

    def test_registered_as_boundary_only(self):
        scene = _scene(_g(0.0))
        (row,) = ParameterRegister.from_scene(scene).rows()
        assert row["gradient_class"] == BOUNDARY_ONLY
        assert row["structural_zero"] is not None

    def test_the_occluder_shape_is_raised_too(self):
        scene = _scene(0.0)
        occluder = scene.component_registry.get("OCC").surfaces[0]
        occluder.geometry.width = _g(10.0)
        with pytest.raises(DeadParameterError) as info:
            _trace(scene)
        ((_owner, name, _stage, reason),) = info.value.dead
        assert name == "geometry.width"
        assert "zero by structure" in reason

    def test_forward_only_evaluation_does_not_raise(self):
        with torch.no_grad():
            result = _trace(_scene(_g(0.0)))
        assert len(result.parameter_register) == 1


class TestSharedPlacement:
    """A frame the occluder shares with a surface that reflects keeps its live path."""

    def test_a_shared_reference_frame_is_not_raised(self):
        z = _g(0.0)
        ref = CoordinateSystem(z=z)
        scene = _scene(0.0, occluder_ref=ref)
        scene.add_mirror(
            "M1",
            CoordinateSystem(x=0.0, z=110.0, ry=0.3, reference_cs=ref),
            MirrorConfig(radius=-800.0, reflectance=1.0, aperture_radius=25.0),
        )
        register = ParameterRegister.from_scene(scene)
        (entry,) = list(register)
        assert entry.gradient_class == INTERIOR_BOUNDARY
        assert entry.structural_zero is None
        result = _trace(scene)
        assert result.parameter_register.find(entry.owner, entry.name).structural_zero is None
