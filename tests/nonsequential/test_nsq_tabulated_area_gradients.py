"""The tabulated source's emitting area, attached by the change of variables.

``docs/theory/09_differentiation.md`` of the research repository, sections 9.7
and 9.13.2 (written before these tests ran), requirement R-09-4; the research
repository's issue 31. A tabulated source emits from a rectangle or a disc
with the extended source's uniform maps, and its flux is a total flux, so the
change of variables' ``|det J|`` cancels the exitance and the birth weight has
no derivative in the area. The bounds are the other source-geometry
parameters' (``test_nsq_source_jacobians.py``): the fourth-order central
difference through the decentred, tilted singlet to
``max(1e-7, 1.5 K u |f| / (h |f'|))`` with ``K = 1e4``; the total flux's
derivative below ``K u`` times the unnormalised factor; values bit-identical.

Before this pass the source stored its area with ``float()``, which detached a
tensor without a word (:class:`TestNoSilentDetach`).
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import (
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.nonsequential.parameter_register import INTERIOR_BOUNDARY, ParameterRegister
from optiland.nonsequential.sources.configs import TabulatedSourceConfig
from optiland.nonsequential.sources.tabulated import TabulatedSource

_NUM_RAYS = 2_000
_SEED = 3
_U = 2.0**-53
_K_OPS = 1e4
_SPECTRUM = Spectrum.monochromatic(0.55)
_LENS = {"x": 0.5, "y": -0.3, "z": 50.0, "rx": 0.01, "ry": -0.02, "rz": 0.0}


@pytest.fixture(autouse=True)
def _torch_float64():
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


def _g(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def _table(width=None, height=None, radius=None, cone_deg=3.0):
    """A narrow cone, uniform in solid angle to ``cone_deg``, from a rectangle or a disc."""
    return TabulatedSourceConfig(
        spectrum=_SPECTRUM,
        polar_angles_deg=[0.0, cone_deg],
        intensity=[1.0, 1.0],
        total_flux=1.0,
        width=width,
        height=height,
        aperture_radius=radius,
    )


def _detector(scene, z: float, width: float = 40.0):
    scene.add_detector(
        "D1",
        CoordinateSystem(z=z),
        IrradianceDetectorConfig(
            width=width, height=width, num_pixels_x=2, num_pixels_y=2, splat="bilinear"
        ),
    )


def _through_the_singlet(config):
    scene = NSQScene()
    scene.add_source("S1", CoordinateSystem(), config)
    scene.add_lens(
        "L1",
        CoordinateSystem(**_LENS),
        LensConfig(
            r1=60.0,
            r2=float("inf"),
            thickness=5.0,
            material="N-BK7",
            front_aperture_radius=12.0,
        ),
    )
    _detector(scene, 150.0)
    scene.sampling_policy = SamplingPolicy(reflect_prob=1e-6)
    return scene


def _centroid(data, width: float):
    q = width / 4.0
    xc = torch.tensor([-q, q, -q, q], dtype=torch.float64)
    yc = torch.tensor([-q, -q, q, q], dtype=torch.float64)
    s = data.sum()
    return (data * xc).sum() / s + 0.5 * (data * yc).sum() / s


def _trace(scene, seed=_SEED):
    return scene.trace(num_rays=_NUM_RAYS, seed=seed, max_depth=8)


#: (id, config builder, nominal, the parameter's name)
_CASES = [
    ("rectangle-width", lambda v: _table(width=v, height=2.0), 3.0, "width"),
    ("rectangle-height", lambda v: _table(width=3.0, height=v), 2.0, "height"),
    ("disc-radius", lambda v: _table(radius=v), 1.5, "aperture_radius"),
]


class TestAgainstFD4:
    @pytest.mark.parametrize(("case", "config_of", "value", "name"), _CASES, ids=[c[0] for c in _CASES])
    def test_against_fd4(self, case, config_of, value, name):
        def loss(v):
            return _centroid(_trace(_through_the_singlet(config_of(v))).detectors["D1"].data, 40.0)

        h = 1e-2
        param = _g(value)
        f = loss(param)
        assert f.requires_grad, f"{case}: the parameter never reached the autograd graph"
        (grad,) = torch.autograd.grad(f, param)
        ad = grad.item()
        with torch.no_grad():
            fp2, fp1, fm1, fm2 = (float(loss(value + m * h)) for m in (2, 1, -1, -2))
        fd = (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)
        assert fd != 0.0, "the loss does not move: the comparison would mean nothing"
        tol = max(1e-7, 1.5 * _K_OPS * _U * abs(f.detach().item()) / (h * abs(fd)))
        rel = abs(ad - fd) / abs(fd)
        assert rel < tol, f"{case}: autograd {ad:.12e} vs FD4 {fd:.12e}: {rel:.2e} > {tol:.1e}"


class TestFixedTotalFlux:
    """The area changes at fixed total flux: the total on a detector that catches every ray does not move."""

    @pytest.mark.parametrize(
        ("config_of", "value", "unnormalised"),
        [
            (lambda v: _table(width=v, height=2.0, cone_deg=0.5), 3.0, lambda v: 1.0 / v),
            (lambda v: _table(radius=v, cone_deg=0.5), 1.5, lambda v: 2.0 / v),
        ],
        ids=["rectangle", "disc"],
    )
    def test_total_flux_does_not_move(self, config_of, value, unnormalised):
        scene = NSQScene()
        param = _g(value)
        scene.add_source("S1", CoordinateSystem(), config_of(param))
        _detector(scene, 10.0)
        total = _trace(scene).detectors["D1"].data.sum()
        assert total.item() == pytest.approx(1.0, rel=1e-12)
        (grad,) = torch.autograd.grad(total, param)
        assert abs(grad.item()) < _K_OPS * _U * unnormalised(value)


class TestAttachingMovesNoValue:
    @pytest.mark.parametrize(("case", "config_of", "value", "name"), _CASES, ids=[c[0] for c in _CASES])
    def test_forward_values_bit_identical(self, case, config_of, value, name):
        plain = _trace(_through_the_singlet(config_of(value)))
        attached = _trace(_through_the_singlet(config_of(_g(value))))
        a = plain.detectors["D1"].data
        b = attached.detectors["D1"].data
        assert b.requires_grad and not a.requires_grad
        assert torch.equal(a, b.detach())
        assert attached.total_flux_detected == plain.total_flux_detected


class TestRegister:
    @pytest.mark.parametrize(("case", "config_of", "value", "name"), _CASES, ids=[c[0] for c in _CASES])
    def test_registered_as_interior_with_boundary(self, case, config_of, value, name):
        scene = _through_the_singlet(config_of(_g(value)))
        entry = ParameterRegister.from_scene(scene).find("S1", name)
        assert entry.gradient_class == INTERIOR_BOUNDARY
        assert "change of variables" in entry.stage


class TestNoSilentDetach:
    def test_a_directly_built_source_keeps_the_tensor(self):
        w = _g(3.0)
        source = TabulatedSource(
            CoordinateSystem(), _SPECTRUM, 1.0, [0.0, 3.0], [1.0, 1.0], width=w, height=2.0
        )
        assert source.width is w

    def test_serialisation_writes_numbers(self):
        from optiland.nonsequential import kinds

        source = TabulatedSource(
            CoordinateSystem(), _SPECTRUM, 1.0, [0.0, 3.0], [1.0, 1.0], aperture_radius=_g(1.5)
        )
        d = kinds.registry("source").for_object(source).to_dict(source)
        assert d["aperture_radius"] == 1.5 and isinstance(d["aperture_radius"], float)
        assert d["width"] is None
