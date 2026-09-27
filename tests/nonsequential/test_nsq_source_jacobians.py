"""Source geometry attached by the change of variables (R-09-4, T-09-8).

``docs/theory/09_differentiation.md`` of the research repository, section 9.7,
requirement R-09-4 and test T-09-8; the research repository's issue 31. A
source's geometry -- the aperture radius of a top-hat collimated beam or of an
extended disc, an extended rectangle's width and height, the half-angle of a
point source's cone or of an extended source's cone -- enters the trace
through where its samples come from. With the source's draws held fixed, the
emission points and directions are smooth functions of the geometry, and the
trace attaches them to it (``parameter_register.attach_source_geometry``).

How the finite differences are kept honest: as in
``test_nsq_interior_gradients.py``, autograd is compared with the fourth-order
central difference at float64 on the CPU with common random numbers, on scenes
whose loss is the flux-weighted landing centroid ``x + y / 2`` on a 2 x 2
bilinear detector (linear in every landing point inside the square between
the pixel centres, where every ray lands), with the Fresnel branch probability
fixed. A top-hat disc is symmetric, so the centroid of the beam alone does not
move with its radius: the scenes send the beam through the decentred, tilted
singlet of the placement tests, whose aberrations make the centroid a smooth
function of the source's size and cone. The tolerance is the same derivation
as there: ``max(1e-7, 1.5 K u |f| / (h |f'|))`` with ``K = 1e4`` operations
and ``u = 2**-53``; it evaluates to between 1e-7 and 2.2e-7 here. Measured on
the development machine (Apple silicon, CPU, float64, 2,000 rays, seed 3):
1e-11 to 3e-10 relative.

The Jacobian in the weight: every source of the engine is specified by its
total flux spread uniformly over its area or solid angle, and its maps from
the draws are uniform, so the change of variables' ``|det J|`` and the
exitance's ``1 / A`` cancel and the birth weight ``Phi / N`` has no
derivative in the geometry (``attach_source_geometry``'s docstring). T-09-8's
case of an area that changes at fixed total flux is
:class:`TestFixedTotalFlux`: the total flux on a detector that catches every
ray has a derivative of zero, which fails if the factor enters unnormalised
(it would be ``2 Phi / a`` for a disc of radius ``a``).
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import (
    CollimatedSourceConfig,
    ExtendedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    PointSourceConfig,
    Spectrum,
)
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.nonsequential.parameter_register import (
    DETACHED,
    INTERIOR_BOUNDARY,
    DeadParameterError,
    ParameterRefused,
    ParameterRegister,
)

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


def _centroid(data, width: float):
    """Flux-weighted landing ``x + y / 2`` on a 2 x 2 bilinear detector of side ``width``."""
    q = width / 4.0
    xc = torch.tensor([-q, q, -q, q], dtype=torch.float64)
    yc = torch.tensor([-q, -q, q, q], dtype=torch.float64)
    s = data.sum()
    return (data * xc).sum() / s + 0.5 * (data * yc).sum() / s


def _detector(scene, z: float, width: float = 40.0):
    scene.add_detector(
        "D1",
        CoordinateSystem(z=z),
        IrradianceDetectorConfig(
            width=width, height=width, num_pixels_x=2, num_pixels_y=2, splat="bilinear"
        ),
    )


def _through_the_singlet(config):
    """The source at the origin, the decentred tilted singlet at z = 50, the detector at 150."""
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


def _collimated_radius(a):
    return CollimatedSourceConfig(spectrum=_SPECTRUM, total_flux=1.0, aperture_radius=a)


def _point_half_angle(alpha):
    return PointSourceConfig(spectrum=_SPECTRUM, total_flux=1.0, half_angle_deg=alpha)


def _extended(width=3.0, height=2.0, radius=None, alpha=3.0):
    return ExtendedSourceConfig(
        spectrum=_SPECTRUM,
        total_flux=1.0,
        width=width,
        height=height,
        aperture_radius=radius,
        half_angle_deg=alpha,
    )


#: (id, config builder, nominal value, the parameter's name on the source)
_CASES = [
    ("collimated-aperture_radius", _collimated_radius, 3.0, "aperture_radius"),
    ("point-half_angle_deg", _point_half_angle, 5.0, "half_angle_deg"),
    ("extended-width", lambda v: _extended(width=v), 3.0, "width"),
    ("extended-height", lambda v: _extended(height=v), 2.0, "height"),
    ("extended-aperture_radius", lambda v: _extended(radius=v), 1.5, "aperture_radius"),
    ("extended-half_angle_deg", lambda v: _extended(alpha=v), 3.0, "half_angle_deg"),
]


def _trace(scene, seed=_SEED):
    return scene.trace(num_rays=_NUM_RAYS, seed=seed, max_depth=8)


def _loss_of(config_of):
    def loss(value):
        return _centroid(_trace(_through_the_singlet(config_of(value))).detectors["D1"].data, 40.0)

    return loss


class TestSourceGeometryGradients:
    """T-09-8: autograd through the change of variables against the fourth-order difference."""

    @pytest.mark.parametrize(("case", "config_of", "value", "name"), _CASES, ids=[c[0] for c in _CASES])
    def test_against_fd4(self, case, config_of, value, name):
        loss = _loss_of(config_of)
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

    @pytest.mark.parametrize(("case", "config_of", "value", "name"), _CASES, ids=[c[0] for c in _CASES])
    def test_forward_mode_equals_reverse(self, case, config_of, value, name):
        """R-09-9: the dual tensor's tangent reaches the output and agrees with reverse mode."""
        from torch.autograd import forward_ad

        loss = _loss_of(config_of)
        param = _g(value)
        (reverse,) = torch.autograd.grad(loss(param), param)
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(
                torch.tensor(value, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
            )
            tangent = forward_ad.unpack_dual(loss(dual)).tangent
        assert tangent is not None, f"{case}: the forward-mode tangent never reached the output"
        rel = abs(tangent.item() - reverse.item()) / abs(reverse.item())
        assert rel < _K_OPS * _U, f"{case}: forward {tangent.item():.17e} reverse {reverse.item():.17e}"


class TestFixedTotalFlux:
    """T-09-8's area case: the emitting area changes at fixed total flux.

    A detector that catches every ray reads the source's total flux whatever
    its size, so the derivative of that reading in the size is zero; a
    Jacobian factor entering the weight unnormalised would give
    ``2 Phi / a`` (a disc) or ``Phi / w`` (a rectangle's side). The reading is
    a sum of the four bilinear weights of every ray, each pair's derivatives
    cancelling to rounding: the bound is ``K u`` times the unnormalised
    factor's derivative, with ``K = 1e4`` as above.
    """

    @pytest.mark.parametrize(
        ("config_of", "value", "unnormalised"),
        [
            (_collimated_radius, 2.0, lambda v: 2.0 / v),
            (lambda v: _extended(width=v, alpha=0.5), 3.0, lambda v: 1.0 / v),
            (lambda v: _extended(radius=v, alpha=0.5), 1.5, lambda v: 2.0 / v),
        ],
        ids=["collimated-disc", "extended-rectangle", "extended-disc"],
    )
    def test_total_flux_does_not_move(self, config_of, value, unnormalised):
        scene = NSQScene()
        param = _g(value)
        scene.add_source("S1", CoordinateSystem(), config_of(param))
        _detector(scene, 10.0)
        data = _trace(scene).detectors["D1"].data
        total = data.sum()
        assert float(total) == pytest.approx(1.0, rel=1e-12)
        (grad,) = torch.autograd.grad(total, param)
        assert abs(grad.item()) < _K_OPS * _U * unnormalised(value)


def _ledger(result):
    return (
        result.total_flux_detected,
        result.total_flux_escaped,
        result.total_flux_absorbed,
        result.total_flux_sampling_residual,
        result.num_rays_escaped,
    )


class TestAttachingMovesNoValue:
    """The emitted values are the host's to the bit (the regression rule)."""

    @pytest.mark.parametrize(("case", "config_of", "value", "name"), _CASES, ids=[c[0] for c in _CASES])
    def test_forward_values_bit_identical(self, case, config_of, value, name):
        plain = _trace(_through_the_singlet(config_of(value)))
        attached = _trace(_through_the_singlet(config_of(_g(value))))
        a = plain.detectors["D1"].data
        b = attached.detectors["D1"].data
        assert b.requires_grad and not a.requires_grad
        assert torch.equal(a, b.detach())
        assert _ledger(attached) == _ledger(plain)


class TestRegisterAndRaise:
    """The register's rows for source geometry, the refusals and the dead-parameter raise."""

    @pytest.mark.parametrize(("case", "config_of", "value", "name"), _CASES, ids=[c[0] for c in _CASES])
    def test_registered_as_interior_with_boundary(self, case, config_of, value, name):
        scene = _through_the_singlet(config_of(_g(value)))
        entry = ParameterRegister.from_scene(scene).find("S1", name)
        assert entry.gradient_class == INTERIOR_BOUNDARY
        assert "change of variables" in entry.stage
        result = _trace(scene)
        assert result.environment["gradient_boundary_term"] == "absent"

    def test_truncated_gaussian_radius_is_refused(self):
        """The truncation edge of a Gaussian beam moves samples across it: not attached."""
        config = CollimatedSourceConfig(
            spectrum=_SPECTRUM, total_flux=1.0, aperture_radius=_g(3.0), profile="gaussian"
        )
        with pytest.raises(NotImplementedError, match="truncates the Gaussian"):
            _through_the_singlet(config)

    def test_gaussian_sigma_stays_detached(self):
        config = CollimatedSourceConfig(
            spectrum=_SPECTRUM,
            total_flux=1.0,
            aperture_radius=3.0,
            profile="gaussian",
            gaussian_sigma=_g(1.0),
        )
        with pytest.raises(NotImplementedError):
            _through_the_singlet(config)
        from optiland.nonsequential.parameter_register import _SOURCE_CONTRACT

        assert _SOURCE_CONTRACT["gaussian_sigma"][0] == DETACHED

    def test_width_of_a_disc_source_is_dead(self):
        """An extended source with a radius ignores its width: the width reaches nothing."""
        scene = _through_the_singlet(_extended(width=_g(3.0), radius=1.5))
        with pytest.raises(DeadParameterError) as info:
            _trace(scene)
        ((owner, name, _stage, reason),) = info.value.dead
        assert (owner, name) == ("S1", "width")
        assert "no output of the trace depends on it" in reason

    def test_half_angle_of_a_lambertian_source_is_dead(self):
        """At 90 degrees and above an extended source is Lambertian: the half-angle does not enter."""
        scene = _through_the_singlet(_extended(alpha=_g(90.0)))
        with pytest.raises(DeadParameterError) as info:
            _trace(scene)
        ((owner, name, _stage, _reason),) = info.value.dead
        assert (owner, name) == ("S1", "half_angle_deg")

    def test_a_backend_without_autograd_refuses(self):
        scene = _through_the_singlet(_collimated_radius(_g(3.0)))
        be.set_backend("numpy")
        with pytest.raises(ParameterRefused, match="S1:aperture_radius"):
            _trace(scene)

    def test_the_translation_of_a_disc_is_one_for_one_in_the_radius(self):
        """An independent route: a disc's emission points scale with its radius, so
        ``d p / d a = p / a`` for every ray, read directly off the born bundle."""
        from optiland.nonsequential.parameter_register import attach_source_geometry

        scene = NSQScene()
        a = _g(2.0)
        scene.add_source("S1", CoordinateSystem(x=0.3, ry=0.1), _collimated_radius(a))
        source = scene.sources[0]
        from optiland.nonsequential.backends.torch_backend import TorchBackend
        from optiland.nonsequential.rng import NSQRng

        rays = source.generate(np.arange(64, dtype=np.int64), NSQRng(seed=_SEED))

        rays = TorchBackend()._prepare_bundle(rays)
        x0 = rays.x.detach().clone()
        rays = attach_source_geometry(rays, source)
        (dx,) = torch.autograd.grad(rays.x.sum(), a)
        # x = 0.3 + x_l cos(0.1) + z_l sin(0.1) with z_l = 0 and x_l proportional to a
        expected = ((x0 - 0.3) / 2.0).sum().item()
        assert dx.item() == pytest.approx(expected, rel=1e-12)

