"""A truncated Gaussian beam's sigma and radius, by the implicit reparameterisation.

``docs/theory/09_differentiation.md`` of the research repository, section
9.13.1 (written before these tests ran), requirement R-09-4; the research
repository's issue 31. The beam's radial coordinate, held at a fixed value
``u = F(r)`` of its truncated distribution, is a smooth function of sigma and
of the truncation radius ``R``; its derivative is the implicit one,
``-dF/dtheta / (dF/dr)``, evaluated at the value the engine's rejection
sampler drew. The derivative is the default (``profile_gradient="implicit"``,
the maintainer's ruling of 2026-10-01); a beam built with
``profile_gradient="refuse"`` refuses a gradient on either parameter.

The bounds, each derived in the chapter before the run:

- the trace's estimate of ``d E[r^2] / d theta`` within 5 standard errors of
  the closed form (:class:`TestUnbiasedAgainstTheClosedForm`);
- the engine's tangent against reverse-mode autograd of the explicit inverse
  distribution to ``1e3 u`` of the formula's largest term
  (:class:`TestTangentFormula`);
- the whole trace of a twin source that draws by the inverse distribution
  against the fourth-order central difference, to
  ``max(1e-7, 1.5 K u |f| / (h |f'|))`` with ``K = 1e4``
  (:class:`TestTraceAgainstFD4`);
- the total flux on a detector that catches every ray, its derivative below
  ``K u`` times ``2 Phi / theta`` (:class:`TestFixedTotalFlux`);
- values bit-identical with and without the gradient.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential._utils import host_float
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.nonsequential.parameter_register import (
    attach_source_geometry,
    truncated_gaussian_tangents,
)
from optiland.nonsequential.rng import EventSlot, NSQRng
from optiland.nonsequential.sources.collimated import CollimatedSource

_NUM_RAYS = 2_000
_SEED = 3
_U = 2.0**-53
_K_OPS = 1e4
_SPECTRUM = Spectrum.monochromatic(0.55)
_LENS = {"x": 0.5, "y": -0.3, "z": 50.0, "rx": 0.01, "ry": -0.02, "rz": 0.0}
_SIGMA = 1.5
_RADIUS = 3.0


@pytest.fixture(autouse=True)
def _torch_float64():
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


def _g(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def _beam(sigma=_SIGMA, radius=_RADIUS, *, follow=False, cls=CollimatedSource, cs=None):
    """A Gaussian beam built with the implicit reparameterisation."""
    return cls(
        cs if cs is not None else CoordinateSystem(),
        _SPECTRUM,
        total_flux=1.0,
        aperture_radius=radius,
        profile="gaussian",
        gaussian_sigma=None if follow else sigma,
        profile_gradient="implicit",
    )


class _InverseCdfBeam(CollimatedSource):
    """The twin: the same beam drawn by the inverse distribution, ``r = F^-1(u)``.

    Its rays are exactly the reparameterised samples, so its trace is a smooth
    function of sigma and the radius under common random numbers, which the
    engine's rejection sampler is not (chapter 09 section 9.13.1).
    """

    def _sample_gaussian_disk(self, ray_id, bounce0, rng):
        u1 = to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1))
        u2 = to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U2))
        s = host_float(self.gaussian_sigma)
        big_r = host_float(self.aperture_radius)
        z = -np.expm1(-big_r * big_r / (2.0 * s * s))
        r = s * np.sqrt(-2.0 * np.log1p(-u1 * z))
        phi = 2.0 * np.pi * u2
        return r * np.cos(phi), r * np.sin(phi)


def _detector(scene, z: float, width: float = 40.0):
    scene.add_detector(
        "D1",
        CoordinateSystem(z=z),
        IrradianceDetectorConfig(
            width=width, height=width, num_pixels_x=2, num_pixels_y=2, splat="bilinear"
        ),
    )


def _through_the_singlet(source):
    """The beam at the origin, the decentred tilted singlet at z = 50, the detector at 150."""
    scene = NSQScene()
    scene.source_registry.add("S1", source)
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


# -- the closed form of chapter 09 section 9.13.1 ----------------------------------


def _closed_form(sigma: float, radius: float):
    """``E[r^2]`` of the truncated beam and its two partial derivatives."""
    big_a = radius * radius / (2.0 * sigma * sigma)
    em1 = math.expm1(big_a)
    e = em1 + 1.0
    m = 2.0 * sigma * sigma - radius * radius / em1
    dm_dsigma = 4.0 * sigma - 2.0 * radius * radius * big_a / sigma * e / (em1 * em1)
    dm_dradius = -2.0 * radius / em1 + 2.0 * big_a * radius * e / (em1 * em1)
    return m, dm_dsigma, dm_dradius


class TestTangentFormula:
    """The engine's implicit tangent against autograd of the explicit inverse distribution."""

    @pytest.mark.parametrize(
        ("sigma", "radius"),
        [(1.5, 3.0), (2.0, 1.0), (0.5, 3.0), (1.0, 40.0)],
        ids=["A=2", "A=0.125", "A=18", "A=800"],
    )
    def test_against_the_explicit_map(self, sigma, radius):
        """The explicit map ``r = sigma sqrt(-2 ln(1 - u Z))`` differentiated at 40 digits.

        The reference is evaluated in 40-digit arithmetic: in float64 the
        explicit map is ill-conditioned near the edge when ``A`` is large
        (``1 - u Z`` cancels: at ``A = 18`` and ``u = 1 - 1e-12`` its relative
        rounding is about ``u / 1.5e-8``), which the first run of this test
        showed; the engine's formula, evaluated at the reference's ``r``
        rounded to float64, is held to the bound derived in the chapter.
        """
        mpmath = pytest.importorskip("mpmath")
        mp = mpmath.mp
        mp.dps = 40
        s_mp, r_big = mpmath.mpf(sigma), mpmath.mpf(radius)

        def explicit(s, big_r, u):
            z = -mpmath.expm1(-big_r * big_r / (2 * s * s))
            return s * mpmath.sqrt(-2 * mpmath.log1p(-u * z))

        us = [*np.linspace(1e-6, 1.0 - 1e-6, 101), 1e-12, 0.5, 1.0 - 1e-12]
        r_list, d_sigma, d_radius = [], [], []
        for u in us:
            u_mp = mpmath.mpf(float(u))
            r_list.append(float(explicit(s_mp, r_big, u_mp)))
            d_sigma.append(float(mpmath.diff(lambda x, u_mp=u_mp: explicit(x, r_big, u_mp), s_mp)))
            d_radius.append(float(mpmath.diff(lambda x, u_mp=u_mp: explicit(s_mp, x, u_mp), r_big)))
        r = torch.tensor(r_list, dtype=torch.float64)
        rel_sigma, rel_radius = truncated_gaussian_tangents(r * r, sigma, radius)
        engine_sigma, engine_radius = r * rel_sigma, r * rel_radius
        q = (rel_radius / radius) * r * r  # Q
        scale_sigma = torch.maximum(r / sigma, radius * radius * q / (sigma * r))
        scale_radius = (radius * q / r).clamp_min(1e-300)
        bound = 1e3 * _U
        ref_sigma = torch.tensor(d_sigma, dtype=torch.float64)
        ref_radius = torch.tensor(d_radius, dtype=torch.float64)
        err_sigma = ((engine_sigma - ref_sigma).abs() / scale_sigma).max().item()
        err_radius = ((engine_radius - ref_radius).abs() / scale_radius).max().item()
        assert err_sigma < bound, f"sigma tangent: {err_sigma / _U:.1f} u of the largest term"
        assert err_radius < bound, f"radius tangent: {err_radius / _U:.1f} u of the term"

    def test_limits(self):
        """At the edge the sigma tangent is 0 and the radius tangent 1; at the axis both are finite."""
        sigma, radius = 1.5, 3.0
        r2 = torch.tensor([radius * radius, 0.0], dtype=torch.float64)
        rel_sigma, rel_radius = truncated_gaussian_tangents(r2, sigma, radius)
        assert (radius * rel_sigma[0]).item() == pytest.approx(0.0, abs=64 * _U * radius / sigma)
        assert (radius * rel_radius[0]).item() == pytest.approx(1.0, rel=64 * _U)
        big_a = radius * radius / (2 * sigma * sigma)
        g0 = math.exp(-big_a) / (2 * sigma * sigma * -math.expm1(-big_a))
        assert rel_radius[1].item() == pytest.approx(radius * g0, rel=64 * _U)
        assert torch.isfinite(rel_sigma).all() and torch.isfinite(rel_radius).all()


class TestUnbiasedAgainstTheClosedForm:
    """The estimate of ``d E[r^2] / d theta`` within 5 standard errors of the closed form."""

    @pytest.mark.parametrize("which", ["sigma", "radius", "radius-with-sigma-following"])
    def test_within_five_standard_errors(self, which):
        from torch.autograd import forward_ad

        from optiland.nonsequential.backends.torch_backend import TorchBackend

        num_rays = 20_000
        with forward_ad.dual_level():
            one = torch.tensor(1.0, dtype=torch.float64)
            if which == "sigma":
                source = _beam(sigma=forward_ad.make_dual(torch.tensor(_SIGMA, dtype=torch.float64), one))
            elif which == "radius":
                source = _beam(radius=forward_ad.make_dual(torch.tensor(_RADIUS, dtype=torch.float64), one))
            else:
                source = _beam(
                    radius=forward_ad.make_dual(torch.tensor(_RADIUS, dtype=torch.float64), one),
                    follow=True,
                )
            rays = source.generate(np.arange(num_rays, dtype=np.int64), NSQRng(seed=_SEED))
            rays = TorchBackend()._prepare_bundle(rays)
            rays = attach_source_geometry(rays, source)
            terms = forward_ad.unpack_dual(rays.x * rays.x + rays.y * rays.y).tangent.detach().clone()
        sigma = _RADIUS / 2.0 if which.endswith("following") else _SIGMA
        _, dm_dsigma, dm_dradius = _closed_form(sigma, _RADIUS)
        expected = {
            "sigma": dm_dsigma,
            "radius": dm_dradius,
            # sigma = R / 2: the total derivative in R
            "radius-with-sigma-following": dm_dradius + 0.5 * dm_dsigma,
        }[which]
        estimate = terms.mean().item()
        se = terms.std().item() / math.sqrt(num_rays)
        assert abs(estimate - expected) < 5.0 * se, (
            f"{which}: estimate {estimate:.6f} closed form {expected:.6f}, "
            f"{abs(estimate - expected) / se:.2f} standard errors"
        )

    def test_reverse_mode_gives_the_same_mean(self):
        """The per-ray forward tangents and reverse mode's gradient of the mean are one derivative."""
        from optiland.nonsequential.backends.torch_backend import TorchBackend

        sigma = _g(_SIGMA)
        source = _beam(sigma=sigma)
        rays = source.generate(np.arange(4_000, dtype=np.int64), NSQRng(seed=_SEED))
        rays = TorchBackend()._prepare_bundle(rays)
        rays = attach_source_geometry(rays, source)
        r2 = rays.x * rays.x + rays.y * rays.y
        (grad,) = torch.autograd.grad(r2.mean(), sigma)
        rel_sigma, _ = truncated_gaussian_tangents(r2.detach(), _SIGMA, _RADIUS)
        assert grad.item() == pytest.approx((2.0 * r2.detach() * rel_sigma).mean().item(), rel=1e3 * _U)


#: (id, keyword of the attached value, nominal, follow)
_FD4_CASES = [
    ("sigma", "sigma", _SIGMA, False),
    ("radius", "radius", _RADIUS, False),
    ("radius-sigma-follows", "radius", _RADIUS, True),
]


class TestTraceAgainstFD4:
    """The whole trace of the inverse-distribution twin against the fourth-order difference."""

    @pytest.mark.parametrize(("case", "key", "value", "follow"), _FD4_CASES, ids=[c[0] for c in _FD4_CASES])
    def test_against_fd4(self, case, key, value, follow):
        def loss(v):
            source = _beam(**{key: v}, follow=follow, cls=_InverseCdfBeam)
            return _centroid(_trace(_through_the_singlet(source)).detectors["D1"].data, 40.0)

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
    """The total flux does not move with sigma or the radius: no Jacobian factor enters."""

    @pytest.mark.parametrize(("key", "value"), [("sigma", _SIGMA), ("radius", _RADIUS)])
    def test_total_flux_does_not_move(self, key, value):
        scene = NSQScene()
        param = _g(value)
        scene.source_registry.add("S1", _beam(**{key: param}))
        _detector(scene, 10.0)
        total = _trace(scene).detectors["D1"].data.sum()
        assert float(total) == pytest.approx(1.0, rel=1e-12)
        (grad,) = torch.autograd.grad(total, param)
        assert abs(grad.item()) < _K_OPS * _U * 2.0 / value


class TestAttachingMovesNoValue:
    """The emitted values are the rejection sampler's, to the bit (the regression rule)."""

    @pytest.mark.parametrize("key", ["sigma", "radius"])
    def test_forward_values_bit_identical(self, key):
        value = _SIGMA if key == "sigma" else _RADIUS
        plain_source = CollimatedSource(
            CoordinateSystem(), _SPECTRUM, total_flux=1.0, aperture_radius=_RADIUS,
            profile="gaussian", gaussian_sigma=_SIGMA,
        )
        plain = _trace(_through_the_singlet(plain_source))
        attached = _trace(_through_the_singlet(_beam(**{key: _g(value)})))
        a = plain.detectors["D1"].data
        b = attached.detectors["D1"].data
        assert b.requires_grad and not a.requires_grad
        assert torch.equal(a, b.detach())
        assert attached.total_flux_detected == plain.total_flux_detected


class TestTheSwitch:
    """``profile_gradient`` is validated, reaches the source through the scene, and defaults to attaching."""

    def test_default_attaches_and_refuse_refuses(self):
        """The default attaches sigma; ``profile_gradient="refuse"`` refuses it.

        Changed under the maintainer's ruling of 2026-10-01 on slide 63,
        question 2 (with question 5 of the rulings R4 of the same day): this
        test asserted that the default refuses. The implicit
        reparameterisation is now the default, so the test asserts that the
        default keeps the tensor given, and that the refusal it guarded is
        still what ``profile_gradient="refuse"`` does.
        """
        sigma = _g(1.5)
        source = CollimatedSource(
            CoordinateSystem(), _SPECTRUM, aperture_radius=3.0, profile="gaussian",
            gaussian_sigma=sigma,
        )
        assert source.profile_gradient == "implicit"
        assert source.gaussian_sigma is sigma
        assert CollimatedSourceConfig(spectrum=_SPECTRUM).profile_gradient == "implicit"
        with pytest.raises(NotImplementedError, match="profile_gradient='implicit'"):
            CollimatedSource(
                CoordinateSystem(), _SPECTRUM, aperture_radius=3.0, profile="gaussian",
                gaussian_sigma=_g(1.5), profile_gradient="refuse",
            )

    def test_unknown_value(self):
        with pytest.raises(ValueError, match="profile_gradient"):
            CollimatedSource(CoordinateSystem(), _SPECTRUM, profile_gradient="relaxed")

    def test_through_the_scene_builder(self):
        scene = NSQScene()
        sigma = _g(_SIGMA)
        scene.add_source(
            "S1",
            CoordinateSystem(),
            CollimatedSourceConfig(
                spectrum=_SPECTRUM, total_flux=1.0, aperture_radius=_RADIUS,
                profile="gaussian", gaussian_sigma=sigma, profile_gradient="implicit",
            ),
        )
        _detector(scene, 10.0, width=4.0)
        data = _trace(scene).detectors["D1"].data
        (grad,) = torch.autograd.grad((data * torch.arange(4.0, dtype=torch.float64)).sum(), sigma)
        assert torch.isfinite(grad)
