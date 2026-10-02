"""The scatter models' weight paths: the Harvey-Shack lobe's l0 and s, every reflect-or-transmit fraction.

The research repository's chapter 09 section 9.14.3 (written and committed
before these tests ran) and its issue 93. A scatter model draws a direction
from a density ``q`` at the parameters' host values (detached) and returns the
weight ``w = f / q``; the weight is attached through the lobe ``f`` only, so
its derivative is ``w * d log f / d theta`` at the drawn direction: the
likelihood-ratio estimator, unbiased for the whole derivative of the
expectation (the directions' movement included), with a higher variance than a
reparameterised direction would have. The reflect-or-transmit fraction ``tau``
is the branch estimator of section 9.2: drawn with the detached ``p``, each
branch's weight carries ``tau / p`` or ``(1 - tau) / (1 - p)``.

What is asserted, with the chapter's bounds:

- the forward values are the bits of the build without a gradient;
- each fraction's derivative of the far-side and near-side flux of a diffuser
  against its closed form, within 5 computed standard errors;
- the Harvey-Shack score ``d log f`` at the drawn directions against closed
  forms and 30-digit quadrature, at normal incidence (1e-9 relative to the
  larger term) and at 30 and 60 degrees (1e-5);
- the Harvey-Shack derivative of a detector's flux against its quadrature,
  within 5 computed standard errors;
- ``b0`` refused as zero by structure, and a gradient-carrying fraction of
  0 or 1 refused;
- the register's class and stage for each attached argument.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402

from scipy import integrate

import optiland.backend as be
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    ReflectiveComponent,
    RefractiveComponent,
    Spectrum,
    kinds,
)
from optiland.nonsequential.bsdf.harvey_shack import HarveyShackBSDF
from optiland.nonsequential.bsdf.lambertian import LambertianBSDF
from optiland.nonsequential.bsdf.tabulated import TabulatedBSDF
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry
from optiland.nonsequential.parameter_register import INTERIOR, ParameterRegister
from optiland.nonsequential.rng import NSQRng

_L = 100.0  # detector distance [mm]
_A = 10.0  # detector half-side [mm]
_TAU = 0.3
_RHO = 0.8
_L0 = 0.05
_N_CLOSED = 100_000


@pytest.fixture(autouse=True)
def _torch_float64():
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


def _g(value) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def _detector(scene, name, z):
    scene.add_detector(
        name,
        CoordinateSystem(z=z),
        IrradianceDetectorConfig(width=2 * _A, height=2 * _A, num_pixels_x=4, num_pixels_y=4),
    )


def _diffuser_scene(bsdf):
    """A point-like collimated beam at normal incidence on a plane diffuser in vacuum.

    The far detector is ``_L`` behind the diffuser, the near one ``_L`` in
    front of it (the source sits between, at z = 50, and emits away from it).
    """
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(z=50.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=1e-6
        ),
    )
    scene.add_component(
        "P", RefractiveComponent(CoordinateSystem(z=100.0), PlaneGeometry(), VACUUM, VACUUM, bsdf=bsdf)
    )
    _detector(scene, "far", 100.0 + _L)
    _detector(scene, "near", 100.0 - _L)
    return scene


def _mirror_scene(bsdf):
    """The same beam on a plane mirror carrying the lobe; the detector ``_L`` in front."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(z=50.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=1e-6
        ),
    )
    scene.add_component(
        "M",
        ReflectiveComponent(CoordinateSystem(z=100.0), PlaneGeometry(), reflectance=1.0, bsdf=bsdf),
    )
    _detector(scene, "near", 100.0 - _L)
    return scene


def _trace(scene, num_rays, seed=3, **kw):
    return scene.trace(num_rays=num_rays, seed=seed, max_depth=8, **kw)


def _ledger(result):
    return (
        result.total_flux_detected,
        result.total_flux_escaped,
        result.total_flux_absorbed,
        result.total_flux_sampling_residual,
        result.num_rays_escaped,
    )


@pytest.fixture(scope="module")
def flat_table(tmp_path_factory):
    """A tabulated BSDF of constant value kappa / pi: its weight is kappa for every ray."""
    path = tmp_path_factory.mktemp("bsdf") / "flat.csv"
    rows = [
        f"{ti},{ts},{_RHO / math.pi!r}"
        for ti in np.linspace(0.0, 90.0, 10)
        for ts in np.linspace(0.0, 90.0, 19)
    ]
    path.write_text("# theta_i, theta_s, bsdf\n" + "\n".join(rows) + "\n")
    return path


# -- values do not move ---------------------------------------------------------------


class TestValuesUnchanged:
    """Attaching the weight paths changes no forward bit (the regression rule)."""

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    @pytest.mark.parametrize("kind", ["lambertian", "tabulated", "harvey_shack"])
    def test_images_and_ledger_bit_identical(self, kind, precision, flat_table):
        be.set_precision(precision)

        def make(attach: bool):
            v = _g if attach else float
            if kind == "lambertian":
                return LambertianBSDF(_RHO, v(_TAU))
            if kind == "tabulated":
                return TabulatedBSDF(flat_table, v(_TAU))
            return HarveyShackBSDF(1.0, v(_L0), v(2.0), v(_TAU))

        plain = _trace(_diffuser_scene(make(False)), 20_000)
        attached = _trace(_diffuser_scene(make(True)), 20_000)
        for name in ("far", "near"):
            a = plain.detectors[name].data
            b = attached.detectors[name].data
            assert b.requires_grad and not a.requires_grad
            assert torch.equal(a, b.detach()), name
        assert _ledger(plain) == _ledger(attached)


# -- the reflect-or-transmit fraction against its closed form ---------------------------


def _view_factor(half_side: float, distance: float) -> float:
    """Fraction of a cosine (Lambertian) lobe from a point that falls on a centred square.

    The differential-area-to-parallel-rectangle form, four corner rectangles:
    ``(4 / pi) X / sqrt(1 + X^2) atan(X / sqrt(1 + X^2))``, ``X = a / L``.
    """
    x = half_side / distance
    r = math.sqrt(1.0 + x * x)
    return 4.0 / math.pi * (x / r) * math.atan(x / r)


def test_view_factor_formula_against_quadrature():
    """The closed form against a direct quadrature of cos(theta) / pi over the square."""
    a, el = _A, _L
    direct, _ = integrate.dblquad(
        lambda y, x: el * el / (math.pi * (x * x + y * y + el * el) ** 2),
        -a, a, -a, a, epsabs=0.0, epsrel=1e-12,
    )
    assert _view_factor(a, el) == pytest.approx(direct, rel=1e-10, abs=0.0)


def _abg_region(l0: float, s: float, dlog_t0: tuple[float, float]):
    """The lobe's detected fraction, its derivatives and second moments at normal incidence.

    Over the detector's square in the gnomonic coordinates tau (chapter 09
    section 9.14.3): ``F = int B / T (1 + |tau|^2)^-2``, ``dF_k = int B / T D_k
    (...)``, ``M_k = int B / T D_k^2 (...)``, by quadrature over the eighth
    0 <= tau_y <= tau_x <= a / L (the integrand has the square's symmetry).
    """
    t_norm, _ = integrate.quad(
        lambda t: 2 * math.pi * t / (1.0 + (t / l0) ** s), 0.0, 1.0, points=[l0], epsabs=0.0, epsrel=1e-13, limit=400
    )
    edge = _A / _L

    def parts(ty, tx):
        r2 = tx * tx + ty * ty
        beta = math.sqrt(r2 / (1.0 + r2))
        x = (beta / l0) ** s
        frac = x / (1.0 + x)
        d_l0 = (s / l0) * frac - dlog_t0[0]
        d_s = (-math.log(beta / l0) * frac if beta > 0 else 0.0) - dlog_t0[1]
        f = 1.0 / (1.0 + x) / t_norm / (1.0 + r2) ** 2
        return f, d_l0, d_s

    def q(fun):
        val, _ = integrate.dblquad(
            lambda ty, tx: fun(*parts(ty, tx)), 0.0, edge, 0.0, lambda tx: tx, epsabs=0.0, epsrel=1e-11
        )
        return 8.0 * val

    return {
        "F": q(lambda f, a, b: f),
        "dF": (q(lambda f, a, b: f * a), q(lambda f, a, b: f * b)),
        "M2": (q(lambda f, a, b: f * a * a), q(lambda f, a, b: f * b * b)),
    }


#: d log T / d l0 and d log T / d s at incidences 0, 30 and 60 degrees, from
#: 30-digit quadrature of the lobe's integrals (the research repository's
#: ``docs/theory/checks_09_scatter.py``, check C9.6; mathematics only, no
#: engine code). At normal incidence and s = 2 the l0 value also equals the closed
#: form 2 / l0 - 2 / (l0 (l0^2 + 1) ln(1 + l0^-2)).
_DLOG_T = {
    (0.05, 2.0, 0): (33.343258901402695, -1.3629372192037657),
    (0.05, 2.0, 30): (33.023490231731725, -1.3126034412849323),
    (0.05, 2.0, 60): (31.699115729048269, -1.2018227978470052),
    (0.05, 1.5, 0): (26.548285508645063, -1.7626016025312586),
    (0.05, 1.5, 30): (26.353994866692469, -1.7331163193117752),
    (0.05, 1.5, 60): (25.866769430902509, -1.7396975710374911),
}


def _closed_dlog_t0_l0_s2(l0: float) -> float:
    return 2.0 / l0 - 2.0 / (l0 * (l0 * l0 + 1.0) * math.log(1.0 + 1.0 / (l0 * l0)))


class TestBranchFractionClosedForm:
    """d(flux)/d(tau) on the far and the near side of a diffuser, against the closed forms."""

    @pytest.mark.parametrize("kind", ["lambertian", "tabulated", "harvey_shack"])
    def test_far_and_near_side(self, kind, flat_table):
        tau = _g(_TAU)
        if kind == "lambertian":
            bsdf = LambertianBSDF(_RHO, tau)
            share = _RHO * _view_factor(_A, _L)
        elif kind == "tabulated":
            bsdf = TabulatedBSDF(flat_table, tau)
            share = _RHO * _view_factor(_A, _L)
        else:
            bsdf = HarveyShackBSDF(1.0, _L0, 2.0, tau)
            dlog = (_closed_dlog_t0_l0_s2(_L0), 0.0)  # the derivatives are not used for F
            share = _abg_region(_L0, 2.0, dlog)["F"]
        result = _trace(_diffuser_scene(bsdf), _N_CLOSED)
        far = result.detectors["far"].data.sum()
        near = result.detectors["near"].data.sum()
        (g_far,) = torch.autograd.grad(far, tau, retain_graph=True)
        (g_near,) = torch.autograd.grad(near, tau)
        # The far side: E = share; one transmitted ray that lands carries 1 / p.
        # A weight of at most 1 per ray (rho, kappa or the lobe's 1): the count
        # is binomial with probability p * share / weight, so the standard
        # error is weight / p * sqrt(q (1 - q) / N), q = p * share / weight.
        weight = _RHO if kind != "harvey_shack" else 1.0
        for got, p, sign in ((g_far.item(), _TAU, 1.0), (g_near.item(), 1.0 - _TAU, -1.0)):
            q = p * share / weight
            se = weight / p * math.sqrt(q * (1.0 - q) / _N_CLOSED)
            assert abs(got - sign * share) < 5.0 * se, (
                f"{kind}: {got:.6f} vs {sign * share:.6f}: {abs(got - sign * share) / se:.2f} SE"
            )


# -- the Harvey-Shack score at fixed directions -----------------------------------------


def _sample_at_incidence(bsdf, deg: float, num: int = 4096):
    """Scatter ``num`` rays at one incidence through ``sample``; return delta, g and the weights."""
    th = math.radians(deg)
    d = torch.tensor([[math.sin(th), 0.0, math.cos(th)]] * num, dtype=torch.float64)
    n = torch.tensor([[0.0, 0.0, -1.0]] * num, dtype=torch.float64)
    ids = torch.arange(num, dtype=torch.int64)
    bounce = torch.zeros(num, dtype=torch.int64)
    out, w, _ = bsdf.sample(num, d, n, torch.full((num,), 0.55, dtype=torch.float64), NSQRng(7), ids, bounce)
    spec = d - 2.0 * (d * n).sum(dim=1, keepdim=True) * n
    beta0 = spec[:, :2]
    beta = out.detach()[:, :2]
    delta = (beta - beta0).norm(dim=1)
    return delta.numpy(), math.sin(th), w


class TestHarveyShackScore:
    """The engine's d log f at the drawn direction against the chapter's closed forms."""

    @pytest.mark.parametrize("deg", [0.0, 30.0, 60.0])
    @pytest.mark.parametrize("slope", [2.0, 1.5])
    @pytest.mark.parametrize("which", ["l0", "s"])
    def test_score(self, which, slope, deg):
        from torch.autograd import forward_ad

        ref = _DLOG_T[(_L0, slope, int(deg))]
        with forward_ad.dual_level():
            l0 = forward_ad.make_dual(torch.tensor(_L0, dtype=torch.float64), torch.tensor(1.0 if which == "l0" else 0.0, dtype=torch.float64))
            s = forward_ad.make_dual(torch.tensor(slope, dtype=torch.float64), torch.tensor(1.0 if which == "s" else 0.0, dtype=torch.float64))
            bsdf = HarveyShackBSDF(1.0, l0, s)
            delta, _g0, w = _sample_at_incidence(bsdf, deg)
            primal, tangent = forward_ad.unpack_dual(w)
            engine = (tangent / primal).numpy()
        x = (delta / _L0) ** slope
        frac = x / (1.0 + x)
        if which == "l0":
            first = (slope / _L0) * frac
            second = ref[0]
        else:
            first = np.where(delta > 0, -np.log(np.where(delta > 0, delta, 1.0) / _L0) * frac, 0.0)
            second = ref[1]
        expected = first - second
        scale = np.maximum(np.abs(first), abs(second))
        bound = 1e-9 if deg == 0.0 else 1e-5
        rel = np.abs(engine - expected) / scale
        assert np.isfinite(engine).all()
        assert rel.max() < bound, f"worst {rel.max():.2e} (bound {bound:.0e})"

    def test_closed_form_matches_the_quadrature_reference(self):
        """The s = 2 closed form for d log T / d l0 at normal incidence equals the 30-digit record."""
        assert _closed_dlog_t0_l0_s2(_L0) == pytest.approx(_DLOG_T[(_L0, 2.0, 0)][0], rel=1e-14)


class TestHarveyShackTrace:
    """d(detected flux)/d(l0, s) through a trace, against quadrature, within 5 computed SE."""

    @pytest.mark.parametrize("slope", [2.0, 1.5])
    def test_detected_flux(self, slope):
        l0, s = _g(_L0), _g(slope)
        result = _trace(_mirror_scene(HarveyShackBSDF(1.0, l0, s)), 200_000)
        flux = result.detectors["near"].data.sum()
        g_l0, g_s = torch.autograd.grad(flux, [l0, s])
        ref = _abg_region(_L0, slope, _DLOG_T[(_L0, slope, 0)])
        for k, got in enumerate((g_l0.item(), g_s.item())):
            mean = ref["dF"][k]
            se = math.sqrt((ref["M2"][k] - mean * mean) / 200_000)
            assert abs(got - mean) < 5.0 * se, (
                f"{('l0', 's')[k]} at s = {slope}: {got:.6f} vs {mean:.6f}: {abs(got - mean) / se:.2f} SE"
            )
        # The forward fraction too, as a check on the scene (binomial SE).
        frac = flux.item()
        se_f = math.sqrt(ref["F"] * (1 - ref["F"]) / 200_000)
        assert abs(frac - ref["F"]) < 5.0 * se_f


# -- refusals and the register -----------------------------------------------------------


class TestRefusalsAndRegister:
    def test_b0_is_refused_as_zero_by_structure(self):
        with pytest.raises(NotImplementedError, match="zero by structure"):
            HarveyShackBSDF(_g(1.0), _L0, 2.0)

    @pytest.mark.parametrize("value", [0.0, 1.0])
    @pytest.mark.parametrize("kind", ["lambertian", "harvey_shack"])
    def test_a_fraction_of_zero_or_one_with_a_gradient_is_refused(self, kind, value):
        with pytest.raises(NotImplementedError, match="never draws one of its two branches"):
            if kind == "lambertian":
                LambertianBSDF(_RHO, _g(value))
            else:
                HarveyShackBSDF(1.0, _L0, 2.0, _g(value))

    def test_plain_numbers_of_zero_and_one_still_build(self):
        LambertianBSDF(_RHO, 0.0)
        LambertianBSDF(_RHO, 1.0)
        HarveyShackBSDF(1.0, _L0, 2.0, 1.0)

    @pytest.mark.parametrize(
        ("kind", "make", "arg"),
        [
            ("lambertian", lambda v: LambertianBSDF(_RHO, v), "transmissive_fraction"),
            ("harvey_shack", lambda v: HarveyShackBSDF(1.0, v, 2.0), "l0"),
            ("harvey_shack", lambda v: HarveyShackBSDF(1.0, _L0, v), "s"),
            ("harvey_shack", lambda v: HarveyShackBSDF(1.0, _L0, 2.0, v), "transmissive_fraction"),
        ],
        ids=["lambertian-tau", "hs-l0", "hs-s", "hs-tau"],
    )
    def test_registered_with_the_rule_class_and_stage(self, kind, make, arg):
        value = _g({"l0": _L0, "s": 2.0}.get(arg, _TAU))
        scene = _mirror_scene(make(value))
        rows = {r["name"]: r for r in ParameterRegister.from_scene(scene).rows()}
        row = rows[f"bsdf.{arg}"]
        rule = kinds.registry("bsdf").by_name(kind).gradients[arg]
        assert rule.is_attached and rule.gradient_class == INTERIOR
        assert row["gradient_class"] == INTERIOR
        assert row["stage"] == rule.text
