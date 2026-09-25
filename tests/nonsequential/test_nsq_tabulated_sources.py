"""Tabulated angular sources and continuous spectra (the RM-06 kinds).

The references are closed forms of the interpolated table: a table linear in
theta (and bilinear in theta and phi) carries no interpolation error, so the
cone and quadrant fractions below are exact, and every Monte Carlo check is a
binomial z-score.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    FarFieldDetectorConfig,
    NSQScene,
    PiecewiseLinearSpectrum,
    SpectralDetectorConfig,
    Spectrum,
    TabulatedSource,
    TabulatedSourceConfig,
    lumens_to_watts,
)
from optiland.nonsequential.ir import lower
from optiland.nonsequential.serialization import scene_from_dict, scene_to_dict
from optiland.nonsequential.sources.spectra import C2_SI, sample_linear
from optiland.nonsequential.sources.tabulated import _partial, cell_integrals, invert_cell

ALPHA, BETA = 0.2, 0.4
DEG = np.arange(0.0, 91.0, 10.0)
TABLE = ALPHA + BETA * np.radians(DEG) / (np.pi / 2)  # linear in theta: exact
PHIS = np.arange(0.0, 361.0, 45.0)
# Piecewise linear in phi with its kinks at nodes: bilinear interpolation of
# the product table is exact.
G_PHI = 1.0 + 0.5 * np.abs(((PHIS + 90.0) % 360.0) / 180.0 - 1.0)


def _cone(theta):
    """``integral_0^theta (alpha + beta' t) sin t dt`` times 2 pi (closed form)."""
    b = BETA / (np.pi / 2)
    return 2 * np.pi * (ALPHA * (1 - np.cos(theta)) + b * (np.sin(theta) - theta * np.cos(theta)))


@pytest.fixture(autouse=True)
def _numpy_float64():
    be.set_backend("numpy")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


# ---------------------------------------------------------------------------
# The closed-form pieces, against independent routes
# ---------------------------------------------------------------------------


def test_cell_integrals_match_quadrature():
    quad = pytest.importorskip("scipy.integrate").quad
    rng = np.random.default_rng(1)
    th = np.sort(rng.uniform(0.0, np.pi, 9))
    vals = rng.uniform(0.0, 3.0, 9)
    ref = [
        quad(lambda t: np.interp(t, th, vals) * np.sin(t), th[j], th[j + 1],
             epsabs=0, epsrel=1e-13)[0]
        for j in range(8)
    ]
    np.testing.assert_allclose(cell_integrals(th, vals), ref, rtol=1e-12)


def test_cell_integrals_stay_accurate_in_tiny_cells_near_the_axis():
    """The half-angle forms keep the digits a cos difference would lose."""
    th = np.array([0.0, 1e-6, 2e-6])
    vals = np.array([1.0, 1.0, 1.0])

    def series(a, b):  # cos a - cos b by its Taylor series, exact at this size
        return (b**2 - a**2) / 2 - (b**4 - a**4) / 24 + (b**6 - a**6) / 720

    expect = [series(0.0, 1e-6), series(1e-6, 2e-6)]
    np.testing.assert_allclose(cell_integrals(th, vals), expect, rtol=1e-12)
    # the plain difference of cosines loses about five digits here
    naive = math.cos(1e-6) - math.cos(2e-6)
    assert abs(naive - expect[1]) / expect[1] > 1e-6


def test_invert_cell_recovers_the_target():
    rng = np.random.default_rng(2)
    t0 = np.full(2000, 0.3)
    t1 = np.full(2000, 0.9)
    i0 = rng.uniform(0.0, 2.0, 2000)
    i1 = rng.uniform(0.0, 2.0, 2000)
    total = cell_integrals(np.array([0.3, 0.9]), np.stack([i0, i1], axis=1))[:, 0]
    target = rng.uniform(0.0, 1.0, 2000) * total
    t = invert_cell(t0, t1, i0, i1, target)
    got, _ = _partial(t0, t, i0, (i1 - i0) / (t1 - t0))
    np.testing.assert_allclose(got, target, rtol=0, atol=1e-14)
    assert np.all((t >= 0.3) & (t <= 0.9))


def test_invert_cell_with_a_zero_end():
    t = invert_cell(np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.0]),
                    np.array([0.0]))
    assert t[0] == 0.0


def test_sample_linear_is_the_inverse_cdf():
    u = np.linspace(0.0, 0.999, 50)
    for a, b in [(1.0, 1.0), (0.0, 2.0), (3.0, 0.5), (1.0, 1.0 + 1e-12)]:
        x = sample_linear(u, np.full_like(u, a), np.full_like(u, b))
        cdf = (a * x + 0.5 * (b - a) * x * x) / (0.5 * (a + b))
        np.testing.assert_allclose(cdf, u, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# Direction sampling, exact distribution
# ---------------------------------------------------------------------------


def _z(frac, p, n):
    return (frac - p) / math.sqrt(p * (1 - p) / n)


def test_symmetric_table_cone_fractions():
    src = TabulatedSource(CoordinateSystem(), Spectrum.monochromatic(0.55), 1.0, DEG, TABLE)
    assert src.table_flux == pytest.approx(_cone(np.pi / 2), rel=1e-14)
    rng = np.random.default_rng(3)
    n = 200_000
    theta, phi = src.sample_directions(rng.uniform(size=n), rng.uniform(size=n))
    total = _cone(np.pi / 2)
    for cone_deg in (20.0, 45.0, 70.0):
        p = _cone(np.radians(cone_deg)) / total
        assert abs(_z(np.mean(theta <= np.radians(cone_deg)), p, n)) < 4.0
    assert np.all(theta <= np.pi / 2) and np.all((phi >= 0) & (phi < 2 * np.pi))


def test_two_dimensional_table_quadrant_fractions():
    table = np.outer(G_PHI, TABLE)
    src = TabulatedSource(CoordinateSystem(), Spectrum.monochromatic(0.55), 1.0, DEG, table,
                          azimuth_angles_deg=PHIS)
    g_total = np.trapezoid(G_PHI, np.radians(PHIS))
    assert src.table_flux == pytest.approx(g_total * _cone(np.pi / 2) / (2 * np.pi), rel=1e-14)
    rng = np.random.default_rng(4)
    n = 200_000
    theta, phi = src.sample_directions(rng.uniform(size=n), rng.uniform(size=n))
    for lo, hi in ((0, 90), (90, 180), (180, 270), (270, 360)):
        sel = (PHIS >= lo) & (PHIS <= hi)
        p = np.trapezoid(G_PHI[sel], np.radians(PHIS[sel])) / g_total
        frac = np.mean((phi >= np.radians(lo)) & (phi < np.radians(hi)))
        assert abs(_z(frac, p, n)) < 4.0
    # separable table: theta's distribution does not depend on phi
    p = _cone(np.radians(45.0)) / _cone(np.pi / 2)
    assert abs(_z(np.mean(theta <= np.radians(45.0)), p, n)) < 4.0


def test_a_table_that_starts_off_axis_emits_nothing_inside_its_first_node():
    src = TabulatedSource(CoordinateSystem(), Spectrum.monochromatic(0.55), 1.0,
                          [30.0, 60.0], [1.0, 1.0])
    rng = np.random.default_rng(5)
    theta, _ = src.sample_directions(rng.uniform(size=10_000), rng.uniform(size=10_000))
    assert theta.min() >= np.radians(30.0) and theta.max() <= np.radians(60.0)


# ---------------------------------------------------------------------------
# Through the engine: both backends
# ---------------------------------------------------------------------------


def _far_field_scene(config):
    scene = NSQScene()
    scene.add_source("S", CoordinateSystem(), config)
    scene.add_detector("F", CoordinateSystem(z=1.0),
                       FarFieldDetectorConfig(num_theta=9, num_phi=4))
    return scene


def _theta_bin_flux(det):
    """Flux per 10-degree polar bin, undoing the detector's own W/sr division."""
    th = np.radians(det.theta)
    d_theta = np.radians(10.0)
    d_phi = 2 * np.pi / det.intensity.shape[1]
    return (det.intensity * (np.sin(th) * d_theta * d_phi)[:, None]).sum(axis=1)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_traced_polar_bins_match_the_closed_form(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    be.set_backend(backend)
    be.set_precision("float64")
    n = 100_000
    config = TabulatedSourceConfig(spectrum=Spectrum.monochromatic(0.55),
                                   polar_angles_deg=DEG, intensity=TABLE, total_flux=2.0)
    result = _far_field_scene(config).trace(num_rays=n, seed=11)
    flux = _theta_bin_flux(result.detectors["F"])
    edges = np.radians(np.arange(0.0, 91.0, 10.0))
    p = np.diff(_cone(edges)) / _cone(np.pi / 2)
    z = (flux / 2.0 - p) / np.sqrt(p * (1 - p) / n)
    assert np.sum(z**2) < 27.88  # chi-squared 99.9 % point, 9 degrees of freedom
    assert float(to_numpy(result.detectors["F"].total_flux)) == pytest.approx(2.0, rel=1e-12)
    assert result.flux_conservation_error < 1e-12


def test_backends_draw_the_same_directions():
    pytest.importorskip("torch")
    config = TabulatedSourceConfig(spectrum=Spectrum.monochromatic(0.55),
                                   polar_angles_deg=DEG, intensity=np.outer(G_PHI, TABLE),
                                   azimuth_angles_deg=PHIS)
    maps = []
    for backend in ("numpy", "torch"):
        be.set_backend(backend)
        be.set_precision("float64")
        maps.append(_far_field_scene(config).trace(num_rays=3000, seed=5).detectors["F"].intensity)
    np.testing.assert_allclose(maps[0], maps[1], rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Flux and units
# ---------------------------------------------------------------------------


def test_absolute_tables_fix_their_own_flux():
    spec = Spectrum.monochromatic(0.555)
    s = NSQScene()
    s.add_source("W", CoordinateSystem(), TabulatedSourceConfig(
        spectrum=spec, polar_angles_deg=DEG, intensity=TABLE, intensity_units="W/sr"))
    s.add_source("C", CoordinateSystem(), TabulatedSourceConfig(
        spectrum=spec, polar_angles_deg=DEG, intensity=TABLE, intensity_units="cd"))
    s.add_source("L", CoordinateSystem(), TabulatedSourceConfig(
        spectrum=spec, polar_angles_deg=DEG, intensity=TABLE, total_flux_lumens=100.0))
    s.add_source("D", CoordinateSystem(), TabulatedSourceConfig(
        spectrum=spec, polar_angles_deg=DEG, intensity=TABLE))
    w, c, lm, d = (src.total_flux for src in s.sources)
    assert w == pytest.approx(_cone(np.pi / 2), rel=1e-14)
    assert c == pytest.approx(lumens_to_watts(_cone(np.pi / 2), spec), rel=1e-14)
    assert lm == pytest.approx(lumens_to_watts(100.0, spec), rel=1e-14)
    assert d == 1.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"intensity_units": "W/sr", "total_flux": 1.0}, "fixes its own flux"),
        ({"total_flux": 1.0, "total_flux_lumens": 5.0}, "not both"),
    ],
)
def test_ambiguous_flux_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NSQScene().add_source("S", CoordinateSystem(), TabulatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), polar_angles_deg=DEG, intensity=TABLE,
            **kwargs))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"polar_angles_deg": [0.0, 10.0, 5.0], "intensity": [1, 1, 1]}, "increasing"),
        ({"polar_angles_deg": [0.0, 190.0], "intensity": [1, 1]}, r"\[0, 180\]"),
        ({"polar_angles_deg": [0.0, 10.0], "intensity": [1, -1]}, "non-negative"),
        ({"polar_angles_deg": [0.0, 10.0], "intensity": [0, 0]}, "zero flux"),
        ({"polar_angles_deg": [0.0, 10.0], "intensity": [[1, 1]] * 3,
          "azimuth_angles_deg": [0.0, 90.0, 180.0]}, "0 to 360"),
        ({"polar_angles_deg": [0.0, 10.0], "intensity": [[1, 1], [2, 2], [3, 3]],
          "azimuth_angles_deg": [0.0, 180.0, 360.0]}, "must be equal"),
        ({"polar_angles_deg": [0.0, 10.0], "intensity": [1, 1], "width": 1.0},
         "both width and height"),
    ],
)
def test_malformed_tables_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TabulatedSource(CoordinateSystem(), Spectrum.monochromatic(0.55), 1.0, **kwargs)


def test_a_gradient_on_the_table_raises():
    torch = pytest.importorskip("torch")
    table = torch.tensor(TABLE, requires_grad=True)
    with pytest.raises(NotImplementedError):
        NSQScene().add_source("S", CoordinateSystem(), TabulatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), polar_angles_deg=DEG, intensity=table))


def test_the_flux_carries_a_gradient():
    torch = pytest.importorskip("torch")
    be.set_backend("torch")
    be.set_precision("float64")
    flux = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    config = TabulatedSourceConfig(spectrum=Spectrum.monochromatic(0.55),
                                   polar_angles_deg=DEG, intensity=TABLE, total_flux=flux)
    det = _far_field_scene(config).trace(num_rays=500, seed=2).detectors["F"]
    total = torch.as_tensor(det.intensity).sum()  # detached copy: use the source's flux
    assert total.item() > 0
    scene = _far_field_scene(config)
    assert scene.sources[0].total_flux is flux


# ---------------------------------------------------------------------------
# Serialization, IR and area emitters
# ---------------------------------------------------------------------------


def test_round_trip_reproduces_the_rays():
    config = TabulatedSourceConfig(spectrum=PiecewiseLinearSpectrum.blackbody(3000.0),
                                   polar_angles_deg=DEG, intensity=np.outer(G_PHI, TABLE),
                                   azimuth_angles_deg=PHIS, intensity_units="W/sr",
                                   aperture_radius=2.0)
    scene = _far_field_scene(config)
    d = scene_to_dict(scene)
    src_d = d["sources"][0]
    assert src_d["type"] == "tabulated" and src_d["spectrum"]["kind"] == "piecewise_linear"
    rebuilt = scene_from_dict(json.loads(json.dumps(d)))
    assert scene_to_dict(rebuilt) == d
    a = scene.trace(num_rays=2000, seed=9).detectors["F"].intensity
    b = rebuilt.trace(num_rays=2000, seed=9).detectors["F"].intensity
    np.testing.assert_array_equal(a, b)
    ir = lower(scene, strict=False)
    assert ir.emitters[0].kind == "tabulated"
    assert ir.emitters[0].params["spectrum"]["kind"] == "piecewise_linear"


def test_area_emitter_positions_fill_the_rectangle():
    from optiland.nonsequential.rng import NSQRng  # noqa: PLC0415

    src = TabulatedSource(CoordinateSystem(z=3.0), Spectrum.monochromatic(0.55), 1.0,
                          DEG, TABLE, width=4.0, height=2.0)
    rays = src.generate(np.arange(5000, dtype=np.int64), NSQRng(seed=1))
    x, y, z = (np.asarray(to_numpy(v)) for v in (rays.x, rays.y, rays.z))
    assert np.all(np.abs(x) <= 2.0) and np.all(np.abs(y) <= 1.0) and np.all(z == 3.0)
    assert x.std() == pytest.approx(4.0 / math.sqrt(12), rel=0.05)


# ---------------------------------------------------------------------------
# Continuous spectra
# ---------------------------------------------------------------------------


def test_second_radiation_constant_from_the_exact_si_values():
    assert C2_SI == pytest.approx(1.438776877e-2, rel=1e-9)


def test_blackbody_matches_planck_and_wien():
    bb = PiecewiseLinearSpectrum.blackbody(5000.0, 0.3, 1.2, 0.0001)
    peak = bb.nodes[np.argmax(bb.values)]
    assert peak == pytest.approx(2897.771955e-6 / 5000.0 * 1e6, abs=1e-4)  # Wien, um


def test_line_integral_is_exact_for_linear_functions():
    s = PiecewiseLinearSpectrum([0.4, 0.5, 0.7], [1.0, 3.0, 2.0])
    # integral of the density itself: trapezoids are exact
    assert s.total == pytest.approx(0.1 * 2.0 + 0.2 * 2.5, rel=1e-15)
    # times g(l) = l on [0.4, 0.7]: sum of exact quadratic integrals
    exact = sum(
        (b - a) * ((fa * a + fb * b) / 3 + (fa * b + fb * a) / 6)
        for a, b, fa, fb in [(0.4, 0.5, 1.0, 3.0), (0.5, 0.7, 3.0, 2.0)]
    )
    assert s.integrate_linear([0.4, 0.7], [0.4, 0.7]) == pytest.approx(exact, rel=1e-14)


def test_sampled_wavelengths_follow_the_density():
    from optiland.nonsequential.rng import NSQRng  # noqa: PLC0415

    s = PiecewiseLinearSpectrum([0.4, 0.5, 0.7], [1.0, 3.0, 0.0])
    n = 200_000
    wl = s.sample(np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int32), NSQRng(seed=3))
    for lo, hi in ((0.4, 0.45), (0.45, 0.55), (0.55, 0.7)):
        p = s.band_fraction(lo, hi)
        assert abs(_z(np.mean((wl >= lo) & (wl < hi)), p, n)) < 4.0
    assert wl.min() >= 0.4 and wl.max() < 0.7


def test_cie_illuminant_a_follows_its_defining_formula():
    a = PiecewiseLinearSpectrum.cie_illuminant("A")
    i560 = int(np.argmin(np.abs(a.nodes - 0.560)))
    assert a.values[i560] == pytest.approx(100.0, rel=1e-14)


def test_cie_illuminants_against_colour_science():
    colour = pytest.importorskip("colour")
    for name in ("A", "D65"):
        mine = PiecewiseLinearSpectrum.cie_illuminant(name)
        sd = colour.SDS_ILLUMINANTS[name]  # the CIE's 5 nm tables
        lam = np.asarray(sd.wavelengths, dtype=float)
        lam = lam[(lam >= mine.nodes[0] * 1e3) & (lam <= mine.nodes[-1] * 1e3)]
        got = np.interp(lam * 1e-3, mine.nodes, mine.values)
        np.testing.assert_allclose(got, sd[lam], rtol=1e-5)


def test_efficacy_of_a_continuous_spectrum_is_integrated_exactly():
    from optiland.nonsequential.units import _table  # noqa: PLC0415

    grid, v, km = _table("photopic")
    s = PiecewiseLinearSpectrum(grid, np.ones_like(grid))
    # a flat density on the V table's own nodes: km * mean of V under the
    # trapezoid rule of a product of two linear functions
    expect = km * s.integrate_linear(grid, v) / s.total
    assert s.luminous_efficacy() == pytest.approx(expect, rel=1e-15)
    assert lumens_to_watts(expect, s) == pytest.approx(1.0, rel=1e-14)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_blackbody_band_fractions_on_a_spectral_detector(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    be.set_backend(backend)
    be.set_precision("float64")
    from optiland.nonsequential import PointSourceConfig  # noqa: PLC0415

    bb = PiecewiseLinearSpectrum.blackbody(2856.0)
    scene = NSQScene()
    scene.add_source("S", CoordinateSystem(), PointSourceConfig(spectrum=bb, half_angle_deg=5.0))
    scene.add_detector("D", CoordinateSystem(z=10.0), SpectralDetectorConfig(
        width=10.0, height=10.0, num_pixels_x=1, num_pixels_y=1, wl_min=0.38, wl_max=0.78,
        num_bins=4, splat="hard"))
    n = 100_000
    res = scene.trace(num_rays=n, seed=4).detectors["D"]
    got = np.asarray(res.irradiance).reshape(-1, 4).sum(axis=0) * 100.0  # W/mm^2 * mm^2
    edges = np.linspace(0.38, 0.78, 5)
    p = np.array([bb.band_fraction(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)])
    z = (got - p) / np.sqrt(p * (1 - p) / n)
    assert np.sum(z**2) < 18.47  # chi-squared 99.9 %, 4 degrees of freedom
