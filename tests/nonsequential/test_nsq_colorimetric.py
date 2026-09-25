"""Photometric and colorimetric detectors (the RM-09 kinds): per-ray CIE weights."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    ColorimetricDetectorConfig,
    ColorimetricFarFieldDetectorConfig,
    FarFieldDetectorConfig,
    IrradianceDetectorConfig,
    NSQScene,
    PiecewiseLinearSpectrum,
    PointSourceConfig,
    Spectrum,
)
from optiland.nonsequential.detectors.colorimetric import cmf_at, cmf_table
from optiland.nonsequential.ir import lower
from optiland.nonsequential.results.colorimetric import (
    chromaticity_uv_prime,
    chromaticity_xy,
    cct_of_xy,
)
from optiland.nonsequential.serialization import scene_from_dict, scene_to_dict
from optiland.nonsequential.units import KM_PHOTOPIC

GRID_UM = (380.0 + np.arange(401)) * 1e-3


@pytest.fixture(autouse=True)
def _numpy_float64():
    be.set_backend("numpy")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _scene(spectrum, config, half_angle=5.0, z=10.0):
    s = NSQScene()
    s.add_source("S", CoordinateSystem(), PointSourceConfig(spectrum=spectrum, half_angle_deg=half_angle))
    s.add_detector("C", CoordinateSystem(z=z), config)
    return s


def test_the_table_is_the_cie_1931_observer_at_its_peak():
    tab = cmf_table()
    assert tab.shape == (3, 401)
    assert tab[1, 555 - 380] == 1.0  # V(555 nm) = 1
    np.testing.assert_allclose(cmf_at([0.555]).ravel(), tab[:, 175], rtol=0)
    assert np.all(cmf_at([0.379, 0.781]) == 0.0)


def test_cmf_against_colour_science():
    colour = pytest.importorskip("colour")
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    lam = np.arange(380, 781)
    np.testing.assert_allclose(cmf_table().T, cmfs[lam], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("lam_um", [0.45, 0.5555, 0.62])
def test_monochromatic_tristimulus_is_exact(lam_um):
    cfg = ColorimetricDetectorConfig(width=10, height=10, num_pixels_x=1, num_pixels_y=1, splat="hard")
    res = _scene(Spectrum.monochromatic(lam_um), cfg).trace(num_rays=5000, seed=1).detectors["C"]
    np.testing.assert_allclose(res.total_tristimulus, cmf_at([lam_um]).ravel(), rtol=1e-12)
    assert res.luminous_flux == pytest.approx(KM_PHOTOPIC * cmf_at([lam_um])[1, 0], rel=1e-12)


def test_candela_at_555_nm():
    cfg = ColorimetricFarFieldDetectorConfig(num_theta=9, num_phi=4)
    res = _scene(Spectrum.monochromatic(0.555), cfg, half_angle=90.0, z=1.0).trace(
        num_rays=20_000, seed=2).detectors["C"]
    hit = res.radiometric.intensity > 0
    np.testing.assert_allclose(res.luminous_intensity[hit] / res.radiometric.intensity[hit],
                               683.002, rtol=1e-13)


def test_the_radiometric_map_is_the_plain_detectors_own():
    spec = PiecewiseLinearSpectrum.blackbody(3000.0)
    plain = _scene(spec, IrradianceDetectorConfig(width=4, height=4, num_pixels_x=8, num_pixels_y=8))
    colour = _scene(spec, ColorimetricDetectorConfig(width=4, height=4, num_pixels_x=8, num_pixels_y=8))
    a = plain.trace(num_rays=4000, seed=5).detectors["C"]
    b = colour.trace(num_rays=4000, seed=5).detectors["C"]
    np.testing.assert_array_equal(b.radiometric.irradiance, a.irradiance)
    assert b.radiometric.num_rays_hit == a.num_rays_hit


def test_far_field_radiometric_pattern_and_count_are_untouched():
    spec = PiecewiseLinearSpectrum.blackbody(3000.0)
    a = _scene(spec, FarFieldDetectorConfig(num_theta=6, num_phi=4), half_angle=60.0, z=1.0).trace(
        num_rays=3000, seed=4).detectors["C"]
    b = _scene(spec, ColorimetricFarFieldDetectorConfig(num_theta=6, num_phi=4), half_angle=60.0,
               z=1.0).trace(num_rays=3000, seed=4).detectors["C"]
    np.testing.assert_array_equal(b.radiometric.intensity, a.intensity)
    assert b.radiometric.total_flux == a.total_flux
    assert b.radiometric.num_rays_hit == a.num_rays_hit


def _batched_xyz(spec, n_batches=25, per_batch=8000):
    cfg = ColorimetricDetectorConfig(width=10, height=10, num_pixels_x=1, num_pixels_y=1, splat="hard")
    vals = []
    for k in range(n_batches):
        res = _scene(spec, cfg).trace(num_rays=per_batch, seed=100 + k).detectors["C"]
        vals.append(res.total_tristimulus)
    vals = np.asarray(vals)
    return vals.mean(axis=0), vals.std(axis=0, ddof=1) / math.sqrt(n_batches)


def test_blackbody_tristimulus_matches_the_exact_integral():
    spec = PiecewiseLinearSpectrum.blackbody(2856.0, 0.36, 0.83)
    tab = cmf_table()
    exact = np.array([spec.integrate_linear(GRID_UM, tab[c]) for c in range(3)]) / spec.total
    mean, se = _batched_xyz(spec)
    z = (mean - exact) / se
    assert np.all(np.abs(z) < 4.0)


@pytest.mark.parametrize("backend", ["torch"])
def test_torch_books_the_same_tristimulus(backend):
    pytest.importorskip("torch")
    spec = PiecewiseLinearSpectrum.blackbody(4000.0)
    cfg = ColorimetricDetectorConfig(width=6, height=6, num_pixels_x=3, num_pixels_y=3, splat="hard")
    ref = _scene(spec, cfg).trace(num_rays=3000, seed=8).detectors["C"].tristimulus
    be.set_backend(backend)
    be.set_precision("float64")
    got = _scene(spec, cfg).trace(num_rays=3000, seed=8).detectors["C"].tristimulus
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)


def test_torch_float32_books_the_tristimulus():
    pytest.importorskip("torch")
    spec = Spectrum.monochromatic(0.5555)
    cfg = ColorimetricDetectorConfig(width=10, height=10, num_pixels_x=1, num_pixels_y=1, splat="hard")
    be.set_backend("torch")
    be.set_precision("float32")
    res = _scene(spec, cfg).trace(num_rays=2000, seed=1).detectors["C"]
    np.testing.assert_allclose(res.total_tristimulus, cmf_at([0.5555]).ravel(), rtol=2e-6)


def test_chromaticity_formulas():
    xyz = np.array([0.95047, 1.0, 1.08883])  # D65 white point
    x, y = chromaticity_xy(xyz)
    assert (x, y) == pytest.approx((0.312727, 0.329023), abs=2e-6)
    u, v = chromaticity_uv_prime(xyz)
    assert (u, v) == pytest.approx((4 * 0.95047 / (0.95047 + 15 + 3 * 1.08883),
                                    9 / (0.95047 + 15 + 3 * 1.08883)), rel=1e-15)


def test_cct_of_a_planckian_through_colour_science():
    pytest.importorskip("colour")
    spec = PiecewiseLinearSpectrum.blackbody(2856.0, 0.36, 0.83, 0.0005)
    tab = cmf_table()
    xyz = np.array([spec.integrate_linear(GRID_UM, tab[c]) for c in range(3)])
    cct, duv = cct_of_xy(chromaticity_xy(xyz))
    assert cct == pytest.approx(2856.0, abs=3.0)
    assert abs(duv) < 5e-4


def test_colorimetric_kinds_round_trip():
    s = NSQScene()
    s.add_source("S", CoordinateSystem(), PointSourceConfig(spectrum=Spectrum.monochromatic(0.5)))
    s.add_detector("C", CoordinateSystem(z=3), ColorimetricDetectorConfig(width=2, height=3))
    s.add_detector("F", CoordinateSystem(z=4), ColorimetricFarFieldDetectorConfig(num_theta=5))
    d = scene_to_dict(s)
    assert [x["type"] for x in d["detectors"]] == ["colorimetric", "colorimetric_far_field"]
    assert scene_to_dict(scene_from_dict(json.loads(json.dumps(d)))) == d
    assert [e.kind for e in lower(s).sensors] == ["colorimetric", "colorimetric_far_field"]
