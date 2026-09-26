"""The Harvey-Shack lobe is normalised over the directions it can reach.

Issue 18 of the research repository: the lobe used to be tabulated over the
direction-cosine disk of radius 2, a sample beyond the unit circle was given
weight zero, and a lossless rough mirror absorbed the scatter beyond the
horizon (13 percent of it at ``l0 = 0.01``). These tests pin the fix to the
theory rather than to a traced case:

- the reported TIS is the hemispherical integral, checked against the
  slope-2 closed form ``pi b0 l^2 ln(1 + 1/l^2)`` (chapter 11 section
  11.4.24) and, off normal incidence, against an independent quadrature;
- the sampled lobe follows ``f`` restricted to the reachable directions, with
  unit weight at normal incidence and unit mean weight off it;
- a lossless rough mirror loses nothing and its ledger closes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    HarveyShackBSDF,
    IrradianceDetectorConfig,
    NSQRng,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.geometry import PlaneGeometry
from optiland.nonsequential.components.reflective import ReflectiveComponent

# The case's lobe (cases/r1_24 of the research repository): 2 nm rms at 0.55 um.
B0 = 0.7216457738659138
L0 = 0.01


def _closed_form_c(beta, b0=B0, l0=L0):
    """Slope-2 lobe integrated over offsets up to ``beta``: pi b0 l^2 ln(1 + beta^2/l^2)."""
    return math.pi * b0 * l0 * l0 * np.log1p((np.asarray(beta, dtype=float) / l0) ** 2)


def _reachable_integral(g, delta_cap=None, n_psi=200_000):
    """Integral of the slope-2 lobe over the unit disk about a reference at |beta0| = g.

    Midpoint rule in the azimuth about the reference, with the closed-form
    radial integral: an independent route from the engine's tables. With
    ``delta_cap`` the offsets are also capped (a cone about the reference).
    """
    psi = (np.arange(n_psi) + 0.5) * (math.pi / n_psi)
    dmax = -g * np.cos(psi) + np.sqrt(np.maximum(1.0 - (g * np.sin(psi)) ** 2, 0.0))
    if delta_cap is not None:
        dmax = np.minimum(dmax, delta_cap)
    return float(np.mean(_closed_form_c(dmax)))


def _sample(bsdf, theta_deg, n, seed=7):
    t = math.radians(theta_deg)
    d = np.tile([math.sin(t), 0.0, math.cos(t)], (n, 1))
    normals = np.tile([0.0, 0.0, -1.0], (n, 1))
    dirs, w, _ = bsdf.sample(
        n, d, normals, np.full(n, 0.55), NSQRng(seed), np.arange(n),
        np.zeros(n, dtype=np.int32),
    )
    return np.asarray(be.to_numpy(dirs)), np.asarray(be.to_numpy(w)), t


class TestTheReportedTis:
    def setup_method(self):
        be.set_backend("numpy")

    def test_is_the_hemispherical_integral(self):
        bsdf = HarveyShackBSDF(B0, L0, 2.0)
        tis = bsdf.total_integrated_scatter
        assert tis == pytest.approx(_closed_form_c(1.0), rel=1e-12)
        # The case file's roughness TIS, (4 pi sigma / lambda)^2 at 2 nm, 0.55 um.
        assert tis == pytest.approx((4 * math.pi * 2e-3 / 0.55) ** 2, rel=1e-12)
        # Not the radius-2 integral, 15 percent more.
        assert abs(tis / _closed_form_c(2.0) - 1.0) > 0.1

    def test_the_radial_table_is_the_closed_form(self):
        bsdf = HarveyShackBSDF(B0, L0, 2.0)
        bsdf._build_tables()
        beta = bsdf._beta_grid[1:]
        c = bsdf._cdf_grid[1:] * bsdf._tis_disk
        np.testing.assert_allclose(c, _closed_form_c(beta), rtol=1e-12)

    @pytest.mark.parametrize("theta_deg", [0.0, 20.0, 45.0, 70.0, 85.0])
    def test_off_normal_incidence(self, theta_deg):
        bsdf = HarveyShackBSDF(B0, L0, 2.0)
        g = math.sin(math.radians(theta_deg))
        got = float(bsdf.total_integrated_scatter_at(math.radians(theta_deg)))
        # The table's own interpolation error is below 3e-7 (the module's
        # measurement); the reference is an independent quadrature.
        assert got == pytest.approx(_reachable_integral(g), rel=1e-6)


class TestTheSampledLobe:
    def setup_method(self):
        be.set_backend("numpy")

    def test_every_weight_is_one_at_normal_incidence(self):
        dirs, w, _ = _sample(HarveyShackBSDF(B0, L0, 2.0), 0.0, 100_000)
        assert np.all(w == 1.0)
        assert np.all(dirs[:, 2] < 0.0)  # every sample on the reflecting side

    def test_every_weight_is_one_at_normal_incidence_in_float32(self):
        torch = pytest.importorskip("torch")
        be.set_backend("torch")
        be.set_precision("float32")
        try:
            n = 50_000
            bsdf = HarveyShackBSDF(B0, L0, 2.0)
            d = torch.tensor([[0.0, 0.0, 1.0]]).repeat(n, 1)
            normals = torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1)
            _, w, _ = bsdf.sample(
                n, d, normals, torch.full((n,), 0.55), NSQRng(7),
                torch.arange(n), torch.zeros(n, dtype=torch.int32),
            )
            assert bool(torch.all(w == 1.0))
        finally:
            be.set_precision("float64")
            be.set_backend("numpy")

    @pytest.mark.parametrize("theta_deg", [40.0, 75.0])
    def test_the_weight_has_unit_mean_off_normal_incidence(self, theta_deg):
        n = 400_000
        dirs, w, _ = _sample(HarveyShackBSDF(B0, L0, 2.0), theta_deg, n)
        se = w.std() / math.sqrt(n)
        assert abs(w.mean() - 1.0) < 4.0 * se + 1e-6
        assert np.all(dirs[:, 2] < 0.0)
        assert w.min() >= 0.0

    @pytest.mark.parametrize("theta_deg", [0.0, 60.0])
    def test_the_cone_fractions_follow_the_reachable_lobe(self, theta_deg):
        """Weighted share of the samples within an offset delta of the specular.

        The target is the lobe restricted to the unit disk, normalised by its
        integral there, from the independent quadrature above.
        """
        n = 400_000
        dirs, w, t = _sample(HarveyShackBSDF(B0, L0, 2.0), theta_deg, n, seed=3)
        # Specular direction cosines in the plane: (sin t, 0); the sample's
        # transverse direction cosines are its x, y components.
        delta = np.hypot(dirs[:, 0] - math.sin(t), dirs[:, 1])
        g = math.sin(t)
        total = _reachable_integral(g)
        for cap in (0.001, 0.01, 0.1, 0.5):
            inside = delta < cap
            share = np.sum(w * inside) / n
            p = _reachable_integral(g, cap) / total
            se = math.sqrt(np.var(w * inside) / n)
            assert abs(share - p) < 4.0 * se, (cap, share, p, se)

    def test_the_old_radius_two_share_is_gone(self):
        """At normal incidence nothing is discarded: the returned share is one,
        where the radius-2 normalisation returned TIS_1 / TIS_2 = 0.8692."""
        _, w, _ = _sample(HarveyShackBSDF(B0, L0, 2.0), 0.0, 100_000)
        assert w.mean() == 1.0
        assert _closed_form_c(1.0) / _closed_form_c(2.0) == pytest.approx(0.8691834, rel=1e-6)


def _rough_mirror_scene(tilt_deg: float) -> NSQScene:
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=2.0
        ),
    )
    scene.add_component(
        "M",
        ReflectiveComponent(
            CoordinateSystem(z=20.0, rx=math.radians(tilt_deg)),
            PlaneGeometry(),
            reflectance=1.0,
            bsdf=HarveyShackBSDF(B0, L0, 2.0),
            name="M",
            scatter_fraction=1.0,
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=-40.0, rx=math.pi),
        IrradianceDetectorConfig(width=400, height=400, num_pixels_x=8, num_pixels_y=8),
    )
    return scene


class TestALosslessRoughMirror:
    def setup_method(self):
        be.set_backend("numpy")

    def test_loses_nothing_at_normal_incidence(self):
        result = _rough_mirror_scene(0.0).trace(
            num_rays=50_000, seed=5, max_depth=4, backend=NumpyBackend(seed=5)
        )
        assert result.total_flux_coating == 0.0
        assert result.flux_conservation_error < 1e-11
        # Everything the lobe returns leaves the mirror, detected or escaped;
        # the rest is the scatter gate's own event residual (its probability
        # is clamped a hair below one), not a surface loss.
        leaving = (
            result.total_flux_detected
            + result.total_flux_escaped
            + result.total_flux_sampling_residual
        )
        assert leaving == pytest.approx(1.0, abs=1e-12)

    def test_the_ledger_closes_off_normal_incidence(self):
        result = _rough_mirror_scene(30.0).trace(
            num_rays=50_000, seed=5, max_depth=4, backend=NumpyBackend(seed=5)
        )
        assert result.flux_conservation_error < 1e-11
        # The lobe's weight is a sampling weight here; what the surface books
        # of it is a zero-mean fluctuation, not a loss of the old 13 percent.
        assert abs(result.total_flux_coating) < 1e-3
