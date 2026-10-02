"""The photopic V(lambda) is the CIE 1 nm table (KronosNSRT issue 82).

The photometric conversions used a 10 nm table whose nodes 550 and 560 nm flank
the 555 nm peak (both 0.995), so a monochromatic 555 nm watt was 679.587 lm
against K_m = 683.002 lm, while the colorimetric detectors read the CIE 1931
ybar at 1 nm. Both routes now read the one 1 nm table.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.nonsequential import Spectrum
from optiland.nonsequential.detectors.colorimetric import cmf_at, cmf_table
from optiland.nonsequential.units import (
    KM_PHOTOPIC,
    _table,
    luminous_efficacy_of_spectrum,
    lumens_to_watts,
    v_lambda,
)

U64 = 2.0**-53


def test_the_peak_is_one_at_555_nm():
    assert v_lambda(0.555) == 1.0
    assert luminous_efficacy_of_spectrum(np.array([0.555]), np.array([1.0])) == KM_PHOTOPIC


def test_a_555_nm_watt_is_km_lumens():
    # lumens / (Km V): one division of Km by Km, exact
    assert lumens_to_watts(KM_PHOTOPIC, Spectrum.monochromatic(0.555)) == 1.0


def test_the_table_is_the_colorimetric_ybar_at_every_node():
    grid, v, km = _table("photopic")
    assert km == KM_PHOTOPIC
    assert grid.size == 401
    np.testing.assert_array_equal(np.round(grid * 1000.0), np.arange(380.0, 781.0))
    np.testing.assert_array_equal(v, cmf_table()[1])


def test_the_two_photometric_routes_agree_between_the_nodes():
    """units interpolates in micrometres on nodes n / 1000, the colorimetric detector in
    nanometres on nodes n: the same linear interpolant up to the rounding of the
    abscissae (a node n / 1000 and lambda * 1000 within 1 ulp each) and of the
    interpolation itself (a subtraction, a division, a product and a sum). With
    |dV/dlambda| < 0.04 per nm on the table (its largest first difference) and
    lambda < 780 nm, each abscissa rounding moves V by less than 0.04 * 780 * u,
    and the interpolation's four roundings by 4 u: 2 * 31.2 u + 4 u < 70 u."""
    lam = np.linspace(0.380, 0.780, 40001)
    a = v_lambda(lam)
    b = cmf_at(lam)[1]
    assert np.max(np.abs(np.diff(_table("photopic")[1]))) < 0.04
    assert np.max(np.abs(a - b)) <= 70.0 * U64


def test_the_scotopic_table_is_the_cie_1nm_table():
    # was test_the_scotopic_table_keeps_its_own_grid (41 nodes at 10 nm); the
    # research repository's issue 89 replaced the grid, under the maintainer's ruling
    grid, v, km = _table("scotopic")
    assert grid.size == v.size == 401
    assert v_lambda(0.500, weighting="scotopic") == pytest.approx(0.982, abs=1e-15)
    assert v_lambda(0.507, weighting="scotopic") == 1.0
