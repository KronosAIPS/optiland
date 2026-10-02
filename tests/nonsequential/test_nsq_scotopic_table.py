"""The CIE scotopic V'(lambda) at 1 nm, frozen with its source (the research repository's issue 89).

The data are the CIE dataset "CIE spectral luminous efficiency for scotopic
vision" (DOI 10.25039/CIE.DS.gr6w4b5g, CIE 018:2019 Table 2, CC BY-SA 4.0).
The checks: the frozen values reproduce the CIE's CSV byte for byte (the
SHA-256 the CIE publishes in the dataset's metadata), V'(507 nm) = 1, and the
peak efficacy follows from the table: K'_m = 683 / V'(555.016 nm), the
wavelength in standard air of the 540 THz that defines the candela.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import optiland
from optiland.nonsequential.units import _scotopic_table_1nm

_DATA = Path(optiland.__file__).parent / "colorimetry" / "scotopic_data_1nm.json"


def test_the_file_states_its_source_and_terms():
    data = json.loads(_DATA.read_text())
    assert "10.25039/CIE.DS.gr6w4b5g" in data["source"]
    assert "CIE 018:2019" in data["source"]
    assert "CC BY-SA 4.0" in data["license"]


def test_the_values_are_the_cie_file_byte_for_byte():
    data = json.loads(_DATA.read_text())
    lines = [f"{380 + i},{v}" for i, v in enumerate(data["values"])]
    blob = ("\r\n".join(lines) + "\r\n").encode()
    assert hashlib.sha256(blob).hexdigest() == data["csv_sha256"]
    assert data["csv_sha256"] == (
        "6a75d3fdbcbf5e9e9a07478511933eefeda953f3e2cc14b74459e5a099ec3759"
    )


def test_the_grid_and_the_peak():
    wl, v = _scotopic_table_1nm()
    assert wl.size == v.size == 401
    np.testing.assert_array_equal(np.round(wl * 1000.0), np.arange(380.0, 781.0))
    assert np.interp(0.507, wl, v) == 1.0
    assert v.max() == 1.0
    assert set(np.round(wl[v == 1.0] * 1000.0)) == {506.0, 507.0, 508.0}
    # the 10 nm nodes the table replaces agree at every 10 nm
    assert v[120] == 0.982 and v[130] == 0.997  # 500 and 510 nm


def test_the_peak_efficacy_follows_from_the_table():
    wl, v = _scotopic_table_1nm()
    k_prime = 683.0 / np.interp(0.555016, wl, v)
    assert k_prime == pytest.approx(1700.06, abs=0.005)
