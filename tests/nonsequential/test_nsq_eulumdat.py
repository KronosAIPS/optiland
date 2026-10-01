"""The native EULUMDAT (.ldt) reader (the research repository's issue 7, KronosSRT issue 67).

The files are composed here, line by line, from the format's description (Stockmar,
1990; DIALux's knowledge base, "Description of the EULUMDAT format"), from a luminaire
whose intensity is a closed form I(gamma, C) with the symmetry each file declares; the
reader's expanded table is compared with the closed form at every listed plane, exactly
(the values are exactly representable sums and products of the composed numbers, so
the comparison is to 1e-12 relative only for the scaling's two roundings).

What is pinned: each symmetry index 0 to 4, the C270-through-C0 order of Isym 3, the
scaling by the conversion factor and the first lamp set's total flux, a decimal comma,
the refusals, and agreement with the IES reader on the same luminaire written as IES.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.nonsequential import read_eulumdat, read_ies, write_ies
from optiland.nonsequential.sources.photometric_files import PhotometricTable

GAMMA = np.arange(0.0, 91.0, 10.0)


def profile(gamma):
    return 100.0 + 200.0 * gamma / 90.0


def g_quadrant(c):
    c = c % 360.0
    c = c if c <= 180.0 else 360.0 - c
    c = c if c <= 90.0 else 180.0 - c
    return 1.0 + c / 90.0


G_C90_270 = {0.0: 1.25, 45.0: 1.125, 90.0: 1.0, 135.0: 1.125, 180.0: 1.25, 225.0: 1.375,
             270.0: 1.5, 315.0: 1.375}
G_C0_180 = {0.0: 1.0, 45.0: 1.5, 90.0: 2.0, 135.0: 1.25, 180.0: 1.75, 225.0: 1.25, 270.0: 2.0,
            315.0: 1.5}


def ldt(isym, c_angles, intensity_rows, conversion=1.0, lamp_flux=1000.0, dims=None,
        comma=False, gamma=GAMMA):
    """An EULUMDAT file's text: the header, one lamp set, the planes the symmetry stores."""
    mc, ng = len(c_angles), len(gamma)
    dc = c_angles[1] - c_angles[0] if mc > 1 else 0.0
    dims = dims or [600, 0, 80, 500, 0, 10, 10, 10, 10]
    fmt = (lambda v: repr(float(v)).replace(".", ",")) if comma else (lambda v: repr(float(v)))
    out = ["test company / catalogue / 1", "1", str(isym), str(mc), fmt(dc), str(ng),
           fmt(gamma[1] - gamma[0]), "report 1", "r1 test luminaire", "TL-1", "test.ldt",
           "2026-10-01 / test"]
    out += [str(v) for v in dims]
    out += ["100", "85", fmt(conversion), "0", "1"]
    out += ["1", "LED module", fmt(lamp_flux), "3000", "80", "12.5"]
    out += ["0.5"] * 10
    out += [fmt(c) for c in c_angles]
    out += [fmt(gm) for gm in gamma]
    for row in intensity_rows:
        out += [fmt(v) for v in row]
    return "\n".join(out) + "\n"


def stored_planes(isym, c_angles):
    mc = len(c_angles)
    if isym == 0:
        return list(range(mc))
    if isym == 1:
        return [0]
    if isym == 2:
        return list(range(mc // 2 + 1))
    if isym == 3:
        return [(3 * mc // 4 + k) % mc for k in range(mc // 2 + 1)]
    return list(range(mc // 4 + 1))


def compose(isym, c_angles, g, **kw):
    rows = [profile(GAMMA) * g(c_angles[i]) for i in stored_planes(isym, c_angles)]
    return ldt(isym, c_angles, rows, **kw)


CASES = {
    0: (list(np.arange(0.0, 360.0, 45.0)), lambda c: G_C90_270[c % 360.0]),
    1: (list(np.arange(0.0, 360.0, 30.0)), lambda c: 1.0),
    2: (list(np.arange(0.0, 360.0, 45.0)), lambda c: G_C0_180[c if c <= 180 else 360 - c]),
    3: (list(np.arange(0.0, 360.0, 45.0)), lambda c: G_C90_270[c % 360.0]),
    4: (list(np.arange(0.0, 360.0, 22.5)), g_quadrant),
}


@pytest.mark.parametrize("isym", sorted(CASES))
def test_each_symmetry_expands_to_the_closed_form(isym):
    c_angles, g = CASES[isym]
    table = read_eulumdat(compose(isym, c_angles, g))
    assert isinstance(table, PhotometricTable)
    assert np.array_equal(table.c_angles_deg, np.append(c_angles, 360.0))
    assert np.array_equal(table.gamma_angles_deg, GAMMA)
    want = np.array([profile(GAMMA) * g(c % 360.0) for c in table.c_angles_deg])
    assert np.array_equal(table.candela, want)
    assert table.keywords["isym"] == isym and table.file_format == "EULUMDAT"


def test_isym_3_lists_from_c270_through_c0():
    """Read in the order C90 to C270 instead, the two halves would swap: 1.5 at C90."""
    c_angles, g = CASES[3]
    table = read_eulumdat(compose(3, c_angles, g))
    by_c = dict(zip(table.c_angles_deg, table.candela[:, 0] / 100.0, strict=True))
    assert by_c[270.0] == 1.5 and by_c[90.0] == 1.0 and by_c[225.0] == 1.375


def test_scaling_by_conversion_factor_and_flux():
    c_angles, g = CASES[4]
    table = read_eulumdat(compose(4, c_angles, g, conversion=2.0, lamp_flux=1500.0))
    want = np.array([profile(GAMMA) * g_quadrant(c) for c in table.c_angles_deg]) * 3.0
    assert np.allclose(table.candela, want, rtol=1e-15, atol=0)
    assert table.keywords["lamp_flux_lm"] == [1500.0]


def test_decimal_comma():
    c_angles, g = CASES[0]
    a = read_eulumdat(compose(0, c_angles, g, conversion=0.5, comma=True))
    b = read_eulumdat(compose(0, c_angles, g, conversion=0.5))
    assert np.array_equal(a.candela, b.candela)


def test_from_a_path(tmp_path):
    c_angles, g = CASES[2]
    path = tmp_path / "x.ldt"
    path.write_text(compose(2, c_angles, g), encoding="ascii")
    assert np.array_equal(read_eulumdat(path).candela, read_eulumdat(str(path)).candela)


def test_agrees_with_the_ies_reader():
    """The quadrant luminaire as EULUMDAT and as an IES file: the same table, the same flux."""
    c_angles, g = CASES[4]
    a = read_eulumdat(compose(4, c_angles, g, conversion=2.0))
    b = read_ies(write_ies(a))
    assert np.array_equal(a.candela, b.candela)
    assert a.luminous_flux() == pytest.approx(b.luminous_flux(), rel=1e-15)
    assert a.luminous_opening_mm == (-500.0, 500.0, 10.0)


def test_refusals():
    c_angles, g = CASES[4]
    text = compose(4, c_angles, g)
    lines = text.splitlines()
    bad = lines.copy()
    bad[2] = "7"
    with pytest.raises(ValueError, match="Isym = 7"):
        read_eulumdat("\n".join(bad))
    with pytest.raises(ValueError, match="numbers after the lamp sets"):
        read_eulumdat("\n".join(lines[:-5]))
    odd = compose(0, [0.0, 120.0, 240.0], lambda c: 1.0).splitlines()
    odd[2] = "4"
    with pytest.raises(ValueError, match="divisible by 4"):
        read_eulumdat("\n".join(odd))
    with pytest.raises(ValueError, match="fewer lines"):
        read_eulumdat("company\n1\n0\n")


def test_to_source_config_is_a_tabulated_source():
    from optiland.nonsequential import Spectrum

    c_angles, g = CASES[3]
    cfg = read_eulumdat(compose(3, c_angles, g)).to_source_config(Spectrum.monochromatic(0.555))
    assert cfg.intensity_units == "cd"
