"""IES LM-63 and EULUMDAT files (the RM-07 reader, writer and frame mapping)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    FarFieldDetectorConfig,
    NSQScene,
    PhotometricTable,
    Spectrum,
    lumens_to_watts,
    read_eulumdat,
    read_ies,
    write_ies,
)
from optiland.nonsequential.sources.photometric_files import expand_lateral_symmetry

GAMMA = np.arange(0.0, 91.0, 10.0)
PROFILE = 200.0 + 400.0 * GAMMA / 90.0  # cd, linear in gamma: exact under interpolation


@pytest.fixture(autouse=True)
def _numpy_float64():
    be.set_backend("numpy")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _ies_text(c_angles, rows, gamma=GAMMA, multiplier=1.0, ballast=1.0, tilt="NONE", ptype=1):
    """An LM-63-2002 file composed here, independently of the engine's writer."""
    lines = ["IESNA:LM-63-2002", "[TEST] fixture", "[MANUFAC] none", f"TILT={tilt}"]
    lines.append(f"1 -1 {multiplier} {len(gamma)} {len(c_angles)} {ptype} 2 0 0 0")
    lines.append(f"{ballast} 1 10")
    lines.append(" ".join(f"{g:g}" for g in gamma))
    lines.append(", ".join(f"{c:g}" for c in c_angles))  # commas are separators too
    for row in rows:
        lines.append(" ".join(repr(float(v)) for v in row))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------


def test_write_then_read_is_exact():
    rng = np.random.default_rng(0)
    c = np.arange(0.0, 361.0, 30.0)
    cd = rng.uniform(0.0, 1000.0, (c.size, GAMMA.size))
    cd[-1] = cd[0]
    table = PhotometricTable(c, GAMMA, cd, luminous_opening_mm=(-20.0, 0.0, 5.0))
    back = read_ies(write_ies(table, keywords={"TEST": "round trip"}))
    np.testing.assert_array_equal(back.candela, table.candela)
    np.testing.assert_array_equal(back.c_angles_deg, table.c_angles_deg)
    np.testing.assert_array_equal(back.gamma_angles_deg, table.gamma_angles_deg)
    assert back.luminous_opening_mm == pytest.approx((-20.0, 0.0, 5.0))
    assert back.keywords["TEST"] == "round trip"


def test_multipliers_are_applied():
    t = read_ies(_ies_text([0.0], [PROFILE], multiplier=2.0, ballast=0.5))
    np.testing.assert_allclose(t.candela[0], PROFILE, rtol=1e-15)


@pytest.mark.parametrize(
    "text, match",
    [
        (_ies_text([0.0], [PROFILE], tilt="INCLUDE"), "TILT"),
        (_ies_text([0.0], [PROFILE], ptype=2), "type B"),
        (_ies_text([0.0], [PROFILE], ptype=3), "type A"),
    ],
)
def test_tilt_and_types_a_b_are_refused(text, match):
    with pytest.raises(NotImplementedError, match=match):
        read_ies(text)


def test_a_truncated_file_is_refused():
    text = _ies_text([0.0], [PROFILE]).rsplit("\n", 2)[0]
    with pytest.raises(ValueError, match="numbers"):
        read_ies(text)


# ---------------------------------------------------------------------------
# Lateral symmetry: every file form expands to the same full table
# ---------------------------------------------------------------------------


def _g(c):
    """A quadrant-symmetric azimuth profile: g(C) = g(180 - C) = g(360 - C)."""
    folded = np.minimum(c % 180.0, 180.0 - c % 180.0)  # 0..90
    return 1.0 + folded / 90.0


def _full(c_step=22.5):
    c = np.arange(0.0, 360.0 + c_step / 2, c_step)
    return c, np.outer(_g(c), PROFILE)


@pytest.mark.parametrize("lo, hi", [(0.0, 90.0), (0.0, 180.0), (90.0, 270.0), (0.0, 360.0)])
def test_symmetric_files_expand_to_the_full_table(lo, hi):
    c_full, rows_full = _full()
    sel = (c_full >= lo) & (c_full <= hi)
    c, rows = expand_lateral_symmetry(c_full[sel], rows_full[sel])
    np.testing.assert_array_equal(c, c_full)
    np.testing.assert_allclose(rows, rows_full, rtol=1e-15)


def test_a_90_270_file_is_mirrored_not_rotated():
    """An asymmetric-looking half: mirroring about the 90-270 plane puts C = 180's
    row at C = 0; a 90 degree rotation would put C = 90's there."""
    c = np.array([90.0, 180.0, 270.0])
    rows = np.array([[1.0] * GAMMA.size, [5.0] * GAMMA.size, [1.0] * GAMMA.size])
    full_c, full_rows = expand_lateral_symmetry(c, rows)
    np.testing.assert_array_equal(full_c, [0.0, 90.0, 180.0, 270.0, 360.0])
    assert full_rows[0, 0] == 5.0 and full_rows[1, 0] == 1.0


def test_a_missing_360_plane_is_the_zero_plane():
    c = np.arange(0.0, 331.0, 30.0)
    rows = np.outer(np.arange(c.size) + 1.0, PROFILE)
    full_c, full_rows = expand_lateral_symmetry(c, rows)
    assert full_c[-1] == 360.0
    np.testing.assert_array_equal(full_rows[-1], full_rows[0])


def test_single_plane_is_rotationally_symmetric():
    t = read_ies(_ies_text([0.0], [PROFILE]))
    assert t.rotationally_symmetric
    cfg = t.to_source_config(Spectrum.monochromatic(0.555))
    assert cfg.azimuth_angles_deg is None


# ---------------------------------------------------------------------------
# The luminaire frame, through the engine
# ---------------------------------------------------------------------------


def _cone_fraction_about_nadir(half_deg):
    a, b = 200.0, 400.0 / (math.pi / 2)
    t = math.radians(half_deg)
    cone = a * (1 - math.cos(t)) + b * (math.sin(t) - t * math.cos(t))
    full = a + b
    return cone / full


def test_luminous_flux_of_the_table():
    t = read_ies(_ies_text([0.0], [PROFILE]))
    expect = 2 * math.pi * (200.0 + 400.0 / (math.pi / 2))
    assert t.luminous_flux() == pytest.approx(expect, rel=1e-13)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_a_downlight_emits_about_the_nadir(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    be.set_backend(backend)
    be.set_precision("float64")
    table = read_ies(_ies_text([0.0], [PROFILE]))
    spec = Spectrum.monochromatic(0.555)
    scene = NSQScene()
    scene.add_source("L", CoordinateSystem(), table.to_source_config(spec))
    # The flat far-field detector folds both sides onto theta = arccos|z|, so in
    # the world frame its polar angle is the vertical angle from the nadir.
    scene.add_detector("F", CoordinateSystem(z=-1.0), FarFieldDetectorConfig(num_theta=3, num_phi=4))
    n = 60_000
    res = scene.trace(num_rays=n, seed=3)
    det = res.detectors["F"]
    th = np.radians(det.theta)
    flux = (det.intensity * (np.sin(th) * math.radians(30.0) * math.pi / 2)[:, None]).sum(axis=1)
    watts = lumens_to_watts(table.luminous_flux(), spec)
    assert scene.sources[0].total_flux == pytest.approx(watts, rel=1e-13)
    frac = flux / watts
    cum = [_cone_fraction_about_nadir(x) for x in (30.0, 60.0, 90.0)]
    p = np.diff([0.0] + cum)
    z = (frac - p) / np.sqrt(p * (1 - p) / n)
    assert np.sum(z**2) < 16.27  # chi-squared 99.9 %, 3 degrees of freedom
    assert res.flux_conservation_error < 1e-12


def test_the_c_planes_turn_counterclockwise_seen_from_above():
    """C = 0 to 90 is the quadrant between local +x and +y (luminaire frame, +z up)."""
    c = np.arange(0.0, 361.0, 90.0)
    rows = np.outer([3.0, 1.0, 1.0, 1.0, 3.0], PROFILE)
    # a peak only in the plane C = 0: the flux leans toward +x
    table = read_ies(_ies_text(c, rows))
    cfg = table.to_source_config(Spectrum.monochromatic(0.555))
    scene = NSQScene()
    scene.add_source("L", CoordinateSystem(), cfg)
    scene.add_detector("F", CoordinateSystem(z=-1.0), FarFieldDetectorConfig(num_theta=1, num_phi=4))
    det = scene.trace(num_rays=20_000, seed=1).detectors["F"]
    by_phi = det.intensity.sum(axis=0)  # bins [-180,-90), [-90,0), [0,90), [90,180)
    # C-plane 0 is +x: the two bins beside phi = 0 hold more than the two beside 180
    assert by_phi[1] + by_phi[2] > 1.3 * (by_phi[0] + by_phi[3])


# ---------------------------------------------------------------------------
# EULUMDAT through the optional reader
# ---------------------------------------------------------------------------

_LDT = """KRONOS FIXTURE
1
0
4
90.0
10
10.0
REPORT
Fixture
0001
FIX
2026-09-25
300
0
100
50
0
0
0
0
0
100
100
1
0
1
1
LED
2000
3000
180
10.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
90.0
180.0
270.0
{gammas}
{rows}"""


def test_eulumdat_matches_the_equivalent_ies(tmp_path):
    pytest.importorskip("pyldt")
    per_klm = PROFILE / 2.0  # cd per 1000 lm; the lamp set is 2000 lm
    rows = "\n".join(repr(float(v)) for _ in range(4) for v in per_klm)
    path = tmp_path / "fixture.ldt"
    path.write_text(_LDT.format(gammas="\n".join(f"{g:g}" for g in GAMMA), rows=rows))
    ldt = read_eulumdat(path)
    ies = read_ies(_ies_text([0.0], [PROFILE]))
    np.testing.assert_allclose(ldt.candela[0], PROFILE, rtol=1e-12)
    assert ldt.luminous_flux() == pytest.approx(ies.luminous_flux(), rel=1e-12)
