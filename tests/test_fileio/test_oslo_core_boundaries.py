"""Core OSLO validation using existing native functionality only."""

from __future__ import annotations

import pytest

from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.fileio.oslo.reader.parser import OsloDataParser
from optiland.physical_apertures import RectangularAperture


def test_surface_resets_clear_only_their_parameter_families(lens_file):
    path = lens_file(
        surface='NOT "retained note"; CC -1; AD .001; ATD; CVX .1; CXD; '
        "GC 0; GCD; RCO; RCD; BEN; BED; PFL 10; PFD; "
        "TLA 4; TDD; PU .1; CSD; PY 0; TSD; APD; DRW 0"
    )
    data = OsloDataParser(path, strict=True).parse()
    surface = data.surfaces[1]
    assert surface["note"] == "retained note"
    assert surface["RD"] == 20
    assert surface["TH"] == 2
    assert (
        not {"CC", "AD", "CVX", "GC", "RCO", "BEN", "PFL", "TLA", "PU", "PY"}
        & surface.keys()
    )


def test_configuration_weight_is_retained(lens_file):
    data = OsloDataParser(lens_file(footer="CFWT 1 .25"), strict=True).parse()
    assert data.configurations[1].weight == 0.25


def test_checked_circular_boundary_survives_export(lens_file, tmp_path):
    optic = load_oslo_file(lens_file(surface="AP CHK 3", system="APCK ON"), strict=True)
    target = tmp_path / "checked.len"
    save_oslo_file(optic, target)
    restored = load_oslo_file(target, strict=True)
    assert restored.surfaces[1].aperture.r_max == 3
    assert not restored.surfaces[1].aperture.contains(4, 0)


def test_custom_aperture_export_preserves_destination(lens_file, tmp_path):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.surfaces[1].aperture = RectangularAperture(-1, 1, -2, 2)
    target = tmp_path / "protected.len"
    target.write_text("retained design")
    with pytest.raises(NotImplementedError, match="aperture shape"):
        save_oslo_file(optic, target)
    assert target.read_text() == "retained design"


def test_direct_index_count_matches_definition_spectrum(lens_file):
    with pytest.raises(ValueError, match="index/wavelength counts differ"):
        load_oslo_file(lens_file(surface="GLA 1.5 1.6"), strict=True)
