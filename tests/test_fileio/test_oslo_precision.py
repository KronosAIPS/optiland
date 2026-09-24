"""Export must not round valid optical data across physical boundaries."""

from __future__ import annotations

import math

import pytest

from optiland.fileio import load_oslo_file, save_oslo_file


def test_export_preserves_an_angular_reference_just_below_ninety(
    lens_file, tmp_path, set_test_backend
):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.fields.fields.clear()
    optic.fields.add(y=89.999999)
    optic.fields.add(y=1)
    output = tmp_path / "near-right-angle.len"
    save_oslo_file(optic, output)
    restored = load_oslo_file(output, strict=True)
    assert restored.fields[0].y == optic.fields[0].y
    assert math.tan(math.radians(restored.fields[1].y)) == pytest.approx(
        math.tan(math.radians(1)), rel=1e-12
    )


@pytest.mark.parametrize("medium,index", [("AIR", 1.0), ("GLA 1.5", 1.5)])
def test_export_preserves_na_below_the_object_medium_index(
    lens_file, tmp_path, set_test_backend, medium, index
):
    optic = load_oslo_file(lens_file(distance="100", system=medium), strict=True)
    value = index - 1e-8
    optic.set_aperture("objectNA", value)
    output = tmp_path / "near-hemisphere.len"
    save_oslo_file(optic, output)
    restored = load_oslo_file(output, strict=True)
    assert restored.aperture.value == value




