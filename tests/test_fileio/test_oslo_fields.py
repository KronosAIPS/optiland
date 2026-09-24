"""Angular field tables must preserve the ray's direction and hemisphere."""

from __future__ import annotations

import math

import pytest

from optiland.fileio import load_oslo_file


@pytest.mark.parametrize("angle", [90, 100, -100])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("footer", ["", "RST NEW\nF 1 1 0 0 0 0 -1 1 -1 1 1"])
def test_angular_reference_cannot_reach_or_cross_ninety_degrees(
    lens_file, angle, strict, footer
):
    # tan(100 degrees) has the same value as tan(-80 degrees). Accepting it
    # would silently put the full-field point in the opposite hemisphere.
    with pytest.raises(ValueError, match="angular reference.*90"):
        load_oslo_file(lens_file(system=f"ANG {angle}", footer=footer), strict=strict)


@pytest.mark.parametrize("fraction", [-1, 0.5, 1, 2])
def test_large_valid_field_table_preserves_direction_tangent(
    lens_file, set_test_backend, fraction
):
    optic = load_oslo_file(
        lens_file(
            system="ANG 89",
            footer=f"RST NEW\nF 1 {fraction} 0 0 0 0 -1 1 -1 1 1",
        ),
        strict=True,
    )
    field = optic.fields[0]
    assert -90 < field.y < 90
    assert math.tan(math.radians(field.y)) == pytest.approx(
        fraction * math.tan(math.radians(89))
    )


def test_large_finite_object_height_is_not_an_angular_limit(lens_file):
    optic = load_oslo_file(
        lens_file(
            distance="1000",
            system="OBH 100",
            footer="RST NEW\nF 1 1 0 0 0 0 -1 1 -1 1 1",
        ),
        strict=True,
    )
    assert optic.fields[0].y == 100
