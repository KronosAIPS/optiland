"""Validate OSLO edge-contact solves against the positioned physical surfaces."""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.fileio import load_oslo_file
from tests.utils import assert_allclose


@pytest.mark.parametrize(
    "surface,second",
    [
        ("", "DCX 1"),
        ("", "DCY 1"),
        ("", "DCZ 3"),
        ("", "TLA 10"),
        ("", "GC 1"),
        ("DCY 1\nRCO", ""),
    ],
)
def test_edge_contact_must_survive_relative_coordinate_transforms(
    lens_file, set_test_backend, surface, second
):
    baseline = load_oslo_file(lens_file(surface=surface, second=second), strict=True)
    path = lens_file(surface=surface + "\nEC 2", second=second)
    with pytest.warns(
        UserWarning, match="EC at surface 1.*retained saved prescription"
    ):
        optic = load_oslo_file(path)
    assert optic.to_dict() == baseline.to_dict()
    with pytest.raises(ValueError, match="EC at surface 1"):
        load_oslo_file(path, strict=True)


@pytest.mark.parametrize("tilt", [0, 30])
@pytest.mark.parametrize("units", [1, 10])
@pytest.mark.parametrize("second", ["", "TLC 90"])
def test_common_surface_frame_preserves_physical_edge_contact(
    lens_file, set_test_backend, tilt, units, second
):
    optic = load_oslo_file(
        lens_file(
            system=f"UNI {units}",
            surface=f"DCX 1\nDCY 2\nDCZ 3\nTLA {tilt}\nEC 2",
            second=second,
        ),
        strict=True,
    )
    # Two spheres of radii +/-20 meet at local height 2 when their
    # vertex separation is twice the positive spherical sag. A common
    # rigid transform must carry the two contact points to the same place.
    # Rolling either sphere around its own axis leaves its shape unchanged.
    sag = (20 - math.sqrt(20**2 - 2**2)) * units
    angle = math.radians(-tilt)  # OSLO tilts around -x.
    edge_points = []
    for index, local_sag in [(1, sag), (2, -sag)]:
        cs = optic.surfaces[index].geometry.cs
        edge_points.append(
            [
                float(cs.x.item()),
                float(cs.y.item())
                + 2 * units * math.cos(angle)
                - local_sag * math.sin(angle),
                float(cs.z.item())
                + 2 * units * math.sin(angle)
                + local_sag * math.cos(angle),
            ]
        )
    assert_allclose(optic.surfaces[1].thickness, 2 * sag, atol=1e-10)
    assert_allclose(edge_points[0], edge_points[1], atol=1e-10)


@pytest.mark.parametrize("origin", [0, 100])
@pytest.mark.parametrize("gap", [0, 0.001])
def test_edge_contact_distinguishes_single_precision_roundoff_from_separation(
    lens_file, set_test_backend, origin, gap
):
    # These radii and edge height come from the official dblgauss2.len demo.
    # Its valid EC solve exposed float32 cancellation in the positioned check.
    is_torch = be.get_backend() == "torch"
    if is_torch:
        be.set_precision("float32")
    try:
        path = lens_file(
            surface=f"RD 46.2463056440498\nDCZ {origin}\nEC 23",
            second=f"RD 766.6798051599121\nDCZ {gap}",
        )
        if gap:
            with pytest.raises(ValueError, match="EC at surface 1"):
                load_oslo_file(path, strict=True)
        else:
            optic = load_oslo_file(path, strict=True)
            sag1 = 46.2463056440498 - math.sqrt(46.2463056440498**2 - 23**2)
            sag2 = 766.6798051599121 - math.sqrt(766.6798051599121**2 - 23**2)
            assert_allclose(optic.surfaces[1].thickness, sag1 - sag2, atol=2e-6)
    finally:
        if is_torch:
            be.set_precision("float64")
