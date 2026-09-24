"""OSLO's image thickness is defocus added to the nominal image distance."""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.fileio.oslo.reader.parser import OsloDataParser
from optiland.rays import RealRays
from tests.utils import assert_allclose


@pytest.mark.parametrize("shift", [-5, 5])
@pytest.mark.parametrize("units", [1, 10])
def test_image_defocus_moves_detector_and_ray_intercept(
    lens_file, set_test_backend, shift, units
):
    optic = load_oslo_file(
        lens_file(system=f"UNI {units}", image=f"TH {shift}"), strict=True
    )
    image = optic.surfaces[-1]
    assert_allclose(image.geometry.cs.z, (22 + shift) * units)
    assert_allclose(optic.surfaces[-2].thickness, (20 + shift) * units)
    assert_allclose(image.thickness, 0)

    # Propagate an independent ray through the final air gap. Its detector
    # height must include the defocus term dy = shift * direction_y/direction_z.
    zeros = be.zeros(1)
    rays = RealRays(zeros, [1], [2 * units], zeros, [0.1], [math.sqrt(0.99)], 1, 0.55)
    image.trace(rays)
    assert_allclose(rays.y, [1 + (20 + shift) * units * 0.1 / math.sqrt(0.99)])


def test_image_defocus_is_applied_after_nominal_focus_solve(
    lens_file, set_test_backend
):
    optic = load_oslo_file(lens_file(second="PY 0", image="TH 5"), strict=True)
    # Independent paraxial refractions for y1=2, n=1.5, R1=20, R2=-20:
    # u1=-1/30; y2=29/15; u2=-59/600; nominal image gap=-y2/u2=1160/59.
    # The saved image TH deliberately defocuses that PY=0 solution by 5 mm.
    assert_allclose(optic.surfaces[-2].thickness, 1160 / 59 + 5)
    assert_allclose(optic.paraxial.marginal_ray()[0][-1], -59 / 120)
    assert_allclose(optic.surfaces[-1].thickness, 0)


def test_image_focus_pickup_uses_the_solved_nominal_thickness(
    lens_file, set_test_backend
):
    optic = load_oslo_file(lens_file(second="PY 0", image="PK TH 2 1"), strict=True)
    # TH pickup adds a constant. The image shift is nominal TH2+1, so the
    # physical final gap is TH2 + (TH2+1), with neither offset applied twice.
    assert_allclose(optic.surfaces[-2].thickness, 2 * 1160 / 59 + 1)
    assert_allclose(optic.surfaces[-1].thickness, 0)


def test_defocus_follows_the_local_image_leg(lens_file, set_test_backend):
    optic = load_oslo_file(lens_file(surface="TLA 30", image="TH 5"), strict=True)
    position, _ = optic.surfaces[-1].geometry.cs.get_effective_transform()
    # Intrinsic OSLO TLA rotates about -x: the following +z leg is
    # (0, sin(30 degrees), cos(30 degrees)), over 2+20+5 lens units.
    assert_allclose(position, [0, 27 / 2, 27 * math.sqrt(3) / 2])
    assert_allclose(optic.surfaces[-2].thickness, 25)


def test_image_global_reference_with_defocus_is_explicitly_rejected(lens_file):
    # GC overrides the preceding distance. Mapping an independent focus shift
    # onto that reference requires a separate verified coordinate convention.
    with pytest.raises(ValueError, match="focus shift.*global reference"):
        load_oslo_file(lens_file(image="GC 1\nDCZ 22\nTH 5"), strict=True)


def test_image_focus_without_an_interior_surface_is_explicitly_rejected(tmp_path):
    path = tmp_path / "no-interior.len"
    path.write_text(
        'LEN NEW "empty" 50 1\nEBR 2\nTH 1e20\nNXT\nTH 5\nEND 1',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="focus shift requires an interior surface"):
        load_oslo_file(path, strict=True)


def test_finite_entrance_beam_stays_fixed_when_the_image_stop_moves(
    lens_file, set_test_backend
):
    optic = load_oslo_file(lens_file(distance="100", image="AST\nTH 5"), strict=True)
    # EBR specifies the radius at surface 1. Moving an image-plane stop moves
    # its entrance pupil; the equivalent native EPD must be recalculated.
    assert_allclose(abs(optic.paraxial.marginal_ray()[0][1]), 2)


def test_defocus_survives_export_without_being_applied_twice(
    lens_file, tmp_path, set_test_backend
):
    optic = load_oslo_file(lens_file(image="TH 5"), strict=True)
    output = tmp_path / "defocused.len"
    save_oslo_file(optic, output)
    model = OsloDataParser(output).parse()
    assert model.surfaces[2]["TH"] == 25
    assert model.surfaces[3]["TH"] == 0
    restored = load_oslo_file(output, strict=True)
    assert_allclose(restored.surfaces[-1].geometry.cs.z, 27)


def test_native_unused_image_thickness_does_not_become_oslo_defocus(
    lens_file, tmp_path, set_test_backend
):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.surfaces[-1].thickness = 5
    # Native thickness advances the next surface, so this final value does not
    # move the existing detector. OSLO would instead move it if TH 5 were saved.
    output = tmp_path / "native-image-thickness.len"
    save_oslo_file(optic, output)
    assert OsloDataParser(output).parse().surfaces[3]["TH"] == 0
    assert_allclose(load_oslo_file(output, strict=True).surfaces[-1].geometry.cs.z, 22)
