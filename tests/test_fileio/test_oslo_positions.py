"""Export the physical axial prescription, including absolute-coordinate lenses."""

from __future__ import annotations

import math

import pytest

from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import RealRays
from tests.utils import assert_allclose


def trace_axial_bundle(optic):
    """Launch identical rays relative to the first physical surface."""
    first_z = float(optic.surfaces[1].geometry.cs.z.item())
    rays = RealRays([0, 0], [0, 1], [first_z - 1] * 2, [0, 0], [0, 0], [1, 1], 1, 0.55)
    optic.surfaces.trace(rays, skip=1)
    return rays


@pytest.mark.parametrize("origin", [0, 13])
@pytest.mark.parametrize("distance", [100, -100, math.inf, -math.inf])
def test_absolute_axial_positions_survive_export(
    tmp_path, set_test_backend, origin, distance
):
    optic = Optic()
    optic.surfaces.add(index=0, z=origin - distance)
    optic.surfaces.add(
        index=1, z=origin, radius=20, material=IdealMaterial(1.5), is_stop=True
    )
    optic.surfaces.add(index=2, z=origin + 5, radius=-20)
    optic.surfaces.add(index=3, z=origin + 25)
    optic.set_aperture("EPD", 4)
    optic.fields.set_type("angle" if math.isinf(distance) else "object_height")
    optic.fields.add(y=0)
    optic.wavelengths.add(0.55, is_primary=True)
    before = trace_axial_bundle(optic)

    output = tmp_path / "absolute-coordinates.len"
    save_oslo_file(optic, output)
    restored = load_oslo_file(output, strict=True)
    after = trace_axial_bundle(restored)

    # OSLO anchors the first surface at zero; the physical gaps and ray
    # intersections must be unchanged by removing a common axial translation.
    assert_allclose(restored.surfaces.global_z_positions.ravel(), [-distance, 0, 5, 25])
    assert_allclose([s.thickness for s in restored.surfaces], [distance, 5, 20, 0])
    for component in ("x", "y", "L", "M", "N", "i", "opd"):
        assert_allclose(
            getattr(after, component), getattr(before, component), atol=1e-10
        )
    assert_allclose(after.z, before.z - origin)
    assert all(s.thickness == 0 for s in optic.surfaces)


def test_export_uses_detector_pose_after_direct_coordinate_edit(
    lens_file, tmp_path, set_test_backend
):
    optic = load_oslo_file(lens_file(), strict=True)
    image_cs = optic.surfaces[-1].geometry.cs
    image_cs.z = image_cs.z + 7
    before = trace_axial_bundle(optic)
    output = tmp_path / "moved-detector.len"
    save_oslo_file(optic, output)
    restored = load_oslo_file(output, strict=True)
    after = trace_axial_bundle(restored)
    assert_allclose(restored.surfaces[-1].geometry.cs.z, 29)
    assert_allclose(restored.surfaces[-2].thickness, 27)
    assert_allclose(after.y, before.y)
    assert_allclose(after.opd, before.opd)


@pytest.mark.parametrize("gap", [9.9e9, -9.9e9])
def test_export_rejects_finite_interior_spacing_at_oslo_infinity_cutoff(
    lens_file, tmp_path, set_test_backend, gap
):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.updater.set_thickness(gap, 2)
    with pytest.raises(NotImplementedError, match="finite.*spacing"):
        save_oslo_file(optic, tmp_path / "finite-gap.len")


def test_finite_interior_spacing_below_cutoff_stays_finite(
    lens_file, tmp_path, set_test_backend
):
    optic = load_oslo_file(lens_file(), strict=True)
    gap = 9.9e9 - 1
    optic.updater.set_thickness(gap, 2)
    output = tmp_path / "finite-gap.len"
    save_oslo_file(optic, output)
    restored = load_oslo_file(output, strict=True)
    assert float(restored.surfaces[2].thickness) == gap


def test_export_rejects_undefined_axial_spacing(lens_file, tmp_path, set_test_backend):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.surfaces[-1].geometry.cs.z = math.nan
    with pytest.raises(ValueError, match="defined axial surface spacings"):
        save_oslo_file(optic, tmp_path / "undefined-gap.len")
