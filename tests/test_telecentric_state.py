"""Regression tests for canonical object-space telecentric state."""

from copy import deepcopy
from pathlib import Path

import pytest

import optiland.backend as be
from optiland.fields import FieldGroup
from optiland.fileio import load_optiland_file, save_optiland_file
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import RealRays
from tests.utils import assert_allclose


@pytest.fixture
def finite_optic(set_test_backend: None) -> Optic:
    """Build a finite system whose ordinary chief ray is not axial."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=32.0, material=IdealMaterial(n=1.0))
    optic.surfaces.add(
        index=1, radius=16.0, thickness=16.0, material=IdealMaterial(n=2.0)
    )
    optic.surfaces.add(
        index=2, thickness=48.0, is_stop=True, material=IdealMaterial(n=2.0)
    )
    optic.surfaces.add(index=3, material=IdealMaterial(n=2.0))
    optic.fields.set_type("object_height")
    optic.fields.add(y=0.25)
    optic.set_aperture("objectNA", 0.05)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _generate_rays(optic: Optic) -> RealRays:
    """Generate a chief ray and two off-axis pupil rays without tracing."""
    return optic.ray_tracer.ray_generator.generate_rays(
        be.zeros(3),
        be.ones(3),
        be.array([0.0, 0.5, -0.5]),
        be.array([0.0, -0.5, 0.5]),
        optic.primary_wavelength,
    )


def _assert_state(optic: Optic, telecentric: bool) -> RealRays:
    """Check both public views and the resulting chief-ray launch."""
    assert optic.obj_space_telecentric is telecentric
    assert optic.fields.telecentric is telecentric
    rays = _generate_rays(optic)
    assert_allclose(rays.x, 0.0, rtol=0.0, atol=1e-12)
    assert_allclose(rays.y, 0.25, rtol=0.0, atol=1e-12)
    assert_allclose(rays.z, -32.0, rtol=0.0, atol=1e-12)
    # The ordinary entrance pupil is at z=16, 48 units from the object.
    slope = 0.0 if telecentric else -0.25 / 48.0
    assert_allclose(rays.M[0] / rays.N[0], slope, rtol=0.0, atol=1e-12)
    assert_allclose(rays.L[0], 0.0, rtol=0.0, atol=1e-12)
    return rays


def test_unconfigured_optic(set_test_backend: None) -> None:
    optic = Optic()
    assert optic.obj_space_telecentric is False
    assert optic.fields.telecentric is False

    optic.obj_space_telecentric = True
    assert optic.fields.telecentric is True
    optic.fields.set_telecentric(False)
    assert optic.obj_space_telecentric is False
    optic.fields.set_telecentric(True)
    assert optic.obj_space_telecentric is True
    optic.obj_space_telecentric = False
    assert optic.fields.telecentric is False

    assert optic.aperture is None
    assert optic.fields.field_definition is None
    assert len(optic.surfaces) == 0
    assert len(optic.wavelengths) == 0
    assert "obj_space_telecentric" not in vars(optic)


@pytest.mark.parametrize("via_fields", [False, True])
def test_last_write_wins_with_warmed_generator(
    finite_optic: Optic, via_fields: bool
) -> None:
    optic = finite_optic
    _assert_state(optic, False)
    generator = optic.ray_tracer.ray_generator
    aimer = generator.aimer

    for telecentric in (True, False, True, False):
        if via_fields:
            optic.fields.set_telecentric(telecentric)
        else:
            optic.obj_space_telecentric = telecentric
        _assert_state(optic, telecentric)
        assert optic.ray_tracer.ray_generator is generator
        assert generator.aimer is aimer
        via_fields = not via_fields


@pytest.mark.parametrize("via_fields", [False, True])
@pytest.mark.parametrize("telecentric", [False, True])
@pytest.mark.parametrize("use_json", [False, True], ids=["dict", "json"])
def test_ray_preserving_roundtrip(
    finite_optic: Optic,
    tmp_path: Path,
    via_fields: bool,
    telecentric: bool,
    use_json: bool,
) -> None:
    optic = finite_optic
    _assert_state(optic, False)
    if via_fields:
        optic.obj_space_telecentric = not telecentric
        optic.fields.set_telecentric(telecentric)
    else:
        optic.fields.set_telecentric(not telecentric)
        optic.obj_space_telecentric = telecentric
    launch = _assert_state(optic, telecentric)
    traced = optic.trace_generic(
        0.0,
        1.0,
        be.array([0.0, 0.5, -0.5]),
        be.array([0.0, -0.5, 0.5]),
        optic.primary_wavelength,
    )
    data = optic.to_dict()
    assert data["version"] == 1.0
    assert data["fields"]["telecentric"] is telecentric
    assert "obj_space_telecentric" not in data

    if use_json:
        filepath = tmp_path / "telecentric.json"
        save_optiland_file(optic, filepath)
        restored = load_optiland_file(filepath)
    else:
        restored = Optic.from_dict(data)

    restored_launch = _assert_state(restored, telecentric)
    restored_traced = restored.trace_generic(
        0.0,
        1.0,
        be.array([0.0, 0.5, -0.5]),
        be.array([0.0, -0.5, 0.5]),
        restored.primary_wavelength,
    )
    assert restored.to_dict()["fields"] == data["fields"]
    for actual, expected in ((restored_launch, launch), (restored_traced, traced)):
        for attribute in ("x", "y", "z", "L", "M", "N", "i", "w", "opd"):
            assert_allclose(
                getattr(actual, attribute),
                getattr(expected, attribute),
                rtol=0.0,
                atol=1e-12,
            )


def test_reset_detaches_old_group(finite_optic: Optic) -> None:
    optic = finite_optic
    optic.obj_space_telecentric = True
    _assert_state(optic, True)
    old_fields = optic.fields

    optic.reset()
    assert optic.fields is not old_fields
    assert optic.obj_space_telecentric is False
    assert optic.fields.telecentric is False
    assert old_fields.telecentric is True
    for telecentric in (False, True):
        old_fields.set_telecentric(telecentric)
        assert optic.obj_space_telecentric is False
    optic.obj_space_telecentric = True
    assert optic.fields.telecentric is True
    optic.obj_space_telecentric = False
    assert old_fields.telecentric is True


@pytest.mark.parametrize("telecentric", [False, True])
def test_field_group_replacement(finite_optic: Optic, telecentric: bool) -> None:
    optic = finite_optic
    optic.obj_space_telecentric = not telecentric
    _assert_state(optic, not telecentric)
    generator = optic.ray_tracer.ray_generator
    aimer = generator.aimer
    old_fields = optic.fields
    replacement = FieldGroup.from_dict(old_fields.to_dict())
    replacement.set_telecentric(telecentric)

    optic.fields = replacement
    _assert_state(optic, telecentric)
    for old_state in (True, False):
        old_fields.set_telecentric(old_state)
        _assert_state(optic, telecentric)
    optic.obj_space_telecentric = not telecentric
    _assert_state(optic, not telecentric)
    assert old_fields.telecentric is False
    replacement.set_telecentric(telecentric)
    _assert_state(optic, telecentric)
    assert optic.ray_tracer.ray_generator is generator
    assert generator.aimer is aimer


def test_deepcopy_independence(finite_optic: Optic) -> None:
    optic = finite_optic
    optic.obj_space_telecentric = True
    _assert_state(optic, True)

    copied = deepcopy(optic)
    assert copied.fields is not optic.fields
    assert copied.ray_tracer.ray_generator.optic is copied
    assert copied.ray_tracer.ray_generator.aimer.optic is copied
    _assert_state(copied, True)
    copied.fields.set_telecentric(False)
    _assert_state(copied, False)
    _assert_state(optic, True)
    copied.obj_space_telecentric = True
    optic.fields.set_telecentric(False)
    _assert_state(copied, True)
    _assert_state(optic, False)


def test_standalone_field_group(set_test_backend: None) -> None:
    fields = FieldGroup()
    assert fields.telecentric is False
    for telecentric in (True, False):
        fields.set_telecentric(telecentric)
        assert fields.telecentric is telecentric
        data = fields.to_dict()
        assert data["telecentric"] is telecentric
        restored = FieldGroup.from_dict(data)
        assert restored.telecentric is telecentric
        restored.set_telecentric(not telecentric)
        assert fields.telecentric is telecentric
