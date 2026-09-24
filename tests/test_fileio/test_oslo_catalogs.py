"""Material identity must come from catalog context, never inferred aliases."""

from __future__ import annotations

import pytest

from optiland.fileio import load_oslo_file
from optiland.fileio.oslo.reader.converter import OsloToOpticConverter
from optiland.materials import IdealMaterial, MatchPolicy, Material
from optiland.optic import Optic


def catalog_lens(tmp_path, glass="GLA SAMPLE_A"):
    path = tmp_path / "catalog.len"
    path.write_text(
        'LEN NEW "catalog" 1 2; EBR 1; TH 1e20; NXT; '
        + glass
        + "; RD 30; TH 3; NXT; AIR; END 2"
    )
    return path




def test_explicit_definition_does_not_replace_direct_index_data(tmp_path):
    optic = load_oslo_file(
        catalog_lens(tmp_path, "GLA SAMPLE_A 1.7"),
        strict=True,
        material_overrides={"SAMPLE_A": IdealMaterial(1.2)},
    )
    assert optic.surfaces[1].material_post.n(0.55).item() == 1.7


@pytest.mark.parametrize(
    "overrides",
    [
        {"A": IdealMaterial(1.5), "a": IdealMaterial(1.6)},
        {"": IdealMaterial(1.5)},
        {"A": "guessed-name"},
    ],
)
def test_invalid_explicit_material_map_fails(overrides):
    with pytest.raises(ValueError, match="material_overrides"):
        OsloToOpticConverter(material_overrides=overrides)


def test_strict_catalog_lookup_does_not_accept_family_or_fuzzy_matches(tmp_path):
    with pytest.raises(ValueError, match="could not be resolved"):
        load_oslo_file(catalog_lens(tmp_path, "GLA Silica"), strict=True)


def test_exact_catalog_identity_does_not_discard_a_name_prefix(tmp_path):
    # Independent public pair: different catalog names are different entries.
    material = Material("N-F2", catalog="schott", match_policy=MatchPolicy.STRICT)
    optic = load_oslo_file(
        catalog_lens(tmp_path, "GLA GLASS_A"),
        strict=True,
        material_overrides={"GLASS_A": material},
    )
    assert optic.surfaces[1].material_post.material_data["filename_no_ext"] == "N-F2"
    restored = Optic.from_dict(optic.to_dict())
    assert restored.surfaces[1].material_post.to_dict() == material.to_dict()
