"""The existing approximate direct-glass path before sampled material support."""

from __future__ import annotations

import pytest

from optiland.fileio import load_oslo_file
from optiland.materials import AbbeMaterial, IdealMaterial


@pytest.mark.parametrize(
    "indices,kind", [("1.5 1.51 1.49", AbbeMaterial), ("1.5 1.49 1.51", IdealMaterial)]
)
def test_legacy_direct_glass_is_explicitly_approximate(lens_file, indices, kind):
    path = lens_file(surface=f"GLA {indices}")
    with pytest.warns(UserWarning, match="approximate Abbe"):
        optic = load_oslo_file(path)
    assert isinstance(optic.surfaces[1].material_post, kind)
    with pytest.raises(ValueError, match="approximate Abbe"):
        load_oslo_file(path, strict=True)
