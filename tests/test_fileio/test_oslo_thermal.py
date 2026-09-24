"""Reference-temperature metadata, without inventing thermal operating changes."""

from __future__ import annotations

import pytest

from optiland.fileio import load_oslo_file
from optiland.fileio.oslo.reader.parser import OsloDataParser


def test_expansion_is_retained_without_changing_nominal_dimensions(tmp_path):
    path = tmp_path / "nominal.len"
    path.write_text(
        'LEN NEW "nominal" 1 2; EBR 1; TH 1e20; TCE 0; NXT; '
        "RD 40; TH 7; TCE 180; TCE -12; NXT; TCE 0; END 2"
    )
    data = OsloDataParser(path, strict=True).parse()
    assert data.to_dict()["surfaces"][1]["TCE"] == -12
    optic = load_oslo_file(path, strict=True)
    assert optic.surfaces[1].geometry.radius == 40
    assert optic.surfaces[2].geometry.cs.z.item() == 7


@pytest.mark.parametrize("command", ["TCE", "TCE 1 2", "TCE bad", "TCE nan"])
def test_invalid_expansion_metadata_is_rejected(tmp_path, command):
    path = tmp_path / "invalid.len"
    path.write_text(f'LEN NEW "invalid" 1 1; {command}; NXT; END 1')
    with pytest.raises(ValueError, match="TCE|could not convert"):
        OsloDataParser(path).parse()
