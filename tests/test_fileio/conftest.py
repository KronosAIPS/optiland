"""Shared prescriptions for OSLO import and export regression tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def lens_file(tmp_path):
    """Write an original, small prescription with independently known power."""

    def write(
        *,
        system="",
        surface="",
        second="",
        image="",
        footer="",
        aperture="EBR 2",
        distance="1e20",
    ):
        path = tmp_path / "edge.len"
        path.write_text(
            'LEN NEW "edge cases" 50 3\n'
            f"{aperture}\nANG 0\n{system}\nTH {distance}\nNXT\n"
            f"GLA 1.5\nRD 20\nTH 2\nAP 3\n{surface}\nNXT\n"
            f"AIR\nRD -20\nTH 20\n{second}\nNXT\nAIR\n{image}\n"
            f"END 3\n{footer}",
            encoding="utf-8",
        )
        return path

    return write
