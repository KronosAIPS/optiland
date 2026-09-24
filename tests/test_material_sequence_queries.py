"""Sequence queries keep the shape of the current call, including cache hits."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from tests.test_material_cache_contracts import SpectralMaterial
from tests.utils import assert_allclose


@pytest.mark.parametrize("property_name", ["n", "k"])
@pytest.mark.parametrize("sequence", [list, tuple])
def test_sequence_cache_distinguishes_flat_and_nested_queries(
    set_test_backend, property_name, sequence
):
    material = SpectralMaterial()
    evaluate = getattr(material, property_name)
    flat = sequence([0.4, 0.5, 0.6, 0.7])
    matrix = sequence([sequence([0.4, 0.5]), sequence([0.6, 0.7])])
    expected = evaluate(be.asarray(matrix))
    evaluate(flat)
    for _ in range(2):
        result = evaluate(matrix)
        assert tuple(result.shape) == (2, 2)
        assert_allclose(result, expected)


@pytest.mark.parametrize("property_name", ["n", "k"])
@pytest.mark.parametrize("sequence", [list, tuple])
def test_uniform_sequence_query_is_broadcast_on_each_backend(
    set_test_backend, property_name, sequence
):
    material = IdealMaterial(1.5, 0.01)
    material.index = be.asarray([1.5])
    material.absorp = be.asarray([0.01])
    evaluate = getattr(material, property_name)
    for shape in [sequence([0.55, 0.55]), sequence([sequence([0.55, 0.55])])]:
        expected = evaluate(be.asarray(shape))
        for _ in range(2):
            result = evaluate(shape)
            assert tuple(result.shape) == tuple(expected.shape)
            assert_allclose(result, expected)
