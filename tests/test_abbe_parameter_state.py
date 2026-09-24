"""Abbe wrappers and derived coefficients must follow their live parameters."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.materials import AbbeMaterial, AbbeMaterialE

from .utils import assert_allclose


def make_material(model, index=1.5, abbe=60.0):
    if model == "e":
        return AbbeMaterialE(index, abbe)
    return AbbeMaterial(index, abbe, model=model)


@pytest.mark.parametrize("model", ["polynomial", "buchdahl", "e"])
def test_wrapper_and_model_share_parameter_storage(set_test_backend, model):
    material = make_material(model)
    assert material.index is material.model.index
    assert material.abbe is material.model.abbe
    material.index = be.asarray([1.6])
    assert material.index is material.model.index
    material.model.abbe = be.asarray([50.0])
    assert material.abbe is material.model.abbe


@pytest.mark.parametrize("model", ["polynomial", "buchdahl", "e"])
@pytest.mark.parametrize("attribute,value", [("index", 1.7), ("abbe", 35.0)])
@pytest.mark.parametrize("owner", ["wrapper", "model"])
def test_in_place_parameter_changes_update_dispersion(
    set_test_backend, model, attribute, value, owner
):
    material = make_material(model)
    target = material if owner == "wrapper" else material.model
    setattr(target, attribute, be.asarray([1.5 if attribute == "index" else 60.0]))
    material.n(0.55)
    getattr(target, attribute)[...] = value
    expected = make_material(
        model,
        index=1.7 if attribute == "index" else 1.5,
        abbe=35.0 if attribute == "abbe" else 60.0,
    )
    assert_allclose(material.n(0.55), expected.n(0.55))
    restored = type(material).from_dict(material.to_dict())
    assert_allclose(restored.n(0.55), material.n(0.55))


@pytest.mark.parametrize("model", ["polynomial", "buchdahl", "e"])
def test_direct_model_updates_derived_coefficients(set_test_backend, model):
    prediction = make_material(model).model
    prediction.index = be.asarray([1.5])
    prediction.abbe = be.asarray([60.0])
    prediction.predict_n(0.55)
    prediction.abbe[...] = 35.0
    expected = make_material(model, abbe=35.0).model.predict_n(0.55)
    assert_allclose(prediction.predict_n(0.55), expected)


@pytest.mark.parametrize("model", ["polynomial", "buchdahl", "e"])
@pytest.mark.parametrize("property_name", ["n", "k"])
def test_custom_model_tracks_additional_mutable_state(
    set_test_backend, monkeypatch, model, property_name
):
    # Use numerical parameters so gradient bypass cannot hide unsafe cache reuse.
    if be.get_backend() == "torch":
        monkeypatch.setattr(be.grad_mode, "requires_grad", False)
    material = make_material(model)
    evaluate = getattr(material, property_name)
    baseline = evaluate(0.55)

    class OffsetModel(type(material.model)):
        offset = 0.0

        def predict_n(self, wavelength):
            return super().predict_n(wavelength) + self.offset

        def predict_k(self, wavelength):
            return super().predict_k(wavelength) + self.offset

    material.model = OffsetModel(1.5, 60.0)
    for offset in (0.01, 0.02):
        material.model.offset = offset
        assert_allclose(evaluate(0.55), baseline + offset)


@pytest.mark.parametrize("model", ["polynomial", "buchdahl", "e"])
def test_repeated_parameter_backward_after_no_grad_query(set_test_backend, model):
    if be.get_backend() != "torch":
        pytest.skip("Torch gradients")
    import torch

    material = make_material(model)
    parameter = torch.nn.Parameter(torch.tensor([1.5], dtype=torch.float64))
    material.index = parameter
    with torch.no_grad():
        material.n(0.55)
    for _ in range(2):
        material.n(0.55).sum().backward()
        step = 1e-5
        upper = make_material(model, index=1.5 + step).n(0.55)
        lower = make_material(model, index=1.5 - step).n(0.55)
        assert_allclose(parameter.grad, (upper - lower) / (2 * step), rtol=1e-6)
        parameter.grad = None


@pytest.mark.parametrize("model", ["polynomial", "buchdahl", "e"])
def test_abbe_parameters_follow_backend_and_precision(set_test_backend, model):
    if "torch" not in be.list_available_backends():
        pytest.skip("requires both backends")
    import torch

    material = make_material(model)
    expected = be.to_numpy(material.n(0.55))
    index = material.index
    for backend, precision in (("numpy", "float32"), ("torch", "float64")):
        be.set_backend(backend)
        be.set_precision(precision)
        if backend == "torch":
            be.set_device("cpu")
        result = material.n(0.55)
        assert isinstance(result, torch.Tensor) == (backend == "torch")
        assert_allclose(result, expected, rtol=1e-6)
        assert material.index is index
    be.set_backend("numpy")
    be.set_precision("float64")
