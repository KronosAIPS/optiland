"""``be.stack`` keeps a complex input complex on the torch backend.

The research repository's issue 97: the torch backend's ``stack`` cast every
element to the real working dtype, which dropped a complex element's imaginary
part with only a ``UserWarning`` (a coating's Jones matrix lost every phase this
way). ``numpy.stack`` keeps it. Now a stack with any complex element is built
in the complex dtype of the working precision; a real stack is unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import optiland.backend as be

torch = pytest.importorskip("torch")


@pytest.fixture(params=["float64", "float32"])
def torch_precision(request):
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision(request.param)
    yield request.param
    be.set_precision("float64")
    be.set_backend("numpy")


def _complex_dtype(precision):
    return torch.complex128 if precision == "float64" else torch.complex64


def test_complex_tensors_keep_their_imaginary_part(torch_precision):
    a = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128)
    b = torch.tensor([0.5j, -1 + 0j], dtype=torch.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the old cast warned and dropped the part
        out = be.stack([a, b])
    assert out.dtype == _complex_dtype(torch_precision)
    np.testing.assert_array_equal(
        out.cpu().numpy(), np.array([[1 + 2j, 3 - 4j], [0.5j, -1 + 0j]])
    )


def test_mixed_real_and_complex_promote_like_numpy(torch_precision):
    real = be.array([1.0, 2.0])
    cplx = torch.tensor([1j, 2 + 1j], dtype=_complex_dtype(torch_precision))
    out = be.stack([real, cplx], axis=1)
    expected = np.stack([np.array([1.0, 2.0]), np.array([1j, 2 + 1j])], axis=1)
    assert out.dtype == _complex_dtype(torch_precision)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


def test_python_and_numpy_complex_inputs(torch_precision):
    out = be.stack([1 + 1j, 2.0, np.complex128(-3j)])
    assert out.dtype == _complex_dtype(torch_precision)
    np.testing.assert_array_equal(out.cpu().numpy(), np.array([1 + 1j, 2.0, -3j]))
    arr = be.stack([np.array([1j, 2.0]), np.array([3.0, 4.0])])
    np.testing.assert_array_equal(
        arr.cpu().numpy(), np.array([[1j, 2.0], [3.0, 4.0]])
    )


def test_real_stack_is_unchanged(torch_precision):
    working = torch.float64 if torch_precision == "float64" else torch.float32
    a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    b = torch.tensor([3, 4])  # an integer tensor is cast to the working dtype
    out = be.stack([a, b])
    assert out.dtype == working
    np.testing.assert_array_equal(out.cpu().numpy(), [[1.0, 2.0], [3.0, 4.0]])


def test_gradient_flows_through_a_complex_stack():
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    try:
        x = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        z = torch.exp(1j * x)
        out = be.stack([z, 2.0 * z])
        loss = out.imag.sum()  # d/dx of 3 sin(x)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), 3.0 * np.cos([0.3, 0.7]))
    finally:
        be.set_backend("numpy")


def test_numpy_backend_agrees():
    be.set_backend("numpy")
    out = be.stack([np.array([1 + 2j]), np.array([3.0])])
    assert np.iscomplexobj(out)
    np.testing.assert_array_equal(out, [[1 + 2j], [3.0]])


def _jones_pupil_centre(backend):
    from optiland.analysis.jones_pupil import JonesPupil  # noqa: PLC0415
    from optiland.coatings import RetarderCoating  # noqa: PLC0415
    from optiland.rays.polarization_state import PolarizationState  # noqa: PLC0415
    from optiland.samples.objectives import CookeTriplet  # noqa: PLC0415

    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
        be.set_precision("float64")
    optic = CookeTriplet()
    optic.updater.set_polarization(
        PolarizationState(is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0)
    )
    # a quarter-wave retarder with its fast axis at 45 degrees on the first face
    optic.surfaces.surfaces[1].interaction_model.coating = RetarderCoating(
        np.pi / 2, (1.0, 1.0, 0.0)
    )
    J = be.to_numpy(JonesPupil(optic, grid_size=5).data[0]["J"])
    return J[12]  # the chief ray of the 5 x 5 grid


def test_jones_pupil_keeps_the_retarders_phase_on_torch():
    # the caller that lost its imaginary part: JonesPupil stacks the four
    # complex Jones entries with be.stack; on torch the quarter-wave phase
    # (Jxy = -i / sqrt 2 on the chief ray) was dropped to 0
    try:
        ref = _jones_pupil_centre("numpy")
        got = _jones_pupil_centre("torch")
    finally:
        be.set_backend("numpy")
    assert np.iscomplexobj(got)
    assert abs(ref[0, 1].imag) > 0.7
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)
