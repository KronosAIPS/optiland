"""Forward mode through ``torch.func.jvp`` (R-09-9).

``docs/theory/09_differentiation.md`` of the research repository, section
9.13.5 (written before these tests ran); the research repository's issue 31.
``torch.func.jvp`` applies the same forward-mode derivative rules as
``torch.autograd.forward_ad``, and the engine takes the same code path under
both, so on every scene the two tangents are the same floating-point number:
the bound is equality. Against reverse mode, ``jvp`` then inherits T-09-6's
forms as ``forward_ad`` has them; the first form (``K u`` at this seed, the
operation counts of ``test_nsq_interior_gradients.py``) is asserted here.

What stopped ``jvp`` before (the research repository's build log of
2026-09-26): inside a ``torch.func`` transform every tensor refuses
``.numpy()``, and the host build of a placement reads its value that way. The
read now clears the transform's interpreter stack and reads the innermost
tensor's value (``optiland.backend.utils.to_numpy``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402

import optiland.backend as be
from tests.nonsequential.test_nsq_interior_gradients import (
    _K_PER_SCENE,
    _T096_CASES,
    _forward_and_reverse,
)
from tests.nonsequential.test_nsq_source_jacobians import _CASES as _SOURCE_CASES
from tests.nonsequential.test_nsq_source_jacobians import _loss_of
from tests.nonsequential.test_nsq_tabulated_area_gradients import _CASES as _TABULATED_CASES

_U = 2.0**-53


@pytest.fixture(autouse=True)
def _torch_float64():
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


def _jvp(loss, value: float):
    """``(value, tangent)`` of ``loss`` at ``value`` by ``torch.func.jvp``."""
    primal, tangent = torch.func.jvp(
        loss,
        (torch.tensor(value, dtype=torch.float64),),
        (torch.tensor(1.0, dtype=torch.float64),),
    )
    return primal.item(), tangent.item()


class TestPlacementScenes:
    """The 20 placement scenes of chapter 09 section 9.12."""

    @pytest.mark.parametrize(("scene", "loss", "value"), _T096_CASES, ids=[c[0] for c in _T096_CASES])
    def test_equals_forward_ad_bit_for_bit(self, scene, loss, value):
        primal, tangent = _jvp(loss, value)
        forward, reverse = _forward_and_reverse(loss, value)
        assert tangent == forward, f"{scene}: jvp {tangent!r} forward_ad {forward!r}"
        with torch.no_grad():
            assert primal == loss(value).item(), f"{scene}: jvp moved the value"
        rel = abs(tangent - reverse) / abs(reverse)
        assert rel < _K_PER_SCENE[scene] * _U, (
            f"{scene}: jvp against reverse {rel / _U:.1f} u > K = {_K_PER_SCENE[scene]}"
        )


_GEOMETRY_CASES = [
    *((f"source-{c[0]}", _loss_of(c[1]), c[2]) for c in _SOURCE_CASES),
]


class TestSourceGeometryScenes:
    """The source-geometry parameters attached by the change of variables (R-09-4)."""

    @pytest.mark.parametrize(("case", "loss", "value"), _GEOMETRY_CASES, ids=[c[0] for c in _GEOMETRY_CASES])
    def test_equals_forward_ad_bit_for_bit(self, case, loss, value):
        _, tangent = _jvp(loss, value)
        forward, _ = _forward_and_reverse(loss, value)
        assert tangent == forward, f"{case}: jvp {tangent!r} forward_ad {forward!r}"

    @pytest.mark.parametrize(
        ("case", "config_of", "value", "name"), _TABULATED_CASES, ids=[c[0] for c in _TABULATED_CASES]
    )
    def test_tabulated_area(self, case, config_of, value, name):
        from tests.nonsequential.test_nsq_tabulated_area_gradients import (
            _centroid,
            _through_the_singlet,
            _trace,
        )

        def loss(v):
            return _centroid(_trace(_through_the_singlet(config_of(v))).detectors["D1"].data, 40.0)

        _, tangent = _jvp(loss, value)
        forward, _ = _forward_and_reverse(loss, value)
        assert tangent == forward, f"{case}: jvp {tangent!r} forward_ad {forward!r}"

    @pytest.mark.parametrize("key", ["sigma", "radius"])
    def test_gaussian_beam(self, key):
        from tests.nonsequential.test_nsq_gaussian_beam_gradients import (
            _RADIUS,
            _SIGMA,
            _beam,
            _centroid,
            _through_the_singlet,
            _trace,
        )

        def loss(v):
            return _centroid(_trace(_through_the_singlet(_beam(**{key: v}))).detectors["D1"].data, 40.0)

        value = _SIGMA if key == "sigma" else _RADIUS
        _, tangent = _jvp(loss, value)
        forward, _ = _forward_and_reverse(loss, value)
        assert tangent == forward, f"gaussian {key}: jvp {tangent!r} forward_ad {forward!r}"


def test_to_numpy_inside_a_transform_reads_the_value():
    """The host read inside ``jvp`` gives the bits it gives outside, for a wrapped and a plain tensor."""
    from optiland.backend.utils import to_numpy

    seen = {}

    def f(x):
        seen["wrapped"] = to_numpy(x * 3.0)
        seen["plain"] = to_numpy(torch.tensor([0.1, 0.2], dtype=torch.float64))
        return x * 3.0

    torch.func.jvp(f, (torch.tensor([0.7], dtype=torch.float64),), (torch.tensor([1.0], dtype=torch.float64),))
    assert seen["wrapped"].tolist() == [0.7 * 3.0]
    assert seen["plain"].tolist() == [0.1, 0.2]
