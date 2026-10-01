"""The entry split as an engine function (T-09-6, second form).

``docs/theory/09_differentiation.md`` of the research repository, section
9.13.6 (written before these tests ran); the research repository's issue 31.
:func:`~optiland.nonsequential.parameter_register.entry_split` numbers the
entry elements of a parameter -- every element of every tangent the trace
attaches -- and keeps one at a time, with a hook on the detector's
scatter-adds recording each hit's value and tangent. The bounds the chapter
states: its contributions are bit-identical to the instrumentation of
``test_nsq_interior_gradients.py`` (which replaces two engine functions) on
the scenes the second form is tested on, and they sum to the forward-mode
derivative to 1e-12 relative. The second form's own bound (``64 u`` of the
absolute sum) is that test's; here it is asserted again through the function,
which needs no instrumentation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import parameter_register
from optiland.nonsequential.detectors import base as detector_base
from optiland.nonsequential.parameter_register import entry_split
from tests.nonsequential.test_nsq_interior_gradients import (
    _SECOND_FORM_CASES,
    _centroid_bin_weights,
    _entry_split_contributions,
    _forward_and_reverse,
    _seeded_loss,
)

_U = 2.0**-53


@pytest.fixture(autouse=True)
def _torch_float64():
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


def _dloss_dbins(split, width):
    bins = split.image()
    g = _centroid_bin_weights(width)
    total = bins.sum()
    return (g - (bins * g).sum() / total) / total


def _build_of(build, nominal, key):
    def b(value):
        p = dict(nominal)
        p[key] = value
        return build(p)[0]

    return b


@pytest.mark.parametrize(
    ("scene", "build", "nominal", "key"), _SECOND_FORM_CASES, ids=[c[0] for c in _SECOND_FORM_CASES]
)
def test_equals_the_test_instrumentation_bit_for_bit(scene, build, nominal, key):
    width = build(dict(nominal))[1]
    split = entry_split(_build_of(build, nominal, key), nominal[key])
    ours = split.contributions(_dloss_dbins(split, width))
    theirs = _entry_split_contributions(_seeded_loss(build, nominal, key, 3), nominal[key], width)
    assert ours.shape == theirs.shape
    assert torch.equal(ours, theirs)


@pytest.mark.parametrize("seed", [3, 5])
@pytest.mark.parametrize(
    ("scene", "build", "nominal", "key"), _SECOND_FORM_CASES, ids=[c[0] for c in _SECOND_FORM_CASES]
)
def test_sums_to_the_derivative_and_holds_the_second_form(scene, build, nominal, key, seed):
    width = build(dict(nominal))[1]
    split = entry_split(
        _build_of(build, nominal, key),
        nominal[key],
        trace=lambda s: s.trace(num_rays=2_000, seed=seed, max_depth=8),
    )
    dloss = _dloss_dbins(split, width)
    forward, reverse = _forward_and_reverse(_seeded_loss(build, nominal, key, seed), nominal[key])
    assert split.derivative(dloss) == pytest.approx(forward, rel=1e-12, abs=0.0)
    gap = abs(forward - reverse)
    assert gap < 64 * _U * split.absolute_sum(dloss)


def test_the_elements_of_a_lens_tilt():
    """A lens tilt reaches the rotations of its surfaces: every live element is a rotation element."""
    build, nominal, key = _SECOND_FORM_CASES[0][1:]
    split = entry_split(_build_of(build, nominal, key), nominal[key])
    assert 1 <= len(split.elements) <= 32
    assert {shape for _call, _element, shape in split.elements} <= {(3,), (3, 3)}
    assert len(split.by_element) == len(split.elements)


def test_leaves_no_state_behind():
    build, nominal, key = _SECOND_FORM_CASES[2][1:]
    entry_split(_build_of(build, nominal, key), nominal[key])
    assert parameter_register._ENTRY_SPLIT is None
    assert detector_base.HIT_OBSERVERS == []
