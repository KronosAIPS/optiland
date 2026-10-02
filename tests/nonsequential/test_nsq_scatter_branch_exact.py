"""A scatter fraction of exactly 0 or 1 draws no branch (the research repository's issue 55).

The clamp ``[1e-6, 1 - 1e-6]`` ran for ``scatter_fraction = 1`` (the default of
every BSDF surface). At float32 its two constants round inconsistently:
``1 - 1e-6`` becomes ``1 - 17 * 2**-24`` and ``1 / (1 - 1e-6)`` becomes
``1 + 8 * 2**-23``, so a scattering event's expected gate was
``(1 - 17 a)(1 + 16 a) = 1 - a - 272 a**2`` with ``a = 2**-24``: a systematic
loss of 6e-8 per event, about 2.9e-6 of r1_17's closed-sphere multiplier
(49 events per watt). With the fraction exactly 1 every hit scatters with a
gate of exactly 1, on both precisions and in the Warp stage's constants.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.nonsequential.components.sampling_support import (
    _SF_EPS,
    scatter_branch,
    scatter_branch_constants,
)
from optiland.nonsequential.rng import NSQRng


def test_constants_at_the_ends_and_inside():
    assert scatter_branch_constants(1.0) == (1.0, 1.0, 0.0)
    assert scatter_branch_constants(0.0) == (0.0, 0.0, 1.0)
    sf = 0.3
    assert scatter_branch_constants(sf) == (sf, 1.0, 1.0)
    tiny = 1e-9
    det, ws, wns = scatter_branch_constants(tiny)
    assert det == _SF_EPS
    assert ws == tiny / _SF_EPS
    assert wns == (1.0 - tiny) / (1.0 - _SF_EPS)


def test_the_old_float32_constants_were_biased():
    # the arithmetic the fix removes, stated so the mechanism stays on record
    a = 2.0**-24
    p32 = np.float32(1.0 - _SF_EPS)
    w32 = np.float32(1.0 / (1.0 - _SF_EPS))
    assert float(p32) == 1.0 - 17 * a  # P(u < p32) for a 24-bit draw, exactly
    assert float(w32) == 1.0 + 16 * a
    expected_gate = float(p32) * float(w32)
    assert expected_gate == pytest.approx(1.0 - a, abs=300 * a * a)


def _branch(sf, backend, precision, n=4096):
    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
        be.set_precision(precision)
    rng = NSQRng(7)
    ray_id = be.arange(n)
    bounce = be.zeros(n)
    hit = be.ones(n) > 0
    hit[::3] = False
    scatters, gate = scatter_branch(sf, hit, rng, ray_id, bounce)
    return be.to_numpy(hit), be.to_numpy(scatters), be.to_numpy(gate)


@pytest.mark.parametrize(
    ("backend", "precision"),
    [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")],
)
def test_fraction_one_scatters_every_hit_with_unit_gate(backend, precision):
    try:
        hit, scatters, gate = _branch(1.0, backend, precision)
    finally:
        be.set_backend("numpy")
    assert np.array_equal(scatters, hit)
    assert np.all(gate[hit] == 1.0)


@pytest.mark.parametrize(
    ("backend", "precision"),
    [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")],
)
def test_fraction_zero_scatters_nothing_with_unit_gate(backend, precision):
    try:
        hit, scatters, gate = _branch(0.0, backend, precision)
    finally:
        be.set_backend("numpy")
    assert not scatters.any()
    assert np.all(gate[hit] == 1.0)


def test_a_gradient_carrying_fraction_keeps_its_compensating_weight():
    torch = pytest.importorskip("torch")
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    try:
        sf = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
        rng = NSQRng(7)
        n = 64
        hit = torch.ones(n, dtype=torch.bool)
        scatters, gate = scatter_branch(sf, hit, rng, be.arange(n), be.zeros(n))
        gate.sum().backward()
        k = int(scatters.sum())
        # d/dsf of sf / p on the scatter branch and (1 - sf) / (1 - p) off it
        assert float(sf.grad) == pytest.approx(k / 0.75 - (n - k) / 0.25)
    finally:
        be.set_backend("numpy")
