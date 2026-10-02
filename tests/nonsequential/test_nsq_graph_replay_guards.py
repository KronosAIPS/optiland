"""The emulated replay's host-transfer check and ``torch.compile``'s own guards.

The research repository's issue 104: with the compiled asphere refinement on
(``OPTILAND_NSQ_COMPILE_REFINE``), ``graph_replay="emulate"`` on the CPU refused
the trace, because dynamo evaluates ``not math.isnan(L[name].item())`` on every
0-dim float CPU tensor a compiled program takes, before it runs the program.
On the CPU that read copies nothing, and on CUDA the capture recorded the
compiled refinement. The check now exempts a generated guard reading a CPU
tensor, and only that: an ``item`` in user code is still recorded.
"""

from __future__ import annotations

import pytest

import optiland.backend as be

torch = pytest.importorskip("torch")

from optiland.nonsequential.backends.graph_replay import (  # noqa: E402
    host_transfer_check,
)


def _clamp(x, floor):
    return torch.maximum(x, floor) * 2.0


@pytest.fixture
def compiled():
    torch._dynamo.reset()
    yield torch.compile(_clamp, backend="eager", dynamic=False)
    torch._dynamo.reset()


def test_compiler_guard_on_a_cpu_scalar_is_not_a_transfer(compiled):
    x = torch.ones(8, dtype=torch.float64)
    floor = torch.tensor(1e-300, dtype=torch.float64)  # a 0-dim CPU constant
    compiled(x, floor)  # compile outside the check
    with host_transfer_check() as seen:
        out = compiled(x, floor)
    assert seen == []
    assert torch.equal(out, 2.0 * x)


def test_a_read_in_user_code_is_still_recorded(compiled):
    x = torch.ones(8, dtype=torch.float64)
    floor = torch.tensor(1e-300, dtype=torch.float64)
    compiled(x, floor)
    with host_transfer_check() as seen:
        compiled(x, floor)
        floor.item()
        float(x.sum())
    kinds = sorted(kind for kind, _ in seen)
    assert kinds == [
        "reads a tensor to the host (__float__)",
        "reads a tensor to the host (item)",
    ]


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_asphere_emulated_replay_with_the_compiled_refinement(monkeypatch, precision):
    # the issue's reproduction: the asphere file's emulated-replay test with
    # the switch on (it refused the trace before the exemption)
    from tests.nonsequential.test_nsq_asphere import TestTraced  # noqa: PLC0415

    monkeypatch.setenv("OPTILAND_NSQ_COMPILE_REFINE", "1")
    try:
        TestTraced().test_emulated_replay_is_the_eager_trace(precision)
    finally:
        be.set_backend("numpy")
