"""The result says whether the bounces were replayed.

``TorchBackend(graph_replay=...)`` hands a batch to a replayed graph after its
eager bounces, but a batch narrower than the compaction ladder's floor runs
eagerly whatever was asked. ``SimulationResult.environment`` records what
ran: ``graph_replay`` (``"cuda"``, ``"emulate"`` or ``"none"``),
``graph_replay_requested`` and ``graph_replay_batches``. Without the request
the keys are absent and the environment is what it was before.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.array_backend import BUCKET_MIN_WIDTH
from optiland.nonsequential.backends.torch_backend import TorchBackend


@pytest.fixture
def torch_state():
    previous = be.get_backend()
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.disable()
    yield
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.set_backend(previous)


def _singlet() -> NSQScene:
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0
        ),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50),
        LensConfig(
            r1=100.0,
            r2=-100.0,
            thickness=5.0,
            material="N-BK7",
            front_aperture_radius=12.5,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=150),
        IrradianceDetectorConfig(
            width=20, height=20, num_pixels_x=8, num_pixels_y=8, splat="bilinear"
        ),
    )
    return scene


def _trace(backend, num_rays, batch_size):
    return _singlet().trace(
        num_rays=num_rays, seed=5, max_depth=8, batch_size=batch_size, backend=backend
    )


def test_an_emulated_replay_is_recorded_with_its_batch_count(torch_state):
    backend = TorchBackend(seed=5, alive_check_every=0, graph_replay="emulate")
    width = BUCKET_MIN_WIDTH
    env = _trace(backend, 3 * width, width).environment
    assert env["graph_replay"] == "emulate"
    assert env["graph_replay_requested"] == "emulate"
    assert env["graph_replay_batches"] == 3
    assert backend.graph_replay_batches == 3


def test_narrow_batches_run_eagerly_and_say_so(torch_state):
    """Below the ladder's floor no batch is replayed, and the record says none."""
    backend = TorchBackend(seed=5, alive_check_every=0, graph_replay="emulate")
    env = _trace(backend, 1000, 500).environment
    assert env["graph_replay"] == "none"
    assert env["graph_replay_batches"] == 0


def test_the_count_starts_again_at_each_trace(torch_state):
    backend = TorchBackend(seed=5, alive_check_every=0, graph_replay="emulate")
    _trace(backend, 2 * BUCKET_MIN_WIDTH, BUCKET_MIN_WIDTH)
    env = _trace(backend, 1000, 1000).environment
    assert env["graph_replay_batches"] == 0
    assert env["graph_replay"] == "none"


def test_without_the_request_the_keys_are_absent(torch_state):
    env = _trace(TorchBackend(seed=5), 1000, 1000).environment
    assert not {k for k in env if k.startswith("graph_replay")}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_cuda_replay_is_recorded(torch_state):
    be.set_device("cuda")
    try:
        backend = TorchBackend(seed=5, alive_check_every=0, graph_replay=True)
        env = _trace(backend, 2 * BUCKET_MIN_WIDTH, BUCKET_MIN_WIDTH).environment
    finally:
        be.set_device("cpu")
    assert env["graph_replay"] == "cuda"
    assert env["graph_replay_requested"] is True
    assert env["graph_replay_batches"] == 2
