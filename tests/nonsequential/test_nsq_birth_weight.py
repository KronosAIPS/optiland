"""A ray's birth weight is total_flux / N whatever batch it is born in (issue 24).

KronosNSRT issue 24: the trace loop gave a ray born in a batch of ``b`` rays
out of a source budget of ``N`` the weight ``(Phi / b) (b / N)``, which rounds
differently from ``Phi / N`` for some ``b``, so a ray of a remainder batch
carried a weight one unit in the last place off its siblings. Section 10.3 of
the theory requires per-ray weights bit-identical across batch sizes.

The scene and the ray count are chosen so that the old formula rounds away
from ``Phi / N`` at every batch size tried (the control below), which is what
makes the per-ray comparison discriminating rather than vacuous.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    PointSourceConfig,
    Spectrum,
)
from optiland.nonsequential.backends.array_backend import birth_flux_like
from optiland.nonsequential.backends.numpy_backend import NumpyBackend

torch = pytest.importorskip("torch")
from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: E402

#: Source flux and ray count at which (Phi / b)(b / N) != Phi / N for a batch
#: of every batch size below (a full batch of 1, 7 or 64, or the remainder).
PHI = 0.7
N = 204
BATCH_SIZES = (1, 7, 64, N)


def _scene(source: str = "collimated") -> NSQScene:
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    if source == "collimated":
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(spectrum=spec, total_flux=PHI, aperture_radius=8.0),
        )
    else:
        scene.add_source(
            "S",
            CoordinateSystem(),
            PointSourceConfig(spectrum=spec, total_flux=PHI, half_angle_deg=5.0),
        )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50.0),
        LensConfig(
            r1=60.0, r2=-60.0, thickness=6.0, material="N-BK7",
            front_aperture_radius=12.0,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=200.0),
        IrradianceDetectorConfig(
            width=40, height=40, num_pixels_x=16, num_pixels_y=16, splat="hard"
        ),
    )
    return scene


def _backend(kind: str):
    if kind == "numpy":
        be.set_backend("numpy")
        return NumpyBackend(seed=9)
    be.set_backend("torch")
    be.set_precision("float64")
    return TorchBackend(seed=9, alive_check_every=0)


@pytest.fixture(autouse=True)
def _restore_backend():
    # Start from forward mode whatever an earlier test left behind: the
    # replay refuses a gradient trace, and these tests trace forward only.
    be.set_backend("torch")
    was_on = bool(be.grad_mode.requires_grad)
    be.grad_mode.disable()
    be.set_backend("numpy")
    yield
    be.set_backend("torch")
    if was_on:
        be.grad_mode.enable()
    be.set_backend("numpy")
    be.set_precision("float64")


def test_the_old_formula_rounds_away_at_every_batch_size():
    """The control: at (PHI, N) the old weight differs from PHI / N for every b tried."""
    for b in BATCH_SIZES[:-1]:
        sizes = {b, N % b} - {0}
        assert any((PHI / s) * (s / N) != PHI / N for s in sizes), b


@pytest.mark.parametrize("kind", ["numpy", "torch"])
@pytest.mark.parametrize("source", ["collimated", "point"])
def test_per_ray_records_are_bit_identical_across_batch_sizes(kind, source):
    paths = {}
    bins = {}
    for batch_size in BATCH_SIZES:
        result = _scene(source).trace(
            num_rays=N, seed=9, max_depth=8, batch_size=batch_size,
            record_paths=True, backend=_backend(kind),
        )
        events = result.ray_paths["events"]
        # One ray's events are logged in the order they happen; a stable sort
        # by ray id keeps that order and removes the batch's interleaving.
        paths[batch_size] = events[np.argsort(events["ray_id"], kind="stable")]
        bins[batch_size] = np.asarray(
            be.to_numpy(result.detectors["D1"].data), dtype=np.float64
        ).copy()

    reference = paths[N]
    births = reference[reference["event_type"] == "birth"]
    assert len(births) == N
    # Every ray is born with the one weight PHI / N, in its IEEE-754 bytes.
    assert np.all(births["flux"] == PHI / N)
    for batch_size, rp in paths.items():
        assert rp.shape == reference.shape, batch_size
        assert rp.tobytes() == reference.tobytes(), (
            f"batch_size={batch_size}: per-ray records differ from the single batch"
        )
        # The detector's bins: bit-identical too, here, because every ray
        # reaches the detector at the same bounce and so in the same order.
        assert bins[batch_size].tobytes() == bins[N].tobytes(), batch_size


def test_birth_flux_like_matches_the_full_batch_construction():
    """The weights a partial batch gets are the bytes a full batch gets, on both libraries."""
    per_ray = PHI / N
    flux_np = np.full(5, PHI / 5)
    out = birth_flux_like(flux_np, per_ray)
    assert out.dtype == np.float64
    assert out.tobytes() == np.full(5, per_ray).tobytes()
    for dtype in (torch.float64, torch.float32):
        ones = torch.ones(5, dtype=dtype)
        got = birth_flux_like(ones * (PHI / 5), per_ray)
        assert got.dtype == dtype
        assert torch.equal(got, ones * per_ray)
    # A tensor total flux keeps its graph.
    phi = torch.tensor(PHI, dtype=torch.float64, requires_grad=True)
    got = birth_flux_like(torch.ones(4, dtype=torch.float64) * (phi / 4), phi / N)
    got.sum().backward()
    assert phi.grad is not None and float(phi.grad) == pytest.approx(4 / N)
