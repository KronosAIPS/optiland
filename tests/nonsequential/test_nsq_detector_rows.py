"""Detector bins are summed in rows keyed by ray id, then pairwise (issue 23).

KronosNSRT issue 23: a scatter-add (``np.add.at``, ``index_add_``) adds a
bin's contributions one after another, so a bin that receives ``K`` of them
rounds as ``K u``: the catalogue's window adds 3.7 million equal weights into
one bin and its ledger closed to 4.0e-11 against the 1e-11 of section 10.3.
A float64 detector buffer is now ``(rows, size)``; ray ``i`` adds into row
``i % rows`` and the rows are reduced pairwise when the detector is read.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    FarFieldDetectorConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    SpectralDetectorConfig,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.detectors.base import (
    _BIN_ROWS_MAX,
    _BIN_ROWS_MAX_ELEMENTS,
    bin_rows,
    bin_values,
)
from optiland.nonsequential.detectors.irradiance import IrradianceDetector

torch = pytest.importorskip("torch")
from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: E402

U64 = 2.0**-53


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def test_rows_depend_on_the_size_alone():
    assert bin_rows(1) == _BIN_ROWS_MAX
    for size in (1, 3, 16, 1024, 1000, 70_000, 1 << 18, 1 << 20):
        rows = bin_rows(size)
        assert rows >= 1 and rows & (rows - 1) == 0
        assert rows == 1 or rows * size <= _BIN_ROWS_MAX_ELEMENTS


def _one_bin_rays(ids, weight, lib):
    n = len(ids)
    if lib == "torch":
        arr = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
        return types.SimpleNamespace(
            x=arr(np.zeros(n)), y=arr(np.zeros(n)), z=arr(np.zeros(n)),
            L=arr(np.zeros(n)), M=arr(np.zeros(n)), N=arr(np.ones(n)),
            flux=arr(np.full(n, weight)), wavelength=arr(np.full(n, 0.55)),
            ray_id=torch.as_tensor(ids, dtype=torch.int64),
        ), torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.float64)
    return types.SimpleNamespace(
        x=np.zeros(n), y=np.zeros(n), z=np.zeros(n),
        L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
        flux=np.full(n, weight), wavelength=np.full(n, 0.55),
        ray_id=np.asarray(ids, dtype=np.int64),
    ), np.ones(n, dtype=bool), np.zeros(n)


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_a_bin_of_millions_of_equal_weights_rounds_as_a_pairwise_sum(lib):
    """The window's case in miniature: 2^21 equal weights into one bin, 16384 per call."""
    if lib == "torch":
        be.set_backend("torch")
        be.set_precision("float64")
    det = IrradianceDetector(
        CoordinateSystem(), width=10.0, height=10.0, num_pixels_x=1, num_pixels_y=1,
    )
    weight = 1.0 / 3_999_997.0  # the shape of Phi / N: not a short binary fraction
    batch, calls = 16_384, 128
    for c in range(calls):
        rays, hit, t = _one_bin_rays(np.arange(c * batch, (c + 1) * batch), weight, lib)
        det.record(rays, t, hit)
    k = batch * calls
    exact = math.fsum([weight] * k)
    got = float(be.to_numpy(det.get_result().data)[0])
    rows = bin_rows(1)
    bound = (k / rows + math.log2(rows) + 1) * U64
    assert abs(got - exact) / exact <= bound
    assert abs(got - exact) / exact <= 1e-14
    # The control: the sequential sum of the same weights, as the plain
    # scatter-add formed it, is far outside that bound.
    plain = np.zeros(1)
    np.add.at(plain, np.zeros(k, dtype=np.int64), np.full(k, weight))
    assert abs(plain[0] - exact) / exact > 100 * bound


def test_bin_values_reduces_the_rows_pairwise():
    rng = np.random.default_rng(2)
    buf = rng.random((8, 5))
    expected = ((buf[0] + buf[1]) + (buf[2] + buf[3])) + (
        (buf[4] + buf[5]) + (buf[6] + buf[7])
    )
    assert bin_values(buf).tobytes() == expected.tobytes()
    flat = rng.random(5)
    assert bin_values(flat) is flat
    assert bin_values(None) is None
    t = torch.as_tensor(buf)
    assert torch.equal(bin_values(t), torch.as_tensor(expected))


def _window(num_pixels: int = 1) -> NSQScene:
    """A plane-parallel window before a one-bin detector: every ray ends there or escapes."""
    scene = NSQScene()
    scene.add_source(
        "S", CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=2.0
        ),
    )
    scene.add_lens(
        "W", CoordinateSystem(z=10.0),
        LensConfig(r1=0.0, r2=0.0, thickness=5.0, material="N-BK7",
                   front_aperture_radius=10.0),
    )
    scene.add_detector(
        "D", CoordinateSystem(z=30.0),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=num_pixels,
                                 num_pixels_y=num_pixels, reflection_bins=3),
    )
    return scene


def test_the_windows_ledger_closes_to_rounding():
    """400,000 rays, 367,362 of them into one bin: the ledger closes to 1e-15.

    The engine before this change closed the same trace to 4.5e-14
    (numpy float64, seed 1), the sequential sum's K u growth; the case r2_01
    runs ten times the rays, where it reached 4.0e-11.
    """
    result = _window().trace(
        num_rays=400_000, seed=1, max_depth=16, backend=NumpyBackend(seed=1)
    )
    assert result.detectors["D"].num_rays_hit > 300_000
    assert result.flux_conservation_error < 1e-15


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_bins_are_bit_identical_across_batch_sizes(kind):
    """The row of a contribution is its ray's, so batching does not reorder a row."""
    num_rays = 3000 if kind == "numpy" else 400
    maps = {}
    for batch_size in (1, 7, 64, num_rays):
        if kind == "torch":
            be.set_backend("torch")
            be.set_precision("float64")
            backend = TorchBackend(seed=4, alive_check_every=0)
        else:
            backend = NumpyBackend(seed=4)
        result = _window(num_pixels=4).trace(
            num_rays=num_rays, seed=4, max_depth=16, batch_size=batch_size,
            backend=backend,
        )
        det = result.detectors["D"]
        hist = result.reflection_histograms["D"]
        maps[batch_size] = (
            np.asarray(be.to_numpy(det.data)).tobytes(),
            hist.flux.tobytes(),
            hist.flux_sq.tobytes(),
        )
    assert len(set(maps.values())) == 1


def test_every_detector_kind_replays():
    """Irradiance with reflection bins, far field, spectral tap: the row buffers under emulate."""
    be.set_backend("torch")
    be.set_precision("float64")

    def scene():
        s = _window(num_pixels=8)
        s.add_detector(
            "TAP", CoordinateSystem(z=5.0),
            SpectralDetectorConfig(width=40, height=40, num_pixels_x=4, num_pixels_y=4,
                                   wl_min=0.4, wl_max=0.7, num_bins=3,
                                   splat="bilinear", absorb=False),
        )
        s.add_detector(
            "FF", CoordinateSystem(z=-20.0),
            FarFieldDetectorConfig(num_theta=8, num_phi=8),
        )
        return s

    kw = dict(num_rays=8192, seed=6, max_depth=12, batch_size=4096)
    eager = scene().trace(
        **kw, backend=TorchBackend(seed=6, alive_check_every=0, compact_every=0)
    )
    emulated = scene().trace(
        **kw, backend=TorchBackend(seed=6, alive_check_every=0, graph_replay="emulate")
    )
    assert emulated.environment["graph_replay_batches"] == 2, emulated.environment
    assert emulated.flux_conservation_error < 1e-14
    for name in ("D", "TAP", "FF"):
        a, b = eager.detectors[name], emulated.detectors[name]
        for field in ("data", "irradiance", "intensity"):
            if hasattr(a, field) and getattr(a, field) is not None:
                x = np.asarray(be.to_numpy(getattr(a, field)))
                y = np.asarray(be.to_numpy(getattr(b, field)))
                assert x.tobytes() == y.tobytes(), (name, field)
