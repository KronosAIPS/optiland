"""Tests for the detector accumulation-buffer fix.

Three things are exercised, matching the fix's own requirements:

1. The splatting detectors (irradiance, far field, spectral) accumulate
   *in place*: the buffer's identity (and, on Torch, its storage) does not
   change across ``record()`` calls, so a bounce never reallocates the
   pixel buffer.
2. The buffer is float64 (the accumulation dtype) wherever the device has
   float64, whatever the working/traversal dtype is -- checked here on the
   Torch backend at float32 precision on the CPU. On Apple's ``mps`` device
   the accumulator is float32 (Metal has no double type) and the scatter-add
   goes through a grouped pairwise reduction whose error is bounded by about
   (K / G + log2 G + 1) units of float32 roundoff per bin instead of K; the
   grouped reduction is checked on the CPU against a float64 sum, and the
   mps dtype choice when the device is present.
3. A gradient still flows through the in-place accumulation on the Torch
   backend.

Kramer Harrison, 2026
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential.detectors.far_field import FarFieldDetector
from optiland.nonsequential.detectors.irradiance import IrradianceDetector
from optiland.nonsequential.detectors.spectral import SpectralDetector

torch = pytest.importorskip("torch", reason="Torch not available -- skip these tests")

import optiland.backend as be  # noqa: E402
from optiland.backend.utils import to_numpy  # noqa: E402
from optiland.nonsequential import (  # noqa: E402
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: E402


def _reset_backend() -> None:
    be.set_backend("numpy")


def _make_rays(n: int, seed: int = 0, backend: str = "numpy"):
    """Synthetic ray bundle covering only the fields record() reads."""
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(-4.0, 4.0, n)
    y_np = rng.uniform(-4.0, 4.0, n)
    flux_np = rng.uniform(0.1, 1.0, n).astype(np.float32)
    z_np = np.zeros(n)
    L_np = np.zeros(n)
    M_np = np.zeros(n)
    N_np = np.ones(n)
    wl_np = np.full(n, 0.55)
    alive_np = np.ones(n, dtype=bool)

    if backend == "torch":
        as_arr = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)  # noqa: E731
        alive = torch.as_tensor(alive_np, dtype=torch.bool)
    else:
        as_arr = lambda a, dt=np.float64: np.asarray(a, dtype=dt)  # noqa: E731
        alive = alive_np

    return types.SimpleNamespace(
        x=as_arr(x_np),
        y=as_arr(y_np),
        z=as_arr(z_np),
        L=as_arr(L_np),
        M=as_arr(M_np),
        N=as_arr(N_np),
        flux=as_arr(flux_np, torch.float32 if backend == "torch" else np.float64),
        wavelength=as_arr(wl_np),
        alive=alive,
    ), flux_np


# ---------------------------------------------------------------------------
# 1. In-place accumulation: buffer identity does not change across record()
# ---------------------------------------------------------------------------


class TestInPlaceAccumulation:
    def test_irradiance_buffer_identity_numpy(self):
        det = IrradianceDetector(
            CoordinateSystem(), width=10.0, height=10.0,
            num_pixels_x=8, num_pixels_y=8, splat="bilinear",
        )
        rays, _ = _make_rays(2000, seed=1, backend="numpy")
        t = np.zeros(2000)
        hit_mask = np.ones(2000, dtype=bool)

        buf_before = det._data
        det.record(rays, t, hit_mask)
        assert det._data is buf_before, "record() must not reallocate the buffer"
        det.record(rays, t, hit_mask)
        assert det._data is buf_before

    def test_irradiance_buffer_identity_torch(self):
        be.set_backend("torch")
        try:
            det = IrradianceDetector(
                CoordinateSystem(), width=10.0, height=10.0,
                num_pixels_x=8, num_pixels_y=8, splat="bilinear",
            )
            rays, _ = _make_rays(2000, seed=1, backend="torch")
            t = torch.zeros(2000)
            hit_mask = torch.ones(2000, dtype=torch.bool)

            buf_before = det._data
            ptr_before = buf_before.data_ptr()
            det.record(rays, t, hit_mask)
            assert det._data is buf_before
            assert det._data.data_ptr() == ptr_before, (
                "in-place index_add_ must not reallocate storage"
            )
            det.record(rays, t, hit_mask)
            assert det._data is buf_before
            assert det._data.data_ptr() == ptr_before
        finally:
            _reset_backend()

    def test_far_field_buffer_identity(self):
        det = FarFieldDetector(
            CoordinateSystem(), theta_max_deg=30.0, num_bins_theta=8, num_bins_phi=8,
        )
        rays, _ = _make_rays(2000, seed=2, backend="numpy")
        t = np.zeros(2000)
        hit_mask = np.ones(2000, dtype=bool)

        buf_before = det._intensity
        det.record(rays, t, hit_mask)
        assert det._intensity is buf_before
        det.record(rays, t, hit_mask)
        assert det._intensity is buf_before

    def test_spectral_buffer_identity(self):
        det = SpectralDetector(
            CoordinateSystem(), width=10.0, height=10.0,
            num_pixels_x=8, num_pixels_y=8,
            wavelength_bins=np.array([0.4, 0.5, 0.6, 0.7]),
        )
        rays, _ = _make_rays(2000, seed=3, backend="numpy")
        t = np.zeros(2000)
        hit_mask = np.ones(2000, dtype=bool)

        buf_before = det._flux_map
        det.record(rays, t, hit_mask)
        assert det._flux_map is buf_before
        det.record(rays, t, hit_mask)
        assert det._flux_map is buf_before


# ---------------------------------------------------------------------------
# 2. Accumulation dtype is float64 even when the working dtype is float32
# ---------------------------------------------------------------------------


class TestAccumulationDtype:
    def test_irradiance_accumulates_in_float64_under_float32_torch(self):
        be.set_backend("torch")
        be.set_precision("float32")
        try:
            det = IrradianceDetector(
                CoordinateSystem(), width=10.0, height=10.0,
                num_pixels_x=16, num_pixels_y=16, splat="hard",
            )
            assert det._data.dtype == torch.float64

            n = 4000
            rays, flux_np = _make_rays(n, seed=7, backend="torch")
            t = torch.zeros(n)
            hit_mask = torch.ones(n, dtype=torch.bool)

            det.record(rays, t, hit_mask)
            assert det._data.dtype == torch.float64

            # Hard splat assigns each ray's flux to exactly one bin with no
            # weight splitting, so the reference is just the float64 sum of
            # the (float32) per-ray flux values that landed inside the grid.
            in_grid = (np.abs(to_numpy(rays.x)) <= 5.0) & (np.abs(to_numpy(rays.y)) <= 5.0)
            expected_total = float(flux_np[in_grid].astype(np.float64).sum())
            actual_total = float(to_numpy(det._data).sum())

            assert expected_total > 0.0
            rel_err = abs(actual_total - expected_total) / expected_total
            assert rel_err < 1e-12, (
                f"float64 accumulation drifted from the float64 sum of the "
                f"float32 contributions: rel_err={rel_err:.3e}"
            )
        finally:
            be.set_precision("float64")
            _reset_backend()


class TestFloat32AccumulatorOnDevicesWithoutFloat64:
    """The accumulator on a device without float64 (Apple's mps): float32, with
    the grouped pairwise reduction bounding the rounding error."""

    def test_grouped_scatter_add_matches_float64_sum_to_machine_precision(self):
        """20,000 equal contributions into one bin: a bare float32 scatter-add
        rounds the running sum once per add (a systematic error of order
        K * u32 = 1.2e-3 relative for equal terms); the grouped reduction keeps
        it within a few units of float32 roundoff. Tolerance derived from the
        stated bound (K / G + log2 G + 1) * u32 with G = 256: about 9e-6, asserted
        at 1e-5; the bare scatter-add is asserted to be worse than 1e-4 so the
        test would notice if the grouped path were bypassed."""
        from optiland.nonsequential.detectors.base import _grouped_index_add_

        k, size = 20_000, 8
        src = torch.full((k,), 5.0e-5, dtype=torch.float32)
        idx = torch.zeros(k, dtype=torch.int64)
        exact = float(src.double().sum())
        grouped = torch.zeros(size, dtype=torch.float32)
        _grouped_index_add_(grouped, idx, src)
        bare = torch.zeros(size, dtype=torch.float32).index_add_(0, idx, src)
        assert abs(float(grouped[0]) - exact) / exact < 1e-5
        assert abs(float(bare[0]) - exact) / exact > 1e-4
        assert float(grouped[1:].abs().sum()) == 0.0

    def test_grouped_scatter_add_keeps_the_gradient(self):
        from optiland.nonsequential.detectors.base import _grouped_index_add_

        src = torch.full((4096,), 1.0e-3, dtype=torch.float32, requires_grad=True)
        idx = torch.arange(4096, dtype=torch.int64) % 3
        buf = torch.zeros(3, dtype=torch.float32)
        _grouped_index_add_(buf, idx, src)
        buf.sum().backward()
        assert torch.allclose(src.grad, torch.ones_like(src))

    def test_accumulator_dtype_follows_the_device(self):
        from optiland.nonsequential.detectors.base import accumulator_dtype

        be.set_backend("torch")
        try:
            be.set_precision("float32")
            be.set_device("cpu")
            assert accumulator_dtype() == be.float64
            if torch.backends.mps.is_available():
                be.set_device("mps")
                assert accumulator_dtype() == be.float32
                det = IrradianceDetector(CoordinateSystem(), width=10.0, height=10.0,
                                         num_pixels_x=4, num_pixels_y=4, splat="hard")
                assert det._data.dtype == torch.float32 and det._data.device.type == "mps"
        finally:
            be.set_device("cpu")
            be.set_precision("float64")
            _reset_backend()


# ---------------------------------------------------------------------------
# 3. Gradients still flow through the in-place accumulation on Torch
# ---------------------------------------------------------------------------


class TestGradientThroughInPlaceAccumulation:
    def test_total_flux_gradient_wrt_width_is_defined(self):
        """``total_flux.backward()`` must run and reach ``width`` unbroken.

        The bilinear splat's four corner weights sum to exactly 1 per ray
        (``wx0 + wx1 == 1`` by construction), so ``total_flux`` -- the sum
        over every pixel -- is analytically independent of the detector
        width as long as no ray's hit/no-hit status changes (a measured
        property of this scene, not a regression from this fix: the
        gradient below is expected to be exactly zero, the same
        "visibility gradient is zero" limitation documented elsewhere in
        this suite for the detector's aperture boundary). What this test
        actually guards is that the in-place accumulation keeps the graph
        connected -- ``width.grad`` must be a defined, real tensor rather
        than ``None`` (which would mean the graph broke).
        """
        be.set_backend("torch")
        try:
            width = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
            spec = Spectrum.monochromatic(0.55)
            scene = NSQScene()
            scene.add_source(
                "S",
                CoordinateSystem(),
                CollimatedSourceConfig(
                    spectrum=spec, total_flux=1.0, aperture_radius=8.0
                ),
            )
            scene.add_detector(
                "D",
                CoordinateSystem(z=10),
                IrradianceDetectorConfig(
                    width=width, height=width,
                    num_pixels_x=16, num_pixels_y=16, splat="bilinear",
                ),
            )
            result = scene.trace(num_rays=2000, seed=0, backend=TorchBackend(seed=0))
            total_flux = result.detectors["D"].total_flux
            assert total_flux.requires_grad
            total_flux.backward()
            assert width.grad is not None
        finally:
            _reset_backend()

    def test_per_pixel_gradient_wrt_width_is_nonzero(self):
        """A genuine, informative gradient does reach a single pixel.

        Unlike ``total_flux`` (see the test above), an individual pixel's
        accumulated value is not protected by the corner-weight identity,
        so its gradient w.r.t. width is nonzero when the ray lands away
        from the pixel centre -- this is the meaningful check that the
        in-place ``index_add_`` genuinely preserves per-element gradient
        information, not just a trivially disconnected zero.
        """
        be.set_backend("torch")
        try:
            width = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
            spec = Spectrum.monochromatic(0.55)
            scene = NSQScene()
            scene.add_source(
                "S",
                CoordinateSystem(),
                CollimatedSourceConfig(
                    spectrum=spec, total_flux=1.0, aperture_radius=8.0
                ),
            )
            scene.add_detector(
                "D",
                CoordinateSystem(z=10),
                IrradianceDetectorConfig(
                    width=width, height=width,
                    num_pixels_x=16, num_pixels_y=16, splat="bilinear",
                ),
            )
            result = scene.trace(num_rays=2000, seed=0, backend=TorchBackend(seed=0))
            data = result.detectors["D"].data
            assert data.requires_grad

            nonzero = (data.detach() != 0).nonzero().flatten().tolist()
            assert nonzero, "expected at least one lit pixel"
            pixel_value = data[nonzero[len(nonzero) // 2]]
            pixel_value.backward()
            assert width.grad is not None
            assert width.grad.abs().item() > 0.0
        finally:
            _reset_backend()


def _skewed_contributions(k=50_000, size=512, seed=3):
    """Contributions piled into a few heavy bins (bin = size * u^3) and spread thinly."""
    g = torch.Generator().manual_seed(seed)
    idx = (torch.rand(k, generator=g) ** 3 * size).long()
    src = torch.rand(k, generator=g) * 1e-3
    return idx, src


def _pairwise_reference(idx, src, size):
    """The ordered accumulation written out per bin in plain Python: each bin's
    contributions in call order, summed by the pairwise tree in float32."""
    out = np.zeros(size, dtype=np.float32)
    idx_np, src_np = idx.numpy(), src.numpy().astype(np.float32)
    for b in np.unique(idx_np):
        vals = list(src_np[idx_np == b])
        step = 1
        while step < len(vals):
            for r in range(0, len(vals) - step, 2 * step):
                vals[r] = np.float32(vals[r] + vals[r + step])
            step *= 2
        out[b] = np.float32(out[b] + vals[0])
    return out


class TestOrderedFloat32Accumulation:
    """The float32 accumulator's scatter-add in a fixed order (research repository
    issue 69): a sort by bin, a pairwise tree per bin, one write per bin.

    The rounding bound per bin is ``(ceil(log2 K_b) + 1) u32`` relative for
    ``K_b`` non-negative contributions: the pairwise tree has depth
    ``ceil(log2 K_b)`` and each level rounds a partial sum once (the standard
    bound for pairwise summation), plus the addition into the buffer."""

    def test_it_is_the_pairwise_sum_of_each_bin_in_call_order_to_the_bit(self):
        from optiland.nonsequential.detectors.base import _ordered_index_add_

        idx, src = _skewed_contributions(k=6000, size=64)
        buf = torch.zeros(64, dtype=torch.float32)
        _ordered_index_add_(buf, idx, src)
        assert np.array_equal(buf.numpy(), _pairwise_reference(idx, src, 64))

    def test_every_bin_is_within_the_pairwise_bound(self):
        import math

        from optiland.nonsequential.detectors.base import _ordered_index_add_

        idx, src = _skewed_contributions()
        buf = torch.zeros(512, dtype=torch.float32)
        _ordered_index_add_(buf, idx, src)
        exact = torch.zeros(512, dtype=torch.float64).index_add_(0, idx, src.double())
        counts = torch.bincount(idx, minlength=512)
        hit = counts > 0
        bound = torch.tensor(
            [math.ceil(math.log2(c)) + 1 for c in counts[hit].tolist()], dtype=torch.float64
        ) * 2.0**-24
        rel = (buf.double()[hit] - exact[hit]).abs() / exact[hit]
        assert bool((rel <= bound).all())
        assert float(buf[~hit].abs().sum()) == 0.0
        assert int(counts.max()) > 5000  # a heavy bin is in the test

    def test_equal_contributions_into_one_bin(self):
        """The grouped test's case (20,000 equal terms, one bin): bound 16 u32."""
        from optiland.nonsequential.detectors.base import _ordered_index_add_

        src = torch.full((20_000,), 5.0e-5, dtype=torch.float32)
        idx = torch.zeros(20_000, dtype=torch.int64)
        exact = float(src.double().sum())
        buf = torch.zeros(8, dtype=torch.float32)
        _ordered_index_add_(buf, idx, src)
        assert abs(float(buf[0]) - exact) / exact <= 16 * 2.0**-24
        assert float(buf[1:].abs().sum()) == 0.0

    def test_it_adds_to_what_the_buffer_holds_and_ignores_an_empty_call(self):
        from optiland.nonsequential.detectors.base import _ordered_index_add_

        buf = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        _ordered_index_add_(buf, torch.zeros(0, dtype=torch.int64), torch.zeros(0))
        assert torch.equal(buf, torch.tensor([1.0, 2.0, 3.0]))
        _ordered_index_add_(buf, torch.tensor([2, 0, 2]), torch.tensor([0.5, 0.25, 0.5]))
        assert torch.equal(buf, torch.tensor([1.25, 2.0, 4.0]))

    def test_it_keeps_the_gradient(self):
        from optiland.nonsequential.detectors.base import _ordered_index_add_

        src = torch.full((4096,), 1.0e-3, dtype=torch.float32, requires_grad=True)
        idx = torch.arange(4096, dtype=torch.int64) % 3
        buf = torch.zeros(3, dtype=torch.float32)
        _ordered_index_add_(buf, idx, src)
        (buf * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
        assert torch.equal(src.grad, (idx + 1).to(torch.float32))

    def test_a_float32_buffer_is_accumulated_in_order_and_a_float64_one_as_before(self):
        from optiland.nonsequential.detectors.base import _accumulate_into, _ordered_index_add_

        idx, src = _skewed_contributions(k=5000, size=64)
        via = torch.zeros(64, dtype=torch.float32)
        _accumulate_into(via, idx.numpy(), src)
        direct = torch.zeros(64, dtype=torch.float32)
        _ordered_index_add_(direct, idx, src)
        assert torch.equal(via, direct)
        wide = torch.zeros(64, dtype=torch.float64)
        _accumulate_into(wide, idx.numpy(), src)
        assert torch.equal(wide, torch.zeros(64, dtype=torch.float64).index_add_(0, idx, src.double()))
