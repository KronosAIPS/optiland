"""Tests for the detector accumulation-buffer fix.

Three things are exercised, matching the fix's own requirements:

1. The splatting detectors (irradiance, far field, spectral) accumulate
   *in place*: the buffer's identity (and, on Torch, its storage) does not
   change across ``record()`` calls, so a bounce never reallocates the
   pixel buffer.
2. The buffer is always float64 (the accumulation dtype), whatever the
   working/traversal dtype is -- checked here on the Torch backend at
   float32 precision.
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
