"""A spectral detector on a device: built, traced, and read like its CPU twin.

The scene builder makes a spectral detector's bin edges with ``be.linspace``,
which on the torch backend is a tensor on the active device; the detector
keeps its edges as a host float64 array and read them with ``np.asarray``,
which NumPy cannot do for a CUDA or an Apple-GPU tensor. Every scene with a
spectral detector therefore failed to build on a device (the catalogue's
r1_31 on CUDA among them), while the CPU never saw it.

Checked here: the edges given as a NumPy array, a CPU tensor or a device
tensor land as the same host float64 values; a scene with a spectral tap
builds and traces on the Apple GPU and on CUDA (each skipped where the
device is absent) with the same edges, hit count and per-bin flux as the same
trace on the CPU at the same precision.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    NSQScene,
    SpectralDetectorConfig,
    Spectrum,
)
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.detectors.spectral import SpectralDetector


def _available(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return bool(getattr(torch.backends, "mps", None)) and (
            torch.backends.mps.is_available()
        )
    return True


@pytest.fixture
def torch_state():
    """Torch on the CPU for the test, and the backend put back afterwards."""
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


def _scene() -> NSQScene:
    """Three spectral lines through a transmissive tap onto an absorbing one."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum(
                wavelengths=np.array([0.45, 0.55, 0.65]),
                weights=np.array([1.0, 2.0, 1.0]),
            ),
            total_flux=1.0,
            aperture_radius=5.0,
        ),
    )
    scene.add_detector(
        "TAP",
        CoordinateSystem(z=40),
        SpectralDetectorConfig(
            width=20,
            height=20,
            num_pixels_x=8,
            num_pixels_y=8,
            wl_min=0.4,
            wl_max=0.7,
            num_bins=3,
            splat="bilinear",
            absorb=False,
        ),
    )
    scene.add_detector(
        "END",
        CoordinateSystem(z=80),
        SpectralDetectorConfig(
            width=20,
            height=20,
            num_pixels_x=1,
            num_pixels_y=1,
            wl_min=0.4,
            wl_max=0.7,
            num_bins=6,
            splat="hard",
        ),
    )
    return scene


def _read(result) -> dict:
    out = {}
    for name, det in result.detectors.items():
        irr = np.asarray(to_numpy(det.irradiance), dtype=np.float64)
        out[name] = {
            "per_bin": irr.reshape(-1, irr.shape[-1]).sum(axis=0),
            "total": float(det.total_flux),
            "hits": int(det.num_rays_hit),
            "wavelengths": np.asarray(det.wavelengths, dtype=np.float64),
        }
    return out


def test_the_edges_land_on_the_host_as_the_same_values(torch_state):
    """NumPy edges, CPU-tensor edges and backend-built edges agree bit for bit."""
    cs = CoordinateSystem()
    for precision in ("float64", "float32"):
        be.set_precision(precision)
        built = be.linspace(0.4, 0.7, 4)
        reference = np.asarray(built.numpy(), dtype=np.float64)
        for edges in (built, built.numpy(), built.clone()):
            det = SpectralDetector(cs, 10.0, 10.0, 2, 2, edges)
            assert isinstance(det.wavelength_bins, np.ndarray)
            assert det.wavelength_bins.dtype == np.float64
            np.testing.assert_array_equal(det.wavelength_bins, reference)
            assert det._n_lambda == 3


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_a_spectral_detector_builds_and_traces_on_a_device(torch_state, device):
    """The same scene on the device and on the CPU: same edges, hits and flux."""
    if not _available(device):
        pytest.skip(f"no {device} device")
    be.set_precision("float32")  # the Apple GPU has no float64
    reference = _read(
        _scene().trace(num_rays=4096, seed=3, max_depth=8, backend=TorchBackend(seed=3))
    )
    be.set_device(device)
    try:
        scene = _scene()
        edges = scene.detectors[0].wavelength_bins
        assert isinstance(edges, np.ndarray) and edges.dtype == np.float64
        on_device = _read(
            scene.trace(
                num_rays=4096, seed=3, max_depth=8, backend=TorchBackend(seed=3)
            )
        )
    finally:
        be.set_device("cpu")
    for name, ref in reference.items():
        got = on_device[name]
        assert got["hits"] == ref["hits"] > 0, name
        np.testing.assert_array_equal(got["wavelengths"], ref["wavelengths"])
        # The same keyed draws land in the same bins; the float32 sums may
        # round differently on the device, by a few units in the last place.
        np.testing.assert_allclose(got["per_bin"], ref["per_bin"], rtol=1e-5, atol=0)
        assert got["total"] == pytest.approx(ref["total"], rel=1e-5)
