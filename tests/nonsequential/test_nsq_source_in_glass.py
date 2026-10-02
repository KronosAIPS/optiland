"""A source placed inside a glass on every device the suite reaches (the research repository's issue 70).

The point, extended and collimated sources read their medium's index with
``np.asarray(medium.n(wavelengths), dtype=float)``. On CUDA the material memo
returns a device tensor, which NumPy cannot read, so a source inside a glass
crashed there (r2_04's reciprocity window on an A100, 2026-09-25). The index is
now read with ``to_numpy`` (detach, cpu, numpy) and cast as before. On the CPU a
tensor that carries a gradient is refused by ``np.asarray`` the same way, which
stands in for the device tensor where no CUDA device is present.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential.materials import NSQMaterial
from optiland.nonsequential.rng import NSQRng
from optiland.nonsequential.sources.base import Spectrum, medium_index_on_host
from optiland.nonsequential.sources.collimated import CollimatedSource
from optiland.nonsequential.sources.extended import ExtendedSource
from optiland.nonsequential.sources.point import PointSource

torch = pytest.importorskip("torch")

N_GLASS = 1.5168

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _AttachedIndex:
    """A medium whose index is a tensor carrying a gradient (np.asarray refuses it)."""

    def __init__(self, n):
        self.value = torch.tensor(n, dtype=torch.float64, requires_grad=True)

    def n(self, wavelengths):
        return self.value * torch.ones(len(wavelengths), dtype=torch.float64)

    def k(self, wavelengths):
        return torch.zeros(len(wavelengths), dtype=torch.float64)


def test_an_index_numpy_cannot_read_is_read_on_the_host():
    medium = _AttachedIndex(N_GLASS)
    wl = np.full(8, 0.55)
    with pytest.raises(RuntimeError):
        np.asarray(medium.n(wl), dtype=float)  # the earlier read
    n, k = medium_index_on_host(medium, wl, 8)
    assert n.dtype == np.float64 and k.dtype == np.float64
    np.testing.assert_array_equal(n, np.full(8, N_GLASS))
    np.testing.assert_array_equal(k, np.zeros(8))


def test_a_scalar_index_is_broadcast():
    class Scalar:
        def n(self, wl):
            return 1.33

        def k(self, wl):
            return 0.0

    n, k = medium_index_on_host(Scalar(), np.full(5, 0.55), 5)
    np.testing.assert_array_equal(n, np.full(5, 1.33))
    np.testing.assert_array_equal(k, np.zeros(5))


def _sources(medium):
    spec = Spectrum.monochromatic(0.5876)
    cs = CoordinateSystem(z=1.0)
    return {
        "point": PointSource(cs, spec, medium=medium),
        "extended": ExtendedSource(cs, spec, width=2.0, height=2.0, medium=medium),
        "collimated": CollimatedSource(cs, spec, aperture_radius=1.0, medium=medium),
    }


@pytest.mark.parametrize("kind", ["point", "extended", "collimated"])
@pytest.mark.parametrize("device", _DEVICES)
def test_each_source_inside_a_glass_on_torch(kind, device):
    be.set_backend("torch")
    be.set_device(device)
    be.set_precision("float64")
    try:
        glass = NSQMaterial(optiland_material=IdealMaterial(n=N_GLASS, k=0.0))
        source = _sources(glass)[kind]
        rays = source.generate(np.arange(64), NSQRng(3))
        n = be.to_numpy(rays.n_current)
        k = be.to_numpy(rays.k_current)
    finally:
        be.set_device("cpu")
        be.set_backend("numpy")
    np.testing.assert_array_equal(n, np.full(64, N_GLASS))
    np.testing.assert_array_equal(k, np.zeros(64))


@pytest.mark.parametrize("kind", ["point", "extended", "collimated"])
def test_numpy_and_torch_cpu_give_the_same_bundle_index(kind):
    out = {}
    for backend in ("numpy", "torch"):
        be.set_backend(backend)
        if backend == "torch":
            be.set_device("cpu")
            be.set_precision("float64")
        try:
            glass = NSQMaterial(optiland_material=IdealMaterial(n=N_GLASS, k=0.0))
            rays = _sources(glass)[kind].generate(np.arange(16), NSQRng(3))
            out[backend] = be.to_numpy(rays.n_current)
        finally:
            be.set_backend("numpy")
    assert np.array_equal(out["numpy"].view(np.int64), out["torch"].view(np.int64))
    assert math.isclose(float(out["numpy"][0]), N_GLASS)
