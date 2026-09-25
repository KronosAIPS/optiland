"""Photometric and colorimetric detectors: CIE tristimulus tallies per ray.

Photometry and colorimetry are linear functionals of the spectral flux, so a
tracer that carries a wavelength per ray books them without a second trace by
weighting each ray at the tally (``docs/theory/13_sources_and_detectors.md``
section 13.6 of the research repository, R-01-6 and R-13-11): three extra
tallies of ``flux * xbar(lambda)``, ``flux * ybar(lambda)`` and
``flux * zbar(lambda)``. Weighting per ray is exact for any spectrum;
converting a finished spectral map bin by bin is exact only when the spectrum
inside every bin is known.

The colour-matching functions are the CIE 1931 2-degree observer at 1 nm from
380 to 780 nm (the engine's frozen table, :mod:`optiland.colorimetry.constants`),
linear between nodes and zero outside; ``ybar`` is the photopic V(lambda) and
``K_m = 683.002 lm/W``.

:class:`ColorimetricDetector` is an :class:`IrradianceDetector` and
:class:`ColorimetricFarFieldDetector` a :class:`FarFieldDetector` that book the
three tristimulus channels beside their radiometric map, by calling their
parent's own tally with each ray's flux weighted: the binning, splatting and
the radiometric result are the parent's, unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential._tally import Tally
from optiland.nonsequential._utils import clamp_int, floor_to_int, int_to_float_like, to_numpy
from optiland.nonsequential.detectors.base import _new_flat_accumulator
from optiland.nonsequential.detectors.far_field import FarFieldDetector
from optiland.nonsequential.detectors.irradiance import IrradianceDetector

if TYPE_CHECKING:
    from optiland.nonsequential.ray_bundle import NSQRayBundle

#: CIE 1931 2-degree observer: first node, node step, number of nodes [nm].
CMF_FIRST_NM = 380.0
CMF_STEP_NM = 1.0
CMF_NAME = "CIE 1931 2-degree, 1 nm, 380-780 nm, linear between nodes"


def cmf_table() -> np.ndarray:
    """The colour-matching functions, shape (3, 401): xbar, ybar, zbar."""
    from optiland.colorimetry.constants import (  # noqa: PLC0415
        CIE_1931_2DEG,
        WAVELENGTHS_STD,
    )

    wl = np.asarray(WAVELENGTHS_STD, dtype=np.float64)
    if wl[0] != CMF_FIRST_NM or not np.all(np.diff(wl) == CMF_STEP_NM):
        raise RuntimeError("the frozen CIE table is not the 1 nm, 380-780 nm grid")
    return np.asarray(CIE_1931_2DEG, dtype=np.float64).T.copy()


def cmf_at(wavelength_um) -> np.ndarray:
    """Host reference: xbar, ybar, zbar at ``wavelength_um``, shape (3, N)."""
    lam = np.asarray(wavelength_um, dtype=np.float64) * 1000.0
    grid = CMF_FIRST_NM + CMF_STEP_NM * np.arange(401)
    tab = cmf_table()
    return np.stack([np.interp(lam, grid, tab[c], left=0.0, right=0.0) for c in range(3)])


class _WeightedRays:
    """A view of a ray bundle whose ``flux`` is replaced by a weighted flux."""

    def __init__(self, rays, flux) -> None:
        self._rays = rays
        self.flux = flux

    def __getattr__(self, name):
        return getattr(self._rays, name)


class _TristimulusMixin:
    """Three tristimulus tallies booked through the parent class's own record."""

    #: Name of the parent's accumulation buffer this mixin swaps per channel.
    _BUFFER = ""
    #: Parent counters to preserve across the three weighted calls.
    _COUNTERS: tuple[str, ...] = ()

    def _init_channels(self, size: int) -> None:
        self._size = size
        self._channels = [_new_flat_accumulator(size) for _ in range(3)]
        self._cmf = cmf_table()

    def _weights(self, wavelength_um):
        """xbar, ybar, zbar at each ray's wavelength, in the active backend."""
        pos = (wavelength_um * 1000.0 - CMF_FIRST_NM) / CMF_STEP_NM
        last = self._cmf.shape[1] - 1
        inside = (pos >= 0.0) & (pos <= float(last))
        i0 = clamp_int(floor_to_int(pos), 0, last - 1)
        frac = pos - int_to_float_like(i0, pos)
        out = []
        for c, key in enumerate(("xbar", "ybar", "zbar")):
            tab = self.table(key, self._cmf[c])
            lo = tab[i0]
            hi = tab[i0 + 1]
            w = lo + (hi - lo) * frac
            out.append(be.where(inside, w, be.zeros_like(w)))
        return out

    def record(self, rays: NSQRayBundle, t, hit_mask) -> None:
        """Book the radiometric map, then the three weighted channels."""
        super().record(rays, t, hit_mask)
        saved_buffer = getattr(self, self._BUFFER)
        saved = {name: getattr(self, name) for name in self._COUNTERS}
        try:
            for c, w in enumerate(self._weights(rays.wavelength)):
                setattr(self, self._BUFFER, self._channels[c])
                for name in self._COUNTERS:
                    setattr(self, name, Tally() if isinstance(saved[name], Tally) else 0)
                super().record(_WeightedRays(rays, rays.flux * w), t, hit_mask)
                self._channels[c] = getattr(self, self._BUFFER)
        finally:
            setattr(self, self._BUFFER, saved_buffer)
            for name, value in saved.items():
                setattr(self, name, value)

    def reset(self) -> None:
        """Clear the radiometric map and the three channels."""
        super().reset()
        self._channels = [_new_flat_accumulator(self._size) for _ in range(3)]

    def _tristimulus(self) -> np.ndarray:
        return np.stack([np.asarray(to_numpy(ch), dtype=np.float64) for ch in self._channels])


class ColorimetricDetector(_TristimulusMixin, IrradianceDetector):
    """A plane of pixels booking irradiance and the CIE tristimulus flux."""

    _BUFFER = "_data"
    _COUNTERS = ("_num_rays_hit",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_channels(self.num_pixels_y * self.num_pixels_x)

    def get_result(self):
        """The radiometric map plus the tristimulus maps (see :class:`ColorimetricMap`)."""
        from optiland.nonsequential.results.colorimetric import (  # noqa: PLC0415
            ColorimetricMap,
        )

        radiometric = super().get_result()
        xyz = self._tristimulus().reshape(3, self.num_pixels_y, self.num_pixels_x)
        return ColorimetricMap(radiometric=radiometric, tristimulus=xyz,
                               pixel_area_mm2=float(self.width * self.height)
                               / (self.num_pixels_x * self.num_pixels_y),
                               cmf=CMF_NAME)


class ColorimetricFarFieldDetector(_TristimulusMixin, FarFieldDetector):
    """A far-field detector booking W/sr and the CIE tristimulus intensity."""

    _BUFFER = "_intensity"
    _COUNTERS = ("_num_rays_hit", "_total_flux")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_channels(self.num_bins_theta * self.num_bins_phi)

    def get_result(self):
        """The far-field pattern plus tristimulus intensity (see :class:`ColorimetricFarField`)."""
        from optiland.nonsequential.results.colorimetric import (  # noqa: PLC0415
            ColorimetricFarField,
        )

        radiometric = super().get_result()
        xyz = self._tristimulus().reshape(3, self.num_bins_theta, self.num_bins_phi)
        return ColorimetricFarField(radiometric=radiometric, tristimulus=xyz, cmf=CMF_NAME)
