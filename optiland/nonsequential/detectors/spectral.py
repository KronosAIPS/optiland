"""Spectral detector for Non-Sequential Raytracing.

Accumulates per-wavelength irradiance on a planar surface.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

import optiland.backend as be
from optiland.nonsequential._tally import masked_count
from optiland.nonsequential._utils import (
    as_detached_param,
    clamp_int,
    floor_to_int,
    int_to_float_like,
    to_numpy,
)
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.detectors.base import (
    BaseDetector,
    _accumulate_into,
    _new_flat_accumulator,
)
from optiland.nonsequential.results.spectral_result import SpectralResult

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.ray_bundle import NSQRayBundle

# Longest wavelength Optiland's material catalogs cover is well under this, so
# any bin edge above it is a nanometre value that slipped through as µm.
_MAX_PLAUSIBLE_WAVELENGTH_UM = 100.0


class SpectralDetector(BaseDetector):
    """Per-wavelength irradiance detector on a planar rectangular surface.

    Records flux in a 3D (x, y, wl) grid.

    Attributes:
        cs: Coordinate system.
        width: Detector width [mm].
        height: Detector height [mm].
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
        wavelength_bins: Wavelength bin edges [µm].
        splat: Spatial splatting mode -- 'bilinear', 'gaussian', or 'hard'.
            Splatting is spatial (x, y) only; the wavelength bin is always
            hard-assigned.
        splat_sigma: Gaussian splat sigma in pixels (used when
            ``splat='gaussian'``).
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        width: float,
        height: float,
        num_pixels_x: int,
        num_pixels_y: int,
        wavelength_bins: np.ndarray,
        splat: Literal["bilinear", "gaussian", "hard"] = "bilinear",
        splat_sigma: float = 0.5,
        name: str = "",
        absorb: bool = True,
    ) -> None:
        """Initialize SpectralDetector.

        Args:
            cs: Coordinate system for detector position/orientation.
            width: Detector width [mm].
            height: Detector height [mm].
            num_pixels_x: Number of pixels along x.
            num_pixels_y: Number of pixels along y.
            wavelength_bins: Wavelength bin edges [µm], shape (n_lambda + 1,).
            splat: Spatial splatting mode. Accepts 'bilinear', 'gaussian',
                or 'hard'.
            splat_sigma: Gaussian splat sigma in pixels. Only used when
                ``splat='gaussian'``.
            name: Optional label.
            absorb: Whether a hit terminates the ray (default True).

        Raises:
            ValueError: If ``wavelength_bins`` are not plausibly in µm. Bin
                edges are compared against ``rays.wavelength``, which is in
                µm; nanometre edges would silently clip every ray into the
                first bin.
        """
        geometry = FinitePlaneGeometry(width=width, height=height)
        super().__init__(cs, geometry, name=name, absorb=absorb)
        _reason = "it accumulates into a NumPy histogram"
        self.width = as_detached_param(width, "width", "SpectralDetector", _reason)
        self.height = as_detached_param(height, "height", "SpectralDetector", _reason)
        self.num_pixels_x = int(num_pixels_x)
        self.num_pixels_y = int(num_pixels_y)
        self.splat = splat
        self.splat_sigma = float(splat_sigma)
        self.wavelength_bins = np.asarray(wavelength_bins, dtype=np.float64)
        if self.wavelength_bins.min() > _MAX_PLAUSIBLE_WAVELENGTH_UM:
            raise ValueError(
                f"wavelength_bins must be in µm, but the smallest bin edge is "
                f"{self.wavelength_bins.min():g}. Values this large look like "
                f"nanometres - divide by 1000 (e.g. 550 nm -> 0.55)."
            )
        n_lambda = len(wavelength_bins) - 1
        self._n_lambda = n_lambda

        # Flat accumulation buffer: shape (ny * nx * n_lambda,). Always
        # float64, on the active backend and device, mutated in place by
        # every record() call -- see detectors/base.py. Flat index for
        # (iy, ix, iwl) is (iy * nx + ix) * n_lambda + iwl.
        self._flux_map = _new_flat_accumulator(num_pixels_y * num_pixels_x * n_lambda)
        self._num_rays_hit = 0

        self._x_edges = np.linspace(
            -self.width / 2.0, self.width / 2.0, num_pixels_x + 1
        )
        self._y_edges = np.linspace(
            -self.height / 2.0, self.height / 2.0, num_pixels_y + 1
        )

    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Accumulate per-wavelength flux from hit rays.

        Written in the active backend's own operations throughout: the
        placement comes from the per-trace cache
        (:meth:`~optiland.nonsequential.detectors.base.BaseDetector.frame`),
        the pixel and wavelength bin edges from tables uploaded once
        (:meth:`~optiland.nonsequential.detectors.base.BaseDetector.table`),
        and a ray that did not hit is masked to a zero landing position and
        zero flux rather than gathered out -- so the method never asks how
        many rays hit and costs no device-to-host transfer.

        Args:
            rays: Current ray bundle.
            t: Hit distances [mm], shape (N,).
            hit_mask: Boolean mask of rays hitting this detector, shape (N,).
        """
        translation, rot = self.frame()

        t_hit = be.where(hit_mask, t, be.zeros_like(t))
        hx_g = rays.x + t_hit * rays.L
        hy_g = rays.y + t_hit * rays.M
        hz_g = rays.z + t_hit * rays.N

        pos_g = be.stack([hx_g, hy_g, hz_g], axis=1)
        pos_l = (pos_g - translation) @ rot
        # Mask the coordinate, not just the weight: a dead ray's origin can
        # be infinite and inf * 0 is NaN, which no later multiplication by a
        # zero flux undoes (docs/theory/08_precision.md R-08-8).
        hx_l = be.where(hit_mask, pos_l[:, 0], be.zeros_like(pos_l[:, 0]))
        hy_l = be.where(hit_mask, pos_l[:, 1], be.zeros_like(pos_l[:, 1]))
        flux_masked = be.where(hit_mask, rays.flux, be.zeros_like(rays.flux))

        # The wavelength bin is always hard-assigned; splatting is spatial.
        wl_edges = self.table("wavelength_bins", self.wavelength_bins)
        iwl = clamp_int(
            be.searchsorted(wl_edges, rays.wavelength, side="right") - 1,
            0,
            self._n_lambda - 1,
        )

        nx, ny = self.num_pixels_x, self.num_pixels_y
        dx = self.width / nx
        dy = self.height / ny
        if self.splat == "hard":
            self._record_hard(hx_l, hy_l, flux_masked, iwl, nx, ny)
        elif self.splat == "gaussian":
            self._record_gaussian(hx_l, hy_l, flux_masked, iwl, nx, ny, dx, dy)
        else:
            self._record_bilinear(hx_l, hy_l, flux_masked, iwl, nx, ny, dx, dy)
        self._num_rays_hit = self._num_rays_hit + masked_count(hit_mask)

    def _flat_index(self, iy, ix, iwl):
        """Flatten (iy, ix, iwl) into the flux-map buffer's flat index."""
        return (iy * self.num_pixels_x + ix) * self._n_lambda + iwl

    def _record_hard(self, hx_l, hy_l, flux_hit, iwl, nx, ny) -> None:
        """Hard-bin spatial accumulation (see ``IrradianceDetector._record_hard``)."""
        x_edges = self.table("x_edges", self._x_edges)
        y_edges = self.table("y_edges", self._y_edges)
        ix = clamp_int(be.searchsorted(x_edges, hx_l, side="right") - 1, 0, nx - 1)
        iy = clamp_int(be.searchsorted(y_edges, hy_l, side="right") - 1, 0, ny - 1)
        _accumulate_into(self._flux_map, self._flat_index(iy, ix, iwl), flux_hit)

    def _record_bilinear(self, hx_l, hy_l, flux_hit, iwl, nx, ny, dx, dy) -> None:
        """Bilinear spatial splat (see ``IrradianceDetector._record_bilinear``)."""
        px = (hx_l + self.width / 2.0) / dx - 0.5
        py = (hy_l + self.height / 2.0) / dy - 0.5
        ix0 = floor_to_int(px)
        iy0 = floor_to_int(py)
        wx1 = px - int_to_float_like(ix0, px)
        wy1 = py - int_to_float_like(iy0, py)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        for dix, diy, wx, wy in (
            (0, 0, wx0, wy0),
            (1, 0, wx1, wy0),
            (0, 1, wx0, wy1),
            (1, 1, wx1, wy1),
        ):
            ix = clamp_int(ix0 + dix, 0, nx - 1)
            iy = clamp_int(iy0 + diy, 0, ny - 1)
            _accumulate_into(
                self._flux_map, self._flat_index(iy, ix, iwl), flux_hit * wx * wy
            )

    def _record_gaussian(self, hx_l, hy_l, flux_hit, iwl, nx, ny, dx, dy) -> None:
        """Gaussian spatial splat, truncated and renormalised per ray so
        truncation never loses energy (see
        ``IrradianceDetector._record_gaussian``)."""
        sigma = self.splat_sigma
        if sigma <= 0.0:
            self._record_hard(hx_l, hy_l, flux_hit, iwl, nx, ny)
            return

        radius = max(1, int(np.ceil(3.0 * sigma)))
        px = (hx_l + self.width / 2.0) / dx - 0.5
        py = (hy_l + self.height / 2.0) / dy - 0.5
        ix0 = floor_to_int(px)
        iy0 = floor_to_int(py)

        offsets = range(-radius, radius + 1)
        gx = {
            d: be.exp(-0.5 * ((int_to_float_like(ix0 + d, px) - px) / sigma) ** 2)
            for d in offsets
        }
        gy = {
            d: be.exp(-0.5 * ((int_to_float_like(iy0 + d, py) - py) / sigma) ** 2)
            for d in offsets
        }
        sx = sum(gx.values())
        sy = sum(gy.values())
        norm = sx * sy

        for dix in offsets:
            ix = clamp_int(ix0 + dix, 0, nx - 1)
            for diy in offsets:
                iy = clamp_int(iy0 + diy, 0, ny - 1)
                weight = (gx[dix] * gy[diy]) / norm
                _accumulate_into(
                    self._flux_map, self._flat_index(iy, ix, iwl), flux_hit * weight
                )

    def get_result(self) -> SpectralResult:
        """Return accumulated spectral result.

        Returns:
            SpectralResult with irradiance [W/mm^2] per pixel per wavelength bin.
        """
        pixel_area = (self.width / self.num_pixels_x) * (
            self.height / self.num_pixels_y
        )
        flux_map_np = to_numpy(self._flux_map).reshape(
            self.num_pixels_y, self.num_pixels_x, self._n_lambda
        )
        irradiance = flux_map_np / pixel_area

        x_centres = 0.5 * (self._x_edges[:-1] + self._x_edges[1:])
        y_centres = 0.5 * (self._y_edges[:-1] + self._y_edges[1:])
        # wl_centres in µm
        wl_centres = 0.5 * (self.wavelength_bins[:-1] + self.wavelength_bins[1:])

        return SpectralResult(
            irradiance=irradiance.copy(),
            x_coords=x_centres,
            y_coords=y_centres,
            wavelengths=wl_centres,
            total_flux=float(flux_map_np.sum()),
            num_rays_hit=int(to_numpy(self._num_rays_hit)),
        )

    def reset(self) -> None:
        """Clear accumulated data."""
        self._flux_map = _new_flat_accumulator(
            self.num_pixels_y * self.num_pixels_x * self._n_lambda
        )
        self._num_rays_hit = 0
        self.invalidate_frame()
