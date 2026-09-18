"""Irradiance detector for Non-Sequential Raytracing.

Accumulates a 2D flux map on a planar rectangular surface with
differentiable splatting support.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential._utils import (
    as_float,
    as_param,
    clamp_int,
    floor_to_int,
    int_to_float_like,
)
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential._tally import masked_count
from optiland.nonsequential.detectors.base import (
    BaseDetector,
    _accumulate_into,
    _new_flat_accumulator,
)
from optiland.nonsequential.results.irradiance_map import IrradianceMap

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class IrradianceDetector(BaseDetector):
    """2D irradiance map detector on a planar rectangular surface.

    Records flux in a pixel grid with differentiable splatting.

    Attributes:
        cs: Coordinate system.
        width: Detector width [mm].
        height: Detector height [mm].
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
        splat: Splatting mode — 'bilinear', 'gaussian', or 'hard'.
        splat_sigma: Gaussian splat sigma in pixels (used when splat='gaussian').
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        width: float,
        height: float,
        num_pixels_x: int,
        num_pixels_y: int,
        splat: Literal["bilinear", "gaussian", "hard"] = "bilinear",
        splat_sigma: float = 0.5,
        name: str = "",
        absorb: bool = True,
        side: str = "both",
    ) -> None:
        """Initialize IrradianceDetector.

        Args:
            cs: Coordinate system for detector position/orientation.
            width: Detector width [mm].
            height: Detector height [mm].
            num_pixels_x: Number of pixels along x.
            num_pixels_y: Number of pixels along y.
            splat: Splatting mode. 'bilinear' (differentiable, default),
                'hard' (forward-only histogram), or 'gaussian' (a true
                Gaussian kernel of width ``splat_sigma``, truncated and
                renormalised -- see :meth:`_record_gaussian`).
            splat_sigma: Gaussian sigma in pixels. Only used when
                ``splat='gaussian'``.
            name: Optional label.
            absorb: Whether a hit terminates the ray (default True).
            side: Which side of the plane is live -- ``"both"`` (default),
                ``"front"`` (only rays arriving on the side the placement's
                normal, local +z, points toward), or ``"back"``. See
                :meth:`BaseDetector.intersect`.
        """
        geometry = FinitePlaneGeometry(width=width, height=height)
        super().__init__(cs, geometry, name=name, absorb=absorb, side=side)
        self.width = as_param(width)
        self.height = as_param(height)
        self.num_pixels_x = int(num_pixels_x)
        self.num_pixels_y = int(num_pixels_y)
        self.splat = splat
        self.splat_sigma = as_float(splat_sigma)

        # Flat accumulation buffer: shape (ny * nx,). Always float64 (the
        # accumulation dtype A), whatever the working/traversal dtype T is
        # (docs/theory/08_precision.md R-08-1) -- and mutated in place by
        # every record() call (see _accumulate_into in detectors/base.py),
        # so a bounce never reallocates the pixel buffer.
        self._data = _new_flat_accumulator(num_pixels_y * num_pixels_x)
        self._num_rays_hit: int = 0

        # Pixel bin edges (NumPy, used for index arithmetic -- always detached)
        w_f = as_float(width)
        h_f = as_float(height)
        self._x_edges = np.linspace(-w_f / 2.0, w_f / 2.0, num_pixels_x + 1)
        self._y_edges = np.linspace(-h_f / 2.0, h_f / 2.0, num_pixels_y + 1)

    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Accumulate flux from hit rays into the pixel grid.

        Computes hit positions in local detector frame and accumulates
        flux using the configured splatting mode.

        Args:
            rays: Current ray bundle (positions not yet advanced to hit
                point).
            t: Hit distances [mm], shape (N,).
            hit_mask: Boolean mask of hitting rays, shape (N,).
        """
        t_arr, R_arr = self.frame()

        # Advance hit rays to intersection point in backend-attached arrays
        t_hit_be = be.where(hit_mask, t, be.zeros_like(t))
        hx_g = rays.x + t_hit_be * rays.L
        hy_g = rays.y + t_hit_be * rays.M
        hz_g = rays.z + t_hit_be * rays.N

        pos_g = be.stack([hx_g, hy_g, hz_g], axis=1)
        pos_l = (pos_g - t_arr) @ R_arr
        # A ray that did not hit this detector contributes nothing, and its
        # landing position is meaningless -- a dead ray's origin can be
        # infinite (it was pushed out of the scene when it escaped), and
        # inf * 0 is NaN, which no later multiplication by a zero flux can
        # undo. Mask the coordinate itself rather than relying on the zero
        # weight (docs/theory/08_precision.md R-08-8): the splat then reads
        # a finite 0 for every non-hit ray and adds exactly zero.
        hx_l = be.where(hit_mask, pos_l[:, 0], be.zeros_like(pos_l[:, 0]))
        hy_l = be.where(hit_mask, pos_l[:, 1], be.zeros_like(pos_l[:, 1]))

        # Zero out non-hit ray contributions while keeping graph attached
        flux_masked = be.where(hit_mask, rays.flux, be.zeros_like(rays.flux))

        nx = self.num_pixels_x
        ny = self.num_pixels_y
        dx = self.width / nx
        dy = self.height / ny

        # Every splat is index arithmetic in the active backend and one
        # in-place scatter-add per touched pixel. A non-hit ray carries zero
        # flux through all three and adds exactly zero, so none of them
        # needs an early exit on an empty mask -- which is what lets a
        # device backend call record() for every detector every bounce
        # without asking whether any ray hit it.
        if self.splat == "bilinear":
            self._record_bilinear(hx_l, hy_l, flux_masked, nx, ny, dx, dy)
        elif self.splat == "gaussian":
            self._record_gaussian(hx_l, hy_l, flux_masked, nx, ny, dx, dy)
        else:
            self._record_hard(hx_l, hy_l, flux_masked, nx, ny)

        self._num_rays_hit = self._num_rays_hit + masked_count(hit_mask)

    def _record_hard(
        self,
        hx_l,
        hy_l,
        flux_masked,
        nx: int,
        ny: int,
    ) -> None:
        """Hard-bin accumulation: each ray's whole flux into one pixel.

        The bin is found by ``searchsorted`` against the stored edges rather
        than by a floor of the continuous pixel coordinate. The two differ
        at a pixel boundary -- an edge is representable, and which side of
        it the division lands on is a rounding question -- so keeping
        ``searchsorted`` keeps the bin assignment exactly what it was, with
        the edges uploaded to the backend instead of the coordinates being
        brought down to them.

        Args:
            hx_l: Local x coordinates, be-array shape (N,).
            hy_l: Local y coordinates, be-array shape (N,).
            flux_masked: Per-ray flux (non-hit rays zeroed), be-array.
            nx: Number of pixels along x.
            ny: Number of pixels along y.
        """
        x_edges = self.table("x_edges", self._x_edges)
        y_edges = self.table("y_edges", self._y_edges)
        ix = clamp_int(be.searchsorted(x_edges, hx_l, side="right") - 1, 0, nx - 1)
        iy = clamp_int(be.searchsorted(y_edges, hy_l, side="right") - 1, 0, ny - 1)
        _accumulate_into(self._data, iy * nx + ix, flux_masked)

    def _record_bilinear(
        self,
        hx_l,
        hy_l,
        flux_masked,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
    ) -> None:
        """Bilinear splat — differentiable w.r.t. landing position and flux.

        Distributes each ray's flux to the four surrounding pixel centres
        with bilinear weights. The index arithmetic is detached (an index
        carries no gradient) but stays in the active backend, on the same
        device as the ray state and the pixel buffer; the flux contribution
        (flux * weight) carries gradients.

        Args:
            hx_l: Local x coordinates, be-array shape (N,).
            hy_l: Local y coordinates, be-array shape (N,).
            flux_masked: Per-ray flux (non-hit rays zeroed), be-array.
            nx: Number of pixels along x.
            ny: Number of pixels along y.
            dx: Pixel width [mm].
            dy: Pixel height [mm].
        """
        # Continuous pixel coordinate — centre of pixel ix is at 0.0 when
        # ix == 0, i.e.  px = (hx_l + width/2) / dx - 0.5
        px = (hx_l + self.width / 2.0) / dx - 0.5
        py = (hy_l + self.height / 2.0) / dy - 0.5

        # Base pixel index (detached — index must not carry gradient)
        ix0 = floor_to_int(px)
        iy0 = floor_to_int(py)

        # Fractional weights (attached to the graph)
        wx1 = px - int_to_float_like(ix0, px)  # fraction toward ix+1
        wy1 = py - int_to_float_like(iy0, py)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        # Distribute flux to all four neighbour pixels.
        for dix, diy, wx, wy in (
            (0, 0, wx0, wy0),
            (1, 0, wx1, wy0),
            (0, 1, wx0, wy1),
            (1, 1, wx1, wy1),
        ):
            ix = clamp_int(ix0 + dix, 0, nx - 1)
            iy = clamp_int(iy0 + diy, 0, ny - 1)
            flat = iy * nx + ix

            contrib = flux_masked * wx * wy  # attached
            _accumulate_into(self._data, flat, contrib)

    def _record_gaussian(
        self,
        hx_l,
        hy_l,
        flux_masked,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
    ) -> None:
        """Gaussian splat — differentiable, energy-conserving.

        Distributes each ray's flux over a ``(2*radius+1) x (2*radius+1)``
        neighbourhood using a separable Gaussian kernel of width
        ``self.splat_sigma`` pixels, truncated at ``radius = ceil(3 *
        splat_sigma)`` pixels. The truncated kernel is renormalised per ray
        (weights sum to 1 over exactly the pixels actually touched), so
        truncating the tail never loses energy -- an untruncated Gaussian
        would only asymptotically conserve flux, and a non-renormalised
        truncation would be a new flux-truncation bias of exactly the kind
        D-9 exists to avoid.

        Args:
            hx_l: Local x coordinates, be-array shape (N,).
            hy_l: Local y coordinates, be-array shape (N,).
            flux_masked: Per-ray flux (non-hit rays zeroed), be-array.
            nx: Number of pixels along x.
            ny: Number of pixels along y.
            dx: Pixel width [mm].
            dy: Pixel height [mm].
        """
        sigma = self.splat_sigma
        if sigma <= 0.0:
            self._record_hard(hx_l, hy_l, flux_masked, nx, ny)
            return

        radius = max(1, int(np.ceil(3.0 * sigma)))

        px = (hx_l + self.width / 2.0) / dx - 0.5
        py = (hy_l + self.height / 2.0) / dy - 0.5
        # Detached (an index carries no gradient) but not copied to the
        # host: the kernel's own offsets are integers beside the ray state.
        ix0 = floor_to_int(px)
        iy0 = floor_to_int(py)

        offsets = range(-radius, radius + 1)
        gx: dict[int, object] = {}
        gy: dict[int, object] = {}
        sx = be.zeros_like(px)
        sy = be.zeros_like(py)
        for d in offsets:
            ddx = int_to_float_like(ix0 + d, px) - px
            wx = be.exp(-0.5 * (ddx / sigma) ** 2)
            gx[d] = wx
            sx = sx + wx

            ddy = int_to_float_like(iy0 + d, py) - py
            wy = be.exp(-0.5 * (ddy / sigma) ** 2)
            gy[d] = wy
            sy = sy + wy

        norm = sx * sy  # separable kernel: total weight = Sx * Sy
        for dix in offsets:
            ix = clamp_int(ix0 + dix, 0, nx - 1)
            for diy in offsets:
                iy = clamp_int(iy0 + diy, 0, ny - 1)
                weight = (gx[dix] * gy[diy]) / norm
                contrib = flux_masked * weight
                _accumulate_into(self._data, iy * nx + ix, contrib)

    def get_result(self) -> IrradianceMap:
        """Return the accumulated irradiance map.

        Returns:
            IrradianceMap with irradiance [W/mm^2] computed from stored flux.
            The ``data`` attribute of the returned map is the attached flat
            flux buffer that supports gradient computation.
        """
        nx = self.num_pixels_x
        ny = self.num_pixels_y
        pixel_area = (self.width / nx) * (self.height / ny)
        data_2d = self._data.reshape(ny, nx) / pixel_area

        x_centres = 0.5 * (self._x_edges[:-1] + self._x_edges[1:])
        y_centres = 0.5 * (self._y_edges[:-1] + self._y_edges[1:])

        return IrradianceMap(
            data=self._data,
            irradiance=to_numpy(data_2d),
            x_coords=x_centres,
            y_coords=y_centres,
            # Attached: be.sum keeps this on the autograd graph, so
            # `result.detectors["D1"].total_flux.backward()` carries a
            # gradient. Use `.total_flux_float` for printing/formatting.
            total_flux=be.sum(self._data),
            num_rays_hit=int(to_numpy(self._num_rays_hit)),
        )

    def reset(self) -> None:
        """Clear accumulated data.

        Re-initialises the internal buffer to a fresh float64 accumulator,
        disconnecting it from the previous trace's computation graph.
        """
        self._data = _new_flat_accumulator(self.num_pixels_y * self.num_pixels_x)
        self._num_rays_hit = 0
        self.invalidate_frame()
