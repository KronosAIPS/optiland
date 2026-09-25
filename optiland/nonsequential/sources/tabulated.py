"""A source whose angular intensity is given as a table.

The intensity is tabulated against the polar angle ``theta`` from the local
+z axis, and optionally against the azimuth ``phi`` measured from local +x
toward local +y. Between nodes it is linear in ``theta`` and, for a 2-D table,
bilinear in ``(theta, phi)``: the convention of the engine's tabulated BSDF,
and one under which a table that is exactly linear between its nodes carries no
interpolation error. Outside the polar range of the table the intensity is zero.

Every ray's direction is drawn exactly from the interpolated distribution
``I(theta, phi) sin(theta) dtheta dphi`` (every ray carries the same weight):

1. ``phi`` (2-D tables only): the azimuthal marginal is linear between rows,
   so a cell is chosen by the cumulative row integrals and the position in it
   by :func:`~optiland.nonsequential.sources.spectra.sample_linear`
   (slot ``SOURCE_U4``);
2. ``theta``: given ``phi`` the intensity is linear in ``theta`` in every
   cell, the cell integrals have the closed form of :func:`cell_integrals`,
   a cell is chosen by their cumulative sum and the angle inside it by
   inverting the cell's closed-form CDF with a bracketed Newton iteration of
   bounded length (slot ``SOURCE_U3``).

The emitter is a point (the default) or a rectangle or disc in the local x-y
plane, every point of which emits the same distribution (slots ``SOURCE_U1``,
``SOURCE_U2`` for the position).

The table is data: its values are detached (a gradient with respect to a table
value would need the implicit derivative of the inverse CDF). The source's
total flux carries a gradient, as for every other source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential._utils import is_tensor
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.rng import EventSlot
from optiland.nonsequential.sources.base import BaseNSQSource
from optiland.nonsequential.sources.spectra import sample_linear

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.rng import NSQRng

#: Bracketed-Newton iterations allowed per ray (a fixed bound, so the sampler
#: has no data-dependent loop length). Newton converges in five to eight from
#: the starting guess on every table the tests use; the rest is headroom for
#: cells where it falls back to bisection, each step of which halves the
#: bracket, so 64 iterations always reach the resolution of float64.
_NEWTON_MAX_ITER = 64

INTENSITY_UNITS = ("relative", "W/sr", "cd")


def _as_array(value) -> np.ndarray:
    """A float64 numpy copy of a table given as a list, an array or a tensor."""
    if is_tensor(value):
        return np.asarray(to_numpy(value), dtype=np.float64)
    return np.array(value, dtype=np.float64)


def cell_integrals(theta: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """``integral I(t) sin(t) dt`` over every cell of a table linear in ``theta``.

    With ``I = I0 + s (t - t0)`` on ``[t0, t1]`` and ``h = (t1 - t0) / 2``,
    ``m = (t0 + t1) / 2``:

        ``I0 (cos t0 - cos t1) + s (sin t1 - sin t0 - (t1 - t0) cos t1)``,

    with the two differences written as ``2 sin(m) sin(h)`` and
    ``2 cos(m) sin(h)`` so that small cells lose no digits to cancellation.

    Args:
        theta: Polar nodes [rad], strictly increasing, shape (n,).
        intensity: Intensity at the nodes, shape (..., n).

    Returns:
        Cell integrals, shape (..., n - 1).
    """
    t0, t1 = theta[:-1], theta[1:]
    d = t1 - t0
    h = 0.5 * d
    m = 0.5 * (t0 + t1)
    i0 = intensity[..., :-1]
    s = (intensity[..., 1:] - i0) / d
    dcos = 2.0 * np.sin(m) * np.sin(h)
    dsin = 2.0 * np.cos(m) * np.sin(h)
    return i0 * dcos + s * (dsin - d * np.cos(t1))


def _partial(t0, t, i0, s):
    """``integral_{t0}^{t} (i0 + s (x - t0)) sin(x) dx`` and its integrand at ``t``."""
    d = t - t0
    h = 0.5 * d
    m = 0.5 * (t0 + t)
    value = i0 * (2.0 * np.sin(m) * np.sin(h)) + s * (
        2.0 * np.cos(m) * np.sin(h) - d * np.cos(t)
    )
    return value, (i0 + s * d) * np.sin(t)


def invert_cell(t0, t1, i0, i1, target):
    """Solve ``integral_{t0}^{t} I sin = target`` for ``t`` in every row.

    ``I`` is linear from ``i0`` at ``t0`` to ``i1`` at ``t1``. The iteration
    keeps a bracket ``[lo, hi]`` and takes a Newton step when it stays inside,
    a bisection otherwise (the scheme of ``NewtonBisection`` in PBRT-v4,
    ``src/pbrt/util/math.h``, Apache-2.0, for arrays and with a fixed
    iteration bound).

    Args:
        t0, t1: Cell edges [rad], arrays of one shape.
        i0, i1: Intensity at the edges.
        target: The partial integral to reach, ``0 <= target <= cell integral``.

    Returns:
        ``t`` in ``[t0, t1]``.
    """
    s = (i1 - i0) / (t1 - t0)
    lo = np.array(t0, dtype=np.float64, copy=True)
    hi = np.array(t1, dtype=np.float64, copy=True)
    # Start where a constant intensity equal to the cell mean would put it:
    # cos t = cos t0 - target / mean.
    mean = 0.5 * (i0 + i1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.cos(t0) - np.where(mean > 0.0, target / mean, 0.0)
    t = np.clip(np.arccos(np.clip(c, -1.0, 1.0)), lo, hi)
    active = np.ones(t.shape, dtype=bool)
    for _ in range(_NEWTON_MAX_ITER):
        f, df = _partial(t0, t, i0, s)
        f = f - target
        lo = np.where(f < 0.0, t, lo)
        hi = np.where(f > 0.0, t, hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            step = np.where(df > 0.0, f / df, np.inf)
        t_new = t - step
        bisect = ~(t_new > lo) | ~(t_new < hi)
        t_new = np.where(bisect, 0.5 * (lo + hi), t_new)
        converged = np.abs(t_new - t) <= 4.0 * np.finfo(np.float64).eps * np.maximum(
            np.abs(t_new), 1.0
        )
        t = np.where(active, t_new, t)
        active = active & ~converged
        if not active.any():
            break
    # A zero target is the cell's first edge exactly (where the integrand
    # vanishes there the iteration would only creep toward it).
    return np.where(target > 0.0, t, t0)


class TabulatedSource(BaseNSQSource):
    """A point or area emitter with a tabulated angular intensity.

    Attributes:
        polar_angles_deg: Polar nodes [deg], increasing in [0, 180].
        azimuth_angles_deg: Azimuth nodes [deg] from 0 to 360, or ``None`` for a
            rotationally symmetric table.
        intensity: Intensity table, shape (n_theta,) or (n_phi, n_theta), in
            ``intensity_units``.
        intensity_units: ``"relative"``, ``"W/sr"`` or ``"cd"``.
        table_flux: The table's own integral ``integral I dOmega`` in its units
            (W for W/sr, lm for cd).
        width, height, aperture_radius: The emitting area (all ``None``: a
            point).
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        spectrum,
        total_flux,
        polar_angles_deg,
        intensity,
        azimuth_angles_deg=None,
        intensity_units: str = "relative",
        width: float | None = None,
        height: float | None = None,
        aperture_radius: float | None = None,
        medium=None,
    ) -> None:
        super().__init__(cs, spectrum, total_flux)
        for label, value in (("polar_angles_deg", polar_angles_deg),
                             ("intensity", intensity),
                             ("azimuth_angles_deg", azimuth_angles_deg)):
            if is_tensor(value) and value.requires_grad:
                raise NotImplementedError(
                    f"TabulatedSource.{label} cannot carry a gradient: the table "
                    "is data sampled by an inverse CDF, so the gradient would be "
                    "silently dropped. Pass plain numbers."
                )
        theta_deg = _as_array(polar_angles_deg)
        table = _as_array(intensity)
        if theta_deg.ndim != 1 or theta_deg.size < 2:
            raise ValueError("TabulatedSource: polar_angles_deg needs two or more nodes.")
        if not np.all(np.diff(theta_deg) > 0.0):
            raise ValueError("TabulatedSource: polar angles must be strictly increasing.")
        if theta_deg[0] < 0.0 or theta_deg[-1] > 180.0:
            raise ValueError("TabulatedSource: polar angles must lie in [0, 180] degrees.")
        if not np.all(np.isfinite(table)) or np.any(table < 0.0):
            raise ValueError("TabulatedSource: intensities must be finite and non-negative.")
        if intensity_units not in INTENSITY_UNITS:
            raise ValueError(
                f"TabulatedSource: intensity_units must be one of {INTENSITY_UNITS}."
            )
        if azimuth_angles_deg is None:
            if table.shape != theta_deg.shape:
                raise ValueError(
                    "TabulatedSource: a 1-D table needs one intensity per polar node."
                )
            phi_deg = None
        else:
            phi_deg = _as_array(azimuth_angles_deg)
            if phi_deg.ndim != 1 or phi_deg.size < 2:
                raise ValueError("TabulatedSource: azimuth_angles_deg needs two or more nodes.")
            if not np.all(np.diff(phi_deg) > 0.0):
                raise ValueError("TabulatedSource: azimuths must be strictly increasing.")
            if phi_deg[0] != 0.0 or phi_deg[-1] != 360.0:
                raise ValueError(
                    "TabulatedSource: azimuths must run from 0 to 360 degrees; expand "
                    "a symmetric table before building the source (the first and last "
                    "rows are the same plane)."
                )
            if table.shape != (phi_deg.size, theta_deg.size):
                raise ValueError(
                    "TabulatedSource: a 2-D table has one row per azimuth and one "
                    f"column per polar node, shape ({phi_deg.size}, {theta_deg.size}); "
                    f"got {table.shape}."
                )
            if not np.allclose(table[0], table[-1], rtol=1e-12, atol=0.0):
                raise ValueError(
                    "TabulatedSource: the rows at 0 and 360 degrees are one plane and "
                    "must be equal."
                )
        self.polar_angles_deg = theta_deg
        self.azimuth_angles_deg = phi_deg
        self.intensity = table
        self.intensity_units = intensity_units
        self.width = None if width is None else float(width)
        self.height = None if height is None else float(height)
        self.aperture_radius = None if aperture_radius is None else float(aperture_radius)
        if (self.width is None) != (self.height is None):
            raise ValueError("TabulatedSource: give both width and height, or neither.")
        if self.aperture_radius is not None and self.width is not None:
            raise ValueError("TabulatedSource: a rectangle or a disc, not both.")
        self.medium = medium

        self._theta = np.radians(theta_deg)
        if phi_deg is None:
            self._cells = cell_integrals(self._theta, table)  # (n_theta - 1,)
            self._cdf = np.cumsum(self._cells)
            flux = 2.0 * np.pi * float(self._cdf[-1])
        else:
            self._phi = np.radians(phi_deg)
            self._row_cells = cell_integrals(self._theta, table)  # (n_phi, n_theta-1)
            rows = self._row_cells.sum(axis=1)  # M_k
            self._rows = rows
            self._phi_cells = 0.5 * (rows[:-1] + rows[1:]) * np.diff(self._phi)
            self._phi_cdf = np.cumsum(self._phi_cells)
            flux = float(self._phi_cdf[-1])
        if not flux > 0.0:
            raise ValueError("TabulatedSource: the table integrates to zero flux.")
        self.table_flux = flux

    # -- the distribution ------------------------------------------------------

    def intensity_at(self, theta_deg, phi_deg=0.0) -> np.ndarray:
        """The interpolated intensity (table units) at the given angles."""
        th = np.asarray(theta_deg, dtype=np.float64)
        if self.azimuth_angles_deg is None:
            return np.interp(th, self.polar_angles_deg, self.intensity, left=0.0, right=0.0)
        ph = np.mod(np.asarray(phi_deg, dtype=np.float64), 360.0)
        k = np.clip(np.searchsorted(self.azimuth_angles_deg, ph, side="right") - 1, 0,
                    self.azimuth_angles_deg.size - 2)
        f = (ph - self.azimuth_angles_deg[k]) / np.diff(self.azimuth_angles_deg)[k]
        row0 = np.array([np.interp(t, self.polar_angles_deg, self.intensity[kk], left=0.0,
                                   right=0.0) for t, kk in zip(np.ravel(th * np.ones_like(ph)),
                                                               np.ravel(k), strict=False)])
        row1 = np.array([np.interp(t, self.polar_angles_deg, self.intensity[kk + 1],
                                   left=0.0, right=0.0)
                         for t, kk in zip(np.ravel(th * np.ones_like(ph)), np.ravel(k),
                                          strict=False)])
        return ((1.0 - np.ravel(f)) * row0 + np.ravel(f) * row1).reshape(np.shape(ph * th))

    def sample_directions(self, u_theta: np.ndarray, u_phi: np.ndarray | None):
        """Local directions from uniforms, exactly distributed as ``I sin``.

        Args:
            u_theta: Uniforms for the polar angle, shape (N,).
            u_phi: Uniforms for the azimuth (2-D tables), shape (N,); for a
                symmetric table the azimuth is ``2 pi u_phi``.

        Returns:
            ``(theta, phi)`` [rad], each shape (N,).
        """
        n = u_theta.shape[0]
        n_cells = self._theta.size - 1
        if self.azimuth_angles_deg is None:
            phi = 2.0 * np.pi * u_phi
            cum = np.broadcast_to(self._cdf, (n, n_cells))
            cells = np.broadcast_to(self._cells, (n, n_cells))
            i_rows = np.broadcast_to(self.intensity, (n, self._theta.size))
        else:
            target_phi = u_phi * self._phi_cdf[-1]
            k = np.clip(np.searchsorted(self._phi_cdf, target_phi, side="right"), 0,
                        self._phi_cells.size - 1)
            before = np.where(k > 0, self._phi_cdf[np.maximum(k - 1, 0)], 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(self._phi_cells[k] > 0.0,
                             (target_phi - before) / self._phi_cells[k], 0.0)
            r = np.clip(r, 0.0, np.nextafter(1.0, 0.0))
            f = sample_linear(r, self._rows[k], self._rows[k + 1])
            dphi = np.diff(self._phi)[k]
            phi = self._phi[k] + f * dphi
            w0 = (1.0 - f)[:, None]
            w1 = f[:, None]
            cells = w0 * self._row_cells[k] + w1 * self._row_cells[k + 1]
            cum = np.cumsum(cells, axis=1)
            i_rows = w0 * self.intensity[k] + w1 * self.intensity[k + 1]
        target = u_theta * cum[:, -1]
        j = np.clip((cum <= target[:, None]).sum(axis=1), 0, n_cells - 1)
        rows = np.arange(n)
        before = np.where(j > 0, cum[rows, np.maximum(j - 1, 0)], 0.0)
        partial = np.clip(target - before, 0.0, cells[rows, j])
        theta = invert_cell(
            self._theta[j], self._theta[j + 1], i_rows[rows, j], i_rows[rows, j + 1], partial
        )
        return theta, phi

    # -- generation --------------------------------------------------------------

    def generate(self, ray_id: np.ndarray, rng: NSQRng) -> NSQRayBundle:
        """Rays from the emitter with directions drawn from the table.

        Args:
            ray_id: Unique ray identifiers, shape (N,).
            rng: Keyed generator.

        Returns:
            The bundle, flux ``total_flux / N`` per ray.
        """
        num = len(ray_id)
        bounce0 = np.zeros(num, dtype=np.int32)
        translation, rot = _get_transform(self.cs)

        if self.aperture_radius is not None:
            u1 = to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1))
            u2 = to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U2))
            r = self.aperture_radius * np.sqrt(u1)
            lx, ly = r * np.cos(2.0 * np.pi * u2), r * np.sin(2.0 * np.pi * u2)
        elif self.width is not None:
            u1 = to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U1))
            u2 = to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U2))
            lx, ly = (u1 - 0.5) * self.width, (u2 - 0.5) * self.height
        else:
            lx = ly = np.zeros(num)

        u3 = np.asarray(to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U3)),
                        dtype=np.float64)
        u4 = np.asarray(to_numpy(rng.uniform(ray_id, bounce0, EventSlot.SOURCE_U4)),
                        dtype=np.float64)
        theta, phi = self.sample_directions(u3, u4)
        st = np.sin(theta)
        dirs_local = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
        pos_local = np.stack([lx, ly, np.zeros(num)], axis=1)
        pos = pos_local @ rot.T + translation
        dirs = dirs_local @ rot.T

        wavelengths = self.spectrum.sample(ray_id, bounce0, rng)
        flux_per_ray = self.total_flux / num
        medium = getattr(self, "medium", None)
        if medium is not None:
            n_init = np.asarray(medium.n(wavelengths), dtype=float)
            if np.ndim(n_init) == 0:
                n_init = np.full(num, float(n_init))
            k_init = np.asarray(medium.k(wavelengths), dtype=float)
            if np.ndim(k_init) == 0:
                k_init = np.full(num, float(k_init))
        else:
            n_init = np.ones(num)
            k_init = np.zeros(num)

        return NSQRayBundle(
            x=pos[:, 0].copy(),
            y=pos[:, 1].copy(),
            z=pos[:, 2].copy(),
            L=dirs[:, 0].copy(),
            M=dirs[:, 1].copy(),
            N=dirs[:, 2].copy(),
            # be.ones * flux keeps the autograd graph when total_flux is a tensor.
            flux=be.ones(num) * flux_per_ray,
            wavelength=wavelengths,
            n_current=n_init,
            bounce=bounce0,
            alive=np.ones(num, dtype=bool),
            ray_id=ray_id,
            k_current=k_init,
        )
