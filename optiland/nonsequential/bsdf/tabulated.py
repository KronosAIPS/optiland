"""Tabulated BSDF for Non-Sequential Raytracing.

Loads scatter data from a CSV or Zemax scatter file and interpolates.

Kramer Harrison, 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential._utils import clamp_int
from optiland.nonsequential.bsdf.base import BaseBSDF
from optiland.nonsequential.components.base import resident_table
from optiland.nonsequential.components.sampling_support import detached
from optiland.nonsequential.ray_bundle import backend_bool_full
from optiland.nonsequential.rng import EventSlot

if TYPE_CHECKING:
    from optiland.nonsequential.rng import NSQRng


class TabulatedBSDF(BaseBSDF):
    """BSDF loaded from tabulated data (CSV or Zemax scatter file).

    The file must contain columns: theta_i [deg], theta_s [deg], bsdf_value.
    The BSDF is assumed azimuthally symmetric (phi-independent).

    Attributes:
        weight_is_albedo: False. This lobe draws its direction from a
            cosine-weighted hemisphere and corrects with ``pi * BSDF``, so
            the weight a given ray carries is a sampling weight, not that
            ray's share of the surface's reflectance: the two agree only in
            expectation. ``1 - weight`` is therefore not what the surface
            absorbed on this realisation, and the ledger splits it -- see
            ``BaseBSDF.weight_is_albedo``.
        path: Path to the scatter data file.
        transmissive_fraction: Probability in [0, 1] that a given scatter
            event samples the transmissive hemisphere (the far side of the
            surface) instead of the reflective one. Defaults to 0.0:
            a purely reflective scatter, identical to this class's
            behaviour before D-5. The tabulated data itself is treated as
            hemisphere-relative (``theta_s`` measured from whichever normal
            the draw lands on), not as a combined BRDF+BTDF table.
    """

    weight_is_albedo = False

    def __init__(self, path: str | Path, transmissive_fraction: float = 0.0) -> None:
        """Load tabulated BSDF from file.

        Args:
            path: Path to CSV file with columns [theta_i, theta_s, bsdf].
            transmissive_fraction: Probability in [0, 1] that a scatter
                event lands in the transmissive hemisphere rather than the
                reflective one.
        """
        self.path = Path(path)
        self.transmissive_fraction = float(transmissive_fraction)
        self._load(self.path)

    def _load(self, path: Path) -> None:
        """Parse and build an interpolator from the data file.

        Args:
            path: Path to the CSV data file.
        """
        data = np.loadtxt(path, delimiter=",", comments="#")
        theta_i = np.unique(data[:, 0])
        theta_s = np.unique(data[:, 1])
        bsdf_grid = data[:, 2].reshape(len(theta_i), len(theta_s))
        self._theta_i_vals = theta_i
        self._theta_s_vals = theta_s
        # Flat, because the evaluation below gathers four corners per ray and
        # a flat gather is one index computation on either array library.
        self._grid_flat = np.ascontiguousarray(bsdf_grid, dtype=np.float64).ravel()
        self._albedo_vals = _albedo_per_incidence(theta_s, bsdf_grid)

    def _evaluate(self, theta_i, theta_s):
        """Bilinear lookup of the measured BSDF, on the device.

        The table is a regular (not necessarily uniform) grid in
        ``(theta_i, theta_s)``, so a linear interpolation over it is exactly
        bilinear: two ``searchsorted`` calls locate the cell, and the value
        is the four corners weighted by the fractional position inside it.
        A query outside the tabulated range returns 0, which is the
        ``fill_value`` the tabulated data has always been read with -- the
        measurement says nothing there, and extrapolating a scatter
        distribution past its data is not a service to anybody.

        The grid stays on the device; only the query angles cross the cell
        arithmetic, and nothing crosses to the host.

        Args:
            theta_i: Incidence angles [deg], shape (N,).
            theta_s: Scatter angles [deg], shape (N,).

        Returns:
            BSDF values, shape (N,), zero outside the tabulated range.
        """
        ti = resident_table(self, "theta_i", self._theta_i_vals)
        ts = resident_table(self, "theta_s", self._theta_s_vals)
        grid = resident_table(self, "grid", self._grid_flat)
        n_i = self._theta_i_vals.size
        n_s = self._theta_s_vals.size

        i0 = clamp_int(
            be.searchsorted(ti, theta_i, side="right") - 1, 0, max(n_i - 2, 0)
        )
        s0 = clamp_int(
            be.searchsorted(ts, theta_s, side="right") - 1, 0, max(n_s - 2, 0)
        )
        i1 = clamp_int(i0 + 1, 0, n_i - 1)
        s1 = clamp_int(s0 + 1, 0, n_s - 1)

        wi = _cell_fraction(theta_i, ti[i0], ti[i1])
        ws = _cell_fraction(theta_s, ts[s0], ts[s1])

        g00 = grid[i0 * n_s + s0]
        g01 = grid[i0 * n_s + s1]
        g10 = grid[i1 * n_s + s0]
        g11 = grid[i1 * n_s + s1]
        value = (
            (1.0 - wi) * ((1.0 - ws) * g00 + ws * g01)
            + wi * ((1.0 - ws) * g10 + ws * g11)
        )

        inside = (
            (theta_i >= ti[0])
            & (theta_i <= ti[n_i - 1])
            & (theta_s >= ts[0])
            & (theta_s <= ts[n_s - 1])
        )
        return be.where(inside, value, be.zeros_like(value))

    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: NSQRng,
        ray_id: np.ndarray,
        bounce: np.ndarray,
        frame=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample scattered directions from the tabulated BSDF.

        Uses importance sampling via a cosine-weighted hemisphere plus a
        BSDF weight. A per-ray draw against :attr:`transmissive_fraction`
        picks whether that hemisphere is centred on ``normals`` (reflective)
        or ``-normals`` (transmissive); ``theta_i``/``theta_s`` are measured
        from whichever normal the ray's draw landed on.

        Sampling is detached, and stays on whichever array library and
        device the ray state is on: the measured grid is uploaded once and
        interpolated with a device-side cell lookup (:meth:`_evaluate`).

        Args:
            num_rays: Number of rays.
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).
            rng: Keyed PCG32 RNG.
            ray_id: Per-ray identifiers, shape (N,).
            bounce: Per-ray bounce/step index, shape (N,).
            frame: The surface's rotation (local to global), or None for the
                world axes (``lambertian._orthonormal_basis``).

        Returns:
            (scattered_dirs, flux_weights, transmitted).
        """
        n_be = detached(normals)
        d_be = detached(incident_dirs)

        if self.transmissive_fraction > 0.0:
            u_lobe = rng.uniform(ray_id, bounce, EventSlot.BSDF_LOBE_BRANCH)
            transmitted = u_lobe < self.transmissive_fraction
            hemisphere = be.where(transmitted[:, None], -n_be, n_be)
        else:
            transmitted = backend_bool_full((n_be.shape[0],), False, like=n_be)
            hemisphere = n_be

        # Angle of incidence relative to the chosen hemisphere.
        cos_i = be.clip((-d_be * hemisphere).sum(axis=1), 0.0, 1.0)
        theta_i = be.degrees(be.arccos(cos_i))

        from optiland.nonsequential.bsdf.lambertian import (  # noqa: PLC0415
            _orthonormal_basis,
        )

        r1 = rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)
        r2 = rng.uniform(ray_id, bounce, EventSlot.BSDF_U2)
        phi = 2.0 * be.pi * r1
        cos_theta = be.sqrt(r2)
        sin_theta = be.sqrt(1.0 - r2)
        theta_s = be.degrees(be.arccos(cos_theta))

        lx = sin_theta * be.cos(phi)
        ly = sin_theta * be.sin(phi)
        lz = cos_theta

        t_vec, b_vec = _orthonormal_basis(hemisphere, frame)
        scattered = (
            lx[:, None] * t_vec + ly[:, None] * b_vec + lz[:, None] * hemisphere
        )
        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        scattered = scattered / norms

        flux_weights = be.clip(be.pi * self._evaluate(theta_i, theta_s), 0.0, 1.0)

        return scattered, flux_weights, transmitted

    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Directional-hemispherical reflectance of the tabulated lobe.

        The mean weight :meth:`sample` hands a ray at this incidence angle,
        which for a cosine-weighted draw corrected by ``pi * BSDF`` is the
        integral of the weight against the sampling density -- the albedo
        the surface actually delivers. It is quadratured from the table once
        per incidence node at load time and interpolated in ``theta_i``
        here; the quadrature is exact for the interpolant because the
        integral is linear in the table values, except where the weight
        clips at 1.

        This is what makes ``weight_is_albedo = False`` worth setting: the
        ledger books ``1 - reflectance()`` as the surface's physical loss
        and the rest as the event residual, and the residual only has its
        documented zero expectation if this is the true mean weight. It was
        previously the BSDF read at the middle tabulated scatter angle,
        which is not.

        Args:
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).

        Returns:
            Reflectance values in [0, 1], shape (N,), zero outside the
            tabulated incidence range (where :meth:`_evaluate` reads zero
            too, so the lobe returns no flux there either).
        """
        d_be = detached(incident_dirs)
        n_be = detached(normals)

        cos_i = be.clip((-d_be * n_be).sum(axis=1), 0.0, 1.0)
        theta_i = be.degrees(be.arccos(cos_i))

        ti = resident_table(self, "theta_i", self._theta_i_vals)
        albedo = resident_table(self, "albedo", self._albedo_vals)
        n_i = self._theta_i_vals.size
        i0 = clamp_int(
            be.searchsorted(ti, theta_i, side="right") - 1, 0, max(n_i - 2, 0)
        )
        i1 = clamp_int(i0 + 1, 0, n_i - 1)
        w = _cell_fraction(theta_i, ti[i0], ti[i1])
        value = (1.0 - w) * albedo[i0] + w * albedo[i1]
        inside = (theta_i >= ti[0]) & (theta_i <= ti[n_i - 1])
        return be.where(inside, value, be.zeros_like(value))


def _cell_fraction(x, lo, hi):
    """Where ``x`` sits between two grid points, as a fraction in [0, 1].

    A repeated grid point (a degenerate cell, or the clamp at the top edge
    of the table) has a zero span; the fraction is zero there, which returns
    the lower corner, as an interpolation between two equal abscissae must.

    Args:
        x: Query values, shape (N,).
        lo: Lower grid point for each query, shape (N,).
        hi: Upper grid point for each query, shape (N,).

    Returns:
        The fraction, shape (N,).
    """
    span = hi - lo
    positive = span > 0
    return be.where(
        positive,
        (x - lo) / be.where(positive, span, be.ones_like(span)),
        be.zeros_like(span),
    )


#: Quadrature points over the scatter hemisphere for the albedo integral.
_ALBEDO_QUADRATURE_POINTS = 4096


def _albedo_per_incidence(theta_s_vals, grid):
    """Mean sampling weight per tabulated incidence angle.

    ``sample`` draws a direction from a cosine-weighted hemisphere, whose
    density is ``cos(theta_s) / pi``, and weights it by
    ``clip(pi * BSDF, 0, 1)``. The mean weight is therefore

        rho(theta_i) = 2 * integral_0^{pi/2}
                        clip(pi * f, 0, 1) cos(theta_s) sin(theta_s) dtheta_s,

    taken over the whole hemisphere: the table says nothing outside its own
    scatter range and the lookup reads zero there, so the integrand is zero
    there too.

    Args:
        theta_s_vals: Tabulated scatter angles [deg], shape (n_s,).
        grid: BSDF values, shape (n_i, n_s).

    Returns:
        The albedo at each tabulated incidence angle, shape (n_i,).
    """
    theta = np.linspace(0.0, 0.5 * np.pi, _ALBEDO_QUADRATURE_POINTS)
    deg = np.degrees(theta)
    kernel = np.cos(theta) * np.sin(theta)
    out = np.empty(grid.shape[0], dtype=np.float64)
    for k in range(grid.shape[0]):
        f = np.interp(deg, theta_s_vals, grid[k], left=0.0, right=0.0)
        weight = np.clip(np.pi * f, 0.0, 1.0)
        out[k] = 2.0 * np.trapezoid(weight * kernel, theta)
    return np.clip(out, 0.0, 1.0)
