"""Continuous source spectra: piecewise-linear densities, Planck, CIE illuminants.

:class:`~optiland.nonsequential.sources.base.Spectrum` is a set of discrete
lines. :class:`PiecewiseLinearSpectrum` is a continuous spectral density: its
values at nodes are joined by straight lines and it is zero outside the first
and last node. Each ray's wavelength is drawn exactly from that density by
inverting its cumulative distribution with one uniform (the source's
``SOURCE_WAVELENGTH`` slot of the keyed generator), so every ray carries the
same weight and a spectral detector's bins see a continuous spectrum, not
lines.

Two families of standard spectra are built on it:

- :meth:`PiecewiseLinearSpectrum.blackbody` -- Planck's law evaluated at the
  nodes of a fine grid, with the second radiation constant
  ``c2 = h c / k`` from the exact SI values (2019) unless another is given.
- :meth:`PiecewiseLinearSpectrum.cie_illuminant` -- CIE standard illuminant A
  (its defining formula), E (equal energy) and D65 (the engine's frozen CIE
  table at 1 nm); any other name is taken from the ``colour-science`` package's
  datasets when that optional dependency is installed.

Linear interpolation between tabulated nodes is this module's stated
convention: a reference computed for a run integrates the same interpolant, so
a table that is exactly linear between its nodes carries no interpolation error.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from optiland.backend.utils import to_numpy
from optiland.nonsequential.rng import EventSlot

if TYPE_CHECKING:
    from optiland.nonsequential.rng import NSQRng

#: Planck constant [J s], speed of light [m/s], Boltzmann constant [J/K]:
#: exact by the definition of the SI (26th CGPM, 2018, in force 2019-05-20).
PLANCK_H = 6.62607015e-34
SPEED_OF_LIGHT = 299792458.0
BOLTZMANN_K = 1.380649e-23

#: Second radiation constant c2 = h c / k [m K], from the exact constants above.
C2_SI = PLANCK_H * SPEED_OF_LIGHT / BOLTZMANN_K

#: The value CIE standard illuminant A is defined with (CIE 15, ISO 11664-2):
#: c2 = 1.435e-2 m K at a distribution temperature of 2848 K, which is the
#: Planckian of about 2856 K with the modern c2.
C2_CIE_A = 1.435e-2
T_CIE_A = 2848.0


def sample_linear(u: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Invert the CDF of the density proportional to ``(1 - x) a + x b`` on [0, 1].

    The closed form of PBRT-v4 (``SampleLinear`` in ``src/pbrt/util/sampling.h``,
    Pharr, Jakob and Humphreys, Apache-2.0), written for arrays:
    ``x = u (a + b) / (a + sqrt(lerp(u, a^2, b^2)))``, which stays accurate when
    ``a`` and ``b`` are close (no cancellation in the quadratic's root).

    Args:
        u: Uniforms in [0, 1).
        a: Density at x = 0 (non-negative).
        b: Density at x = 1 (non-negative), ``a + b > 0``.

    Returns:
        x in [0, 1).
    """
    u = np.asarray(u, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = a + np.sqrt((1.0 - u) * a * a + u * b * b)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(denom > 0.0, u * (a + b) / denom, 0.0)
    return np.minimum(x, np.nextafter(1.0, 0.0))


def _check_nodes(x: np.ndarray, y: np.ndarray, what: str) -> None:
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"{what}: nodes and values must be 1-D arrays of one length.")
    if x.size < 2:
        raise ValueError(f"{what}: at least two nodes are needed.")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError(f"{what}: node wavelengths must be strictly increasing.")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError(f"{what}: values must be finite and non-negative.")


class PiecewiseLinearSpectrum:
    """A continuous spectral density, linear between nodes, zero outside them.

    Attributes:
        nodes: Node wavelengths [µm], strictly increasing, shape (n,).
        values: Relative spectral density at the nodes (per µm, any scale),
            shape (n,).
        label: What the spectrum is (for reports): ``"blackbody 2856 K"``,
            ``"CIE A"``, or the caller's own.
    """

    def __init__(self, wavelengths, values, label: str = "") -> None:
        nodes = np.asarray(wavelengths, dtype=np.float64)
        vals = np.asarray(values, dtype=np.float64)
        _check_nodes(nodes, vals, "PiecewiseLinearSpectrum")
        if np.any(nodes <= 0.0) or np.any(nodes > 20.0):
            raise ValueError("Wavelengths must be in µm (expected range 0.1-20 µm).")
        widths = np.diff(nodes)
        cells = 0.5 * (vals[:-1] + vals[1:]) * widths
        total = float(cells.sum())
        if not total > 0.0:
            raise ValueError("PiecewiseLinearSpectrum: the density integrates to zero.")
        self.nodes = nodes
        self.values = vals
        self.label = label
        self._widths = widths
        self._cells = cells
        self._cdf = np.cumsum(cells)
        self._total = total

    # -- the line-spectrum interface the rest of the engine reads -------------

    @property
    def wavelengths(self) -> np.ndarray:
        """The node wavelengths [µm] (the line-spectrum interface)."""
        return self.nodes

    @property
    def weights(self) -> np.ndarray:
        """Trapezoid weights of the nodes: a line spectrum whose weighted sums
        are the trapezoid rule of the density (for code that reads spectra as
        lines; the exact integrals are :meth:`integrate_linear`)."""
        w = np.zeros_like(self.values)
        half = 0.5 * self._widths
        w[:-1] += self.values[:-1] * half
        w[1:] += self.values[1:] * half
        return w

    # -- density and integrals -------------------------------------------------

    @property
    def total(self) -> float:
        """Integral of the (unnormalised) density over its support [value x µm]."""
        return self._total

    def density(self, wavelength_um) -> np.ndarray:
        """The normalised probability density [1/µm] at ``wavelength_um``."""
        return (
            np.interp(wavelength_um, self.nodes, self.values, left=0.0, right=0.0)
            / self._total
        )

    def integrate_linear(self, grid_um, grid_values) -> float:
        """Exact integral of this density times another piecewise-linear function.

        Both functions are linear between their own nodes, so on the merged
        node set their product is a quadratic per interval and
        ``h ((a0 b0 + a1 b1) / 3 + (a0 b1 + a1 b0) / 6)`` integrates it exactly.
        The other function is zero outside its own nodes, as this one is.

        Args:
            grid_um: The other function's nodes [µm], increasing.
            grid_values: Its values there.

        Returns:
            ``integral S(l) g(l) dl`` over the common support, with ``S`` the
            unnormalised density.
        """
        gx = np.asarray(grid_um, dtype=np.float64)
        gy = np.asarray(grid_values, dtype=np.float64)
        lo = max(self.nodes[0], gx[0])
        hi = min(self.nodes[-1], gx[-1])
        if not hi > lo:
            return 0.0
        pts = np.union1d(self.nodes, gx)
        pts = pts[(pts >= lo) & (pts <= hi)]
        a = np.interp(pts, self.nodes, self.values)
        b = np.interp(pts, gx, gy)
        h = np.diff(pts)
        return float(
            np.sum(
                h
                * (
                    (a[:-1] * b[:-1] + a[1:] * b[1:]) / 3.0
                    + (a[:-1] * b[1:] + a[1:] * b[:-1]) / 6.0
                )
            )
        )

    def band_fraction(self, lo_um: float, hi_um: float) -> float:
        """Fraction of the density's integral between ``lo_um`` and ``hi_um``."""
        return self.integrate_linear([lo_um, hi_um], [1.0, 1.0]) / self._total

    def luminous_efficacy(self, weighting: str = "photopic") -> float:
        """Luminous efficacy [lm/W] of this spectrum, integrated exactly.

        ``Km integral S V / integral S`` with the engine's V(lambda) table
        (:mod:`optiland.nonsequential.units`, linear between its nodes, as
        :func:`~optiland.nonsequential.units.v_lambda` reads it): both factors
        are piecewise linear, so :meth:`integrate_linear` is exact.
        """
        from optiland.nonsequential.units import _table  # noqa: PLC0415

        grid, v, km = _table(weighting)
        return km * self.integrate_linear(grid, v) / self._total

    # -- sampling ----------------------------------------------------------------

    def sample(self, ray_id: np.ndarray, bounce: np.ndarray, rng: NSQRng) -> np.ndarray:
        """Draw one wavelength per ray from the density, exactly.

        One uniform per ray (slot ``SOURCE_WAVELENGTH``) selects the cell by the
        cumulative cell integrals and, rescaled to the cell, the position in it
        by :func:`sample_linear`: a single inversion of the continuous CDF.

        Args:
            ray_id: Per-ray identifiers, shape (N,).
            bounce: Per-ray bounce index (0 at birth).
            rng: Keyed generator.

        Returns:
            Wavelengths [µm], float64, shape (N,).
        """
        u = to_numpy(rng.uniform(ray_id, bounce, EventSlot.SOURCE_WAVELENGTH))
        target = np.asarray(u, dtype=np.float64) * self._total
        cell = np.searchsorted(self._cdf, target, side="right")
        cell = np.clip(cell, 0, self._cells.size - 1)
        before = np.where(cell > 0, self._cdf[np.maximum(cell - 1, 0)], 0.0)
        mass = self._cells[cell]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(mass > 0.0, (target - before) / mass, 0.0)
        r = np.clip(r, 0.0, np.nextafter(1.0, 0.0))
        t = sample_linear(r, self.values[cell], self.values[cell + 1])
        return self.nodes[cell] + t * self._widths[cell]

    # -- standard spectra ------------------------------------------------------

    @staticmethod
    def planck(wavelength_um, temperature_k: float, c2_m_k: float = C2_SI) -> np.ndarray:
        """Planck's spectral radiance shape ``lambda^-5 / (exp(c2 / (lambda T)) - 1)``.

        Relative (the constant ``2 h c^2`` is dropped); wavelength in µm.
        ``expm1`` keeps the long-wavelength tail exact.
        """
        lam_m = np.asarray(wavelength_um, dtype=np.float64) * 1e-6
        x = c2_m_k / (lam_m * float(temperature_k))
        return 1.0 / (lam_m**5 * np.expm1(x))

    @classmethod
    def blackbody(
        cls,
        temperature_k: float,
        wl_min_um: float = 0.38,
        wl_max_um: float = 0.78,
        step_um: float = 0.0005,
        c2_m_k: float = C2_SI,
    ) -> PiecewiseLinearSpectrum:
        """A Planckian radiator's spectrum on ``[wl_min_um, wl_max_um]``.

        The nodes are spaced ``step_um`` (0.5 nm by default); between them the
        density is linear, which departs from Planck's curve by at most
        ``step^2 / 8`` times its second derivative, below 1e-6 relative on the
        visible band at the default step.

        Args:
            temperature_k: Temperature [K].
            wl_min_um: Lower edge [µm].
            wl_max_um: Upper edge [µm].
            step_um: Node spacing [µm].
            c2_m_k: Second radiation constant [m K]; the exact SI value by
                default.

        Returns:
            The spectrum, scaled so its largest node value is 1.
        """
        if not temperature_k > 0.0:
            raise ValueError("blackbody: the temperature must be positive.")
        n = max(2, int(round((wl_max_um - wl_min_um) / step_um)) + 1)
        nodes = np.linspace(wl_min_um, wl_max_um, n)
        vals = cls.planck(nodes, temperature_k, c2_m_k)
        vals = vals / vals.max()
        return cls(nodes, vals, label=f"blackbody {temperature_k:g} K")

    @classmethod
    def cie_illuminant(cls, name: str) -> PiecewiseLinearSpectrum:
        """A CIE standard illuminant as a spectrum.

        ``"A"``: the defining formula, ``100 (560 / l)^5 (exp(c2 / (2848 x 560))
        - 1) / (exp(c2 / (2848 l)) - 1)`` with ``c2 = 1.435e7 nm K``, at 1 nm
        from 300 to 830 nm. ``"E"``: constant from 380 to 780 nm. ``"D65"``:
        the engine's frozen CIE table (1 nm, 380-780 nm, the file the sequential
        colorimetry module reads). Any other name (``"D50"``, ``"FL2"``,
        ``"LED-B3"``, ...) is read from the ``colour-science`` package's
        ``SDS_ILLUMINANTS`` if it is installed.

        Raises:
            ImportError: For a name that needs ``colour-science`` when it is
                not installed.
            KeyError: For a name ``colour-science`` does not know either.
        """
        key = name.strip()
        if key.upper() == "A":
            lam_nm = np.arange(300.0, 831.0, 1.0)
            c2 = C2_CIE_A * 1e9  # nm K
            vals = (
                100.0
                * (560.0 / lam_nm) ** 5
                * math.expm1(c2 / (T_CIE_A * 560.0))
                / np.expm1(c2 / (T_CIE_A * lam_nm))
            )
            return cls(lam_nm * 1e-3, vals, label="CIE A")
        if key.upper() == "E":
            return cls([0.38, 0.78], [1.0, 1.0], label="CIE E")
        if key.upper() == "D65":
            from optiland.colorimetry.constants import (  # noqa: PLC0415
                ILLUMINANT_D65,
                WAVELENGTHS_STD,
            )

            return cls(
                np.asarray(WAVELENGTHS_STD, dtype=np.float64) * 1e-3,
                np.asarray(ILLUMINANT_D65, dtype=np.float64),
                label="CIE D65",
            )
        try:
            import colour  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"CIE illuminant {name!r} is read from the optional package "
                "'colour-science' (BSD-3-Clause); install it, or build the "
                "spectrum from your own table with PiecewiseLinearSpectrum."
            ) from err
        sd = colour.SDS_ILLUMINANTS[key]
        return cls(
            np.asarray(sd.wavelengths, dtype=np.float64) * 1e-3,
            np.asarray(sd.values, dtype=np.float64),
            label=f"CIE {key} (colour-science {colour.__version__})",
        )

    def __repr__(self) -> str:
        return (
            f"PiecewiseLinearSpectrum({self.label or 'table'}, {self.nodes.size} nodes, "
            f"{self.nodes[0]:g}-{self.nodes[-1]:g} um)"
        )
