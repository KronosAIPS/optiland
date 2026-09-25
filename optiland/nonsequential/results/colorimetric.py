"""Results of the colorimetric detectors: tristimulus tallies and what follows from them.

``X``, ``Y``, ``Z`` here are flux-weighted tristimulus values in W (the sum over
arriving rays of flux times xbar, ybar, zbar at the ray's wavelength); the
photometric quantities are ``K_m`` times ``Y`` per area or per solid angle, and
chromaticity is scale-free. Correlated colour temperature is taken from the
optional package ``colour-science`` (BSD-3-Clause), Ohno's 2013 method by
default, rather than re-implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optiland.nonsequential.units import KM_PHOTOPIC


def chromaticity_xy(xyz: np.ndarray) -> np.ndarray:
    """CIE 1931 (x, y) of tristimulus values along axis 0; NaN where X+Y+Z = 0."""
    xyz = np.asarray(xyz, dtype=np.float64)
    s = xyz.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([xyz[0] / s, xyz[1] / s])


def chromaticity_uv_prime(xyz: np.ndarray) -> np.ndarray:
    """CIE 1976 (u', v') = (4X, 9Y) / (X + 15Y + 3Z); NaN where the denominator is 0."""
    xyz = np.asarray(xyz, dtype=np.float64)
    d = xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([4.0 * xyz[0] / d, 9.0 * xyz[1] / d])


def cct_of_xy(xy, method: str = "Ohno 2013") -> tuple[float, float]:
    """Correlated colour temperature [K] and Duv of a chromaticity, via colour-science.

    Raises:
        ImportError: If the optional package ``colour-science`` is not installed.
    """
    try:
        import colour  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Correlated colour temperature is computed with the optional package "
            "'colour-science' (BSD-3-Clause); install it."
        ) from err
    x, y = (float(v) for v in xy)
    d = -2.0 * x + 12.0 * y + 3.0
    uv = np.array([4.0 * x / d, 6.0 * y / d])  # CIE 1960 UCS
    cct, duv = colour.temperature.uv_to_CCT(uv, method=method)
    return float(cct), float(duv)


@dataclass
class ColorimetricMap:
    """A planar colorimetric detector's result.

    Attributes:
        radiometric: The irradiance map (W, W/mm^2) of the same pixels.
        tristimulus: X, Y, Z per pixel [W], shape (3, ny, nx).
        pixel_area_mm2: One pixel's area [mm^2].
        cmf: The colour-matching table and its interpolation, by name.
    """

    radiometric: object
    tristimulus: np.ndarray
    pixel_area_mm2: float
    cmf: str

    @property
    def luminous_flux(self) -> float:
        """Luminous flux on the detector [lm]: ``K_m sum Y``."""
        return float(KM_PHOTOPIC * self.tristimulus[1].sum())

    @property
    def illuminance(self) -> np.ndarray:
        """Illuminance per pixel [lm/mm^2]; times 1e6 for lux."""
        return KM_PHOTOPIC * self.tristimulus[1] / self.pixel_area_mm2

    @property
    def illuminance_lux(self) -> np.ndarray:
        """Illuminance per pixel [lx], for display (R-01-7)."""
        return self.illuminance * 1.0e6

    @property
    def total_tristimulus(self) -> np.ndarray:
        """X, Y, Z summed over the detector [W]."""
        return self.tristimulus.reshape(3, -1).sum(axis=1)

    def chromaticity(self, per_pixel: bool = False) -> np.ndarray:
        """CIE 1931 (x, y) of the whole detector, or per pixel, shape (2, ny, nx)."""
        return chromaticity_xy(self.tristimulus if per_pixel else self.total_tristimulus)

    def uv_prime(self, per_pixel: bool = False) -> np.ndarray:
        """CIE 1976 (u', v') of the whole detector, or per pixel."""
        return chromaticity_uv_prime(self.tristimulus if per_pixel else self.total_tristimulus)

    def cct(self, method: str = "Ohno 2013") -> tuple[float, float]:
        """Correlated colour temperature [K] and Duv of the whole detector."""
        return cct_of_xy(self.chromaticity(), method)


@dataclass
class ColorimetricFarField:
    """A colorimetric far-field detector's result.

    Attributes:
        radiometric: The far-field pattern (W/sr) of the same bins.
        tristimulus: X, Y, Z intensity per bin [W/sr], shape (3, n_theta, n_phi),
            divided by the same per-bin solid angle as the radiometric pattern.
        cmf: The colour-matching table and its interpolation, by name.
    """

    radiometric: object
    tristimulus: np.ndarray
    cmf: str

    @property
    def luminous_intensity(self) -> np.ndarray:
        """Luminous intensity per bin [cd]: ``K_m Y``."""
        return KM_PHOTOPIC * self.tristimulus[1]

    def chromaticity(self, per_bin: bool = True) -> np.ndarray:
        """CIE 1931 (x, y) per bin, shape (2, n_theta, n_phi), or of the bins' sum."""
        return chromaticity_xy(self.tristimulus if per_bin else self._total())

    def uv_prime(self, per_bin: bool = True) -> np.ndarray:
        """CIE 1976 (u', v') per bin, or of the bins' sum."""
        return chromaticity_uv_prime(self.tristimulus if per_bin else self._total())

    def _total(self) -> np.ndarray:
        # intensity times the pattern's own per-bin solid angle recovers flux
        th = np.radians(np.asarray(self.radiometric.theta, dtype=float))
        n_t, n_p = self.tristimulus.shape[1:]
        d_theta = (th[1] - th[0]) if n_t > 1 else 2 * th[0]
        omega = np.sin(th)[:, None] * d_theta * (2 * np.pi / n_p)
        return (self.tristimulus * omega).reshape(3, -1).sum(axis=1)
