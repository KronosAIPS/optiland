"""Irradiance map result for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class IrradianceMap:
    """2D irradiance distribution on a planar detector.

    Attributes:
        data: Flat accumulated flux buffer (be-array, shape ny*nx).
            This is the differentiable handle; call ``.data.backward()``
            to propagate gradients through the detector image.
            ``None`` when constructed from legacy hard-splatted data.
        irradiance: Irradiance [W/mm^2], shape (ny, nx), as a NumPy array.
        x_coords: Bin centre x-coordinates [mm], shape (nx,).
        y_coords: Bin centre y-coordinates [mm], shape (ny,).
        total_flux: Total flux recorded [W]. Attached to the active
            backend's autograd graph -- a torch.Tensor when the
            underlying data is, so ``result.total_flux.backward()``
            propagates a gradient. Use :attr:`total_flux_float` for
            printing or any consumer that expects a plain Python float.
        num_rays_hit: Number of rays recorded on this detector.
        stokes: The Stokes tallies (:class:`StokesMaps`) of a detector built
            with ``stokes=True`` and traced with ``polarization="stokes"``;
            ``None`` otherwise.
    """

    def __init__(
        self,
        irradiance: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        total_flux,
        num_rays_hit: int,
        data=None,
        stokes: StokesMaps | None = None,
    ) -> None:
        """Initialize IrradianceMap.

        Args:
            irradiance: 2D irradiance array [W/mm^2], shape (ny, nx).
            x_coords: Bin centre x-coordinates [mm], shape (nx,).
            y_coords: Bin centre y-coordinates [mm], shape (ny,).
            total_flux: Total detected flux [W]. May be a plain float or a
                backend array/tensor; kept attached if the latter.
            num_rays_hit: Number of rays that contributed.
            data: Flat accumulated flux be-array (shape ny*nx), optional.
                When provided, this is the attached differentiable buffer
                from which irradiance was computed. Defaults to None.
            stokes: Optional :class:`StokesMaps`.
        """
        self.data = data  # attached tensor or numpy array (flat, ny*nx)
        self.irradiance = irradiance
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.total_flux = total_flux
        self.num_rays_hit = int(num_rays_hit)
        self.stokes = stokes

    @property
    def total_flux_float(self) -> float:
        """Total detected flux [W] as a plain, detached Python float.

        Use this for printing, formatting, or any code path that cannot
        accept a gradient-carrying tensor; :attr:`total_flux` stays attached
        for differentiation.
        """
        from optiland.backend.utils import to_numpy  # noqa: PLC0415

        return float(to_numpy(self.total_flux))

    def plot(self, ax=None, **kwargs):
        """Plot the irradiance map.

        Args:
            ax: Optional Matplotlib Axes. If None, a new figure is created.
            **kwargs: Additional arguments passed to imshow.

        Returns:
            The Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        extent = [
            self.x_coords[0],
            self.x_coords[-1],
            self.y_coords[0],
            self.y_coords[-1],
        ]
        im = ax.imshow(
            self.irradiance,
            origin="lower",
            extent=extent,
            aspect="equal",
            **kwargs,
        )
        plt.colorbar(im, ax=ax, label="Irradiance [W/mm^2]")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(
            f"Irradiance Map -- {self.num_rays_hit} rays, {self.total_flux_float:.3g} W"
        )
        return fig

    def save(self, path: str | Path) -> None:
        """Save irradiance map to a .npz file.

        Args:
            path: Output file path.
        """
        np.savez(
            path,
            irradiance=self.irradiance,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
            total_flux=self.total_flux_float,
            num_rays_hit=self.num_rays_hit,
        )

    def to_numpy(self) -> np.ndarray:
        """Return the irradiance array as a NumPy array.

        Returns:
            The irradiance array, shape (ny, nx).
        """
        return np.asarray(self.irradiance)


class StokesMaps:
    """The Stokes tallies of a polarization-resolving detector (the research repository's issue 5).

    Per pixel, the sums over the arriving rays of ``I``, ``Q = I q'``,
    ``U = I u'`` and ``V = I v'``, with ``(q', u', v')`` each ray's reduced
    state in the detector's frame: ``Q > 0`` is polarization along the
    detector's local x axis, ``U > 0`` along the direction 45 degrees from it
    towards ``k x x``, and ``V`` has the sign of chapter 06 (a quarter-wave
    retarder with its fast axis at +45 degrees takes ``(1, 1, 0, 0)`` to
    ``(1, 0, 0, -1)``). The sums are accumulated as the flux is, in float64
    where the device has it. The degree and the angle of polarization are
    derived on read, never accumulated: a ratio of sums, not a sum of ratios.

    Attributes:
        i, q, u, v: Flat per-pixel sums [W], shape ``ny * nx``; backend
            arrays, attached to the trace's graph where the flux is.
        shape: ``(ny, nx)``.
    """

    def __init__(self, i, q, u, v, shape: tuple[int, int]) -> None:
        self.i, self.q, self.u, self.v = i, q, u, v
        self.shape = tuple(shape)

    def totals(self) -> tuple:
        """``(I, Q, U, V)`` summed over the detector, as backend scalars (attached)."""
        import optiland.backend as be  # noqa: PLC0415

        return tuple(be.sum(x) for x in (self.i, self.q, self.u, self.v))

    def totals_float(self) -> tuple[float, float, float, float]:
        """``(I, Q, U, V)`` summed over the detector, as Python floats."""
        from optiland.backend.utils import to_numpy  # noqa: PLC0415

        return tuple(float(to_numpy(x)) for x in self.totals())

    def reduced(self) -> tuple[float, float, float]:
        """``(Q/I, U/I, V/I)`` of the whole detector (zero where nothing arrived)."""
        i, q, u, v = self.totals_float()
        if i == 0.0:
            return 0.0, 0.0, 0.0
        return q / i, u / i, v / i

    def degree_of_polarization(self) -> float:
        """``sqrt(Q^2 + U^2 + V^2) / I`` of the whole detector."""
        q, u, v = self.reduced()
        return float(np.sqrt(q * q + u * u + v * v))

    def angle_of_polarization_deg(self) -> float:
        """``atan2(U, Q) / 2`` of the whole detector, in degrees from the local x axis."""
        q, u, _ = self.reduced()
        return float(np.degrees(0.5 * np.arctan2(u, q)))

    def maps(self) -> dict[str, np.ndarray]:
        """The per-pixel ``I``, ``Q``, ``U``, ``V`` [W] and ``dop`` as NumPy arrays of ``shape``."""
        from optiland.backend.utils import to_numpy  # noqa: PLC0415

        out = {
            name: np.asarray(to_numpy(x), dtype=np.float64).reshape(self.shape)
            for name, x in (("I", self.i), ("Q", self.q), ("U", self.u), ("V", self.v))
        }
        pol = np.sqrt(out["Q"] ** 2 + out["U"] ** 2 + out["V"] ** 2)
        safe = np.where(out["I"] > 0, out["I"], 1.0)
        out["dop"] = np.where(out["I"] > 0, pol / safe, 0.0)
        return out
