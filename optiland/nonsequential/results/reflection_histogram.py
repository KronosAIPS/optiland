"""Arriving flux by reflection count, for Non-Sequential Raytracing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReflectionHistogram:
    """The flux a detector received, binned by the rays' reflection count.

    Bin ``k`` holds the flux that arrived with exactly ``k`` reflections in
    its history (``NSQRayBundle.reflections``), for ``k = 0 .. K-1``; the
    overflow bin holds everything with ``K`` or more. The bins and the
    overflow together are the whole flux the detector recorded, so a ghost
    order is never merged into the direct beam and nothing is dropped.

    Attributes:
        flux: Flux per exact bin [W], shape (K,).
        flux_sq: Sum of squared per-ray flux per exact bin [W^2], shape (K,).
        num_rays_hit: Ray hits per exact bin, shape (K,).
        overflow_flux: Flux with ``K`` or more reflections [W].
        overflow_flux_sq: Sum of squared per-ray flux in the overflow bin.
        overflow_num_rays_hit: Ray hits in the overflow bin.
        data: The accumulation buffer (flux, overflow last) on the backend
            that filled it -- a ``torch.Tensor`` keeps its autograd graph,
            as ``IrradianceMap.data`` does.
    """

    flux: np.ndarray
    flux_sq: np.ndarray
    num_rays_hit: np.ndarray
    overflow_flux: float = 0.0
    overflow_flux_sq: float = 0.0
    overflow_num_rays_hit: int = 0
    data: object = None

    @property
    def num_bins(self) -> int:
        """Number of exact bins ``K``."""
        return int(self.flux.shape[0])

    @property
    def total_flux(self) -> float:
        """All the flux the histogram holds, overflow included [W]."""
        return float(self.flux.sum() + self.overflow_flux)

    def standard_error(self, num_rays_launched: int) -> np.ndarray:
        """Standard error of each exact bin's flux, from the run.

        ``docs/theory/03_monte_carlo.md`` R-03-7: with ``N`` launched rays,
        each contributing an independent weight ``w_i`` (zero when it does
        not reach this bin), the bin total ``F = sum w_i`` has

            SE^2 = N / (N - 1) * (sum w_i^2 - F^2 / N).

        That independence holds when every launched ray reaches the detector
        at most once and no ray is split -- roulette and importance-biased
        branches are fine, bounded splitting is not (a split tree is
        deterministic in its branches, and its children share a parent).

        Args:
            num_rays_launched: ``N``, the rays the trace launched.

        Returns:
            Standard error per exact bin [W], shape (K,).
        """
        n = float(num_rays_launched)
        if n <= 1.0:
            return np.zeros_like(self.flux)
        var = (self.flux_sq - self.flux**2 / n) * n / (n - 1.0)
        return np.sqrt(np.maximum(var, 0.0))

    def to_numpy(self) -> np.ndarray:
        """Flux per exact bin followed by the overflow bin, shape (K + 1,)."""
        return np.append(self.flux, self.overflow_flux)
