"""NumPy CPU backend for Non-Sequential Raytracing.

The trace loop itself lives in :class:`~optiland.nonsequential.backends
.array_backend.ArrayBackend`. This class supplies the two things that are
specific to keeping the ray state in host memory: dead rays are compacted
out of the bundle after every bounce, and a reduction to a Python bool is
free, so the loop may skip an empty block and stop as soon as no ray is
alive.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.nonsequential.backends.array_backend import ArrayBackend
from optiland.nonsequential.rng import NSQRng

if TYPE_CHECKING:
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class NumpyBackend(ArrayBackend):
    """CPU backend using NumPy for all array operations.

    This is the default fallback backend. All ray data remains in host
    (CPU) memory throughout the simulation.

    Attributes:
        rng: Keyed PCG32 RNG (see :mod:`optiland.nonsequential.rng`).
        seed: RNG seed stored for internal use.
    """

    host_reads_free = True
    supports_splitting = True

    def __init__(self, seed: int | None = None) -> None:
        """Initialize NumpyBackend.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.seed = seed
        self.rng = NSQRng(seed)

    def _maybe_compact(self, rays: NSQRayBundle) -> NSQRayBundle:
        """Compact dead rays after every bounce for the NumPy fast path.

        Removes rays where ``alive=False`` so subsequent intersection tests
        skip them, giving a significant speedup when many rays die early.

        Args:
            rays: Current ray bundle.

        Returns:
            Compacted ray bundle containing only alive rays.
        """
        return rays.compact()
