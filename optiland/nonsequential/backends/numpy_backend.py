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
from optiland.nonsequential.polarization import mode_for_backend
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

    def __init__(
        self, seed: int | None = None, polarization: str | bool | None = None
    ) -> None:
        """Initialize NumpyBackend.

        Args:
            seed: Optional random seed for reproducibility.
            polarization: ``"off"`` or ``"stokes"`` (``True``/``False`` are
                accepted); ``None`` reads
                :data:`~optiland.nonsequential.polarization.POLARIZATION_ENV`,
                unset meaning ``"off"``.
        """
        self.seed = seed
        self.rng = NSQRng(seed)
        self.polarization = mode_for_backend(polarization)

    def _maybe_compact(self, rays: NSQRayBundle, depth: int) -> NSQRayBundle:
        """Compact dead rays after every bounce for the NumPy fast path.

        Removes rays where ``alive=False`` so subsequent intersection tests
        skip them, giving a significant speedup when many rays die early.
        Not bucketed: a host array's width is a Python int, so an exact
        width costs nothing here and there is no kernel to recompile.

        Args:
            rays: Current ray bundle.
            depth: Bounces already run for this batch. Unused -- the host
                path compacts every bounce.

        Returns:
            Compacted ray bundle containing only alive rays.
        """
        return rays.compact()
