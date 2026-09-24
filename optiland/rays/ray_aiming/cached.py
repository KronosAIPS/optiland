"""Cached Ray Aiming Module

This module implements a caching wrapper for ray aiming algorithms.
It stores previous results to speed up repetitive calculations when the inputs
and optical system remain unchanged.

Retained for backward compatibility (explicit ``cache=True``) but no longer
the default warm-start mechanism for ``"robust"``, which has its own
intrinsic, always-on ``PupilMapCache`` (see ``pupil_map.py`` and
``robust.py``).

Kramer Harrison, 2025
"""

from __future__ import annotations

import hashlib
import pickle
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.rays.ray_aiming.base import BaseRayAimer

if TYPE_CHECKING:
    from optiland.optic import Optic


class CachedRayAimer(BaseRayAimer):
    """Cached ray aiming strategy.

    This class wraps another ray aimer and caches its results. It checks
    if the inputs and the optical system state have changed. If they match
    a cached entry in the same numerical execution context, the result is
    returned immediately. Otherwise, the wrapped aimer computes a fresh result
    without an automatic initial guess. Explicit caller guesses bypass caching.

    Coordinate-frame note: cached entries are full launch states in global
    coordinates. The system hash covers every surface, so any rigid pose
    change (translation, fold reorientation) misses the exact-reuse path.
    After a system change, a cached launch is not used as a starting guess:
    its fixed field-dependent components may no longer represent the requested
    field, even if the ray still satisfies the stop constraint.

    Torch calls clear and bypass this cache whenever either backend gradient
    tracking or ambient autograd is enabled, so computation graphs are never
    reused. Backend, precision, device, and Torch inference-mode changes also
    clear numerical entries.
    The wrapped robust aimer's detached pupil-map seed cache is independent.

    Attributes:
        optic (Optic): The optical system being traced.
        wrapped_aimer (BaseRayAimer): The actual aiming strategy being cached.
        max_cache_size (int): Maximum number of entries in the cache.
    """

    def __init__(
        self,
        optic: Optic,
        wrapped_aimer: BaseRayAimer,
        max_cache_size: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize the CachedRayAimer.

        Args:
            optic (Optic): The optical system instance.
            wrapped_aimer (BaseRayAimer): The aimer instance to wrap.
            max_cache_size (int, optional): Max cache entries. Defaults to 128.
            **kwargs: Additional arguments passed to BaseRayAimer.
        """
        super().__init__(optic, **kwargs)
        self.wrapped_aimer = wrapped_aimer
        self.max_cache_size = max_cache_size
        self._cache: dict[str, tuple[str, tuple]] = {}
        self._cache_context: tuple[str, int, str | None, bool] | None = None

    def aim_rays(
        self,
        fields: tuple,
        wavelengths: Any,
        pupil_coords: tuple,
        initial_guess: tuple | None = None,
    ) -> tuple:
        """Calculate ray starting coordinates, using cache if available.

        Args:
            fields (tuple): Field coordinates.
            wavelengths (Any): Wavelengths.
            pupil_coords (tuple): Pupil coordinates.
            initial_guess (tuple | None, optional): Explicit starting guess.

        Returns:
            tuple: Ray parameters (x, y, z, L, M, N).
        """
        # Invalidate before any early return, including explicit caller guesses,
        # so a bypass cannot leave incompatible entries for subsequent calls.
        backend = be.get_backend()
        bypass_cache = False
        if backend == "torch":
            import torch

            # Either Optiland's gradient control or ambient Torch autograd can
            # require a fresh graph. Equal numerical values do not identify
            # its lifetime or the parameter leaves to which it is attached.
            bypass_cache = be.grad_mode.requires_grad or torch.is_grad_enabled()

        if bypass_cache:
            # Discard earlier entries too, so returning to numerical execution
            # starts a new cache rather than reviving pre-gradient results.
            self.clear_cache()
        else:
            # Results must match the active backend, precision and device.
            # Inference mode also differs from no_grad: inference tensors cannot
            # be mutated outside inference mode or saved for backward.
            context = (
                backend,
                be.get_precision(),
                be.get_device() if backend == "torch" else None,
                torch.is_inference_mode_enabled() if backend == "torch" else False,
            )
            if context != self._cache_context:
                self.clear_cache()
                self._cache_context = context

        # If an explicit initial guess is provided, we skip the cache check
        if initial_guess is not None:
            return self.wrapped_aimer.aim_rays(
                fields, wavelengths, pupil_coords, initial_guess=initial_guess
            )
        if bypass_cache:
            # Do not hash, read or store results in a differentiable call.
            return self.wrapped_aimer.aim_rays(fields, wavelengths, pupil_coords)

        # 1. Generate Input Hash
        input_key = self._get_input_hash(fields, wavelengths, pupil_coords)

        # 2. Generate System Hash
        current_sys_hash = self._get_system_hash()

        # 3. Check Cache
        cached_entry = self._cache.get(input_key)
        if cached_entry is not None:
            cached_sys_hash, cached_result = cached_entry
            if cached_sys_hash == current_sys_hash:
                # Exact match: inputs same, system same
                return cached_result

        # 4. Delegate to wrapped aimer
        # A full launch fixes the object point for finite conjugates or the
        # incident direction for infinite conjugates. The old launch can still
        # satisfy the stop constraint after a field edit, so do not use it as
        # an automatic initial guess. Omitting the keyword also supports aimers
        # whose public interface accepts only the three required arguments.
        result = self.wrapped_aimer.aim_rays(fields, wavelengths, pupil_coords)

        # 5. Update Cache
        self._cache[input_key] = (current_sys_hash, result)

        # 6. Manage Cache Size
        if len(self._cache) > self.max_cache_size:
            # Simple FIFO removal (Python dicts preserve insertion order)
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        return result

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()

    def _get_input_hash(
        self, fields: tuple, wavelengths: Any, pupil_coords: tuple
    ) -> str:
        """Generate a hash for the input parameters."""

        def _to_hashable(obj: Any) -> Any:
            if hasattr(obj, "tobytes"):
                return obj.tobytes()
            elif isinstance(obj, list | tuple):
                return tuple(_to_hashable(x) for x in obj)
            return obj

        data = (
            _to_hashable(fields),
            _to_hashable(wavelengths),
            _to_hashable(pupil_coords),
        )
        return hashlib.md5(pickle.dumps(data)).hexdigest()

    def _get_system_hash(self) -> str:
        """Generate a hash for the current state of the optical system."""
        data = (
            self.optic.surfaces.to_dict(),
            self.optic.fields.to_dict(),
            self.optic.wavelengths.to_dict(),
            self.optic.aperture.to_dict() if self.optic.aperture else None,
            self.optic.ray_tracer.ray_aiming_config,
        )
        # Array and tensor display formatting rounds values, so str(data) can
        # hide small field or aperture changes and produce a stale exact hit.
        # Serialize the numeric contents instead; no pickle payload is loaded.
        return hashlib.md5(pickle.dumps(data)).hexdigest()
