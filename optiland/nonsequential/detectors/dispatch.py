"""Shared nearest-detector dispatch for Non-Sequential Raytracing.

Both reference backends need to find, for every ray, the nearest detector it
hits (if any) among ``scene.detectors``. Before PR10 this routine was
duplicated almost verbatim in ``ArrayBackend._intersect_detectors`` and
``TorchBackend._intersect_detectors``, with subtly different grad-attachment
semantics: the NumPy version cast ``t``/normals to float64 NumPy (harmless
there, since NumPy has no autograd), while the Torch version kept ``t``
attached to the graph because the splatted landing position is
``origin + t * direction`` and detaching ``t`` silently drops the
``direction * dt/dtheta`` term from every spatial loss.

This module keeps exactly one implementation, using the Torch-safe (grad
-preserving) semantics unconditionally -- harmless under the NumPy backend,
required under Torch.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy

if TYPE_CHECKING:
    from optiland.nonsequential.detectors.base import BaseDetector
    from optiland.nonsequential.ray_bundle import NSQRayBundle


def intersect_detectors(
    rays: NSQRayBundle,
    detectors: list[BaseDetector],
) -> tuple[object, object, object, object, object]:
    """Find the nearest detector intersection for every ray.

    A running minimum over the detector list, in the active backend's own
    operations: the winning detector's index and its ``absorb`` flag are
    carried alongside the distance in arrays of the same library and device
    as the ray state, so the dispatch never leaves the device. The index is
    an integer array and the absorb flag a boolean one, so no gradient can
    flow through the choice of detector; ``t_min``/``hit_normals`` stay
    attached to the autograd graph, because the splatted landing position
    is ``origin + t * direction`` and detaching ``t`` silently drops the
    ``direction * dt/dtheta`` term from every spatial loss.

    Args:
        rays: Current ray bundle.
        detectors: ``scene.detectors``.

    Returns:
        ``(t_min, hit_normals, detector_indices, absorbs, hit_n_geom)``:
        backend arrays, with ``detector_indices`` an integer array holding
        ``-1`` where no detector was hit, ``absorbs`` the hit detector's
        ``absorb`` flag (``True`` where nothing was hit -- the caller only
        consults it where a detector actually was), and ``hit_n_geom`` the
        geometric (unflipped) normal of the winning detector's surface,
        which is what a transmitted ray's origin is offset along (R-07-6,
        R-07-9).
    """
    from optiland.nonsequential.ray_bundle import (  # noqa: PLC0415
        backend_bool_full,
        backend_int_full,
    )

    n = rays.num_rays
    t_min = be.ones(n) * be.inf
    hit_normals = be.zeros((n, 3))
    hit_n_geom = be.zeros((n, 3))
    det_indices = backend_int_full((n,), -1, like=rays.x, bits=32)
    absorbs = backend_bool_full((n,), True, like=rays.alive)

    for i, det in enumerate(detectors):
        t_d, normals_d, hit_d, n_geom_d = det.intersect(rays)
        better = hit_d & (t_d < t_min)

        t_min = be.where(better, t_d, t_min)
        hit_normals = be.where(better[:, None], normals_d, hit_normals)
        hit_n_geom = be.where(better[:, None], n_geom_d, hit_n_geom)
        det_indices = be.where(
            better, backend_int_full((n,), i, like=rays.x, bits=32), det_indices
        )
        absorbs = be.where(
            better, backend_bool_full((n,), det.absorb, like=rays.alive), absorbs
        )

    return t_min, hit_normals, det_indices, absorbs, hit_n_geom


def detector_absorb_mask(
    det_idx: np.ndarray, detectors: list[BaseDetector]
) -> np.ndarray:
    """Per-ray absorb flag of the detector each ray hit (D-10 ``absorb``).

    Retained for callers that hold a host-side detector index; the trace
    loop no longer uses it, because :func:`intersect_detectors` now carries
    the flag alongside the index without leaving the device.

    Args:
        det_idx: Per-ray index into ``detectors`` of the nearest-hit
            detector, or ``-1`` where no detector was hit. NumPy int array.
        detectors: ``scene.detectors``.

    Returns:
        Boolean NumPy array, shape matching ``det_idx``: True where the hit
        detector (if any) absorbs the ray. Rays with no detector hit are
        reported as ``True`` (irrelevant -- the caller only consults this
        where a detector was actually hit).
    """
    det_idx = to_numpy(det_idx)
    if len(detectors) == 0:
        return np.ones_like(det_idx, dtype=bool)
    absorb_per_detector = np.array([bool(d.absorb) for d in detectors], dtype=bool)
    safe_idx = np.clip(det_idx, 0, len(detectors) - 1)
    return absorb_per_detector[safe_idx]
