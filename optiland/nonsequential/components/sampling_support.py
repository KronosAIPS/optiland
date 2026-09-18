"""Detached-decision helpers shared by the interacting components.

Every stochastic branch in this engine is drawn from a *detached*
probability and compensated by an *attached* weight, so the estimator stays
unbiased and the gradient still reaches the physical parameter. Detaching
used to be done by copying the probability to the host with ``to_numpy``,
which conflates two unrelated things: "this value must not be
differentiated through" (a property of the autograd graph) and "this value
must live in host memory" (a property of the device). Only the first is
wanted. These helpers keep the first and drop the second, so a branch
decision costs no device synchronisation.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.nonsequential.rng import EventSlot

if TYPE_CHECKING:
    from optiland.nonsequential.rng import NSQRng

# The scatter-branch probability is clamped away from 0 and 1 for the same
# reason the Fresnel branch probability is: scatter_fraction exactly 1 (or
# 0) would otherwise divide by zero for the vanishing fraction of draws the
# clamp itself puts on the "wrong" side.
_SF_EPS = 1e-6


def detached(value):
    """Return ``value`` with no gradient attached, on its own device.

    Args:
        value: A backend array or tensor.

    Returns:
        The same values, detached from the autograd graph. A NumPy array is
        returned unchanged -- it never carried a graph.
    """
    if be.is_torch_tensor(value):
        return value.detach()
    return value


def scatter_branch(scatter_fraction, hit_mask, rng: NSQRng, ray_id, bounce):
    """Draw the BSDF scatter branch, and its compensating weight gate.

    Routes a ``scatter_fraction`` of the hit rays into the BSDF lobe and
    leaves the rest on the deterministic (specular or refracted) direction.
    The branch is drawn from a detached probability, matching the Fresnel
    split -- and, like it, carries a compensating attached weight so
    ``d(flux)/d(scatter_fraction)`` is correct rather than silently zero.

    Args:
        scatter_fraction: The surface's scatter fraction: a Python float, or
            a backend scalar that may carry a gradient.
        hit_mask: Per-ray mask of rays hitting this surface, shape (N,).
        rng: Keyed PCG32 RNG.
        ray_id: Per-ray identifiers, shape (N,).
        bounce: Per-ray bounce index as of this event, shape (N,).

    Returns:
        ``(scatters, sf_gate)``: the per-ray branch mask, and the per-ray
        attached weight to multiply into the flux of every hit ray.
    """
    sf = scatter_fraction
    sf_val = detached(sf)
    if be.is_torch_tensor(sf_val):
        sf_det = be.clip(sf_val, _SF_EPS, 1.0 - _SF_EPS)
    else:
        sf_det = min(max(float(sf_val), _SF_EPS), 1.0 - _SF_EPS)
    u_scatter = rng.uniform(ray_id, bounce, EventSlot.SCATTER_BRANCH)
    scatters = hit_mask & (u_scatter < sf_det)

    weight_scatter_branch = sf / sf_det
    weight_nonscatter_branch = (1.0 - sf) / (1.0 - sf_det)
    sf_gate = be.where(scatters, weight_scatter_branch, weight_nonscatter_branch)
    return scatters, sf_gate
