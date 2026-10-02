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
from optiland.nonsequential._utils import resident_scalar
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


def scatter_branch(
    scatter_fraction, hit_mask, rng: NSQRng, ray_id, bounce, owner=None
):
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
        owner: The surface drawing the branch. When given, two weights that
            are Python numbers are held on the device on it
            (:func:`~optiland.nonsequential._utils.resident_scalar`) instead
            of being uploaded at every call; the values are the same.

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
    if owner is not None and not be.is_torch_tensor(weight_scatter_branch):
        weight_scatter_branch = resident_scalar(
            owner, "scatter_weight", weight_scatter_branch, u_scatter
        )
        weight_nonscatter_branch = resident_scalar(
            owner, "scatter_weight", weight_nonscatter_branch, u_scatter
        )
    sf_gate = be.where(scatters, weight_scatter_branch, weight_nonscatter_branch)
    return scatters, sf_gate


def attachable_fraction(owner: str, name: str, value):
    """A reflect-or-transmit fraction: ``(the value kept, its host float)``.

    A fraction that carries a derivative (``requires_grad`` or a forward-mode
    tangent) is kept as given so the lobe's weight can attach to it
    (:func:`lobe_branch_gate`); any other value is read as a float. The
    research repository's chapter 09 section 9.14.3: the branch is drawn with
    the detached probability ``p`` (the host float) and each branch's weight
    carries its share ``tau / p`` or ``(1 - tau) / (1 - p)``. A fraction of
    exactly 0 or 1 never draws one of the two branches, so the estimator would
    have no sample for that branch's derivative: a gradient-carrying fraction
    of 0 or 1 is refused.

    Raises:
        NotImplementedError: For a gradient-carrying fraction of 0 or 1.
    """
    from optiland.nonsequential._utils import (  # noqa: PLC0415
        _carries_derivative,
        host_float,
    )

    host = host_float(value)
    if not _carries_derivative(value):
        return host, host
    if not 0.0 < host < 1.0:
        raise NotImplementedError(
            f"{owner}.{name} = {host} carries a gradient, but a fraction of exactly 0 "
            "or 1 never draws one of its two branches, so the derivative of the "
            "branch it never draws has no sample (chapter 09 section 9.14.3). Use a "
            "value strictly between 0 and 1, or a plain number."
        )
    return value, host


def lobe_branch_gate(fraction, host: float, transmitted, like):
    """The attached share of a lobe's reflect-or-transmit branch, or None.

    ``tau / p`` on the transmissive branch and ``(1 - tau) / (1 - p)`` on the
    reflective one, with ``p = host`` the detached probability the branch was
    drawn with and ``tau`` the fraction itself (chapter 09 sections 9.2 and
    9.14.3). Both are exactly 1 in value, so a weight multiplied by the gate
    keeps its bits; their derivatives are ``1 / p`` and ``-1 / (1 - p)``.

    Args:
        fraction: The fraction as kept by :func:`attachable_fraction`.
        host: Its host float.
        transmitted: The per-ray branch mask, shape (N,).
        like: The weights the gate multiplies, for the dtype.

    Returns:
        The gate, shape (N,), in ``like``'s dtype; None when ``fraction`` is a
        plain number (nothing to attach).
    """
    if not be.is_torch_tensor(fraction):
        return None
    gate = be.where(transmitted, fraction / host, (1.0 - fraction) / (1.0 - host))
    return gate.to(like.dtype)
