"""Dtype-aware tolerances for the non-sequential ray tracer.

Every numerical tolerance in ``optiland.nonsequential`` used to be a bare
literal (``1e-9``, ``1e-12``, ``1e-30``, ...) that did not consult the active
dtype or the coordinate magnitude it was being compared against. At float32,
several of those literals sit below the resolvable step of the numbers they
guard -- a self-intersection epsilon of ``1e-9`` mm is four orders of
magnitude below the float32 step at 50 mm -- so a ray immediately re-accepts
the surface it just left. At float64 the same absolute epsilon fails once the
scene is large enough (roughly 5e4 mm) for the same reason. See
``docs/theory/07_geometry.md`` section 7.7 and ``docs/theory/08_precision.md``
section 8.7 for the derivation.

The rule this module implements, in one sentence: **no tolerance is an
absolute length, angle, or denominator constant; every one is a small
multiple of the working dtype's own resolution at the magnitude in play.**

Three primitives:

- :func:`ulp` -- the spacing between adjacent representable numbers at a
  given magnitude, in the input's own dtype and backend (NumPy or Torch).
  This is the exact IEEE-754 "unit in the last place", via ``np.spacing`` or
  ``torch.nextafter``; it is *not* the cruder "dtype epsilon times the next
  power of two" shortcut that a pure-Python reference script uses when it
  cannot ask NumPy or Torch directly. The two agree to within a factor of 2
  (rounding to the exact binade vs. the next one up) -- either is a safe
  self-intersection guard at the scale this module defaults to,
  because both sit comfortably above the self-intersection residual they
  exist to reject, which is half an ulp of the ray's own coordinate once
  the hit point is built in the surface's frame (see
  :data:`DEFAULT_ACCEPT_K`). [measured]

- :func:`accept_t_min` -- the minimum accepted ray parameter ``t``, expressed
  as ``k`` ulps of a representative coordinate magnitude (with a 1.0 mm floor
  so a ray at the local origin still gets a non-zero threshold). Replaces
  every bare ``t > 1e-9`` / ``eps = 1e-9`` self-intersection guard.

- :func:`tiny_for` -- a division-guard epsilon that is safe for the backward
  pass: the square root of the dtype's smallest normal number, so squaring it
  (which reverse-mode autodiff does for a bare reciprocal's local derivative)
  does not underflow to zero and turn a ``where``'s discarded zero-cotangent
  branch into ``0 * inf = NaN``. See ``docs/theory/08_precision.md`` section
  8.8. Replaces every bare ``+ 1e-30`` additive division guard, and every
  bare ``1e-30``/``1e-14`` comparison threshold used to decide whether a
  denominator is degenerate.

And two helpers for a value about to be square-rooted: :func:`radicand_min`
(``k u_T``, below which a radicand is a domain error; the refraction radicand
uses it) and :func:`radicand_floor` (a float64-calibrated clamp scaled to the
dtype; the conic and sphere geometries use it).

Two exceptions, both intentional and both flagged where used: a physical
fraction (a Russian-roulette flux floor, a scatter-branch probability clamp)
is dtype-independent by definition and stays an absolute constant; and a
few construction-time geometry checks (e.g. a frustum's axial height) are
evaluated once, off the hot ray loop, in whatever precision their inputs
happen to carry, so they use :func:`ulp` directly with a Python-float floor
rather than backend dispatch.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from optiland.nonsequential._utils import is_tensor

if TYPE_CHECKING:
    from optiland._types import ScalarOrArrayT

# Default k for accept_t_min: docs/theory/07_geometry.md R-07-5's own
# default, inside its documented [8, 64].
#
# This was 16384 while the hit point was computed as p + t*d in global
# coordinates. A hit distance is one number of the size of the whole leg
# travelled, so it carries u*|t| of rounding, and the ray landed that far
# off the surface it had just hit -- 5.8e-11 mm in float64 after a 1e6 mm
# leg, eight thousand ulps of the 50 mm coordinate it landed at -- which
# only a threshold that large could reject. The hit point is now rebuilt in
# the surface's own frame from the advance and the residual separately
# (BaseComponent.advance_to_hit), which puts the ray back on the surface to
# half an ulp of its own coordinate: measured over the singlet's conic
# surfaces, max 3.60e-15 mm at a 55 mm coordinate in float64 (0.51 ulp) and
# max 2.00e-06 mm in float32 (0.52 ulp), at every source distance from 0 to
# 1e6 mm. [measured] k=16 leaves a factor of 30 above that floor. See
# docs/build/X1_threshold_arithmetic.md.
DEFAULT_ACCEPT_K = 16

# Coordinate-magnitude floor [mm] for accept_t_min: a ray whose local origin
# is at (or very near) the coordinate origin still needs a non-zero minimum
# accepted t, so the ulp is never taken at magnitude 0.
_MAGNITUDE_FLOOR = 1.0

# k_delta of docs/theory/07_geometry.md R-07-6 and section 7.7 (cure 2), in
# units of the unit roundoff u: the secondary ray's origin is offset off the
# surface by delta = k_delta * u * |p|_inf. Section 7.7 quotes k_delta = 32.
# One ulp spans 2u at the magnitude it is taken at, so the same bound written
# in ulps is delta = (k_delta / 2) * ulp(|p|_inf) = 16 ulp -- the same 16 the
# accept threshold uses, which is what makes the two complementary: the
# offset moves the new origin to the edge of the band the threshold rejects,
# so a ray leaving a surface is outside that band by construction and a ray
# still inside it is rejected. Raising one without the other would either
# blind the engine to a near surface (threshold) or displace the ray
# measurably (offset).
DEFAULT_OFFSET_K_DELTA = 32

# ulp(1) at float64 -- the reference radicand_floor() scales against.
_ULP1_F64 = 2.220446049250313e-16

# k of docs/theory/08_precision.md section 8.7's radicand row ("reject w < k u_T
# as a domain error", k = 4), in units of the unit roundoff u_T. Used for the
# refraction radicand w = 1 - sin^2(theta_t), an O(1) dimensionless quantity
# whose own resolution is u_T. Near the critical angle w is formed as
# 1 - r^2 (1 - c^2) (r = n1/n2, c = cos(theta_i)); both subtractions are exact
# there (Sterbenz), and the three products carry at most (1 + r^2) u_T of
# absolute error once r is given (3.3 u_T for N-BK7 against vacuum), or
# (3 + r^2) u_T if the index ratio itself is rounded (5.3 u_T). [derived] k = 4
# is the theory's integer and sits in that range: below it the sign of w, and
# so whether a transmitted wave exists at all, is not resolved by the working
# dtype. It does not cover the error cos(theta_i) brings in from the direction
# and the normal (2 r^2 c per unit of error in c, 3.5 u_T per ulp of c for
# N-BK7), which is the incidence angle's own uncertainty, not the formula's.
DEFAULT_RADICAND_K = 4

# The Fresnel split's branch-probability clamp (docs/theory/08_precision.md
# section 8.7, the branch-probability row, and 09_differentiation.md R-09-1:
# p in [k u, 1 - k u], k = 4). The engine's clamp was the float64 literal pair
# [1e-12, 1 - 1e-12], and at float64 it stays exactly that: 1e-12 is looser
# than 4 u = 4.4e-16 there, so keeping it moves no float64 number. At float32
# the upper literal is exactly 1.0 (1 - 1e-12 rounds up), so a probability at
# the clamp could not leave the reflect branch except on the one draw in
# ~3e7 that itself rounds to 1.0 -- and then took the transmit branch with a
# weight of T / tiny_for, about 9e18 T (issue 59 of the research repository).
# The upper bound is therefore the smaller of the literal and 1 - k u, which
# is exact in the working dtype (1 - 2**-22 at float32). The lower literal is
# kept in every dtype: 1e-12 is a normal float32 number, well away from 0,
# and the reflect weight R / p it bounds is finite with it.
DEFAULT_BRANCH_K = 4
_BRANCH_P_LITERAL = 1e-12


def ulp(x: ScalarOrArrayT) -> ScalarOrArrayT:
    """Spacing between adjacent representable numbers at magnitude ``|x|``.

    Dispatches on backend so the result is in ``x``'s own dtype: a NumPy
    array or scalar uses :func:`numpy.spacing`; a Torch tensor uses
    :func:`torch.nextafter` toward positive infinity, which is the exact
    successor distance in that tensor's dtype (and its device, and detached
    from any autograd graph on ``x`` -- this is a constant, not a quantity to
    differentiate through).

    Inside a ``torch.compile`` region (the torch backend's compiled bounce
    step) the successor is formed by adding one to the bit pattern of ``|x|``
    read as an integer of the same width, because the inductor's Apple GPU
    code generator has no ``nextafter``. For a finite non-negative float the
    two are the same number, bit for bit, on the CPU and on the Apple GPU
    (measured on 100,006 float32 magnitudes from 0 to 3.4e38, and on the
    same set at float64 on the CPU).

    Args:
        x: Coordinate magnitude(s). A plain Python float is treated as a
            NumPy float64 scalar (Python has no other float width).

    Returns:
        The ulp at ``|x|``, same backend and dtype as ``x`` (float64 for a
        plain Python float or int).
    """
    if is_tensor(x):
        import torch  # noqa: PLC0415

        ax = torch.abs(x).detach()
        if torch.compiler.is_compiling():
            same_width = {torch.float32: torch.int32, torch.float64: torch.int64}
            if ax.dtype in same_width:
                return (ax.view(same_width[ax.dtype]) + 1).view(ax.dtype) - ax
        return torch.nextafter(ax, torch.full_like(ax, float("inf"))) - ax
    return np.spacing(np.abs(np.asarray(x)))


def accept_t_min(
    origin_magnitude: ScalarOrArrayT, k: int = DEFAULT_ACCEPT_K
) -> ScalarOrArrayT:
    """Minimum accepted ray parameter ``t``, ``k`` ulps of a coordinate scale.

    See docs/theory/07_geometry.md section 7.7 (cure 1) and
    docs/theory/08_precision.md section 8.7. A ray is accepted as hitting a
    surface only when its solved parameter exceeds this value; below it, the
    hit cannot be distinguished, in the working dtype, from the surface the
    ray has just left.

    Args:
        origin_magnitude: A representative coordinate magnitude (e.g. the
            infinity norm of the local-frame ray origins the intersection is
            about to be solved for) in the working backend/dtype. A plain
            Python float is accepted too (treated as float64). Floored at
            1.0 (in whatever units ``origin_magnitude`` carries -- mm
            throughout this package) so a ray already at the local origin
            still gets a non-zero threshold.
        k: Multiple of the local ulp. Default
            :data:`DEFAULT_ACCEPT_K`, inside the documented [8, 64] range.

    Returns:
        The threshold, same backend/dtype as ``origin_magnitude``.
    """
    if is_tensor(origin_magnitude):
        import torch  # noqa: PLC0415

        mag = torch.clamp(torch.abs(origin_magnitude).detach(), min=_MAGNITUDE_FLOOR)
    else:
        mag = np.maximum(np.abs(np.asarray(origin_magnitude)), _MAGNITUDE_FLOOR)
    return k * ulp(mag)


def origin_offset(
    origin_magnitude: ScalarOrArrayT, k_delta: int = DEFAULT_OFFSET_K_DELTA
) -> ScalarOrArrayT:
    """Distance to push a secondary ray's origin off the surface it leaves.

    Cure 2 of docs/theory/07_geometry.md section 7.7, required by R-07-6:
    ``o' = p +/- delta * n_geom`` with ``delta = k_delta * u * |p|_inf``, the
    sign taken from the hemisphere the outgoing ray leaves into. It is the
    complement of :func:`accept_t_min`, not a substitute: the threshold
    rejects a root that is still inside the position's error ball, while the
    offset moves the origin out of that ball so a root never forms there in
    the first place.

    The threshold alone is not enough for a ray that leaves at a grazing
    angle. The rebuilt hit point can land half an ulp on the wrong side of
    the surface, and a ray leaving at ``alpha`` from the surface plane
    re-crosses it after ``delta_perp / sin(alpha)`` -- a path length the
    threshold sees, not the perpendicular error it was sized for. At one
    millidegree inside the critical angle of an N-BK7/air interface the exit
    is 0.36 degrees from grazing, which multiplies the residual by 159 and
    lifts it above ``k`` ulps. The offset removes the amplification instead
    of chasing it with a larger ``k``.

    ``delta_shape``, the per-primitive term for the error of the
    intersection itself, is zero here: the hit point is rebuilt in the
    surface's own frame (:meth:`BaseComponent.advance_to_hit`), which was
    measured at at most 0.4 ulp of the ray's coordinate over every leg
    length from 0 to 1e6 mm, so the coordinate term already bounds it.

    Args:
        origin_magnitude: A representative coordinate magnitude of the hit
            point, per ray, in the working backend/dtype. Floored at 1.0 mm
            like :func:`accept_t_min`, so a hit at the coordinate origin
            still gets a non-zero offset.
        k_delta: Multiple of the unit roundoff. Default
            :data:`DEFAULT_OFFSET_K_DELTA`.

    Returns:
        The offset distance [mm], same backend/dtype as the input, always
        non-negative -- the caller applies the sign.
    """
    if is_tensor(origin_magnitude):
        import torch  # noqa: PLC0415

        mag = torch.clamp(torch.abs(origin_magnitude).detach(), min=_MAGNITUDE_FLOOR)
    else:
        mag = np.maximum(np.abs(np.asarray(origin_magnitude)), _MAGNITUDE_FLOOR)
    return (k_delta / 2.0) * ulp(mag)


def unit_roundoff(like: ScalarOrArrayT) -> ScalarOrArrayT:
    """Unit roundoff ``u_T = ulp(1) / 2`` of ``like``'s dtype, in its backend.

    A 0-d array (NumPy) or 0-d tensor (Torch, on ``like``'s device), so it
    compares against arrays of the same dtype without a host round trip.
    ``2^-24`` at float32 and ``2^-53`` at float64 (docs/theory/08_precision.md
    table 8.1).

    Args:
        like: Any array, tensor or Python float in the working dtype; only
            its dtype (and device) are read.

    Returns:
        ``u_T`` as a 0-d array or tensor in ``like``'s dtype.
    """
    if is_tensor(like):
        import torch  # noqa: PLC0415

        one = torch.ones((), dtype=like.dtype, device=like.device)
    else:
        one = np.ones((), dtype=np.asarray(like).dtype)
    return 0.5 * ulp(one)


def branch_probability_bounds(
    like: Any, k: int = DEFAULT_BRANCH_K
) -> tuple[float, float]:
    """The interval a detached branch probability is clamped to, for ``like``'s dtype.

    ``[1e-12, min(1 - 1e-12, 1 - k u_T)]``: at float64 exactly the literal
    pair the engine always used, at float32 ``[1e-12, 1 - 2**-22]``, so that
    ``1 - p`` stays a representable, non-zero number and the transmit weight
    ``T / (1 - p)`` is bounded by ``T / (k u_T)`` (see
    :data:`DEFAULT_BRANCH_K` for the derivation). Read from the dtype's
    machine limits, so nothing is read from a device.

    Args:
        like: The probability (an array or tensor in the working dtype), or
            anything of that dtype.
        k: Multiple of the unit roundoff kept between ``p`` and 1.

    Returns:
        ``(lo, hi)`` as Python floats; ``hi`` is exact in ``like``'s dtype.
    """
    if is_tensor(like):
        import torch  # noqa: PLC0415

        eps = float(torch.finfo(like.dtype).eps)
    else:
        eps = float(np.finfo(np.asarray(like).dtype).eps)
    u = 0.5 * eps
    return _BRANCH_P_LITERAL, min(1.0 - _BRANCH_P_LITERAL, 1.0 - k * u)


def radicand_min(like: ScalarOrArrayT, k: int = DEFAULT_RADICAND_K) -> ScalarOrArrayT:
    """Smallest radicand the working dtype resolves as positive, ``k * u_T``.

    docs/theory/08_precision.md section 8.7, the radicand row: a value about
    to be square-rooted that is below ``k u_T`` is a domain error, not a
    number to clamp and carry on with. For the refraction radicand
    ``w = 1 - sin^2(theta_t)`` the domain error is total internal
    reflection: no transmitted wave is asserted where the dtype cannot tell
    ``w`` from zero (see ``components.refractive.refraction_cosine``).

    Unlike :func:`radicand_floor`, this does not change the value of any
    radicand the dtype resolves: it only decides which ones are not there.

    Args:
        like: The radicand (or anything in its dtype and on its device).
        k: Multiple of the unit roundoff. Default
            :data:`DEFAULT_RADICAND_K`.

    Returns:
        ``k * u_T`` as a 0-d array or tensor in ``like``'s dtype:
        ``2.38e-7`` at float32, ``4.44e-16`` at float64.
    """
    return k * unit_roundoff(like)


def tiny_for(dtype_or_array: Any) -> float:
    """Division-guard epsilon safe for the backward pass, for ``dtype``.

    Returns ``sqrt(smallest_normal)`` of the given dtype. See
    docs/theory/08_precision.md section 8.8: a guarded reciprocal
    ``1 / (2*a + eps)`` forms ``eps**2`` in its backward pass (the local
    derivative of ``1/x`` is ``-1/x**2``); if ``eps`` is much smaller than
    this bound, ``eps**2`` underflows to exactly zero, the reciprocal's local
    derivative becomes signed infinity, and a ``where`` multiplying that by a
    discarded zero cotangent produces ``0 * inf = NaN`` that then propagates
    through the whole reverse sweep -- even though the forward pass was
    masked correctly. ``eps >= sqrt(x_min_normal)`` keeps ``eps**2`` at or
    above the smallest normal, so the local derivative stays finite and the
    zero cotangent actually zeroes it out.

    Args:
        dtype_or_array: A NumPy array/dtype, a Torch tensor/dtype, or a
            string/``numpy.dtype`` NumPy accepts (``"float32"``, ...). An
            array or tensor has its own ``.dtype`` used automatically.

    Returns:
        A plain Python float: ``1.0842e-19`` for float32, ``1.4917e-154``
        for float64.
    """
    if is_tensor(dtype_or_array):
        import torch  # noqa: PLC0415

        return float(torch.finfo(dtype_or_array.dtype).tiny) ** 0.5
    try:
        import torch  # noqa: PLC0415

        if isinstance(dtype_or_array, torch.dtype):
            return float(torch.finfo(dtype_or_array).tiny) ** 0.5
    except ImportError:
        pass
    # A NumPy array uses its own dtype; anything else (a numpy dtype object,
    # a scalar type like np.float32, or a string like "float32") is handed
    # to np.dtype() directly -- np.float32.dtype (the class, not an instance)
    # is an unbound attribute descriptor, not a dtype, so arrays must be
    # special-cased rather than blindly read through getattr(..., "dtype").
    dt = dtype_or_array.dtype if isinstance(dtype_or_array, np.ndarray) else dtype_or_array
    return float(np.finfo(np.dtype(dt)).tiny) ** 0.5


def radicand_floor(ones_like_x: ScalarOrArrayT, floor_at_f64: float = 1e-12) -> ScalarOrArrayT:
    """Floor for a value about to be square-rooted, calibrated at float64.

    A secondary helper alongside the three primitives above, for the one
    tolerance shape that is not simply "k ulps of a coordinate": a handful of
    radicand clamps (``sqrt(max(w, floor))``) whose floor was originally a
    bare ``1e-12`` chosen so a specific forward quantity (a lens-edge sag)
    moves by no more than ``sqrt(1e-12) = 1`` micron. That budget is a design
    choice, not a resolution limit, so it should not simply shrink to the
    working dtype's own ulp(1) (replacing it with ``k * ulp(1)`` for a small
    k changes the regularization strength and, empirically, the traced
    result -- see docs/build/W4_tolerances.md). Instead this scales the
    *same* calibrated budget by the working dtype's own coarseness relative
    to float64, so it is unchanged at float64 (the ratio is exactly 1.0) and
    grows at lower precision, where the un-scaled 1e-12 floor is already
    below the dtype's own resolution at magnitude 1 (float32 epsilon is
    1.19e-7) and so acts as if it were zero -- the sqrt(0) singularity it
    exists to avoid still reaches the backward pass.

    Not for a radicand whose *value* is the physics. At float32 the scaled
    floor is 5.37e-4, which is 9000 u_T: the refraction radicand
    ``w = 1 - sin^2(theta_t)`` used to be clamped with it, so ``cos(theta_t)``
    never fell below 0.0232 and every refraction within 1.33 degrees of
    grazing was bent and weighted as if it were 1.33 degrees from grazing --
    measured one millidegree inside the critical angle of N-BK7 against
    vacuum, a transmittance of 0.124 where float64 reads 0.0373 on the same
    rays. That radicand now uses :func:`radicand_min` (a domain test that
    leaves every resolved value alone); the geometry clamps below still use
    this floor.

    Args:
        ones_like_x: An array of ones in the working backend/dtype (e.g.
            ``be.ones_like(w)``) -- only its dtype matters.
        floor_at_f64: The calibrated float64 floor this reduces to.

    Returns:
        ``floor_at_f64`` at float64; scaled up at lower precision.
    """
    return floor_at_f64 * (ulp(ones_like_x) / _ULP1_F64)
