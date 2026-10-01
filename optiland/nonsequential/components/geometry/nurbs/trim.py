"""Trim loops of the NURBS kind on the host: the library's polygons, per Bezier piece.

KronosNSRT issue 66, ticket C (device trimming). The array contract carries each
patch's trim loops as 2-D NURBS curves in ``(u, v)``. The family's geometry
library classifies a parameter point against them (``kgeom.nurbs.point_in_trim``)
by the even-odd rule on polygons that follow each curve to a stated tolerance:
``TRIM_TOLERANCE`` (1e-6) times the diagonal of the patch's ``uv_bounds``, the
largest distance a chord may fall from the curve. The library promises nothing
closer to a curve than that tolerance (KronosLIB issue 286).

This module builds the same polygons from the contract, with the library's
operations in the library's order (Piegl and Tiller A2.1 and A2.2 for the
basis, its chord-halving polyline), so the engine and ``point_in_trim`` agree to
the bit at float64 without the engine importing the library; the kind's tests
compare the two. It then cuts the polygon edges per Bezier piece: a lane's hit
lies in its leaf's piece, and only the edges a horizontal ray from a point of
that piece can cross are kept (:func:`piece_edges`), so the device tests a few
edges per lane instead of every edge of the patch.

The band. Inside the tolerance band around a trim curve the engine answers what
the polygon says (the library's answer, to the bit at float64); the true curve
may say otherwise. Between a chord and its arc the band is at most ``tol`` wide
in ``(u, v)``, so the surface area it covers is at most ``tol`` times the loop's
length in ``(u, v)`` times the largest ``|S_u x S_v|`` along it, and the flux the
engine can assign to the wrong side of a trim curve is at most that area times
the largest irradiance on it. On the device the polygon and the hit's ``(u, v)``
are rounded to the working dtype; at float32 that widens the band by a few
units of float32 at the parameters' scale (about 1e-7 for a unit domain).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

#: The polygonisation tolerance as a fraction of the ``uv_bounds`` diagonal (the
#: library's ``kgeom.nurbs.TRIM_TOLERANCE``).
TRIM_TOLERANCE = 1e-6

#: Slack of the edge filters, relative to the patch's parameter scale. A lane's
#: ``(u, v)`` is clipped to its piece, but rounding (float32 on the device, and
#: the slab index computed in the working dtype) may leave it a few units of
#: the dtype outside; an edge is kept unless it is farther than this from the
#: piece or slab. Keeping an edge that cannot be crossed adds nothing to the
#: parity, so the slack costs time, never correctness.
_FILTER_SLACK = 1e-6

#: Most edges a slab should hold on average; the slab count of a set is chosen
#: from its largest piece (at most :data:`MAX_SLABS`).
_EDGES_PER_SLAB = 4
MAX_SLABS = 256


# ---------------------------------------------------------------------------
# The library's curve evaluation and polyline (same operations, same order)
# ---------------------------------------------------------------------------


def _find_span(knots: np.ndarray, degree: int, n_ctrl: int, t: np.ndarray) -> np.ndarray:
    span = np.searchsorted(knots, t, side="right") - 1
    span = np.clip(span, degree, n_ctrl - 1)
    empty = knots[span] >= knots[span + 1]
    while np.any(empty):
        span = np.where(empty, span - 1, span)
        empty = (knots[span] >= knots[span + 1]) & (span > degree)
    return span.astype(np.int64)


def _basis(knots: np.ndarray, degree: int, span: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The ``degree + 1`` non-zero basis functions at each ``t`` (A2.2, the ``ndu`` table)."""
    p = int(degree)
    n = t.shape[0]
    ndu = np.zeros((n, p + 1, p + 1))
    ndu[:, 0, 0] = 1.0
    left = np.zeros((n, p + 1))
    right = np.zeros((n, p + 1))
    for j in range(1, p + 1):
        left[:, j] = t - knots[span + 1 - j]
        right[:, j] = knots[span + j] - t
        saved = np.zeros(n)
        for r in range(j):
            ndu[:, j, r] = right[:, r + 1] + left[:, j - r]
            temp = ndu[:, r, j - 1] / ndu[:, j, r]
            ndu[:, r, j] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        ndu[:, j, j] = saved
    return ndu[:, :, p].copy()


class TrimCurve:
    """One trim curve of the contract: degree, knots, ``(u, v)`` points, weights."""

    def __init__(self, degree: int, knots, points, weights) -> None:
        self.degree = int(degree)
        self.knots = np.asarray(knots, dtype=np.float64).reshape(-1)
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        n = self.points.shape[0]
        if self.degree < 1 or n < self.degree + 1 or self.knots.size != n + self.degree + 1:
            raise ValueError("a trim curve needs degree >= 1, degree + 1 points and n + degree + 1 knots")
        if np.any(self.weights <= 0) or not np.all(np.isfinite(self.points)):
            raise ValueError("a trim curve needs finite points and strictly positive weights")

    @property
    def domain(self) -> tuple[float, float]:
        return float(self.knots[self.degree]), float(self.knots[self.points.shape[0]])

    def evaluate(self, t) -> np.ndarray:
        """``(k, 2)`` points at parameters ``t`` (clipped to the domain within 1e-12, as the library)."""
        tf = np.asarray(t, dtype=np.float64).reshape(-1)
        a, b = self.domain
        slack = 1e-12 * max(b - a, abs(a), abs(b), 1.0)
        if np.any((tf < a - slack) | (tf > b + slack) | ~np.isfinite(tf)):
            raise ValueError("a trim curve parameter lies outside its domain")
        tf = np.clip(tf, a, b)
        n = self.points.shape[0]
        span = _find_span(self.knots, self.degree, n, tf)
        N = _basis(self.knots, self.degree, span, tf)
        idx = span[:, None] - self.degree + np.arange(self.degree + 1)[None, :]
        w = self.weights[idx]
        num = np.einsum("nk,nkc->nc", N * w, self.points[idx])
        den = np.sum(N * w, axis=1, keepdims=True)
        return num / den


def _point_segment_distance(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.sum(ab * ab, axis=-1)
    s = np.where(denom > 0, np.sum((x - a) * ab, axis=-1) / np.where(denom > 0, denom, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    return np.linalg.norm(x - (a + s[:, None] * ab), axis=-1)


def curve_polyline(curve: TrimCurve, tol: float, max_rounds: int = 30) -> np.ndarray:
    """``(k, 2)`` points along the curve whose chords pass within ``tol`` of the
    curve's point at each chord's middle parameter (the library's rule: four
    chords per knot span to start, every failing chord halved)."""
    a, b = curve.domain
    knots = np.unique(curve.knots[(curve.knots >= a) & (curve.knots <= b)])
    t = np.unique(np.concatenate([np.linspace(knots[i], knots[i + 1], 5) for i in range(knots.size - 1)]))
    for _ in range(int(max_rounds)):
        pts = curve.evaluate(t)
        tm = 0.5 * (t[:-1] + t[1:])
        mid = curve.evaluate(tm)
        d = _point_segment_distance(mid, pts[:-1], pts[1:])
        bad = d > tol
        if not np.any(bad):
            return pts
        t = np.sort(np.concatenate([t, tm[bad]]))
    return curve.evaluate(t)


# ---------------------------------------------------------------------------
# The contract's loops
# ---------------------------------------------------------------------------


def patch_loops(arrays: Mapping[str, Any], i: int) -> list[tuple[list[TrimCurve], bool]]:
    """Patch ``i``'s loops from the flattened contract: ``[(curves, outer), ...]``."""
    out = []
    lo, hi = int(arrays["patch_loop_offset"][i]), int(arrays["patch_loop_offset"][i + 1])
    for L in range(lo, hi):
        c0, c1 = int(arrays["loop_curve_offset"][L]), int(arrays["loop_curve_offset"][L + 1])
        curves = []
        for C in range(c0, c1):
            k0, k1 = int(arrays["curve_ctrl_offset"][C]), int(arrays["curve_ctrl_offset"][C + 1])
            n0, n1 = int(arrays["curve_knot_offset"][C]), int(arrays["curve_knot_offset"][C + 1])
            curves.append(TrimCurve(
                int(arrays["curve_degree"][C]), np.asarray(arrays["curve_knots"][n0:n1]),
                np.asarray(arrays["curve_ctrl_points"][k0:k1]), np.asarray(arrays["curve_weights"][k0:k1]),
            ))
        out.append((curves, bool(arrays["loop_outer"][L])))
    return out


def trim_tolerance(uv_bounds) -> float:
    """The library's default polygonisation tolerance for a patch's ``uv_bounds``."""
    u0, u1, v0, v1 = (float(x) for x in uv_bounds)
    return TRIM_TOLERANCE * math.hypot(u1 - u0, v1 - v0)


def trim_polygons(arrays: Mapping[str, Any], i: int, uv_bounds, tol: float | None = None):
    """Each loop of patch ``i`` as a closed polygon ``(k, 2)``, last point equal to
    the first, with its ``outer`` flag: ``kgeom.nurbs.trim_polygons`` from the contract."""
    if tol is None:
        tol = trim_tolerance(uv_bounds)
    out = []
    for curves, outer in patch_loops(arrays, i):
        parts = [curve_polyline(c, tol) for c in curves]
        pts = np.concatenate([parts[0]] + [q[1:] for q in parts[1:]], axis=0)
        pts = np.concatenate([pts, pts[:1]], axis=0)
        out.append((pts, outer))
    return out


def polygon_edges(polygons) -> np.ndarray:
    """``(E, 4)`` edges ``(a_u, a_v, b_u, b_v)`` of every polygon, in loop order."""
    if not polygons:
        return np.zeros((0, 4))
    return np.concatenate([np.concatenate([p[:-1], p[1:]], axis=1) for p, _ in polygons], axis=0)


def even_odd(edges: np.ndarray, u, v) -> np.ndarray:
    """The library's crossing rule on edges ``(E, 4)``: a ray toward ``+u`` counts
    the edges with one end strictly above ``v`` and the other not, crossed beyond
    ``u``. Host float64; the device kernel applies the same expression."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    count = np.zeros(u.shape[0], dtype=np.int64)
    if edges.shape[0] == 0:
        return count % 2 == 1
    au, av, bu, bv = (edges[None, :, k] for k in range(4))
    pu, pv = u[:, None], v[:, None]
    straddle = (av > pv) != (bv > pv)
    with np.errstate(invalid="ignore", divide="ignore"):
        cross_u = au + (pv - av) * (bu - au) / (bv - av)
    count += np.sum(straddle & (pu < cross_u), axis=1)
    return count % 2 == 1


def box_meets_edges(edges: np.ndarray, box, margin: float) -> bool:
    """Whether any edge meets the box ``(u0, u1, v0, v1)`` grown by ``margin``.

    Exact for a segment against an axis-aligned rectangle: the segment's own box
    overlaps the rectangle and the rectangle's corners do not all lie strictly on
    one side of the segment's line (the separating axes are u, v and the
    segment's normal).
    """
    if edges.shape[0] == 0:
        return False
    u0, u1, v0, v1 = box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin
    au, av, bu, bv = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]
    overlap = (np.maximum(au, bu) >= u0) & (np.minimum(au, bu) <= u1) & (np.maximum(av, bv) >= v0) & (
        np.minimum(av, bv) <= v1
    )
    if not np.any(overlap):
        return False
    du, dv = bu - au, bv - av
    side = [du * (cv - av) - dv * (cu - au) for cu, cv in ((u0, v0), (u1, v0), (u0, v1), (u1, v1))]
    side = np.stack(side, 1)
    one_side = np.all(side > 0, axis=1) | np.all(side < 0, axis=1)
    return bool(np.any(overlap & ~one_side))


def slab_edges(edges: np.ndarray, v_lo: float, v_hi: float, n_slabs: int, scale: float) -> list[np.ndarray]:
    """The piece's edges cut into ``n_slabs`` equal slabs of ``v`` over ``[v_lo, v_hi]``.

    A point in a slab can only cross an edge whose ``v``-range meets the slab,
    so the parity over a slab's edges equals the parity over the piece's. The
    device finds a lane's slab from its ``v`` and tests that slab only.
    """
    m = _FILTER_SLACK * max(scale, 1.0)
    h = (v_hi - v_lo) / n_slabs
    lo_e = np.minimum(edges[:, 1], edges[:, 3])
    hi_e = np.maximum(edges[:, 1], edges[:, 3])
    out = []
    for k in range(n_slabs):
        lo, hi = v_lo + k * h, v_lo + (k + 1) * h
        out.append(edges[(hi_e >= lo - m) & (lo_e <= hi + m)])
    return out


def piece_edges(edges: np.ndarray, piece_uv, scale: float) -> np.ndarray:
    """The edges a horizontal ray (toward ``+u``) from a point of the piece
    ``(u0, u1, v0, v1)`` can count.

    An edge counts for a point ``(u, v)`` only if it straddles ``v`` (so its
    ``v``-range meets the piece's) and its crossing lies beyond ``u`` (so its
    largest ``u`` is not left of the piece). The rest contribute nothing to any
    point of the piece, and the parity over the kept edges equals the parity
    over all of them; the slack keeps that true for a point an ulp outside.
    """
    if edges.shape[0] == 0:
        return edges
    m = _FILTER_SLACK * max(scale, 1.0)
    u0, _u1, v0, v1 = (float(x) for x in piece_uv)
    keep = (
        (np.maximum(edges[:, 1], edges[:, 3]) >= v0 - m)
        & (np.minimum(edges[:, 1], edges[:, 3]) <= v1 + m)
        & (np.maximum(edges[:, 0], edges[:, 2]) >= u0 - m)
    )
    return edges[keep]
