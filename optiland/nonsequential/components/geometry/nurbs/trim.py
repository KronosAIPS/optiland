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

The band (KronosNSRT issue 107). Between a chord and its arc the polygon and the
true curve disagree; where the chord lies beyond the curve (a hole's inscribed
polygon, a D-cut cap's chords on the cut side) the polygon keeps a sliver of
surface the solid does not have. A ray entering a closed CAD solid through that
sliver is carried as inside glass while it is outside the solid, meets the cut
face from outside at grazing incidence and is refracted into the glass at the
critical angle, where it is trapped (case r1_41 of the research repository lost
4, 1 and 2 of 1e5 rays at three seeds that way). The rule that closes the solid
needs nothing beyond the contract: every edge of a curve that runs inside the
patch's rectangle carries a band, a bound on how far its curve lies from it
(:func:`chord_bands`: the departure sampled at 15 interior points plus the
interpolation term, at most about ``tol``), and a point within its edge's band
of any edge is on the trim curve as far as the polygon can tell and is refused
(given to the adjacent face, the cut). The true curve lies inside the union of
the bands, so no kept point lies beyond it: the faces can leave a gap of at
most twice the band at a cut, never an overlap. Edges of curves along the
rectangle (seams, poles, rims) carry no band; the rectangle is tested exactly
by the patch's ``uv_bounds``. On the device the band grows by
``BAND_ROUNDING_K`` units of the working dtype at the patch's parameter scale
(the polygon's and the hit's ``(u, v)`` rounding).

Outside the bands the engine answers what the library's polygon answers (to the
bit at float64). The surface area of the bands is at most twice the loop's band
times its length in ``(u, v)`` times the largest ``|S_u x S_v|`` along it, and the
flux the engine can assign to the cut instead of the face is at most that area
times the largest irradiance on it.
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


def curve_polyline(curve: TrimCurve, tol: float, max_rounds: int = 30, return_params: bool = False):
    """``(k, 2)`` points along the curve whose chords pass within ``tol`` of the
    curve's point at each chord's middle parameter (the library's rule: four
    chords per knot span to start, every failing chord halved). With
    ``return_params`` also the ``(k,)`` curve parameters of the points (the
    points are the same)."""
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
            return (pts, t) if return_params else pts
        t = np.sort(np.concatenate([t, tm[bad]]))
    pts = curve.evaluate(t)
    return (pts, t) if return_params else pts


# ---------------------------------------------------------------------------
# The band about a trim curve (KronosNSRT issue 107)
# ---------------------------------------------------------------------------

#: Interior samples per chord for the band's measure.
BAND_SAMPLES = 16

#: Units of the working dtype, at the patch's parameter scale, added to an
#: interior edge's band on the device: the rounding of the polygon's vertices
#: on upload (``sqrt(2) u S``), of the hit's ``(u, v)`` formed from its leaf's
#: range (about ``7 u S`` per coordinate: the range's two ends rounded, their
#: difference, the product with ``s`` up to 1.25, the sum; ``7 sqrt(2) u S``),
#: and of the point-to-edge distance's two differences (``2 u S``): 13.3,
#: rounded up to 16.
BAND_ROUNDING_K = 16.0


def _on_rectangle(curve: TrimCurve, rect) -> bool:
    """Whether every control point of the curve lies on one side line of the
    rectangle ``(u0, u1, v0, v1)`` (the curve is then that side: a NURBS curve
    lies in its control polygon's hull). The same rule as the leaf build's
    test of loops along the domain."""
    u0, u1, v0, v1 = (float(x) for x in rect)
    tol = 1e-12 * max(u1 - u0, v1 - v0, abs(u0), abs(u1), abs(v0), abs(v1), 1.0)
    p = curve.points
    return bool(
        np.all(np.abs(p[:, 0] - u0) <= tol)
        or np.all(np.abs(p[:, 0] - u1) <= tol)
        or np.all(np.abs(p[:, 1] - v0) <= tol)
        or np.all(np.abs(p[:, 1] - v1) <= tol)
    )


def chord_bands(curve: TrimCurve, pts: np.ndarray, t: np.ndarray, m: int = BAND_SAMPLES) -> np.ndarray:
    """``(k - 1,)`` a bound on how far the curve departs from each chord of its polyline.

    For the chord from ``c(t_j)`` to ``c(t_{j+1})`` the departure is
    ``f(t) = n . (c(t) - c(t_j))`` with ``n`` the chord's unit normal, zero at
    both ends. It is sampled at ``m - 1`` equally spaced interior parameters;
    between two samples spaced ``h`` the function exceeds the larger sample by at
    most ``h^2 max|f''| / 8``, and ``h^2 |f''|`` is the samples' second
    difference to first order, so the bound is the largest ``|f|`` sampled plus
    a quarter of the largest second difference (twice the interpolation term,
    for the change of ``f''`` between samples), plus the host's rounding of the
    evaluation, ``8 u_h`` at the curve's coordinate scale.
    """
    k = pts.shape[0] - 1
    if k <= 0:
        return np.zeros(0)
    frac = np.linspace(0.0, 1.0, m + 1)
    ts = t[:-1, None] + (t[1:] - t[:-1])[:, None] * frac[None, :]  # (k, m + 1)
    ts[:, 0], ts[:, -1] = t[:-1], t[1:]
    c = curve.evaluate(ts.reshape(-1)).reshape(k, m + 1, 2)
    a, b = pts[:-1], pts[1:]
    e = b - a
    ln = np.linalg.norm(e, axis=1)
    nrm = np.stack([-e[:, 1], e[:, 0]], axis=1) / np.where(ln > 0, ln, 1.0)[:, None]
    f = np.einsum("kmc,kc->km", c - a[:, None, :], nrm)
    # a zero-length chord: the departure is the distance from its point
    f = np.where((ln > 0)[:, None], f, np.linalg.norm(c - a[:, None, :], axis=2))
    f[:, 0] = 0.0
    f[:, -1] = 0.0
    d2 = np.abs(f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]).max(axis=1) if m >= 2 else np.zeros(k)
    scale = max(float(np.abs(curve.points).max()), 1.0)
    return np.abs(f).max(axis=1) + 0.25 * d2 + 8.0 * 2.0**-53 * scale


def trim_polygons_and_bands(arrays: Mapping[str, Any], i: int, uv_bounds, tol: float | None = None):
    """:func:`trim_polygons` (the same polygons, bit for bit) and, per edge in
    :func:`polygon_edges`' order, the band about it on the host.

    The band of an edge is how far the trim curve can lie from it
    (:func:`chord_bands`), ``-1`` for an edge of a curve that runs along the
    ``uv_bounds`` rectangle (a seam, a pole, a rim: the rectangle itself, which
    the device tests exactly and which no adjacent face's edge needs to meet).
    Where two curves of a loop meet, the edge into an interior curve also
    carries the gap between the previous curve's end and its start; the closing
    edge's band is its own length when a curve beside it is interior.
    """
    if tol is None:
        tol = trim_tolerance(uv_bounds)
    polys, bands = [], []
    for curves, outer in patch_loops(arrays, i):
        parts, pb = [], []
        for c in curves:
            pts, t = curve_polyline(c, tol, return_params=True)
            parts.append(pts)
            pb.append(np.full(pts.shape[0] - 1, -1.0) if _on_rectangle(c, uv_bounds) else chord_bands(c, pts, t))
        pts = np.concatenate([parts[0]] + [q[1:] for q in parts[1:]], axis=0)
        pts = np.concatenate([pts, pts[:1]], axis=0)
        interior = [bool(b.size and b.max() >= 0.0) for b in pb]
        band = [pb[0]]
        for j in range(1, len(parts)):
            b = pb[j].copy()
            gap = float(np.linalg.norm(parts[j - 1][-1] - parts[j][0]))
            if b.size and interior[j]:
                b[0] = max(b[0], 0.0) + gap
            band.append(b)
        # the closing edge, from the last curve's end to the first curve's start
        gap = float(np.linalg.norm(parts[-1][-1] - parts[0][0]))
        band.append(np.array([gap if (interior[-1] or interior[0]) else -1.0]))
        polys.append((pts, outer))
        bands.append(np.concatenate(band))
    return polys, (np.concatenate(bands) if bands else np.zeros(0))


def band_filter_margin(bands: np.ndarray, scale: float) -> float:
    """The edge filters' margin that keeps every edge whose band can reach a
    point of a piece or slab, at either working dtype (the band on the device is
    at most the host band plus ``BAND_ROUNDING_K`` units of float32)."""
    b = float(bands.max()) if bands.size else -1.0
    if b < 0.0:
        return 0.0
    return b + BAND_ROUNDING_K * 2.0**-24 * max(scale, 1.0)


def in_band(edges: np.ndarray, bands: np.ndarray, u, v) -> np.ndarray:
    """Host float64: whether each point lies within its band of any edge
    (a band ``< 0`` never holds). The device kernel applies the same expression."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if edges.shape[0] == 0:
        return np.zeros(u.shape[0], dtype=bool)
    au, av, bu, bv = (edges[None, :, k] for k in range(4))
    pu, pv = u[:, None], v[:, None]
    eu, ev = bu - au, bv - av
    qu, qv = pu - au, pv - av
    l2 = eu * eu + ev * ev
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(l2 > 0, (qu * eu + qv * ev) / np.where(l2 > 0, l2, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    du, dv = qu - s * eu, qv - s * ev
    b = bands[None, :]
    return np.any((b >= 0.0) & (du * du + dv * dv <= b * b), axis=1)


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


def slab_edges(edges: np.ndarray, v_lo: float, v_hi: float, n_slabs: int, scale: float,
               margin: float = 0.0) -> list[np.ndarray]:
    """The piece's edges cut into ``n_slabs`` equal slabs of ``v`` over ``[v_lo, v_hi]``.

    A point in a slab can only cross an edge whose ``v``-range meets the slab,
    so the parity over a slab's edges equals the parity over the piece's. The
    device finds a lane's slab from its ``v`` and tests that slab only. With a
    ``margin`` (the band's, :func:`band_filter_margin`) a slab also keeps every
    edge within it, so the band test sees every edge whose band can reach it.
    Extra columns of ``edges`` (the band) travel with their edge.
    """
    m = max(_FILTER_SLACK * max(scale, 1.0), float(margin))
    h = (v_hi - v_lo) / n_slabs
    lo_e = np.minimum(edges[:, 1], edges[:, 3])
    hi_e = np.maximum(edges[:, 1], edges[:, 3])
    out = []
    for k in range(n_slabs):
        lo, hi = v_lo + k * h, v_lo + (k + 1) * h
        out.append(edges[(hi_e >= lo - m) & (lo_e <= hi + m)])
    return out


def piece_edges(edges: np.ndarray, piece_uv, scale: float, margin: float = 0.0) -> np.ndarray:
    """The edges a horizontal ray (toward ``+u``) from a point of the piece
    ``(u0, u1, v0, v1)`` can count.

    An edge counts for a point ``(u, v)`` only if it straddles ``v`` (so its
    ``v``-range meets the piece's) and its crossing lies beyond ``u`` (so its
    largest ``u`` is not left of the piece). The rest contribute nothing to any
    point of the piece, and the parity over the kept edges equals the parity
    over all of them; the slack keeps that true for a point an ulp outside.
    A ``margin`` larger than the slack (the band's) keeps the edges within it
    of the piece as well; an edge kept that cannot be crossed changes no parity.
    """
    if edges.shape[0] == 0:
        return edges
    m = max(_FILTER_SLACK * max(scale, 1.0), float(margin))
    u0, _u1, v0, v1 = (float(x) for x in piece_uv)
    keep = (
        (np.maximum(edges[:, 1], edges[:, 3]) >= v0 - m)
        & (np.minimum(edges[:, 1], edges[:, 3]) <= v1 + m)
        & (np.maximum(edges[:, 0], edges[:, 2]) >= u0 - m)
    )
    return edges[keep]
