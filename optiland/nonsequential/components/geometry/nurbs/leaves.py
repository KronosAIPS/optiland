"""Host build of the NURBS kind: the library's array contract to flat leaves.

Once per geometry, in float64 NumPy, outside any autograd tape (the leaf tree,
the boxes and the leaf choice are topology and are detached,
``docs/theory/09_differentiation.md`` R-09-11 in the research repository):

1. **Bezier extraction as a linear map.** Every patch's knot vectors are
   refined to Bezier form by inserting each distinct knot of the domain up to
   multiplicity ``degree`` (Boehm's insertion, Piegl and Tiller A5.1 and 5.3,
   the same refinement ``kgeom.nurbs.bezier_patches`` performs). The insertion
   runs on the identity matrix, so each Bezier piece's net is a matrix times
   the patch's homogeneous net; a piece depends only on the ``(p + 1) x
   (q + 1)`` block of control points whose basis functions are non-zero on its
   span, and the map is the tensor product ``E_u (x) E_v`` of two small
   matrices. The pieces are checked against the contract's own
   ``bezier_ctrl_points``/``bezier_weights`` when present.
2. **One degree for every leaf.** Patches of lower degree are raised to the
   largest degree of the set by Bernstein degree elevation (exact, linear), so
   the device arrays have one shape.
3. **Leaves.** Each piece is split at parameter midpoints (de Casteljau, again
   a matrix) until its sampled normal cone is at most ``cone_deg`` (15 degrees)
   and the spread of each tangent direction at most ``tangent_deg`` (45
   degrees), or ``max_depth`` splits. The tangent rule exists because an
   annular sector near a pole has a small normal cone but a tangent that
   turns by 90 degrees, and an affine start guess is then poor (the CAD study
   N1 of 2026-09-25, section 5).
4. **Per leaf:** its net, the linear map from the original block (``mu``,
   ``mv``, ``block``), its range inside its piece (``sub``) and in the patch's
   parameters (``prange``), an oriented box from the control points (the
   convex hull holds: every weight is positive), the sampled normal cone about
   the box normal (times 1.25 plus 1 degree), an affine least-squares map from
   the box's in-plane coordinates to the leaf's ``(s, r)`` (Newton's start),
   and the unit normal ``S_s x S_r`` at its centre (the orientation a
   degenerate point's normal is signed by).

The normal cone is sampled on a 9 x 9 grid, not bounded; the rigorous bound is
the cone of the hodograph nets (Sederberg and Meyers 1988). This is the
prototype's rule, measured there; it is kept and stated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Largest half-angle of a leaf's sampled normal cone [degrees].
DEFAULT_CONE_DEG = 15.0
#: Largest spread of a leaf's tangent directions [degrees].
DEFAULT_TANGENT_DEG = 45.0
#: Largest number of midpoint splits of one Bezier piece.
DEFAULT_MAX_DEPTH = 10
#: Samples per direction of the cone, spread and start-map fits.
_GRID = 9
#: Slack on the angle tests [rad], so an angle equal to its bound to rounding passes.
_ANGLE_SLACK = 1e-9
#: Relative agreement required between this build's Bezier pieces and the
#: contract's own, at the patch's coordinate scale.
_PIECE_AGREEMENT = 1e-9


@dataclass
class LeafSet:
    """Flat leaves of a patch set, host float64 (see the module docstring).

    ``L`` leaves of one degree ``(P, Q)``; ``K`` control points in total over
    the patches (the contract's flat ``ctrl_points`` order).
    """

    degree: tuple[int, int]
    net: np.ndarray  # (L, P+1, Q+1, 4) homogeneous (w x, w y, w z, w), the geometry's frame
    mu: np.ndarray  # (L, P+1, P+1) leaf rows from the original block's u columns (zero-padded)
    mv: np.ndarray  # (L, Q+1, Q+1) the same in v
    block: np.ndarray  # (L, P+1, Q+1) int64: flat control-point index of block entry (i, j)
    prange: np.ndarray  # (L, 4) u0, u1, v0, v1 of the leaf in its patch's parameters
    sub: np.ndarray  # (L, 4) the leaf's range inside its Bezier piece, in the piece's [0, 1]^2
    centre: np.ndarray  # (L, 3) oriented-box centre
    frame: np.ndarray  # (L, 3, 3) rows: world -> box frame; row 2 is the box normal
    half: np.ndarray  # (L, 3) box half extents
    cone: np.ndarray  # (L,) normal-cone half-angle about frame[:, 2] [rad]
    uvmap: np.ndarray  # (L, 2, 3) (y1, y2, 1) in the box frame -> (s, r)
    orient: np.ndarray  # (L, 3) unit S_s x S_r at the leaf's centre
    patch: np.ndarray  # (L,) int64 patch index
    piece: np.ndarray  # (L,) int64 Bezier piece index (over the whole set)
    depth: np.ndarray  # (L,) int64
    patch_sign: np.ndarray  # (n_patches,) +1, or -1 for a reversed patch (outward = -S_u x S_v)
    patch_uv_bounds: np.ndarray  # (n_patches, 4) the trimmed region's box
    n_pieces: int

    @property
    def n(self) -> int:
        return int(self.net.shape[0])


# ---------------------------------------------------------------------------
# One-dimensional operators
# ---------------------------------------------------------------------------


def _insert_knot(knots: np.ndarray, pw: np.ndarray, degree: int, t: float, times: int):
    """Boehm's insertion along axis 0 (Piegl and Tiller A5.1), any trailing shape.

    The same operation order as ``kgeom.nurbs.insert_knot``.
    """
    U = np.asarray(knots, dtype=np.float64)
    Q = np.asarray(pw, dtype=np.float64)
    p = int(degree)
    for _ in range(int(times)):
        k = int(np.searchsorted(U, t, side="right")) - 1
        s = int(np.count_nonzero(U == t))
        n = Q.shape[0]
        R = np.empty((n + 1,) + Q.shape[1:])
        R[: k - p + 1] = Q[: k - p + 1]
        R[k - s + 1 :] = Q[k - s :]
        for i in range(k - p + 1, k - s + 1):
            alpha = (t - U[i]) / (U[i + p] - U[i])
            R[i] = alpha * Q[i] + (1.0 - alpha) * Q[i - 1]
        U = np.insert(U, k + 1, t)
        Q = R
    return U, Q


def extraction_1d(knots: np.ndarray, degree: int, n_ctrl: int):
    """The Bezier extraction operators of one direction.

    Returns:
        A list, one entry per non-empty span of the domain in order, of
        ``(j0, E, (a, b))``: the first original control point of the span's
        block, the ``(p + 1, p + 1)`` operator from that block to the Bezier
        segment's points, and the span's parameter interval.

    Raises:
        ValueError: If a segment depends on control points outside a block of
            ``p + 1`` (impossible for a valid knot vector).
    """
    p = int(degree)
    U = np.asarray(knots, dtype=np.float64)
    a, b = float(U[p]), float(U[n_ctrl])
    Q = np.eye(n_ctrl)
    for t in np.unique(U[(U >= a) & (U <= b)]):
        s = int(np.count_nonzero(U == t))
        if s < p:
            U, Q = _insert_knot(U, Q, p, float(t), p - s)
    out = []
    for i in range(p, U.size - p - 1):
        if not (U[i] < U[i + 1] and U[i] >= a and U[i + 1] <= b):
            continue
        rows = Q[i - p : i + 1]
        cols = np.nonzero(np.any(rows != 0.0, axis=0))[0]
        j0 = int(min(cols.min(), n_ctrl - (p + 1)))
        if cols.max() - j0 > p:
            raise ValueError("a Bezier segment depends on more than degree + 1 control points")
        E = rows[:, j0 : j0 + p + 1].copy()
        out.append((j0, E, (float(U[i]), float(U[i + 1]))))
    return out


def elevation_matrix(p: int, P: int) -> np.ndarray:
    """``(P + 1, p + 1)`` Bernstein degree elevation from ``p`` to ``P`` (exact)."""
    E = np.zeros((P + 1, p + 1))
    r = P - p
    for i in range(P + 1):
        for j in range(max(0, i - r), min(p, i) + 1):
            E[i, j] = math.comb(p, j) * math.comb(r, i - j) / math.comb(P, i)
    return E


def split_matrices(P: int) -> tuple[np.ndarray, np.ndarray]:
    """``(P + 1, P + 1)`` de Casteljau operators of the two halves at the midpoint."""
    work = np.eye(P + 1)
    L = np.empty((P + 1, P + 1))
    R = np.empty((P + 1, P + 1))
    L[0] = work[0]
    R[P] = work[P]
    for r in range(1, P + 1):
        work = 0.5 * (work[:-1] + work[1:])
        L[r] = work[0]
        R[P - r] = work[-1]
    return L, R


# ---------------------------------------------------------------------------
# Evaluation of one rational Bezier net (host)
# ---------------------------------------------------------------------------


def _bernstein(n: int, s: np.ndarray):
    one = 1.0 - s
    B = np.stack([math.comb(n, i) * s**i * one ** (n - i) for i in range(n + 1)], -1)
    if n == 0:
        return B, np.zeros_like(B)
    Bm = [math.comb(n - 1, i) * s**i * one ** (n - 1 - i) for i in range(n)]
    zero = 0.0 * s
    dB = np.stack(
        [n * ((Bm[i - 1] if i >= 1 else zero) - (Bm[i] if i < n else zero)) for i in range(n + 1)], -1
    )
    return B, dB


def eval_net(net: np.ndarray, s: np.ndarray, r: np.ndarray):
    """``S, S_s, S_r`` of one homogeneous net ``(P+1, Q+1, 4)`` at points ``(s, r)``."""
    P, Q = net.shape[0] - 1, net.shape[1] - 1
    Bu, dBu = _bernstein(P, s)
    Bv, dBv = _bernstein(Q, r)
    A = np.einsum("pi,pj,ijc->pc", Bu, Bv, net)
    Au = np.einsum("pi,pj,ijc->pc", dBu, Bv, net)
    Av = np.einsum("pi,pj,ijc->pc", Bu, dBv, net)
    w = A[:, 3:4]
    S = A[:, :3] / w
    return S, (Au[:, :3] - S * Au[:, 3:4]) / w, (Av[:, :3] - S * Av[:, 3:4]) / w


def _grid(n: int = _GRID):
    g = np.linspace(0.0, 1.0, n)
    s, r = np.meshgrid(g, g, indexing="ij")
    return s.ravel(), r.ravel()


def _obb(X: np.ndarray):
    """Oriented box of a leaf's projected control points ``(P+1, Q+1, 3)``."""
    c00, c10, c01, c11 = X[0, 0], X[-1, 0], X[0, -1], X[-1, -1]
    eu = (c10 + c11) - (c00 + c01)
    ev = (c01 + c11) - (c00 + c10)
    n = np.cross(eu, ev)
    if np.linalg.norm(n) < 1e-14 * max(1.0, float(np.abs(X).max())) ** 2:
        n = np.cross(c11 - c00, c01 - c10)
    if np.linalg.norm(n) == 0.0:
        raise ValueError("a leaf has no area (all its corners coincide)")
    n = n / np.linalg.norm(n)
    e1 = eu - np.dot(eu, n) * n
    if np.linalg.norm(e1) < 1e-300:
        e1 = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    R = np.stack([e1, e2, n])
    pts = X.reshape(-1, 3)
    cen0 = pts.mean(axis=0)
    Y = (pts - cen0) @ R.T
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    return cen0 + 0.5 * (lo + hi) @ R, R, 0.5 * (hi - lo)


def _cone_and_spread(net: np.ndarray, axis: np.ndarray):
    s, r = _grid()
    _, Ss, Sr = eval_net(net, s, r)
    nn = np.cross(Ss, Sr)
    mag = np.linalg.norm(nn, axis=1)
    # A sample is degenerate (a collapsed edge) when its normal is at the rounding level of the
    # leaf's largest tangents, not of its own: at a pole S_s is rounding noise of |S_r| scale
    # and its cross product points anywhere.
    ns = np.linalg.norm(Ss, axis=1)
    nr = np.linalg.norm(Sr, axis=1)
    ok = mag > 1e-9 * ns.max() * nr.max() + 1e-300
    if ok.any():
        c = np.abs(nn[ok] @ axis) / mag[ok]
        ang = float(np.arccos(np.clip(c.min(), -1.0, 1.0)))
    else:
        ang = math.pi / 2
    cone = min(math.pi / 2, 1.25 * ang + math.radians(1.0))
    spreads = []
    # How far the tangents turn along each parameter: along s (grid rows of fixed r) and
    # along r (columns of fixed s). The split goes across the parameter that turns them
    # most; the spread of one tangent alone does not say which (at a pole the r tangent
    # turns with the azimuth s, not with r).
    turn = [0.0, 0.0]
    for T in (Ss, Sr):
        m = np.linalg.norm(T, axis=1)
        keep = m > 1e-9 * m.max()
        Tn = np.where(keep[:, None], T / np.where(keep, m, 1.0)[:, None], np.nan)
        spreads.append(float(np.arccos(np.clip(np.nanmin(Tn[keep] @ Tn[keep].T), -1.0, 1.0))))
        G = Tn.reshape(_GRID, _GRID, 3)  # [s index, r index]
        for axis_, k in ((0, 0), (1, 1)):
            A = np.moveaxis(G, axis_, 0)  # lines along parameter k
            c = np.einsum("aic,bic->iab", A, A)
            turn[k] = max(turn[k], float(np.arccos(np.clip(np.nanmin(c), -1.0, 1.0))))
    return cone, spreads[0], spreads[1], turn[0], turn[1]


def _uvmap(net: np.ndarray, centre: np.ndarray, R: np.ndarray) -> np.ndarray:
    s, r = _grid()
    S, _, _ = eval_net(net, s, r)
    Y = (S - centre) @ R.T
    A = np.stack([Y[:, 0], Y[:, 1], np.ones_like(s)], 1)
    coef, *_ = np.linalg.lstsq(A, np.stack([s, r], 1), rcond=None)
    return coef.T


def _centre_normal(net: np.ndarray, frame_n: np.ndarray) -> np.ndarray:
    """Unit ``S_s x S_r`` at the leaf's centre (the box normal's sign if degenerate)."""
    _, Ss, Sr = eval_net(net, np.array([0.5]), np.array([0.5]))
    n = np.cross(Ss[0], Sr[0])
    m = np.linalg.norm(n)
    if m == 0.0 or not np.isfinite(m):
        return frame_n.copy()
    return n / m


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def _loops_on_domain_boundary(a: Mapping[str, Any], i: int, domain) -> bool:
    """Whether every trim curve of patch ``i`` runs along its domain's rectangle.

    A curve whose control points all lie on one edge line of the rectangle is
    that edge (a NURBS curve lies in its control polygon's convex hull); such
    loops keep the whole domain, as every face of the CAD study's lenses does.
    """
    u0, u1, v0, v1 = domain
    lo, hi = int(a["patch_loop_offset"][i]), int(a["patch_loop_offset"][i + 1])
    scale = max(u1 - u0, v1 - v0)
    tol = 1e-12 * max(scale, abs(u0), abs(u1), abs(v0), abs(v1), 1.0)
    for L in range(lo, hi):
        if not bool(a["loop_outer"][L]):
            return False
        c0, c1 = int(a["loop_curve_offset"][L]), int(a["loop_curve_offset"][L + 1])
        for C in range(c0, c1):
            k0, k1 = int(a["curve_ctrl_offset"][C]), int(a["curve_ctrl_offset"][C + 1])
            pts = np.asarray(a["curve_ctrl_points"][k0:k1], dtype=np.float64)
            on_edge = [
                np.all(np.abs(pts[:, 0] - u0) <= tol),
                np.all(np.abs(pts[:, 0] - u1) <= tol),
                np.all(np.abs(pts[:, 1] - v0) <= tol),
                np.all(np.abs(pts[:, 1] - v1) <= tol),
            ]
            if not any(on_edge):
                return False
    return True


def build_leaves(
    arrays: Mapping[str, Any],
    control_points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    *,
    cone_deg: float = DEFAULT_CONE_DEG,
    tangent_deg: float = DEFAULT_TANGENT_DEG,
    max_depth: int = DEFAULT_MAX_DEPTH,
    check_pieces: bool = True,
) -> LeafSet:
    """The leaves of a patch set given as the library's array contract.

    Args:
        arrays: ``PatchSet.to_arrays(bezier=True)`` of ``kgeom.nurbs`` (the
            ``bezier_*`` keys are optional; when present the pieces built here
            are checked against them).
        control_points: ``(K, 3)`` float64 to use instead of the contract's
            ``ctrl_points`` (a geometry whose net has been moved), or ``None``.
        weights: ``(K,)`` likewise.
        cone_deg, tangent_deg, max_depth: The leaf rule (module docstring).
        check_pieces: Compare the Bezier pieces with the contract's.

    Returns:
        The :class:`LeafSet`.

    Raises:
        ValueError: A non-positive weight, a piece that disagrees with the
            contract, an unknown array format.
        NotImplementedError: A patch trimmed by anything but its domain's
            rectangle (device trimming is ticket C of KronosNSRT issue 66).
    """
    fmt = int(np.asarray(arrays.get("format", 1)))
    if fmt != 1:
        raise ValueError(f"NURBS array format {fmt} is not the one this build reads (1)")
    deg = np.asarray(arrays["patch_degree"], dtype=np.int64).reshape(-1, 2)
    nctrl = np.asarray(arrays["patch_n_ctrl"], dtype=np.int64).reshape(-1, 2)
    off = np.asarray(arrays["patch_ctrl_offset"], dtype=np.int64)
    ctrl = np.asarray(arrays["ctrl_points"] if control_points is None else control_points, dtype=np.float64)
    wts = np.asarray(arrays["ctrl_weights"] if weights is None else weights, dtype=np.float64)
    ctrl = ctrl.reshape(-1, 3)
    wts = wts.reshape(-1)
    if np.any(wts <= 0) or not np.all(np.isfinite(wts)) or not np.all(np.isfinite(ctrl)):
        raise ValueError("control points must be finite and weights strictly positive")
    n_p = deg.shape[0]
    if n_p == 0:
        raise ValueError("a NURBS geometry needs at least one patch")
    P, Q = int(deg[:, 0].max()), int(deg[:, 1].max())
    Lsplit_u, Rsplit_u = split_matrices(P)
    Lsplit_v, Rsplit_v = split_matrices(Q)
    reversed_ = np.asarray(arrays.get("patch_reversed", np.zeros(n_p, bool)), dtype=bool).reshape(-1)
    domains = np.asarray(arrays["patch_domain"], dtype=np.float64).reshape(-1, 4)
    uvb = np.asarray(arrays.get("patch_uv_bounds", domains), dtype=np.float64).reshape(-1, 4)
    has_bezier = check_pieces and "bezier_ctrl_points" in arrays

    out: dict[str, list] = {k: [] for k in (
        "net", "mu", "mv", "block", "prange", "sub", "centre", "frame", "half", "cone",
        "uvmap", "orient", "patch", "piece", "depth",
    )}
    piece_id = 0
    for i in range(n_p):
        p, q = int(deg[i, 0]), int(deg[i, 1])
        nu, nv = int(nctrl[i, 0]), int(nctrl[i, 1])
        if int(arrays["patch_loop_offset"][i + 1]) > int(arrays["patch_loop_offset"][i]):
            if not _loops_on_domain_boundary(arrays, i, domains[i]):
                raise NotImplementedError(
                    f"patch {i} is trimmed inside its domain; device trimming is not built yet "
                    "(ticket C of KronosNSRT issue 66)"
                )
        ku = np.asarray(arrays["knots_u"][int(arrays["patch_knot_u_offset"][i]) : int(arrays["patch_knot_u_offset"][i + 1])])
        kv = np.asarray(arrays["knots_v"][int(arrays["patch_knot_v_offset"][i]) : int(arrays["patch_knot_v_offset"][i + 1])])
        sl = slice(int(off[i]), int(off[i + 1]))
        Pp = ctrl[sl].reshape(nu, nv, 3)
        Wp = wts[sl].reshape(nu, nv)
        Pw = np.concatenate([Pp * Wp[..., None], Wp[..., None]], axis=-1)
        idx = np.arange(int(off[i]), int(off[i + 1]), dtype=np.int64).reshape(nu, nv)
        eu = extraction_1d(ku, p, nu)
        ev = extraction_1d(kv, q, nv)
        Gu = elevation_matrix(p, P)
        Gv = elevation_matrix(q, Q)
        scale = max(float(np.abs(Pp).max()), 1.0)
        b0 = int(arrays["patch_bezier_offset"][i]) if has_bezier else 0
        if has_bezier and int(arrays["patch_bezier_offset"][i + 1]) - b0 != len(eu) * len(ev):
            raise ValueError(f"patch {i}: {len(eu) * len(ev)} Bezier pieces built, the contract lists "
                             f"{int(arrays['patch_bezier_offset'][i + 1]) - b0}")
        for a_, (ju, Eu, (ua, ub)) in enumerate(eu):
            for b_, (jv, Ev, (va, vb)) in enumerate(ev):
                block_pw = Pw[ju : ju + p + 1, jv : jv + q + 1]
                block_idx = idx[ju : ju + p + 1, jv : jv + q + 1]
                if has_bezier:
                    k = b0 + a_ * len(ev) + b_
                    c0, c1 = int(arrays["bezier_ctrl_offset"][k]), int(arrays["bezier_ctrl_offset"][k + 1])
                    ref_p = np.asarray(arrays["bezier_ctrl_points"][c0:c1]).reshape(p + 1, q + 1, 3)
                    ref_w = np.asarray(arrays["bezier_weights"][c0:c1]).reshape(p + 1, q + 1)
                    mine = np.einsum("ai,bj,ijc->abc", Eu, Ev, block_pw)
                    dp = np.abs(mine[..., :3] / mine[..., 3:4] - ref_p).max() / scale
                    dw = np.abs(mine[..., 3] - ref_w).max() / max(float(np.abs(ref_w).max()), 1e-300)
                    if dp > _PIECE_AGREEMENT or dw > _PIECE_AGREEMENT:
                        raise ValueError(
                            f"patch {i}, piece ({a_}, {b_}): the extraction disagrees with the contract's "
                            f"Bezier piece by {max(dp, dw):.3g} relative"
                        )
                Mu0 = Gu @ Eu  # (P+1, p+1)
                Mv0 = Gv @ Ev
                # zero-padded columns and a valid (repeated) index for them
                blk = np.empty((P + 1, Q + 1), dtype=np.int64)
                blk[:] = block_idx[0, 0]
                blk[: p + 1, : q + 1] = block_idx
                pw_blk = np.zeros((P + 1, Q + 1, 4))
                pw_blk[: p + 1, : q + 1] = block_pw

                def pad(M, n_cols):
                    out_m = np.zeros((M.shape[0], n_cols))
                    out_m[:, : M.shape[1]] = M
                    return out_m

                stack = [(pad(Mu0, P + 1), pad(Mv0, Q + 1), (0.0, 1.0, 0.0, 1.0), 0)]
                while stack:
                    Mu, Mv, (s0, s1, r0, r1), depth = stack.pop()
                    net = np.einsum("ai,bj,ijc->abc", Mu, Mv, pw_blk)
                    if np.any(net[..., 3] <= 0):
                        raise ValueError("non-positive weight in a leaf: the convex-hull bound does not hold")
                    X = net[..., :3] / net[..., 3:4]
                    centre, R, half = _obb(X)
                    cone, spread_s, spread_r, turn_s, turn_r = _cone_and_spread(net, R[2])
                    # 1e-9 rad of slack: a quarter-turn piece halved is 45 degrees to rounding
                    tan_ok = max(spread_s, spread_r) <= math.radians(tangent_deg) + _ANGLE_SLACK
                    final = cone <= math.radians(cone_deg) + _ANGLE_SLACK and tan_ok
                    if final or depth >= max_depth:
                        out["net"].append(net)
                        out["mu"].append(Mu)
                        out["mv"].append(Mv)
                        out["block"].append(blk)
                        out["sub"].append((s0, s1, r0, r1))
                        out["prange"].append((ua + s0 * (ub - ua), ua + s1 * (ub - ua),
                                              va + r0 * (vb - va), va + r1 * (vb - va)))
                        out["centre"].append(centre)
                        out["frame"].append(R)
                        out["half"].append(half)
                        out["cone"].append(cone)
                        out["uvmap"].append(_uvmap(net, centre, R))
                        out["orient"].append(_centre_normal(net, R[2]))
                        out["patch"].append(i)
                        out["piece"].append(piece_id)
                        out["depth"].append(depth)
                        continue
                    c00, c10, c01, c11 = X[0, 0], X[-1, 0], X[0, -1], X[-1, -1]
                    lu = np.linalg.norm(c10 - c00) + np.linalg.norm(c11 - c01)
                    lv = np.linalg.norm(c01 - c00) + np.linalg.norm(c11 - c10)
                    cone_ok = cone <= math.radians(cone_deg) + _ANGLE_SLACK
                    split_u = (turn_s >= turn_r) if (cone_ok and not tan_ok) else (lu >= lv)
                    if split_u:
                        sm = 0.5 * (s0 + s1)
                        # pushed right first so the left half is built first (leaf order is u-major, left to right)
                        stack.append((Rsplit_u @ Mu, Mv, (sm, s1, r0, r1), depth + 1))
                        stack.append((Lsplit_u @ Mu, Mv, (s0, sm, r0, r1), depth + 1))
                    else:
                        rm = 0.5 * (r0 + r1)
                        stack.append((Mu, Rsplit_v @ Mv, (s0, s1, rm, r1), depth + 1))
                        stack.append((Mu, Lsplit_v @ Mv, (s0, s1, r0, rm), depth + 1))
                piece_id += 1

    return LeafSet(
        degree=(P, Q),
        net=np.array(out["net"]), mu=np.array(out["mu"]), mv=np.array(out["mv"]),
        block=np.array(out["block"], dtype=np.int64), prange=np.array(out["prange"]),
        sub=np.array(out["sub"]), centre=np.array(out["centre"]), frame=np.array(out["frame"]),
        half=np.array(out["half"]), cone=np.array(out["cone"]), uvmap=np.array(out["uvmap"]),
        orient=np.array(out["orient"]), patch=np.array(out["patch"], dtype=np.int64),
        piece=np.array(out["piece"], dtype=np.int64), depth=np.array(out["depth"], dtype=np.int64),
        patch_sign=np.where(reversed_, -1.0, 1.0), patch_uv_bounds=uvb, n_pieces=piece_id,
    )
