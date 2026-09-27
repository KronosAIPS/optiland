"""Device side of the NURBS kind: fixed shapes, masked Newton, no host read.

Written once for NumPy and torch (either dtype, any device) through a small
operation table (:class:`_Ops`); every array it builds per call comes from the
input arrays or from the leaf arrays uploaded once per backend, precision and
device, so a CUDA graph can record it and the emulated replay finds no host
transfer (``backends/graph_replay.py``).

Per call (one chunk of rays at a time, the chunk size fixed by the ray count):

1. **Every (ray, leaf) box test** (the slab method on each leaf's oriented
   box). The bounding volume hierarchy of the research repository's issue 1
   replaces this list later.
2. **Candidates, in two rounds of fixed shape.** The boxes a ray meets
   beyond its start ``t0`` sorted by entry distance (a stable sort, so ties
   keep leaf order); the nearest ``n_candidates`` (8) are solved from the
   leaf's mid-plane, and the nearest ``n_ambiguous`` (4) of those the ray
   meets within the leaf's normal cone ("ambiguous": it can cross the leaf
   twice) also from the box's two ends. A dynamic loop would stop at the
   first round whose boxes all start beyond the best hit; fixed shapes solve
   them all and keep the nearest root, which is the same answer. A ray whose
   best root lies beyond the first box the round did not solve is *in doubt*
   (a ray through the many small leaves around a pole, a grazing ray); the
   rays in doubt are gathered into a queue of fixed capacity and solved again
   over 32 boxes, 16 of them from both ends (:func:`intersect`). A ray still
   in doubt, or beyond the queue's capacity, is flagged ``overflow``: it may
   have lost a nearer root, and the geometry counts it.
3. **Newton in (s, r, t)** on each lane: ``F = S(s, r) - o - t d``, ``J = [S_s,
   S_r, -d]`` solved by Cramer's rule; ``n_iter`` fixed iterations, masked, the
   origin advanced to the box entry and the net stored relative to the leaf's
   centre. A lane whose box holds the ray's start (a ray leaving the surface)
   deflates the known root at ``t = 0`` (the step scaled by ``1 / (1 - m' dt /
   m)``, ``m = 1 + (l / t)^2``, ``l = 1e-4`` of the leaf's scale; Farrell,
   Birkisson and Funke 2015) and never starts at it.
4. **Acceptance:** residual at most ``k_tol`` (32) units of the working dtype at
   the larger of the leaf's scale and the ray's coordinate scale; ``(s, r)``
   inside the leaf's Bezier piece within ``4 tol / |S_s|`` (a root beyond the
   leaf but inside the piece is the same polynomial's root); a collapsed edge
   (the whole leaf moves the point by less than ``tol`` along a parameter)
   clamps that parameter; beyond the start ``t0``; for a ray leaving the
   surface, beyond ``4 tol / |cos|``, the self root's own uncertainty; inside
   the patch's ``uv_bounds`` rectangle.
5. **One winner per ray:** the smallest accepted ``t``, ties to the lowest
   lane index, and every field gathered from that one lane.

The numbers of the CAD study N1 of 2026-09-25 (sections 5 and 6) are the
prototype's; this is a fixed-shape port of it with the changes named in the
kind's module docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Fixed Newton iteration count (the prototype's 16; its measured maximum to
#: convergence was 14 on the sphere and 16 on the bicubic patch in float64).
DEFAULT_N_ITER = 16
#: Residual tolerance in units of the working dtype at the coordinate scale.
DEFAULT_K_TOL = 32.0
#: Boxes solved per ray from the mid-plane start, first round.
DEFAULT_N_CANDIDATES = 8
#: Ambiguous boxes per ray also solved from both box ends, first round.
DEFAULT_N_AMBIGUOUS = 4
#: The same for the rays in doubt after it (second round).
DEFAULT_N_CANDIDATES_2 = 32
DEFAULT_N_AMBIGUOUS_2 = 16
#: The second round's queue capacity, a share of the rays of the call.
DEFAULT_SECOND_ROUND_SHARE = 0.25
#: The deflation length, a fraction of the leaf's scale.
ELL_FACTOR = 1e-4
#: The clamp on (s, r) during the iterations, beyond [0, 1] on each side.
PARAM_CLAMP = 0.25
#: Rays per chunk are chosen so a chunk holds at most this many (ray, leaf) box tests ...
_MAX_BOX_PAIRS = 1 << 20
#: ... and at most this many Newton lanes.
_MAX_LANES = 1 << 17
#: A tangent shorter than this many units of the dtype at the leaf's scale is
#: rounding noise: the point is on a collapsed edge and the normal comes from
#: the edge rule.
DEGENERATE_K = 256.0


class _Ops:
    """The operations the kernel needs, for NumPy or torch arrays."""

    def __init__(self, like: Any) -> None:
        self.torch = None
        if type(like).__module__.startswith("torch"):
            import torch  # noqa: PLC0415

            self.torch = torch
            self.xp = torch
        else:
            self.xp = np
        self.dtype = like.dtype
        fin = np.finfo(np.float32 if str(self.dtype).endswith("float32") else np.float64)
        self.u = float(fin.eps) / 2.0
        self.tiny = float(fin.tiny)

    # construction from the inputs
    def full_like(self, a, value):
        return self.xp.full_like(a, value)

    def zeros_like(self, a):
        return self.xp.zeros_like(a)

    def arange_like(self, n: int, like_int):
        if self.torch is not None:
            return self.torch.arange(n, dtype=self.torch.int64, device=like_int.device)
        return np.arange(n, dtype=np.int64)

    # arithmetic
    def where(self, c, a, b):
        return self.xp.where(c, a, b)

    def fmin(self, a, b):
        return self.xp.fmin(a, b)

    def fmax(self, a, b):
        return self.xp.fmax(a, b)

    def minimum(self, a, b):
        return self.xp.minimum(a, b)

    def maximum(self, a, b):
        return self.xp.maximum(a, b)

    def abs(self, a):
        return self.xp.abs(a)

    def sqrt(self, a):
        return self.xp.sqrt(a)

    def isfinite(self, a):
        return self.xp.isfinite(a)

    def clip(self, a, lo, hi):
        if self.torch is not None:
            return self.torch.clamp(a, lo, hi)
        return np.clip(a, lo, hi)

    def cross(self, a, b):
        if self.torch is not None:
            return self.torch.linalg.cross(a, b, dim=-1)
        return np.cross(a, b)

    def einsum(self, spec, *ops):
        return self.xp.einsum(spec, *ops)

    def stack(self, xs, axis):
        if self.torch is not None:
            return self.torch.stack(xs, dim=axis)
        return np.stack(xs, axis=axis)

    def cat(self, xs, axis):
        if self.torch is not None:
            return self.torch.cat(xs, dim=axis)
        return np.concatenate(xs, axis=axis)

    def amax_abs(self, a):
        if self.torch is not None:
            return a.abs().amax(dim=-1)
        return np.abs(a).max(axis=-1)

    def norm(self, a):
        return self.sqrt((a * a).sum(-1))

    def argsort_stable(self, a):
        if self.torch is not None:
            return self.torch.sort(a, dim=1, stable=True).indices
        return np.argsort(a, axis=1, kind="stable")

    def take(self, a, idx):
        """``a[i, idx[i, j]]`` along axis 1."""
        if self.torch is not None:
            return self.torch.gather(a, 1, idx)
        return np.take_along_axis(a, idx, axis=1)

    def argmin(self, a):
        if self.torch is not None:
            return self.torch.argmin(a, dim=1)
        return np.argmin(a, axis=1)

    def amin(self, a):
        if self.torch is not None:
            return a.amin(dim=1)
        return a.min(axis=1)

    def rows(self, a, idx):
        """``a[idx]`` (gather rows by an integer index array)."""
        return a[idx]

    def errstate(self):
        if self.torch is not None:
            import contextlib  # noqa: PLC0415

            return contextlib.nullcontext()
        return np.errstate(all="ignore")


@dataclass
class DeviceLeaves:
    """The leaf arrays in the working dtype and device (uploaded once)."""

    net: Any  # (L, P+1, Q+1, 4) relative to the leaf centre: (w (x - c), w)
    centre: Any  # (L, 3)
    frame: Any  # (L, 3, 3)
    half: Any  # (L, 3) padded outward
    scale: Any  # (L,)
    sin_cone: Any  # (L,)
    uvmap: Any  # (L, 2, 3)
    sub: Any  # (L, 4)
    prange: Any  # (L, 4)
    uvb: Any  # (L, 4) the leaf's patch's uv_bounds
    orient: Any  # (L, 3)
    P: int
    Q: int
    n: int


def upload(leaves, to_array, pad_ulps: float = 8.0, u: float = 2.0**-53) -> DeviceLeaves:
    """The leaf arrays of a :class:`~.leaves.LeafSet` in the working dtype.

    Boxes are padded outward by ``pad_ulps`` units of the working dtype at the
    box's coordinate scale, so rounding the box never culls a root; nets are
    stored relative to the leaf's centre, so the residual lives at the leaf's
    scale, not the scene's.

    Args:
        leaves: The host leaf set.
        to_array: NumPy float64 -> backend array (the working dtype and device).
        pad_ulps: The box padding.
        u: The working dtype's unit roundoff.
    """
    lv = leaves
    net = lv.net.copy()
    net[..., :3] = net[..., :3] - lv.centre[:, None, None, :] * net[..., 3:4]
    coord = np.abs(lv.centre).max(axis=1) + np.abs(lv.half).max(axis=1)
    half = lv.half + (pad_ulps * u * 2.0 * coord + 1e-300)[:, None]
    scale = np.abs(lv.half).max(axis=1) * 2.0 * 1.8
    uvb = lv.patch_uv_bounds[lv.patch]
    return DeviceLeaves(
        to_array(net), to_array(lv.centre), to_array(lv.frame), to_array(half), to_array(scale),
        to_array(np.sin(lv.cone)), to_array(lv.uvmap), to_array(lv.sub), to_array(lv.prange),
        to_array(uvb), to_array(lv.orient), lv.degree[0], lv.degree[1], lv.n,
    )


# ---------------------------------------------------------------------------
# Rational Bezier evaluation
# ---------------------------------------------------------------------------


def bernstein(ops: _Ops, n: int, s):
    """Bernstein basis ``B_i^n(s)`` and its derivative, each ``(M, n + 1)``."""
    one = 1.0 - s
    B = [math.comb(n, i) * s**i * one ** (n - i) for i in range(n + 1)]
    zero = 0.0 * s
    if n == 0:
        return ops.stack(B, -1), ops.stack([zero], -1)
    Bm = [math.comb(n - 1, i) * s**i * one ** (n - 1 - i) for i in range(n)]
    dB = [n * ((Bm[i - 1] if i >= 1 else zero) - (Bm[i] if i < n else zero)) for i in range(n + 1)]
    return ops.stack(B, -1), ops.stack(dB, -1)


def eval_leaf(ops: _Ops, net, s, r, P: int, Q: int):
    """``S, S_s, S_r`` of rational Bezier nets ``(M, P+1, Q+1, 4)`` at ``(s, r)``."""
    Bu, dBu = bernstein(ops, P, s)
    Bv, dBv = bernstein(ops, Q, r)
    A = ops.einsum("pi,pj,pijc->pc", Bu, Bv, net)
    Au = ops.einsum("pi,pj,pijc->pc", dBu, Bv, net)
    Av = ops.einsum("pi,pj,pijc->pc", Bu, dBv, net)
    w = A[:, 3:4]
    S = A[:, :3] / w
    Ss = (Au[:, :3] - S * Au[:, 3:4]) / w
    Sr = (Av[:, :3] - S * Av[:, 3:4]) / w
    return S, Ss, Sr


def solve3(ops: _Ops, a, b, c, y):
    """Solve ``[a b c] x = y`` row-wise by Cramer's rule (all ``(M, 3)``)."""
    bc = ops.cross(b, c)
    det = (a * bc).sum(-1)
    x1 = (y * bc).sum(-1) / det
    x2 = (a * ops.cross(y, c)).sum(-1) / det
    x3 = (a * ops.cross(b, y)).sum(-1) / det
    return x1, x2, x3, det


def unit_normal(ops: _Ops, net, s, r, P: int, Q: int, scale, orient, u: float):
    """Unit ``S_s x S_r`` at ``(s, r)``, with the collapsed-edge rule.

    Where one tangent is rounding noise (a pole of a surface of revolution),
    the tangent plane is spanned by the other tangent at the leaf's two ends
    across the collapsed parameter: the normal is their cross product, signed
    to agree with the leaf's centre normal (the leaf's normal cone is within
    15 degrees, so the sign is unambiguous). This is the limit of ``S_s x S_r``
    at the edge, exact for any surface with a tangent plane there.
    """
    _, Ss, Sr = eval_leaf(ops, net, s, r, P, Q)
    n = ops.cross(Ss, Sr)
    ns = ops.norm(Ss)
    nr = ops.norm(Sr)
    lim = DEGENERATE_K * u * scale
    deg_s = ns <= lim  # s collapsed: S_r at s = 0 and s = 1 span the plane
    deg_r = (nr <= lim) & ~deg_s
    zero = ops.zeros_like(s)
    one = zero + 1.0
    _, _, Sr0 = eval_leaf(ops, net, zero, r, P, Q)
    _, _, Sr1 = eval_leaf(ops, net, one, r, P, Q)
    _, Ss0, _ = eval_leaf(ops, net, s, zero, P, Q)
    _, Ss1, _ = eval_leaf(ops, net, s, one, P, Q)
    n_s = ops.cross(Sr1, Sr0)
    n_r = ops.cross(Ss0, Ss1)
    n = ops.where(deg_s[:, None], n_s, ops.where(deg_r[:, None], n_r, n))
    sign = ops.where(((n * orient).sum(-1) < 0.0) & (deg_s | deg_r), -one, one)
    n = n * sign[:, None]
    m = ops.norm(n)
    return n / ops.where(m > 0.0, m, one)[:, None]


# ---------------------------------------------------------------------------
# The intersection
# ---------------------------------------------------------------------------


def _newton(ops, net, oo, dd, tl, tspan, tol, P, Q, n_iter, s, r, defl, ell):
    """Fixed-count masked Newton from ``(s, r, tl)``; no host read.

    Returns ``s, r, tl``, the final residual, the step at which the residual
    first met ``tol`` (-1: never), ``S_s``, ``S_r`` at the final iterate.
    """
    conv = ops.full_like(tl, -1.0)
    one = ops.full_like(tl, 1.0)
    tiny_t = ops.full_like(tl, ops.tiny)
    lo = ops.where(defl, tol, -tol)
    hi = tspan + tol
    Ss = Sr = res = None
    for it in range(n_iter + 1):
        S, Ss, Sr = eval_leaf(ops, net, s, r, P, Q)
        F = S - oo - tl[:, None] * dd
        res = ops.amax_abs(F)
        conv = ops.where((res <= tol) & (conv < 0.0), one * it, conv)
        if it == n_iter:
            break
        x1, x2, x3, _ = solve3(ops, Ss, Sr, -dd, -F)
        # Deflation of the root at t = 0, the point the ray leaves from.
        tt = ops.where(tl > 0.0, tl, tiny_t)
        m = 1.0 + (ell / tt) ** 2
        dm = -2.0 * ell**2 / tt**3
        alpha = 1.0 / (1.0 - dm * x3 / m)
        alpha = ops.where(defl & ops.isfinite(alpha), alpha, one)
        x1, x2, x3 = alpha * x1, alpha * x2, alpha * x3
        s = ops.clip(ops.where(ops.isfinite(x1), s + x1, s), -PARAM_CLAMP, 1.0 + PARAM_CLAMP)
        r = ops.clip(ops.where(ops.isfinite(x2), r + x2, r), -PARAM_CLAMP, 1.0 + PARAM_CLAMP)
        tl = ops.minimum(ops.maximum(ops.where(ops.isfinite(x3), tl + x3, tl), lo), hi)
    return s, r, tl, res, conv, Ss, Sr


def _chunk(ops, dl: DeviceLeaves, o, d, t0, n_iter, k_tol, n_cand, n_amb):
    """One chunk of rays: see the module docstring. Returns a dict of per-ray arrays."""
    n = o.shape[0]
    L = dl.n
    C = min(n_cand, L)
    A = min(n_amb, C)
    inf = float("inf")
    # -- 1. every (ray, leaf) box ---------------------------------------------
    rel = o[:, None, :] - dl.centre[None, :, :]
    ol = ops.einsum("lij,nlj->nli", dl.frame, rel)
    dlc = ops.einsum("lij,nj->nli", dl.frame, d)
    inv = 1.0 / dlc
    t1 = (-dl.half[None] - ol) * inv
    t2 = (dl.half[None] - ol) * inv
    lo_ = ops.fmin(t1, t2)
    hi_ = ops.fmax(t1, t2)
    tn = ops.fmax(ops.fmax(lo_[..., 0], lo_[..., 1]), lo_[..., 2])
    tf = ops.fmin(ops.fmin(hi_[..., 0], hi_[..., 1]), hi_[..., 2])
    t0c = t0[:, None]
    tent = ops.fmax(tn, t0c)
    inbox = (tf >= tent) & (tf > t0c)
    te_all = ops.where(inbox, tent, ops.full_like(tent, inf))
    # -- 2. candidates ------------------------------------------------------------
    order = ops.argsort_stable(te_all)
    cand = order[:, :C]
    te = ops.take(te_all, cand)
    tfc = ops.take(tf, cand)
    tnc = ops.take(tn, cand)
    cz = ops.abs(ops.take(dlc[..., 2], cand))
    valid = te < inf
    next_box = ops.take(te_all, order[:, C : C + 1])[:, 0] if L > C else ops.full_like(t0, inf)
    amb = valid & (cz <= dl.sin_cone[cand])
    akey = ops.where(amb, te, ops.full_like(te, inf))
    aord = ops.argsort_stable(akey)
    acol = aord[:, :A]
    a_valid = ops.take(amb, acol)
    next_amb = ops.take(akey, aord[:, A : A + 1])[:, 0] if C > A else ops.full_like(t0, inf)

    # -- 3. lanes: n x C mid starts, then n x A from each box end ---------------------
    def flat(x):
        return x.reshape(-1)

    ray_c = ops.arange_like(n, cand)[:, None] + 0 * cand  # (n, C) ray index
    lane_ray = ops.cat([flat(ray_c), flat(ray_c[:, :A]), flat(ray_c[:, :A])], 0)
    leaf_c = cand
    leaf_a = ops.take(cand, acol)
    lane_leaf = ops.cat([flat(leaf_c), flat(leaf_a), flat(leaf_a)], 0)
    te_a = ops.take(te, acol)
    tf_a = ops.take(tfc, acol)
    tn_a = ops.take(tnc, acol)
    lane_te = ops.cat([flat(te), flat(te_a), flat(te_a)], 0)
    lane_tf = ops.cat([flat(tfc), flat(tf_a), flat(tf_a)], 0)
    lane_tn = ops.cat([flat(tnc), flat(tn_a), flat(tn_a)], 0)
    lane_on = ops.cat([flat(valid), flat(a_valid), flat(a_valid)], 0)
    nC, nA = n * C, n * A
    zeros_c = ops.zeros_like(flat(te))
    zeros_a = ops.zeros_like(flat(te_a))
    lane_frac = ops.cat([zeros_c + 0.5, zeros_a, zeros_a + 1.0], 0)
    lane_mid = ops.cat([zeros_c + 1.0, zeros_a, zeros_a], 0) > 0.5

    lt0 = t0[lane_ray]
    tpre = ops.where(lane_on, lane_te, lt0)
    tspan = ops.where(lane_on, lane_tf - tpre, ops.zeros_like(tpre))
    # a lane whose box holds the ray's start: the ray may be leaving this surface
    defl = lane_on & (lane_tn <= lt0)
    od = o[lane_ray]
    dd = d[lane_ray]
    cen = dl.centre[lane_leaf]
    frm = dl.frame[lane_leaf]
    oo = od + tpre[:, None] * dd - cen
    tol = k_tol * ops.u * ops.maximum(dl.scale[lane_leaf], ops.amax_abs(od) + ops.amax_abs(cen))
    # starts
    tl0 = lane_frac * tspan
    tl0 = ops.where(defl & (tl0 <= 0.0), 0.05 * tspan, tl0)
    pl3 = (frm[:, 2] * oo).sum(-1)
    dl3 = (frm[:, 2] * dd).sum(-1)
    tm = -pl3 / dl3
    use_tm = lane_mid & ops.isfinite(tm) & (tm >= 0.0) & (tm <= tspan)
    tl0 = ops.where(use_tm, tm, tl0)
    pt = oo + tl0[:, None] * dd
    y = ops.einsum("pij,pj->pi", frm, pt)
    Am = dl.uvmap[lane_leaf]
    s0 = ops.clip(Am[:, 0, 0] * y[:, 0] + Am[:, 0, 1] * y[:, 1] + Am[:, 0, 2], 0.0, 1.0)
    r0 = ops.clip(Am[:, 1, 0] * y[:, 0] + Am[:, 1, 1] * y[:, 1] + Am[:, 1, 2], 0.0, 1.0)
    net = dl.net[lane_leaf]
    ell = dl.scale[lane_leaf] * ELL_FACTOR
    s, r, tl, res, conv, Ss, Sr = _newton(
        ops, net, oo, dd, tl0, tspan, tol, dl.P, dl.Q, n_iter, s0, r0, defl, ell
    )
    # -- 4. acceptance -------------------------------------------------------------
    ns = ops.norm(Ss)
    nr = ops.norm(Sr)
    nv = ops.cross(Ss, Sr)
    one = ops.full_like(ns, 1.0)
    cosn = ops.abs((nv * dd).sum(-1)) / ops.maximum(ops.norm(nv), one * 1e-30)
    s = ops.where(ns <= tol, ops.clip(s, 0.0, 1.0), s)
    r = ops.where(nr <= tol, ops.clip(r, 0.0, 1.0), r)
    eps_s = 4.0 * tol / ops.maximum(ns, one * 1e-30)
    eps_r = 4.0 * tol / ops.maximum(nr, one * 1e-30)
    sb = dl.sub[lane_leaf]
    ws = sb[:, 1] - sb[:, 0]
    wr = sb[:, 3] - sb[:, 2]
    sp = sb[:, 0] + s * ws
    rp = sb[:, 2] + r * wr
    es = eps_s * ws
    er = eps_r * wr
    dom = (sp >= -es) & (sp <= 1.0 + es) & (rp >= -er) & (rp <= 1.0 + er)
    s = (ops.clip(sp, 0.0, 1.0) - sb[:, 0]) / ws
    r = (ops.clip(rp, 0.0, 1.0) - sb[:, 2]) / wr
    ttot = tpre + tl
    ok = lane_on & (res <= tol) & dom & (ttot > lt0)
    self_excl = 4.0 * tol / ops.maximum(cosn, one * 1e-6)
    ok = ok & ~(defl & (ttot - lt0 <= self_excl))
    pr = dl.prange[lane_leaf]
    uu = pr[:, 0] + s * (pr[:, 1] - pr[:, 0])
    vv = pr[:, 2] + r * (pr[:, 3] - pr[:, 2])
    ub = dl.uvb[lane_leaf]
    ok = ok & (uu >= ub[:, 0]) & (uu <= ub[:, 1]) & (vv >= ub[:, 2]) & (vv <= ub[:, 3])
    # -- 5. one winner per ray ----------------------------------------------------
    tt = ops.where(ok, ttot, ops.full_like(ttot, inf))

    def per_ray(x):
        return ops.cat([x[:nC].reshape(n, C), x[nC : nC + nA].reshape(n, A), x[nC + nA :].reshape(n, A)], 1)

    T = per_ray(tt)
    w = ops.argmin(T)[:, None]
    best = ops.take(T, w)[:, 0]
    hit = best < inf

    def pick(x):
        return ops.take(per_ray(x), w)[:, 0]

    leaf = pick(lane_leaf)
    out = {
        "t": best,
        "hit": hit,
        "leaf": ops.where(hit, leaf, ops.zeros_like(leaf) - 1),
        "leaf_safe": ops.where(hit, leaf, ops.zeros_like(leaf)),
        "s": pick(s),
        "r": pick(r),
        "u": pick(uu),
        "v": pick(vv),
        "tpre": pick(tpre),
        "steps": pick(conv),
        "residual": pick(res),
        "doubt": (best > next_box) | (best > next_amb),
    }
    # the Jacobian columns at the winner, for the adjoint
    out["Ss"] = ops.stack([pick(Ss[:, k]) for k in range(3)], -1)
    out["Sr"] = ops.stack([pick(Sr[:, k]) for k in range(3)], -1)
    return out


def _rounds(ops, dl, o, d, t0, n_iter, k_tol, C, A):
    """One round over every ray given, chunked so a chunk's arrays stay bounded."""
    n = o.shape[0]
    L = max(dl.n, 1)
    lanes_per_ray = min(C, dl.n) + 2 * min(A, C, dl.n)
    size = max(1, min(_MAX_BOX_PAIRS // L, _MAX_LANES // max(lanes_per_ray, 1)))
    parts = [
        _chunk(ops, dl, o[a : min(n, a + size)], d[a : min(n, a + size)], t0[a : min(n, a + size)],
               n_iter, k_tol, C, A)
        for a in range(0, n, size)
    ]
    if len(parts) == 1:
        return parts[0]
    return {k: ops.cat([p[k] for p in parts], 0) for k in parts[0]}


def intersect(ops: _Ops, dl: DeviceLeaves, o, d, t0, *, n_iter=DEFAULT_N_ITER, k_tol=DEFAULT_K_TOL,
              n_candidates=DEFAULT_N_CANDIDATES, n_ambiguous=DEFAULT_N_AMBIGUOUS,
              n_candidates_2=DEFAULT_N_CANDIDATES_2, n_ambiguous_2=DEFAULT_N_AMBIGUOUS_2,
              second_round_share=DEFAULT_SECOND_ROUND_SHARE):
    """Nearest root of every ray beyond its start ``t0`` (module docstring).

    Two rounds of fixed shape. The first solves every ray's nearest
    ``n_candidates`` boxes (``n_ambiguous`` of them also from both ends). A
    ray is *in doubt* when its best root lies beyond the first box that round
    did not solve (or the first ambiguous box it did not solve from its ends):
    a ray through the many small leaves around a pole, or a grazing ray. The
    rays in doubt are gathered, by a stable sort on the doubt flag, into a
    queue of fixed capacity (``second_round_share`` of the rays, at least 64)
    and solved again over their nearest ``n_candidates_2`` boxes
    (``n_ambiguous_2`` from both ends); their results replace the first
    round's. A ray still in doubt after it, or in doubt beyond the queue's
    capacity, is flagged ``overflow``.

    Args:
        ops: The operation table of the working arrays.
        dl: The uploaded leaves.
        o, d: ``(N, 3)`` ray origins and unit directions in the geometry's frame.
        t0: ``(N,)`` the ray's start: roots at or before it are refused.

    Returns:
        A dict of per-ray arrays: ``t`` (inf for a miss), ``hit``, ``leaf``
        (-1 for a miss), ``leaf_safe`` (0 for a miss), the leaf's ``s, r``,
        the patch's ``u, v``, ``steps``, ``residual``, ``overflow``,
        ``second_round`` (the ray was solved again), and the Jacobian columns
        ``Ss``, ``Sr`` at the winning lane's last iterate.
    """
    n = o.shape[0]
    with ops.errstate():
        out = _rounds(ops, dl, o, d, t0, n_iter, k_tol, n_candidates, n_ambiguous)
        doubt = out.pop("doubt")
        more = dl.n > n_candidates or n_ambiguous_2 > n_ambiguous
        if not more or n_candidates_2 <= 0 or n == 0:
            out["overflow"] = doubt
            out["second_round"] = doubt & ~doubt
            return out
        cap = min(n, max(64, int(math.ceil(second_round_share * n))))
        key = ops.where(doubt, ops.zeros_like(out["steps"]), ops.zeros_like(out["steps"]) + 1.0)
        sel = ops.argsort_stable(key[None, :])[0, :cap]
        taken = doubt[sel]
        o2 = _rounds(ops, dl, o[sel], d[sel], t0[sel], n_iter, k_tol,
                     max(n_candidates_2, n_candidates), max(n_ambiguous_2, n_ambiguous))
        doubt2 = o2.pop("doubt")
        in_queue = doubt & ~doubt
        in_queue = _put(ops, in_queue, sel, taken)
        for k, v in o2.items():
            cur = out[k][sel]
            mask = taken if v.ndim == 1 else taken[:, None]
            out[k] = _put(ops, out[k], sel, ops.where(mask, v, cur))
        still = _put(ops, doubt & ~doubt, sel, taken & doubt2)
        out["overflow"] = (doubt & ~in_queue) | still
        out["second_round"] = in_queue
    return out


def _put(ops, target, idx, values):
    """``target`` with ``target[idx] = values`` (a copy on torch, fixed shapes)."""
    if ops.torch is not None:
        return target.index_put((idx,), values)
    out = target.copy()
    out[idx] = values
    return out
