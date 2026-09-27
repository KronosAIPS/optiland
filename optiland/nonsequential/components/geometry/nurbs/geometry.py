"""NURBS surfaces in the device loop: the geometry kind ``nurbs``.

KronosNSRT issue 66, the research card N1 of 2026-09-25 (tickets B and D). The
kind takes a set of NURBS patches in the array contract of the family's
geometry library (``kgeom.nurbs.PatchSet.to_arrays(bezier=True)``: control
points, weights, knots and per-patch offsets, trim loops flattened, face keys,
Bezier pieces) and intersects rays with it in the batched loop, on NumPy and
torch, in float64 and float32, with no host read inside the bounce.

The host side (:mod:`.leaves`) flattens every Bezier piece into leaves with
oriented boxes, normal cones, start maps and the linear map from the original
net; the device side (:mod:`.kernel`) runs the box test, the candidates and a
fixed-count masked Newton in ``(s, r, t)``. The engine does not import the
library: it reads the arrays.

Orientation (the ``n_geom`` contract of
:meth:`~optiland.nonsequential.components.geometry.base.ComponentGeometry.ray_intersect`)
------------------------------------------------------------------------------------
A patch's outward normal is ``S_u x S_v``, or ``-S_u x S_v`` when the patch is
``reversed`` (the library's convention: outward is out of the solid the face
bounds). ``n_geom`` is the *inward* normal, pointing from ``material_front``
(the outside) to ``material_back`` (the solid): the sphere kind's convention,
so the library's exact rational sphere and the analytic sphere give the same
``n_geom``. A sheet that bounds no solid follows the same rule with its
``S_u x S_v`` side as the outside.

Trimming
--------
Device trimming is ticket C of issue 66 and is not built: a patch whose trim
loops run inside its domain is refused at build. Loops that run along the
domain's rectangle (every face of the CAD study's lenses) keep the whole
domain and are accepted; a hit outside a patch's ``uv_bounds`` rectangle is
refused on the device.

Changes against the prototype (the research card N1, section 5)
----------------------------------------------------------------
- Fixed shapes: every candidate of a ray is solved in one masked pass (the
  prototype processed rounds of four while any box was nearer than the best
  hit, a host read per round); a ray with more boxes than candidates whose
  best root lies beyond the first unsolved box is flagged (:attr:`last_overflow`).
- A ray leaving the surface is recognised without a caller flag: a lane whose
  box holds the ray's start ``t0`` deflates the root at ``t0`` (the prototype
  was told which rays started on the surface).
- The leaf rule's degenerate samples are judged against the leaf's largest
  tangents, and a split goes across the parameter that turns the tangents
  most: the prototype's pole slivers (64 leaves of the sphere at the depth
  cap, each 1/2048 of the meridian) are gone; the sphere has 336 leaves, the
  bicubic patch of the study 123 (the prototype's count).
- The normal at a collapsed edge (a pole) is the edge's limit (:func:`.kernel.unit_normal`).

The adjoint (ticket D)
----------------------
The iterations run without a tape. At the accepted root ``x* = (s*, r*, t*)``
one attached Newton step with the Jacobian detached is added in zero-valued
form::

    x = x* - J^-1 (F(x*; theta) - stopgrad F(x*; theta)),   J = [S_s, S_r, -d] detached

so the value is ``x*`` to the bit and the derivative is the implicit one,
``dx*/dtheta = -J^-1 dF/dtheta``. ``F`` is evaluated on the leaf's net built
*attached* from the original control points and weights through the leaf's
linear map (knot insertion, degree elevation and subdivision are linear), and
from the attached ray, through which the placement reaches it. The normal is
evaluated attached at the moved ``(s, r)`` and added in the same zero-valued
form, so a forward number never changes with or without a gradient. The leaf
tree, the boxes and the choice of leaf are detached (R-09-11). The gradient
rule: ``control_points`` and ``weights`` are the kind's attached parameters
(class ``interior+boundary`` in the parameter register).
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential._utils import is_tensor
from optiland.nonsequential.components.geometry.base import AABB, ComponentGeometry
from optiland.nonsequential.components.geometry.nurbs import kernel as K
from optiland.nonsequential.components.geometry.nurbs.leaves import (
    DEFAULT_CONE_DEG,
    DEFAULT_MAX_DEPTH,
    DEFAULT_TANGENT_DEG,
    LeafSet,
    build_leaves,
)

#: The contract keys a geometry keeps (everything but the float nets it
#: owns as parameters and the derived Bezier pieces).
_NET_KEYS = ("ctrl_points", "ctrl_weights")


def _requires_grad(value) -> bool:
    """A tensor that carries a derivative: reverse mode, or a forward-mode tangent."""
    if not is_tensor(value):
        return False
    if bool(value.requires_grad):
        return True
    from optiland.nonsequential.parameter_register import _has_tangent  # noqa: PLC0415

    return _has_tangent(value)


def _host(value) -> np.ndarray:
    if is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def _no_grad(like):
    if is_tensor(like):
        import torch  # noqa: PLC0415

        return torch.no_grad()
    return contextlib.nullcontext()


class NurbsGeometry(ComponentGeometry):
    """A set of NURBS patches (see the module docstring).

    Attributes:
        control_points: ``(K, 3)`` the patches' control points, concatenated
            in the contract's order (float64 array, or a tensor, which may
            require a gradient).
        weights: ``(K,)`` their weights (positive), likewise.
        leaves: The host :class:`~.leaves.LeafSet` built from their current
            values.
        last_overflow: Per ray of the last call, whether a nearer root may lie
            in a box beyond the solved candidates (a device array).
        last_steps: Newton steps to convergence of each ray's winning lane (-1
            for a miss).
        last_leaf, last_u, last_v: The winning leaf (-1 for a miss) and the
            patch parameters of each hit.
    """

    def __init__(
        self,
        arrays: Mapping[str, Any] | Any,
        *,
        control_points=None,
        weights=None,
        cone_deg: float = DEFAULT_CONE_DEG,
        tangent_deg: float = DEFAULT_TANGENT_DEG,
        max_depth: int = DEFAULT_MAX_DEPTH,
        n_iter: int = K.DEFAULT_N_ITER,
        k_tol: float = K.DEFAULT_K_TOL,
        n_candidates: int = K.DEFAULT_N_CANDIDATES,
        n_ambiguous: int = K.DEFAULT_N_AMBIGUOUS,
        n_candidates_2: int = K.DEFAULT_N_CANDIDATES_2,
        n_ambiguous_2: int = K.DEFAULT_N_AMBIGUOUS_2,
        second_round_share: float = K.DEFAULT_SECOND_ROUND_SHARE,
    ) -> None:
        """Build the kind from the library's arrays.

        Args:
            arrays: ``PatchSet.to_arrays(bezier=True)`` of ``kgeom.nurbs``, or
                any object with that ``to_arrays`` method (a ``PatchSet``).
            control_points: ``(K, 3)`` to use instead of the contract's
                ``ctrl_points``; a tensor with ``requires_grad`` stays attached.
            weights: ``(K,)`` likewise for ``ctrl_weights``.
            cone_deg, tangent_deg, max_depth: The leaf rule.
            n_iter: Fixed Newton iteration count.
            k_tol: Residual tolerance in units of the working dtype.
            n_candidates, n_ambiguous: Boxes solved per ray in the first round
                (see :func:`.kernel.intersect`).
            n_candidates_2, n_ambiguous_2, second_round_share: The second
                round, for the rays in doubt after the first.

        Raises:
            ValueError: A bad array, a non-positive weight, an option out of range.
            NotImplementedError: A patch trimmed inside its domain.
        """
        if not isinstance(arrays, Mapping) and hasattr(arrays, "to_arrays"):
            arrays = arrays.to_arrays(bezier=True)
        self._arrays = {k: np.asarray(v) for k, v in dict(arrays).items()}
        base_cp = self._arrays["ctrl_points"].astype(np.float64).reshape(-1, 3)
        base_w = self._arrays["ctrl_weights"].astype(np.float64).reshape(-1)
        self.control_points = base_cp if control_points is None else control_points
        self.weights = base_w if weights is None else weights
        if tuple(self.control_points.shape) != base_cp.shape:
            raise ValueError(f"control_points must have shape {base_cp.shape}")
        if tuple(self.weights.shape) != base_w.shape:
            raise ValueError(f"weights must have shape {base_w.shape}")
        if int(n_iter) < 1 or float(k_tol) <= 0 or int(n_candidates) < 1 or int(n_ambiguous) < 0:
            raise ValueError("n_iter, k_tol and n_candidates must be positive, n_ambiguous non-negative")
        self.cone_deg = float(cone_deg)
        self.tangent_deg = float(tangent_deg)
        self.max_depth = int(max_depth)
        self.n_iter = int(n_iter)
        self.k_tol = float(k_tol)
        self.n_candidates = int(n_candidates)
        self.n_ambiguous = int(n_ambiguous)
        self.n_candidates_2 = int(n_candidates_2)
        self.n_ambiguous_2 = int(n_ambiguous_2)
        if not 0.0 < float(second_round_share) <= 1.0:
            raise ValueError("second_round_share must lie in (0, 1]")
        self.second_round_share = float(second_round_share)
        self._leaf_key = None
        self._generation = 0
        self._device_cache: dict = {}
        self.leaves = self._current_leaves()
        self.last_overflow = None
        self.last_steps = None
        self.last_leaf = None
        self.last_u = None
        self.last_v = None

    @classmethod
    def from_patch_set(cls, patch_set, **kw) -> NurbsGeometry:
        """The kind from a ``kgeom.nurbs.PatchSet`` (through its array contract)."""
        return cls(patch_set.to_arrays(bezier=True), **kw)

    # -- the leaves ------------------------------------------------------------

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """The contract as given, with the current net (host float64)."""
        out = dict(self._arrays)
        out["ctrl_points"] = _host(self.control_points)
        out["ctrl_weights"] = _host(self.weights)
        return out

    def _key(self):
        def one(v):
            if is_tensor(v):
                return (id(v), int(v._version))
            return (id(v),)

        return (one(self.control_points), one(self.weights))

    def _current_leaves(self) -> LeafSet:
        """The leaves of the current net, rebuilt when a parameter changed.

        A tensor parameter is recognised as changed by its identity and its
        in-place version counter (an optimiser step), read from Python
        without a host transfer; only a rebuild reads its values.
        """
        key = self._key()
        if key != self._leaf_key:
            cp, w = _host(self.control_points), _host(self.weights)
            # the contract's Bezier pieces describe its own net: compare only against that
            own = np.array_equal(cp, self._arrays["ctrl_points"].reshape(-1, 3)) and np.array_equal(
                w, self._arrays["ctrl_weights"].reshape(-1)
            )
            self.leaves = build_leaves(
                self._arrays, cp, w,
                cone_deg=self.cone_deg, tangent_deg=self.tangent_deg, max_depth=self.max_depth,
                check_pieces=own,
            )
            self._leaf_key = key
            self._generation += 1
            self._device_cache = {}
        return self.leaves

    def _device(self, like):
        """The leaves uploaded to ``like``'s library, dtype and device (cached)."""
        leaves = self._current_leaves()
        dev = str(getattr(like, "device", "cpu"))
        key = (type(like).__module__.split(".")[0], str(like.dtype), dev, self._generation)
        cached = self._device_cache.get(key)
        if cached is not None:
            return cached
        ops = K._Ops(like)
        if ops.torch is not None:
            torch = ops.torch

            def to_array(x):
                return torch.as_tensor(np.asarray(x, dtype=np.float64), device=like.device).to(like.dtype)

            def to_index(x):
                return torch.as_tensor(np.asarray(x, dtype=np.int64), device=like.device)
        else:

            def to_array(x):
                return np.asarray(x, dtype=np.float64).astype(like.dtype)

            def to_index(x):
                return np.asarray(x, dtype=np.int64)

        dl = K.upload(leaves, to_array, u=ops.u)
        adj = {
            "mu": to_array(leaves.mu),
            "mv": to_array(leaves.mv),
            "block": to_index(leaves.block),
            "sign": to_array(leaves.patch_sign[leaves.patch]),
        }
        cached = (dl, adj)
        self._device_cache[key] = cached
        return cached

    # -- the interface ---------------------------------------------------------

    def ray_intersect(self, origins, directions, eps=None):
        """Intersect rays with the patches (module docstring).

        Args:
            origins: Ray origins in the local frame, ``(N, 3)`` [mm].
            directions: Unit directions, ``(N, 3)``.
            eps: The start of each ray (scalar or ``(N,)``): roots at or
                before it are refused, and a box holding it is treated as the
                surface the ray may be leaving. ``None``: a few ulps of the
                origins' scale (a direct call).

        Returns:
            ``(t, normals, hit_mask, n_geom)``; ``n_geom`` is the inward
            normal (module docstring).
        """
        ops = K._Ops(origins)
        dl, adj = self._device(origins)
        if eps is None:
            eps = _tol.accept_t_min(be.abs(origins).max())
        with _no_grad(origins):
            o = origins.detach() if is_tensor(origins) else origins
            d = directions.detach() if is_tensor(directions) else directions
            e = eps.detach() if is_tensor(eps) else eps
            t0 = ops.zeros_like(o[:, 0]) + e
            out = K.intersect(
                ops, dl, o, d, t0, n_iter=self.n_iter, k_tol=self.k_tol,
                n_candidates=self.n_candidates, n_ambiguous=self.n_ambiguous,
                n_candidates_2=self.n_candidates_2, n_ambiguous_2=self.n_ambiguous_2,
                second_round_share=self.second_round_share,
            )
            leaf = out["leaf_safe"]
            hit = out["hit"]
            n_par = K.unit_normal(
                ops, dl.net[leaf], out["s"], out["r"], dl.P, dl.Q, dl.scale[leaf], dl.orient[leaf], ops.u
            )
            sign = adj["sign"][leaf]
        t = out["t"]
        if ops.torch is not None and self._needs_adjoint(origins, directions):
            t, n_par = self._attached_step(ops, dl, adj, origins, directions, out, n_par)
        self.last_overflow = out["overflow"]
        self.last_second_round = out["second_round"]
        self.last_steps = ops.where(hit, out["steps"], ops.full_like(out["steps"], -1.0))
        self.last_leaf = out["leaf"]
        self.last_u = out["u"]
        self.last_v = out["v"]

        inf_arr = ops.full_like(out["t"], float("inf"))
        t_out = ops.where(hit, t, inf_arr)
        n_geom = -(sign[:, None] * n_par)
        n_geom = ops.where(hit[:, None], n_geom, ops.zeros_like(n_geom))
        dot = (directions * n_geom).sum(-1)
        normals = ops.where((dot > 0.0)[:, None], -n_geom, n_geom)
        return t_out, normals, hit, n_geom

    def _needs_adjoint(self, origins, directions) -> bool:
        import torch  # noqa: PLC0415

        from torch.autograd import forward_ad  # noqa: PLC0415

        if not torch.is_grad_enabled() and forward_ad._current_level < 0:
            return False
        return (
            _requires_grad(origins)
            or _requires_grad(directions)
            or _requires_grad(self.control_points)
            or _requires_grad(self.weights)
        )

    def _attached_step(self, ops, dl, adj, origins, directions, out, n_par):
        """The zero-valued attached Newton step at the root (module docstring)."""
        torch = ops.torch
        dtype, device = origins.dtype, origins.device
        leaf = out["leaf_safe"]
        hit = out["hit"]
        cp = self.control_points
        w = self.weights
        cp = cp.to(dtype=dtype, device=device) if is_tensor(cp) else torch.as_tensor(cp, dtype=dtype, device=device)
        w = w.to(dtype=dtype, device=device) if is_tensor(w) else torch.as_tensor(w, dtype=dtype, device=device)
        pw = torch.cat([cp * w[:, None], w[:, None]], dim=1)
        blk = adj["block"][leaf]
        pw_blk = pw[blk]
        net = torch.einsum("nai,nbj,nijc->nabc", adj["mu"][leaf], adj["mv"][leaf], pw_blk)
        cen = dl.pivot[leaf]
        net = torch.cat([net[..., :3] - cen[:, None, None, :] * net[..., 3:4], net[..., 3:4]], dim=-1)
        s0 = out["s"].detach()
        r0 = out["r"].detach()
        t0 = torch.where(hit, out["t"], torch.zeros_like(out["t"])).detach()
        S, Ss, Sr = K.eval_leaf(ops, net, s0, r0, dl.P, dl.Q)
        F = S - (origins - cen) - t0[:, None] * directions
        Fz = F - F.detach()
        x1, x2, x3, det = K.solve3(ops, Ss.detach(), Sr.detach(), -directions.detach(), Fz)
        good = hit & (det != 0.0) & torch.isfinite(det)
        zero = torch.zeros_like(x1)
        x1 = torch.where(good, x1, zero)
        x2 = torch.where(good, x2, zero)
        x3 = torch.where(good, x3, zero)
        t = out["t"] - x3
        s = s0 - x1
        r = r0 - x2
        n_att = K.unit_normal(ops, net, s, r, dl.P, dl.Q, dl.scale[leaf], dl.orient[leaf], ops.u)
        n = n_par + (n_att - n_att.detach())
        return t, n

    # -- bookkeeping -----------------------------------------------------------

    def overflow_count(self) -> int:
        """Rays of the last call that may have lost a nearer root (reads to the host)."""
        if self.last_overflow is None:
            return 0
        v = self.last_overflow
        return int((v.detach().cpu().numpy() if is_tensor(v) else np.asarray(v)).sum())

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """AABB of the control points in global coordinates (the convex hull
        holds: every weight is positive). Host bookkeeping, detached."""
        t_vec = np.array(transform[0], dtype=float)
        R = np.array(transform[1], dtype=float)
        pts = _host(self.control_points) @ R.T + t_vec
        return AABB(pts.min(axis=0), pts.max(axis=0))

    def detached_copy(self) -> NurbsGeometry:
        """A copy whose net is plain float64 arrays."""
        return NurbsGeometry(
            self.arrays, cone_deg=self.cone_deg, tangent_deg=self.tangent_deg,
            max_depth=self.max_depth, n_iter=self.n_iter, k_tol=self.k_tol,
            n_candidates=self.n_candidates, n_ambiguous=self.n_ambiguous,
            n_candidates_2=self.n_candidates_2, n_ambiguous_2=self.n_ambiguous_2,
            second_round_share=self.second_round_share,
        )

    def lowered_params(self) -> dict:
        """The kind's IR params: the contract as lists, the net as given
        (a tensor stays a tensor), and the solver's settings."""
        arrays = {
            k: v.tolist()
            for k, v in self._arrays.items()
            if k not in _NET_KEYS and not k.startswith("bezier_") and k != "patch_bezier_offset"
        }
        cp, w = self.control_points, self.weights
        return {
            "arrays": arrays,
            "control_points": cp if is_tensor(cp) else np.asarray(cp).tolist(),
            "weights": w if is_tensor(w) else np.asarray(w).tolist(),
            "cone_deg": self.cone_deg,
            "tangent_deg": self.tangent_deg,
            "max_depth": self.max_depth,
            "n_iter": self.n_iter,
            "k_tol": self.k_tol,
            "n_candidates": self.n_candidates,
            "n_ambiguous": self.n_ambiguous,
            "n_candidates_2": self.n_candidates_2,
            "n_ambiguous_2": self.n_ambiguous_2,
            "second_round_share": self.second_round_share,
        }


# ---------------------------------------------------------------------------
# The JSON forms
# ---------------------------------------------------------------------------


def arrays_to_json(arrays: Mapping[str, Any]) -> dict:
    """The array contract as JSON-ready lists (the derived ``bezier_*`` keys
    dropped: the build recomputes them)."""
    out = {}
    for k, v in arrays.items():
        if k.startswith("bezier_") or k == "patch_bezier_offset":
            continue
        out[k] = v.detach().cpu().numpy().tolist() if is_tensor(v) else np.asarray(v).tolist()
    return out


def _offsets(counts) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))]).astype(np.int64)


def contract_from_patch_set_dict(d: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The array contract from the geometry library's JSON form of a patch set
    (``kgeom.nurbs.PatchSet.to_dict()``: ``{"kind": "patch_set", "unit", "patches":
    [{"surface": {...}, "loops": [...], "reversed", "uv_bounds", ...}]}``),
    without importing the library. Face keys are not carried (the kind does
    not read them)."""
    patches = list(d.get("patches", ()))
    if not patches:
        raise ValueError("a NURBS patch set needs at least one patch")
    deg, nctrl, cps, ws, ku, kv, dom, uvb, rev = [], [], [], [], [], [], [], [], []
    loops_per, outer, curves_per, c_deg, c_pts, c_w, c_knots = [], [], [], [], [], [], []
    for p in patches:
        s = p["surface"]
        P = np.asarray(s["control_points"], dtype=np.float64)
        nu, nv = P.shape[:2]
        W = np.ones((nu, nv)) if s.get("weights") is None else np.asarray(s["weights"], dtype=np.float64)
        U = np.asarray(s["knots_u"], dtype=np.float64)
        V = np.asarray(s["knots_v"], dtype=np.float64)
        pu, qv = int(s["degree_u"]), int(s["degree_v"])
        deg.append((pu, qv))
        nctrl.append((nu, nv))
        cps.append(P.reshape(-1, 3))
        ws.append(W.reshape(-1))
        ku.append(U)
        kv.append(V)
        domain = (U[pu], U[nu], V[qv], V[nv])
        dom.append(domain)
        uvb.append(tuple(p.get("uv_bounds") or domain))
        rev.append(bool(p.get("reversed", False)))
        loops = list(p.get("loops", ()))
        loops_per.append(len(loops))
        for lp in loops:
            outer.append(bool(lp.get("outer", True)))
            curves_per.append(len(lp["curves"]))
            for c in lp["curves"]:
                pts = np.asarray(c["control_points"], dtype=np.float64).reshape(-1, 2)
                c_deg.append(int(c["degree"]))
                c_pts.append(pts)
                c_w.append(np.ones(len(pts)) if c.get("weights") is None else np.asarray(c["weights"], dtype=np.float64))
                c_knots.append(np.asarray(c["knots"], dtype=np.float64))
    return {
        "format": np.array(1), "unit": np.array(str(d.get("unit", "mm"))),
        "provenance": np.array(str(d.get("provenance", ""))),
        "patch_degree": np.array(deg, dtype=np.int32), "patch_n_ctrl": np.array(nctrl, dtype=np.int32),
        "patch_ctrl_offset": _offsets([len(w) for w in ws]), "ctrl_points": np.concatenate(cps),
        "ctrl_weights": np.concatenate(ws),
        "patch_knot_u_offset": _offsets([len(k) for k in ku]), "knots_u": np.concatenate(ku),
        "patch_knot_v_offset": _offsets([len(k) for k in kv]), "knots_v": np.concatenate(kv),
        "patch_domain": np.array(dom), "patch_uv_bounds": np.array(uvb), "patch_reversed": np.array(rev, dtype=bool),
        "patch_loop_offset": _offsets(loops_per), "loop_outer": np.array(outer, dtype=bool),
        "loop_curve_offset": _offsets(curves_per), "curve_degree": np.array(c_deg, dtype=np.int32),
        "curve_ctrl_offset": _offsets([len(w) for w in c_w]),
        "curve_ctrl_points": np.concatenate(c_pts) if c_pts else np.zeros((0, 2)),
        "curve_weights": np.concatenate(c_w) if c_w else np.zeros(0),
        "curve_knot_offset": _offsets([len(k) for k in c_knots]),
        "curve_knots": np.concatenate(c_knots) if c_knots else np.zeros(0),
    }


def arrays_from_json(value: Any) -> dict[str, np.ndarray]:
    """The array contract from any form a scene may carry: the contract itself
    (arrays or lists, as :func:`arrays_to_json` writes it), the library's
    ``PatchSet.to_dict()`` form, or a ``PatchSet`` (anything with
    ``to_arrays``)."""
    if not isinstance(value, Mapping) and hasattr(value, "to_arrays"):
        return dict(value.to_arrays(bezier=True))
    if not isinstance(value, Mapping):
        raise TypeError(f"a NURBS surface is a mapping or a patch set, got {type(value).__name__}")
    if value.get("kind") == "patch_set":
        return contract_from_patch_set_dict(value)
    if "patch_degree" not in value:
        raise ValueError("a NURBS mapping is either the array contract or a patch set's to_dict form")
    return {k: np.asarray(v) for k, v in value.items()}
