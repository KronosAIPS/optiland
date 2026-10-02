"""The NURBS geometry kind (KronosNSRT issue 66, tickets B and D).

What each class pins, and the route it uses:

- ``TestBuild``: the host leaves. Bezier extraction against the contract's own
  pieces and against the geometry library's closed-form evaluation (when the
  library is installed); a trimmed patch is refused; mixed degrees are raised
  to one without moving a point; the leaf counts of the study's two surfaces.
- ``TestSphere``: the exact rational sphere (degree 2 x 2, 9 x 5 net), written
  out here as the library writes it, against the engine's analytic sphere
  kind on 1e4 rays in the CAD study's three families (exterior, leaving the
  sphere, aimed at a pole): hit sets and roots within the root's own
  conditioning bound, normals against the exact radial normal, at numpy
  float64, torch float64 and torch float32. The float32 limit of ticket F
  (short re-hits of rays leaving the surface at grazing incidence) is asserted
  as the limit it is, not hidden.
- ``TestPole``: the on-axis ray and rays 1e-12 to 1e-3 mm off it hit the pole of
  a revolved surface at the right distance with the axis as normal.
- ``TestAsphere``: the library's ``revolve_asphere`` against the engine's
  even-asphere kind, within a tolerance derived from the generator's stated
  sag error before any engine run.
- ``TestAdjoint``: dt/dR on the sphere against the closed form R / (p . d);
  dt and the normal with respect to a control point, a weight and the ray
  origin against fourth-order central differences of the primal root.
- ``TestTraced``: a NURBS sphere in the bounce loop against the analytic
  sphere in the same scene; the emulated graph replay of a scene with a NURBS
  surface equals the eager fixed-width trace and transfers nothing to or from
  the host.
- ``TestRegistration``: the kind, its lowering, the parameter register's
  contract, the value unchanged in gradient mode.
- ``TestAppleGPU``: the kind at float32 on the Apple GPU (research repository
  issue 99): the sphere against the same rays on the CPU, and reverse mode
  reaching float64 host parameters, with a control of the torch defect behind
  the second. Skipped where the device is not reachable; the upload's host
  rounding is checked on the CPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    NurbsGeometry,
    ReflectiveComponent,
    SphereGeometry,
    Spectrum,
    kinds,
)
from optiland.nonsequential import _tol
from optiland.nonsequential.components.geometry.nurbs import build_leaves
from optiland.nonsequential.components.geometry.nurbs import kernel as K
from optiland.nonsequential.ir.lower import _lower_geometry

torch = pytest.importorskip("torch", reason="Torch not available")

SQ = math.sqrt(0.5)


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _set(backend: str, precision: str) -> None:
    be.set_backend(backend)
    be.set_precision(precision)


def _np(x):
    return np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x)


# ---------------------------------------------------------------------------
# The contract, written out (the geometry library's layout, format 1)
# ---------------------------------------------------------------------------


def contract(surfaces, loops=None):
    """The array contract of ``kgeom.nurbs.PatchSet.to_arrays`` for untrimmed
    patches ``(p, q, U, V, P (nu, nv, 3), W (nu, nv), reversed)``; ``loops``
    maps a patch index to a list of trim curves (degree-1 polylines in (u, v))
    making one outer loop."""
    loops = loops or {}
    P = len(surfaces)
    a = {"format": np.array(1), "unit": np.array("mm"), "provenance": np.array("test")}
    deg, nctrl, cps, ws, ku, kv, dom, rev = [], [], [], [], [], [], [], []
    loop_count, curve_count, c_deg, c_pts, c_w, c_knots = [], [], [], [], [], []
    for i, (p, q, U, V, Pts, W, reversed_) in enumerate(surfaces):
        Pts = np.asarray(Pts, float)
        nu, nv = Pts.shape[:2]
        deg.append((p, q))
        nctrl.append((nu, nv))
        cps.append(Pts.reshape(-1, 3))
        ws.append(np.asarray(W, float).reshape(-1))
        ku.append(np.asarray(U, float))
        kv.append(np.asarray(V, float))
        dom.append((U[p], U[nu], V[q], V[nv]))
        rev.append(reversed_)
        curves = loops.get(i, [])
        loop_count.append(1 if curves else 0)
        if curves:
            curve_count.append(len(curves))
            for poly in curves:
                poly = np.asarray(poly, float)
                c_deg.append(1)
                c_pts.append(poly)
                c_w.append(np.ones(len(poly)))
                c_knots.append(np.concatenate([[0.0], np.linspace(0, 1, len(poly)), [1.0]]))

    def off(xs):
        return np.concatenate([[0], np.cumsum([len(x) for x in xs])]).astype(np.int64)

    a.update(
        patch_degree=np.array(deg, np.int32), patch_n_ctrl=np.array(nctrl, np.int32),
        patch_ctrl_offset=off(cps), ctrl_points=np.concatenate(cps), ctrl_weights=np.concatenate(ws),
        patch_knot_u_offset=off(ku), knots_u=np.concatenate(ku), patch_knot_v_offset=off(kv),
        knots_v=np.concatenate(kv), patch_domain=np.array(dom), patch_uv_bounds=np.array(dom),
        patch_reversed=np.array(rev, bool),
        patch_loop_offset=np.concatenate([[0], np.cumsum(loop_count)]).astype(np.int64),
        loop_outer=np.ones(sum(loop_count), bool),
        loop_curve_offset=np.concatenate([[0], np.cumsum(curve_count)]).astype(np.int64),
        curve_degree=np.array(c_deg, np.int32),
        curve_ctrl_offset=off(c_pts), curve_ctrl_points=np.concatenate(c_pts) if c_pts else np.zeros((0, 2)),
        curve_weights=np.concatenate(c_w) if c_w else np.zeros(0),
        curve_knot_offset=off(c_knots), curve_knots=np.concatenate(c_knots) if c_knots else np.zeros(0),
    )
    return a


CIRCLE_U = [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]
CIRCLE_P = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]], float)
CIRCLE_W = np.array([1, SQ, 1, SQ, 1, SQ, 1, SQ, 1])


def revolved(profile_knots, profile_rz, profile_w, degree):
    """A surface of revolution about z (Piegl and Tiller A8.1), as the library builds it."""
    prof = np.asarray(profile_rz, float)
    cp = np.empty((9, len(prof), 3))
    cp[..., 0] = CIRCLE_P[:, None, 0] * prof[None, :, 0]
    cp[..., 1] = CIRCLE_P[:, None, 1] * prof[None, :, 0]
    cp[..., 2] = prof[None, :, 1]
    return (2, degree, CIRCLE_U, list(profile_knots), cp, CIRCLE_W[:, None] * np.asarray(profile_w)[None, :])


def sphere_surface(R):
    """The exact rational sphere: degree (2, 2), 9 x 5 net, S_u x S_v outward."""
    s = revolved([0, 0, 0, 0.5, 0.5, 1, 1, 1], [[0, -R], [R, -R], [R, 0], [R, R], [0, R]], [1, SQ, 1, SQ, 1], 2)
    return s + (False,)


def wavy_surface(seed=3):
    """The CAD study's bicubic rational patch: 6 x 6 net, 9 Bezier pieces, slopes near 1."""
    rng = np.random.default_rng(seed)
    n = 6
    xs = np.linspace(-10, 10, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Z = 3.0 * np.sin(X / 3.5) * np.cos(Y / 4.5) + 0.04 * (X**2 - Y**2) / 4 + rng.uniform(-1.0, 1.0, (n, n))
    X = X + rng.uniform(-0.6, 0.6, (n, n))
    Y = Y + rng.uniform(-0.6, 0.6, (n, n))
    W = rng.uniform(0.7, 1.4, (n, n))
    return (3, 3, [0, 0, 0, 0, 0.3, 0.65, 1, 1, 1, 1], [0, 0, 0, 0, 0.42, 0.7, 1, 1, 1, 1],
            np.stack([X, Y, Z], -1), W, False)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sphere_rays(n, R=10.0, seed=20260927):
    """The CAD study's families: 60 % exterior from 30 mm into a disc of 1.15 R,
    25 % leaving the sphere inward, 15 % aimed within 0.5 mm of a pole."""
    rng = np.random.default_rng(seed)
    nA, nB = int(0.60 * n), int(0.25 * n)
    nC = n - nA - nB
    w = unit(rng.normal(size=(nA, 3)))
    oA = 30.0 * w
    a = np.where(np.abs(w[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = unit(np.cross(w, a))
    e2 = np.cross(w, e1)
    rr = 1.15 * R * np.sqrt(rng.uniform(size=nA))
    ph = rng.uniform(0, 2 * np.pi, nA)
    dA = unit(rr[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2) - oA)
    nrm = unit(rng.normal(size=(nB, 3)))
    oB = R * nrm
    dB = unit(rng.normal(size=(nB, 3)))
    dB = np.where((dB * nrm).sum(1, keepdims=True) > 0, -dB, dB)
    pole = np.where(rng.uniform(size=nC) < 0.5, 1.0, -1.0)
    rr = 0.5 * np.sqrt(rng.uniform(size=nC))
    ph = rng.uniform(0, 2 * np.pi, nC)
    tgt = np.stack([rr * np.cos(ph), rr * np.sin(ph), pole * R * np.sqrt(1 - (rr / R) ** 2)], 1)
    wC = unit(rng.normal(size=(nC, 3)))
    wC[:, 2] = np.abs(wC[:, 2]) * pole
    oC = tgt + 25.0 * wC
    dC = unit(tgt - oC)
    fam = np.concatenate([np.zeros(nA), np.ones(nB), 2 * np.ones(nC)]).astype(int)
    return np.concatenate([oA, oB, oC]), np.concatenate([dA, dB, dC]), fam


def _arrays(*surfaces):
    return contract(list(surfaces))


# ---------------------------------------------------------------------------


class TestBuild:
    def test_leaf_counts(self):
        # The sphere: 8 pieces; the leaf rule with degenerate samples judged at
        # the leaf's own tangent scale gives 336 leaves (the prototype's 512
        # included 64 pole slivers at its depth cap). The bicubic patch: 123,
        # the prototype's count.
        assert build_leaves(_arrays(sphere_surface(10.0))).n == 336
        lv = build_leaves(_arrays(wavy_surface()))
        assert lv.n == 123 and lv.n_pieces == 9
        assert np.degrees(lv.cone.max()) <= 15.0 + 1e-6

    def test_leaf_nets_are_the_surface(self):
        # Every leaf's net evaluated at random (s, r) equals the patch at the
        # leaf's (u, v): the extraction, elevation and subdivision are exact.
        from optiland.nonsequential.components.geometry.nurbs.leaves import eval_net  # noqa: PLC0415

        p, q, U, V, P, W, _ = wavy_surface()
        lv = build_leaves(_arrays(wavy_surface()))
        rng = np.random.default_rng(0)
        worst = 0.0
        for k in range(lv.n):
            s, r = rng.uniform(0, 1, (2, 3))
            S, _, _ = eval_net(lv.net[k], s, r)
            pr = lv.prange[k]
            ref = _cox_de_boor(P, W, p, q, np.array(U, float), np.array(V, float),
                               pr[0] + s * (pr[1] - pr[0]), pr[2] + r * (pr[3] - pr[2]))
            worst = max(worst, float(np.abs(S - ref).max()))
        assert worst < 1e-13

    def test_against_the_library(self):
        kn = pytest.importorskip("kgeom.nurbs", reason="the geometry library is not installed")
        # the written-out sphere is the library's, to the bit
        a = kn.PatchSet((kn.TrimmedPatch.untrimmed(kn.sphere(10.0)),), unit="mm").to_arrays(bezier=True)
        mine = _arrays(sphere_surface(10.0))
        for key in ("ctrl_points", "ctrl_weights", "knots_u", "knots_v", "patch_domain"):
            assert np.array_equal(a[key], mine[key]), key
        # pieces checked against the contract's inside the build, and points
        for surface in (kn.sphere(10.0), kn.revolve_conic(50.0, -1.0, 12.0),
                        kn.revolve_asphere(50.0, -0.6, (1e-5, -2e-8), 12.5).surface):
            arr = kn.PatchSet((kn.TrimmedPatch.untrimmed(surface),), unit="mm").to_arrays(bezier=True)
            lv = build_leaves(arr)
            from optiland.nonsequential.components.geometry.nurbs.leaves import eval_net  # noqa: PLC0415

            rng = np.random.default_rng(1)
            for k in rng.integers(0, lv.n, 40):
                s, r = rng.uniform(0, 1, (2, 4))
                S, _, _ = eval_net(lv.net[k], s, r)
                pr = lv.prange[k]
                ref = kn.evaluate(surface, pr[0] + s * (pr[1] - pr[0]), pr[2] + r * (pr[3] - pr[2]))
                assert np.abs(S - ref).max() < 1e-12

    def test_trimmed_patch_is_trimmed_and_the_rectangle_untrimmed(self):
        # Until ticket C (device trimming) a loop inside the domain was refused;
        # it now builds a trimmed set (tests in test_nsq_nurbs_trim.py). A loop
        # on the domain's rectangle still builds no trim data at all.
        s = wavy_surface()
        rect = [[[0, 0], [1, 0]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [0, 0]]]
        assert NurbsGeometry(contract([s], {0: rect})).leaves.piece_edges is None
        inner = [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]]]
        lv = NurbsGeometry(contract([s], {0: inner})).leaves
        assert lv.piece_edges is not None and lv.patch_trimmed[0] and lv.n_dropped > 0

    def test_bad_weight_and_pieces(self):
        a = _arrays(sphere_surface(10.0))
        a["ctrl_weights"] = a["ctrl_weights"].copy()
        a["ctrl_weights"][3] = -1.0
        with pytest.raises(ValueError, match="positive"):
            NurbsGeometry(a)

    def test_mixed_degrees_are_raised_without_moving_a_point(self):
        # A bilinear patch (degree 1 x 1) and the bicubic patch in one set: the
        # bilinear one is raised to 3 x 3 and still evaluates to its plane.
        plane = (1, 1, [0, 0, 1, 1], [0, 0, 1, 1],
                 np.array([[[-5, -5, 20.0], [-5, 5, 20.0]], [[5, -5, 20.0], [5, 5, 20.0]]]), np.ones((2, 2)), False)
        lv = build_leaves(contract([wavy_surface(), plane]))
        assert lv.degree == (3, 3)
        from optiland.nonsequential.components.geometry.nurbs.leaves import eval_net  # noqa: PLC0415

        k = np.where(lv.patch == 1)[0]
        S, _, _ = eval_net(lv.net[k[0]], np.array([0.3, 0.9]), np.array([0.6, 0.1]))
        assert np.allclose(S[:, 2], 20.0, rtol=0, atol=1e-14)
        # a ray down the axis meets the plane first
        g = NurbsGeometry(contract([wavy_surface(), plane]))
        t, _, hit, _ = g.ray_intersect(np.array([[1.0, 1.0, 40.0]]), np.array([[0.0, 0.0, -1.0]]))
        assert hit[0] and t[0] == pytest.approx(20.0, abs=1e-12)


def _cox_de_boor(P, W, p, q, U, V, u, v):
    """Reference evaluation, span-local Cox-de Boor (Piegl and Tiller A2.1, A2.2, A4.3)."""
    def span(n, deg, K, t):
        if t >= K[n + 1]:
            return n
        lo, hi = deg, n + 1
        mid = (lo + hi) // 2
        while t < K[mid] or t >= K[mid + 1]:
            if t < K[mid]:
                hi = mid
            else:
                lo = mid
            mid = (lo + hi) // 2
        return mid

    def basis(i, t, deg, K):
        N = np.zeros(deg + 1)
        left = np.zeros(deg + 1)
        right = np.zeros(deg + 1)
        N[0] = 1.0
        for j in range(1, deg + 1):
            left[j] = t - K[i + 1 - j]
            right[j] = K[i + j] - t
            saved = 0.0
            for r in range(j):
                tmp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * tmp
                saved = left[j - r] * tmp
            N[j] = saved
        return N

    out = []
    Pw = np.concatenate([P * W[..., None], W[..., None]], -1)
    for a, b in zip(np.atleast_1d(u), np.atleast_1d(v)):
        su = span(P.shape[0] - 1, p, U, a)
        sv = span(P.shape[1] - 1, q, V, b)
        A = np.einsum("i,j,ijc->c", basis(su, a, p, U), basis(sv, b, q, V), Pw[su - p : su + 1, sv - q : sv + 1])
        out.append(A[:3] / A[3])
    return np.array(out)


# ---------------------------------------------------------------------------


def _trace_sphere(backend, precision, o, d, R=10.0):
    """NURBS and analytic sphere on the same rays, each with the loop's per-ray threshold."""
    _set(backend, precision)
    dt = np.float64 if precision == "float64" else np.float32
    oc, dc = o.astype(dt), d.astype(dt)
    if backend == "torch":
        O, D = torch.as_tensor(oc), torch.as_tensor(dc)
        eps = _tol.accept_t_min(O.abs().amax(dim=1))
    else:
        O, D = oc, dc
        eps = _tol.accept_t_min(np.abs(O).max(axis=1))
    g = NurbsGeometry(_arrays(sphere_surface(R)))
    t, n, hit, ng = g.ray_intersect(O, D, eps=eps)
    ta, _, ha, nga = SphereGeometry(R).ray_intersect(O, D, eps=eps)
    return (g, oc.astype(float), dc.astype(float), _np(t).astype(float), _np(hit), _np(ng).astype(float),
            _np(ta).astype(float), _np(ha), _np(nga).astype(float))


class TestSphere:
    """kgeom's exact rational sphere against the analytic sphere kind, 1e4 rays."""

    N = 10000

    @pytest.mark.parametrize(("backend", "precision"), [("numpy", "float64"), ("torch", "float64"),
                                                         ("torch", "float32")])
    def test_against_the_analytic_kind(self, backend, precision):
        o, d, fam = sphere_rays(self.N)
        g, oc, dc, t, hit, ng, ta, ha, nga = _trace_sphere(backend, precision, o, d)
        assert g.overflow_count() == 0
        u = 2.0**-53 if precision == "float64" else 2.0**-24
        leaving = fam == 1
        # The analytic kind's own root for a ray leaving the sphere can be the
        # point it leaves from (|t| ~ 1e-13 mm at float64, beyond the loop's
        # threshold when rounding put the origin inside): not a comparison.
        a_self = leaving & ha & (ta < 1e-9 if precision == "float64" else ta < 1e-4)
        cos = np.abs((nga * dc).sum(1))
        # The bound: a root is conditioned by 1/|cos| at the coordinate scale
        # (30 mm): 64 units of the dtype there over |cos|, never below 1e-3.
        bound = 64 * u * 30.0 / np.maximum(cos, 1e-3)
        cmp = ha & hit & ~a_self
        assert np.all(np.abs(t[cmp] - ta[cmp]) <= bound[cmp])
        # hit sets: equal at float64; at float32 the only difference allowed is
        # ticket F's: a ray leaving the surface at grazing incidence whose
        # re-hit lies inside the self root's float32 uncertainty.
        lost = ha & ~hit & ~a_self
        false = hit & ~ha
        assert not false.any()
        if precision == "float64":
            assert not lost.any()
        else:
            assert np.all(leaving[lost])
            chord = ta[lost]
            assert np.all(chord < 0.06), chord.max()
            assert lost.sum() <= 0.01 * leaving.sum()
        # normals against the exact inward normal at the kind's own hit point
        # (the radial direction there; the analytic float32 root is itself off
        # by up to 1e-3 mm near a pole, which would tilt the reference)
        both = cmp & (np.abs(t - ta) <= bound)
        p = oc[both] + t[both, None] * dc[both]
        exact = -unit(p)
        ngu = unit(ng[both])
        ang = np.arctan2(np.linalg.norm(np.cross(exact, ngu), axis=1), (exact * ngu).sum(1))
        assert ang.max() < (1e-7 if precision == "float64" else 1e-4)


class TestPole:
    """The collapsed edge of a revolved surface lies on the axis."""

    @pytest.mark.parametrize(("backend", "precision"), [("numpy", "float64"), ("torch", "float32")])
    def test_on_axis_rays(self, backend, precision):
        _set(backend, precision)
        g = NurbsGeometry(_arrays(sphere_surface(10.0)))
        off = np.array([0.0, 1e-12, 1e-9, 1e-6, 1e-3, 0.1])
        o = np.stack([off, np.zeros_like(off), np.full_like(off, 30.0)], 1)
        d = np.tile([0.0, 0.0, -1.0], (off.size, 1))
        dt = np.float64 if precision == "float64" else np.float32
        O, D = (torch.as_tensor(o.astype(dt)), torch.as_tensor(d.astype(dt))) if backend == "torch" else (o, d)
        t, _, hit, ng = g.ray_intersect(O, D)
        t, hit, ng = _np(t).astype(float), _np(hit), _np(ng).astype(float)
        assert hit.all()
        exact_t = 30.0 - np.sqrt(100.0 - off**2)
        tol = 1e-12 if precision == "float64" else 1e-4
        assert np.abs(t - exact_t).max() <= tol
        exact_n = -unit(np.stack([off, np.zeros_like(off), np.sqrt(100 - off**2)], 1))
        ang = np.arccos(np.clip((unit(ng) * exact_n).sum(1), -1, 1))
        assert ang.max() < (1e-7 if precision == "float64" else 1e-3)


class TestAsphere:
    """The library's revolved asphere against the engine's even-asphere kind."""

    def test_within_the_stated_representation_error(self):
        kn = pytest.importorskip("kgeom.nurbs", reason="the geometry library is not installed")
        from optiland.nonsequential import EvenAsphereGeometry  # noqa: PLC0415

        R, k, coeffs, h = 50.0, -0.6, (0.0, 1e-5, -2e-8), 12.5
        # both conventions put coefficients[0] on r^2 (the library's
        # kgeom.optics.sag and the engine's EvenAsphere)
        fit = kn.revolve_asphere(R, k, coeffs, h)
        err = fit.max_sag_error
        # the generator's statement for this surface at 65 samples (3.4e-8 mm);
        # the comparison's tolerance below is built from it, whatever it is
        assert np.isfinite(err) and err < 1e-6
        # The NURBS face runs toward -z as the library's sag normal; the kind's
        # hit is the same point either way.
        g = NurbsGeometry(kn.PatchSet((kn.TrimmedPatch.untrimmed(fit.surface),), unit="mm").to_arrays(bezier=True))
        a = EvenAsphereGeometry(R, k, h, list(coeffs))
        # rays aimed at points of the asphere inside 0.9 of the aperture, from
        # 30 mm in directions within about 22 degrees of the axis
        rng = np.random.default_rng(4)
        n = 4000
        xy = rng.uniform(-0.9 * h, 0.9 * h, (n, 2))
        xy = xy[np.hypot(xy[:, 0], xy[:, 1]) < 0.9 * h]
        tgt = np.column_stack([xy, a.sag(xy[:, 0], xy[:, 1])])
        d = unit(np.column_stack([rng.uniform(-0.4, 0.4, (len(tgt), 2)), np.ones(len(tgt))]))
        o = tgt - 30.0 * d
        t, _, hit, _ = g.ray_intersect(o, d)
        ta, _, ha, nga = a.ray_intersect(o, d)
        assert np.array_equal(hit, ha) and ha.sum() == len(tgt)
        # Tolerance, from the stated sag error before any engine run: a surface
        # displaced by at most err along z moves a root by err |n_z| / |n . d|
        # to first order; 1.05 covers the sampled maximum and the second order,
        # 64 ulps of 30 mm the two roots' own rounding.
        cos = np.abs((nga * d).sum(1))
        bound = 1.05 * err * np.abs(nga[:, 2]) / cos + 64 * np.spacing(30.0)
        assert np.all(np.abs(t - ta) <= bound)


class TestAdjoint:
    """The attached Newton step against a closed form and finite differences."""

    def test_sphere_dt_dR_closed_form(self):
        _set("torch", "float64")
        fw = torch.autograd.forward_ad
        a = _arrays(sphere_surface(1.0))
        cp1 = torch.as_tensor(a["ctrl_points"])
        rng = np.random.default_rng(7)
        n = 2000
        w = unit(rng.normal(size=(n, 3)))
        o = 30.0 * w
        aa = np.where(np.abs(w[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = unit(np.cross(w, aa))
        e2 = np.cross(w, e1)
        rr = 9.99 * np.sqrt(rng.uniform(size=n))
        ph = rng.uniform(0, 2 * np.pi, n)
        d = unit(rr[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2) - o)
        with fw.dual_level():
            R = fw.make_dual(torch.tensor(10.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
            g = NurbsGeometry(a, control_points=cp1 * R)
            t, _, hit, _ = g.ray_intersect(torch.as_tensor(o), torch.as_tensor(d))
            tp, dt = fw.unpack_dual(t)
        hit = _np(hit)
        assert hit.all()
        p = o + _np(tp)[:, None] * d
        closed = 10.0 / (p * d).sum(1)
        rel = np.abs(_np(dt) - closed) / np.abs(closed)
        assert np.median(rel) < 1e-14 and rel.max() < 1e-12

    @pytest.mark.parametrize("name", ["control_point_z", "weight", "origin_z"])
    def test_patch_against_finite_differences(self, name):
        _set("torch", "float64")
        fw = torch.autograd.forward_ad
        a = _arrays(wavy_surface())
        cp0, w0 = a["ctrl_points"].copy(), a["ctrl_weights"].copy()
        rng = np.random.default_rng(11)
        n = 120
        o = np.stack([rng.uniform(-9, 9, n), rng.uniform(-9, 9, n), np.full(n, 40.0)], 1)
        d = unit(np.stack([rng.uniform(-8, 8, n), rng.uniform(-8, 8, n), np.zeros(n)], 1) - o)
        k_cp, k_w = 3 * 6 + 2, 2 * 6 + 3
        h = {"control_point_z": 1e-3, "weight": 1e-4, "origin_z": 1e-4}[name]

        def perturbed(x):
            cp, wt, oz = torch.as_tensor(cp0), torch.as_tensor(w0), torch.as_tensor(o)
            if name == "control_point_z":
                e = torch.zeros_like(cp)
                e[k_cp, 2] = 1.0
                cp = cp + x * e
            elif name == "weight":
                e = torch.zeros_like(wt)
                e[k_w] = 1.0
                wt = wt + x * e
            else:
                e = torch.zeros_like(oz)
                e[:, 2] = 1.0
                oz = oz + x * e
            g = NurbsGeometry(a, control_points=cp, weights=wt)
            return g.ray_intersect(oz, torch.as_tensor(d))

        with fw.dual_level():
            x = fw.make_dual(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
            t, _, hit, ng = perturbed(x)
            ad = np.stack([_np(fw.unpack_dual(t)[1]), _np(fw.unpack_dual(ng)[1])[:, 2]])
        hit = _np(hit)
        vals = {}
        with torch.no_grad():
            for k in (-2, -1, 1, 2):
                t_, _, h_, n_ = perturbed(torch.tensor(k * h, dtype=torch.float64))
                assert np.array_equal(_np(h_), hit)
                vals[k] = np.stack([_np(t_), _np(n_)[:, 2]])
        fd = (vals[-2] - 8 * vals[-1] + 8 * vals[1] - vals[2]) / (12 * h)
        rows = np.where(hit)[0]
        assert rows.size > 60
        scale = np.abs(fd[:, rows]).max(axis=1, keepdims=True)
        err = np.abs(ad[:, rows] - fd[:, rows]) / scale
        assert err.max() < 1e-8, err.max()
        assert np.abs(ad[0, rows]).max() > 0

    def test_reverse_mode_reaches_control_points_and_weights(self):
        _set("torch", "float64")
        a = _arrays(wavy_surface())
        cp = torch.tensor(a["ctrl_points"], requires_grad=True)
        w = torch.tensor(a["ctrl_weights"], requires_grad=True)
        g = NurbsGeometry(a, control_points=cp, weights=w)
        o = torch.tensor([[1.0, -2.0, 40.0], [3.0, 2.0, 40.0]], dtype=torch.float64)
        d = torch.tensor([[0.0, 0.0, -1.0], [0.1, 0.0, -1.0]], dtype=torch.float64)
        d = d / d.norm(dim=1, keepdim=True)
        t, _, hit, _ = g.ray_intersect(o, d)
        assert bool(hit.all())
        t.sum().backward()
        assert cp.grad is not None and float(cp.grad.abs().sum()) > 0
        assert w.grad is not None and float(w.grad.abs().sum()) > 0

    def test_gradient_mode_does_not_move_a_value(self):
        _set("torch", "float64")
        a = _arrays(wavy_surface())
        o, d, _ = sphere_rays(400)
        o = o * 0.5 + np.array([0, 0, 25.0])
        O, D = torch.as_tensor(o), torch.as_tensor(d)
        g0 = NurbsGeometry(a)
        with torch.no_grad():
            t0, n0, h0, g0n = g0.ray_intersect(O, D)
        cp = torch.tensor(a["ctrl_points"], requires_grad=True)
        g1 = NurbsGeometry(a, control_points=cp)
        t1, n1, h1, g1n = g1.ray_intersect(O, D)
        assert np.array_equal(_np(h0), _np(h1))
        assert _np(t0).tobytes() == _np(t1).tobytes()
        assert _np(g0n).tobytes() == _np(g1n).tobytes()


class TestTraced:
    """The kind inside the bounce loop."""

    @staticmethod
    def _scene(geometry, detector):
        # A beam wider than the sphere fires down onto it from z = 15; the
        # reflected rays that go up cross the detector plane at z = 30.
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=15.0, rx=np.pi),
            CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=12.0),
        )
        scene.add_component("M", ReflectiveComponent(CoordinateSystem(), geometry, reflectance=0.9, name="M"))
        scene.add_detector("D", CoordinateSystem(z=30.0), detector)
        return scene

    @staticmethod
    def _det():
        return IrradianceDetectorConfig(width=60.0, height=60.0, num_pixels_x=24, num_pixels_y=24, absorb=True)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_same_ledger_as_the_analytic_sphere(self, backend):
        _set(backend, "float64")
        res_n = self._scene(NurbsGeometry(_arrays(sphere_surface(10.0))), self._det()).trace(
            num_rays=4000, seed=5, max_depth=6)
        res_a = self._scene(SphereGeometry(10.0), self._det()).trace(num_rays=4000, seed=5, max_depth=6)
        irr_n = _np(res_n.detectors["D"].irradiance)
        irr_a = _np(res_a.detectors["D"].irradiance)
        assert irr_n.sum() > 0
        assert np.allclose(irr_n, irr_a, rtol=1e-9, atol=0)
        assert res_n.flux_conservation_error < 1e-12

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_emulated_replay_is_the_eager_trace(self, precision):
        from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: PLC0415

        _set("torch", precision)
        det = IrradianceDetectorConfig(width=60.0, height=60.0, num_pixels_x=16, num_pixels_y=16,
                                       splat="bilinear", absorb=False)

        def ledger(backend):
            res = self._scene(NurbsGeometry(_arrays(sphere_surface(10.0))), det).trace(
                num_rays=6000, seed=11, max_depth=8, batch_size=2048, backend=backend
            )
            return _np(res.detectors["D"].irradiance).tobytes(), float(res.flux_conservation_error)

        eager = ledger(TorchBackend(seed=11, alive_check_every=0, compact_every=0))
        emulated = ledger(TorchBackend(seed=11, alive_check_every=0, graph_replay="emulate"))
        assert eager == emulated


class TestRegistration:
    def test_kind_registered_and_lowered(self):
        assert "nurbs" in kinds.registered_kinds()["geometry"]
        g = NurbsGeometry(_arrays(sphere_surface(10.0)))
        kind, params = _lower_geometry(g)
        assert kind == "nurbs"
        assert np.asarray(params["control_points"]).shape == (45, 3)
        assert params["n_iter"] == K.DEFAULT_N_ITER
        assert params["arrays"]["patch_degree"] == [[2, 2]]

    def test_parameter_register_classifies_the_net(self):
        from optiland.nonsequential.parameter_register import INTERIOR_BOUNDARY, _classify  # noqa: PLC0415

        assert _classify("surface", "geometry.control_points")[0] == INTERIOR_BOUNDARY
        assert _classify("surface", "geometry.weights")[0] == INTERIOR_BOUNDARY

    def test_bounding_box_holds_the_surface(self):
        g = NurbsGeometry(_arrays(sphere_surface(10.0)))
        box = g.bounding_box((np.array([1.0, 2.0, 3.0]), np.eye(3)))
        assert box.xmin <= 1.0 - 10.0 and box.xmax >= 1.0 + 10.0 and box.zmax >= 13.0

    def test_leaves_rebuilt_after_an_in_place_step(self):
        _set("torch", "float64")
        a = _arrays(sphere_surface(10.0))
        cp = torch.tensor(a["ctrl_points"], requires_grad=True)
        g = NurbsGeometry(a, control_points=cp)
        o = torch.tensor([[0.0, 0.0, 30.0]], dtype=torch.float64)
        d = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
        t0 = float(g.ray_intersect(o, d)[0].detach()[0])
        with torch.no_grad():
            cp.mul_(1.1)  # an optimiser's in-place step: the sphere is now 11 mm
        t1 = float(g.ray_intersect(o, d)[0].detach()[0])
        assert t0 == pytest.approx(20.0, abs=1e-12) and t1 == pytest.approx(19.0, abs=1e-12)


class TestMirrorFace:
    """A NURBS mirror placed in a scene through the builder and its JSON form."""

    F = 50.0

    def _paraboloid(self, h=10.0):
        # z = r^2 / (4 f) exactly: the parabola is a polynomial quadratic
        # Bezier arc (0, 0), (h / 2, 0), (h, h^2 / (4 f)), revolved
        return contract([revolved([0, 0, 0, 1, 1, 1], [[0, 0], [h / 2, 0], [h, h * h / (4 * self.F)]], [1, 1, 1], 2)
                         + (False,)])

    def _scene(self, mirror_cfg):
        from optiland.nonsequential import MirrorConfig, RayDatabaseConfig  # noqa: PLC0415, F401

        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=0.5 * self.F, rx=np.pi),
            CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=8.0),
        )
        scene.add_mirror("M", CoordinateSystem(), mirror_cfg)
        scene.add_detector("D", CoordinateSystem(z=self.F), RayDatabaseConfig(width=40.0, height=40.0, absorb=True))
        return scene

    def test_focuses_as_the_conic_mirror(self):
        from optiland.nonsequential import MirrorConfig  # noqa: PLC0415

        res_n = self._scene(MirrorConfig(radius=0.0, reflectance=1.0, nurbs=self._paraboloid())).trace(
            num_rays=2000, seed=3, max_depth=4)
        res_c = self._scene(MirrorConfig(radius=2 * self.F, conic=-1.0, reflectance=1.0, aperture_radius=10.0)).trace(
            num_rays=2000, seed=3, max_depth=4)
        xn, yn = _np(res_n.detectors["D"].x), _np(res_n.detectors["D"].y)
        xc, yc = _np(res_c.detectors["D"].x), _np(res_c.detectors["D"].y)
        assert xn.size == xc.size and xn.size > 1000
        # every reflected ray passes through the focus to within the two
        # kinds' root and normal accuracy (1e-9 mm at 50 mm)
        assert np.max(np.hypot(xn, yn)) < 1e-9
        assert np.max(np.abs(xn - xc)) < 1e-9 and np.max(np.abs(yn - yc)) < 1e-9

    def test_json_round_trip(self):
        from optiland.nonsequential import MirrorConfig  # noqa: PLC0415
        from optiland.nonsequential.serialization import scene_from_dict, scene_to_dict  # noqa: PLC0415

        d = scene_to_dict(self._scene(MirrorConfig(radius=0.0, reflectance=1.0, nurbs=self._paraboloid())))
        mirror = next(c for c in d["components"] if c["name"] == "M")["config"]
        assert mirror["nurbs"]["patch_degree"] == [[2, 2]]
        back = scene_from_dict(d)
        geom = back.component_registry._registry["M"].surfaces[0].geometry
        assert isinstance(geom, NurbsGeometry)
        assert scene_to_dict(back) == d

    def test_library_json_form_and_conic_json_unchanged(self):
        from optiland.nonsequential import MirrorConfig  # noqa: PLC0415
        from optiland.nonsequential.components.geometry.nurbs.geometry import (  # noqa: PLC0415
            contract_from_patch_set_dict,
        )
        from optiland.nonsequential.serialization import scene_to_dict  # noqa: PLC0415

        p, q, U, V, P, W, _ = sphere_surface(10.0)
        lib = {"kind": "patch_set", "unit": "mm", "patches": [{"surface": {
            "degree_u": p, "degree_v": q, "knots_u": U, "knots_v": V, "control_points": P.tolist(),
            "weights": W.tolist()}, "loops": [], "reversed": False}]}
        a = contract_from_patch_set_dict(lib)
        mine = _arrays(sphere_surface(10.0))
        for key in ("ctrl_points", "ctrl_weights", "knots_u", "knots_v", "patch_domain", "patch_degree"):
            assert np.array_equal(a[key], mine[key]), key
        d = scene_to_dict(self._scene(MirrorConfig(radius=-100.0, reflectance=1.0)))
        assert "nurbs" not in next(c for c in d["components"] if c["name"] == "M")["config"]


# ---------------------------------------------------------------------------


needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="the Apple GPU (torch mps) is not reachable on this host",
)


@pytest.fixture
def _mps_float32():
    be.set_backend("torch")
    be.set_precision("float32")  # before the device: the Apple GPU refuses float64
    be.set_device("mps")
    yield
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.set_backend("numpy")


class TestAppleGPU:
    """The kind at float32 on the Apple GPU (research repository issue 99).

    The leaves were uploaded as a float64 tensor on the target device and cast
    there; the Apple GPU holds no float64 tensor, so every trace raised before
    the first stage ran. They are now rounded on the host and moved. The
    attached Newton step cast and moved a float64 host parameter in one call,
    whose backward is the device-to-host float64 copy of issue 54 (it writes
    zeros); it now casts where the parameter lives, then moves.
    """

    N = 2000

    def test_upload_rounds_on_the_host_as_numpy_does(self):
        _set("torch", "float32")
        g = NurbsGeometry(_arrays(sphere_surface(10.0)))
        dl, adj = g._device(torch.zeros(1, dtype=torch.float32))
        assert np.array_equal(_np(adj["mu"]), g.leaves.mu.astype(np.float32))
        assert np.array_equal(_np(adj["sign"]), g.leaves.patch_sign[g.leaves.patch].astype(np.float32))

    @needs_mps
    def test_sphere_against_the_cpu(self, _mps_float32):
        o, d, fam = sphere_rays(self.N)
        oc, dc = o.astype(np.float32), d.astype(np.float32)
        out = {}
        for dev in ("mps", "cpu"):
            be.set_device(dev)
            O, D = torch.as_tensor(oc, device=dev), torch.as_tensor(dc, device=dev)
            g = NurbsGeometry(_arrays(sphere_surface(10.0)))
            t, _, hit, ng = g.ray_intersect(O, D, eps=_tol.accept_t_min(O.abs().amax(dim=1)))
            assert g.overflow_count() == 0
            out[dev] = (_np(t).astype(float), _np(hit), _np(ng).astype(float))
        t, hit, ng = out["mps"]
        tc, hc, ngc = out["cpu"]
        assert np.isfinite(t[hit]).all() and np.isfinite(ng[hit]).all()
        # hit sets: a ray may differ only where TestSphere allows a float32
        # difference at all (a ray leaving the surface at grazing incidence)
        differ = hit != hc
        assert np.all(fam[differ] == 1) and differ.sum() <= 0.01 * (fam == 1).sum()
        # roots: each device within TestSphere's bound of the root, so within
        # twice it of each other
        both = hit & hc
        cos = np.abs((ngc * dc.astype(float)).sum(1))
        bound = 2 * 64 * 2.0**-24 * 30.0 / np.maximum(cos, 1e-3)
        assert np.all(np.abs(t[both] - tc[both]) <= bound[both])
        ang = np.arccos(np.clip((unit(ng[both]) * unit(ngc[both])).sum(1), -1, 1))
        assert ang.max() < 2e-4

    @needs_mps
    def test_reverse_mode_reaches_float64_host_parameters(self, _mps_float32):
        """A lost gradient (zeros, or the uninitialised values the defect can
        leave) fails by orders of magnitude; 1e-3 of the largest component is a
        detection threshold, far above the two float32 legs' own differences."""
        a = _arrays(wavy_surface())
        o = [[1.0, -2.0, 40.0], [3.0, 2.0, 40.0]]
        d = np.array([[0.0, 0.0, -1.0], [0.1, 0.0, -1.0]])
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        grads = {}
        for dev in ("mps", "cpu"):
            be.set_device(dev)
            cp = torch.tensor(a["ctrl_points"], dtype=torch.float64, requires_grad=True)
            w = torch.tensor(a["ctrl_weights"], dtype=torch.float64, requires_grad=True)
            g = NurbsGeometry(a, control_points=cp, weights=w)
            O = torch.tensor(o, dtype=torch.float32, device=dev)
            D = torch.tensor(d, dtype=torch.float32, device=dev)
            t, _, hit, _ = g.ray_intersect(O, D)
            assert bool(hit.all())
            t.sum().backward()
            grads[dev] = (cp.grad.numpy().copy(), w.grad.numpy().copy())
        for got, ref in zip(grads["mps"], grads["cpu"]):
            scale = np.abs(ref).max()
            assert scale > 0
            assert np.abs(got - ref).max() <= 1e-3 * scale

    @needs_mps
    def test_control_the_one_call_cast_and_move_loses_the_gradient(self):
        """The defect itself, on this torch. If this starts failing, torch has
        fixed it, and the kind's two-step cast and this control can be dated and
        retired."""
        x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
        x.to(dtype=torch.float32, device="mps").sum().backward()
        assert not torch.equal(x.grad, torch.ones_like(x))
