"""Trimmed NURBS patches on the device (KronosNSRT issue 66, ticket C).

What each class pins, and the route it uses:

- ``TestPolygons``: the engine's trim polygons, built from the array contract
  without the geometry library, equal the library's ``trim_polygons`` to the
  bit, and the engine's even-odd rule equals ``point_in_trim`` on random
  points (when the library is installed).
- ``TestHole`` and ``TestDCut``: a plane patch with a circular hole, and a
  D-cut disc (an arc and a chord as the outer loop), shot with a uniform beam
  along -z. The hit fraction against its closed form (a binomial standard
  error), and every ray farther than the polygonisation band from the trim
  curve classified as the exact circle or chord classifies it, at numpy
  float64, torch float64 and torch float32.
- ``TestSphereCap``: the exact rational sphere with its north cap trimmed away
  by a loop inside its domain (the ``uv_bounds`` left at the whole domain, so
  the polygon decides). A ray through the opening is not stopped by the
  trimmed-away root and finds the next one, the inside of the far wall, at
  its closed-form distance; the share of the beam that does so against its
  closed form; leaves wholly inside the opening are not built.
- ``TestUntrimmedUnchanged``: a patch whose loops run along its domain's
  rectangle builds no trim data and gives the same bytes as the same patch
  with no loop, on both backends.
- ``TestTracedTrimmed``: the emulated graph replay of a scene with a trimmed
  sphere equals the eager trace at both precisions; the adjoint at hits away
  from the trim curve equals the untrimmed sphere's to the bit.
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
    Spectrum,
)
from optiland.nonsequential.components.geometry.nurbs import trim as T

torch = pytest.importorskip("torch", reason="Torch not available")

SQ = math.sqrt(0.5)
CIRCLE_U = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], float)
CIRCLE_P = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]], float)
CIRCLE_W = np.array([1, SQ, 1, SQ, 1, SQ, 1, SQ, 1])

CONFIGS = [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")]


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _set(backend, precision):
    be.set_backend(backend)
    be.set_precision(precision)


def _np(x):
    return np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x)


def _u(precision):
    return 2.0**-53 if precision == "float64" else 2.0**-24


def _cast(backend, precision, *arrays):
    dt = np.float64 if precision == "float64" else np.float32
    out = [a.astype(dt) for a in arrays]
    if backend == "torch":
        out = [torch.as_tensor(a) for a in out]
    return out


# ---------------------------------------------------------------------------
# Contracts (the geometry library's format 1), written out
# ---------------------------------------------------------------------------


def _off(xs):
    return np.concatenate([[0], np.cumsum([len(x) for x in xs])]).astype(np.int64)


def line(a, b):
    """A degree-1 trim curve from ``a`` to ``b`` in (u, v)."""
    return (1, np.array([0, 0, 1, 1.0]), np.array([a, b], float), np.ones(2))


def circle(centre, rho, clockwise=False):
    """The full circle as a rational quadratic, starting at angle 0."""
    P = np.asarray(centre, float) + rho * CIRCLE_P
    W = CIRCLE_W.copy()
    if clockwise:
        P, W = P[::-1].copy(), W[::-1].copy()
    return (2, CIRCLE_U.copy(), P, W)


def arc(centre, rho, a0, a1, n_seg=4):
    """A circular arc from angle ``a0`` to ``a1`` (counterclockwise), ``n_seg``
    rational quadratic segments of equal angle (each under 90 degrees)."""
    c = np.asarray(centre, float)
    dth = (a1 - a0) / n_seg
    pts, w = [], []
    for k in range(n_seg):
        t0 = a0 + k * dth
        tm = t0 + 0.5 * dth
        if k == 0:
            pts.append(c + rho * np.array([math.cos(t0), math.sin(t0)]))
            w.append(1.0)
        pts.append(c + rho / math.cos(0.5 * dth) * np.array([math.cos(tm), math.sin(tm)]))
        w.append(math.cos(0.5 * dth))
        t1 = t0 + dth
        pts.append(c + rho * np.array([math.cos(t1), math.sin(t1)]))
        w.append(1.0)
    knots = [0.0, 0.0, 0.0]
    for k in range(1, n_seg):
        knots += [k / n_seg, k / n_seg]
    knots += [1.0, 1.0, 1.0]
    return (2, np.array(knots), np.array(pts), np.array(w))


RECT = [line((0, 0), (1, 0)), line((1, 0), (1, 1)), line((1, 1), (0, 1)), line((0, 1), (0, 0))]


def contract(p, q, U, V, P, W, loops, uv_bounds=None, reversed_=False):
    """One patch with ``loops`` = [(curves, outer), ...]."""
    P = np.asarray(P, float)
    nu, nv = P.shape[:2]
    curves = [c for lp, _ in loops for c in lp]
    dom = (U[p], U[nu], V[q], V[nv])
    return {
        "format": np.array(1), "unit": np.array("mm"), "provenance": np.array("test"),
        "patch_degree": np.array([[p, q]], np.int32), "patch_n_ctrl": np.array([[nu, nv]], np.int32),
        "patch_ctrl_offset": np.array([0, nu * nv]), "ctrl_points": P.reshape(-1, 3),
        "ctrl_weights": np.asarray(W, float).reshape(-1),
        "patch_knot_u_offset": np.array([0, len(U)]), "knots_u": np.asarray(U, float),
        "patch_knot_v_offset": np.array([0, len(V)]), "knots_v": np.asarray(V, float),
        "patch_domain": np.array([dom]), "patch_uv_bounds": np.array([uv_bounds or dom]),
        "patch_reversed": np.array([reversed_]),
        "patch_loop_offset": np.array([0, len(loops)]), "loop_outer": np.array([o for _, o in loops], bool),
        "loop_curve_offset": _off([lp for lp, _ in loops]),
        "curve_degree": np.array([c[0] for c in curves], np.int32),
        "curve_ctrl_offset": _off([c[2] for c in curves]),
        "curve_ctrl_points": np.concatenate([c[2] for c in curves]) if curves else np.zeros((0, 2)),
        "curve_weights": np.concatenate([c[3] for c in curves]) if curves else np.zeros(0),
        "curve_knot_offset": _off([c[1] for c in curves]),
        "curve_knots": np.concatenate([c[1] for c in curves]) if curves else np.zeros(0),
    }


L_PLATE = 10.0


def plate(loops):
    """The square [-10, 10]^2 at z = 0 as a bilinear patch: (x, y) = 20 (u, v) - 10."""
    L = L_PLATE
    P = np.array([[[-L, -L, 0.0], [-L, L, 0.0]], [[L, -L, 0.0], [L, L, 0.0]]])
    return contract(1, 1, [0, 0, 1, 1.0], [0, 0, 1, 1.0], P, np.ones((2, 2)), loops)


HOLE_C, HOLE_RHO = (0.55, 0.45), 0.3
DCUT_C, DCUT_RHO, DCUT_ALPHA = (0.5, 0.5), 0.4, math.radians(50.0)


def holed_plate():
    return plate([(RECT, True), ([circle(HOLE_C, HOLE_RHO, clockwise=True)], False)])


def dcut_plate():
    c = np.array(DCUT_C)
    a0, a1 = DCUT_ALPHA, 2 * math.pi - DCUT_ALPHA
    end = c + DCUT_RHO * np.array([math.cos(a1), math.sin(a1)])
    start = c + DCUT_RHO * np.array([math.cos(a0), math.sin(a0)])
    return plate([([arc(c, DCUT_RHO, a0, a1), line(tuple(end), tuple(start))], True)])


R_SPHERE = 10.0
V_CUT = 0.875


def sphere_profile_z(v, R=R_SPHERE):
    """z of the library's rational sphere at profile parameter ``v`` (the arc from
    the equator to the north pole for v in [0.5, 1])."""
    s = 2.0 * (v - 0.5)
    b0, b1, b2 = (1 - s) ** 2, 2 * s * (1 - s) * SQ, s * s
    return (b1 * R + b2 * R) / (b0 + b1 + b2)


def sphere_net(R=R_SPHERE):
    prof = np.array([[0, -R], [R, -R], [R, 0], [R, R], [0, R]], float)
    wp = np.array([1, SQ, 1, SQ, 1])
    cp = np.empty((9, 5, 3))
    cp[..., 0] = CIRCLE_P[:, None, 0] * prof[None, :, 0]
    cp[..., 1] = CIRCLE_P[:, None, 1] * prof[None, :, 0]
    cp[..., 2] = prof[None, :, 1]
    return 2, 2, CIRCLE_U, [0, 0, 0, 0.5, 0.5, 1, 1, 1], cp, CIRCLE_W[:, None] * wp[None, :]


def capped_sphere():
    """The sphere with v > V_CUT (the north cap) trimmed away; uv_bounds left at the domain."""
    loop = [line((0, 0), (1, 0)), line((1, 0), (1, V_CUT)), line((1, V_CUT), (0, V_CUT)), line((0, V_CUT), (0, 0))]
    return contract(*sphere_net(), [(loop, True)])


def _library_patch(kn, a):
    """The library's TrimmedPatch for a one-patch contract written above."""
    p, q = (int(x) for x in a["patch_degree"][0])
    nu, nv = (int(x) for x in a["patch_n_ctrl"][0])
    surf = kn.NurbsSurface(p, q, a["knots_u"], a["knots_v"], a["ctrl_points"].reshape(nu, nv, 3),
                           a["ctrl_weights"].reshape(nu, nv))
    loops = []
    lo = a["loop_curve_offset"]
    for L in range(len(a["loop_outer"])):
        curves = []
        for C in range(int(lo[L]), int(lo[L + 1])):
            k0, k1 = int(a["curve_ctrl_offset"][C]), int(a["curve_ctrl_offset"][C + 1])
            n0, n1 = int(a["curve_knot_offset"][C]), int(a["curve_knot_offset"][C + 1])
            curves.append(kn.NurbsCurve2d(int(a["curve_degree"][C]), a["curve_knots"][n0:n1],
                                          a["curve_ctrl_points"][k0:k1], a["curve_weights"][k0:k1]))
        loops.append(kn.TrimLoop(tuple(curves), bool(a["loop_outer"][L])))
    return kn.TrimmedPatch(surf, tuple(loops), uv_bounds=tuple(a["patch_uv_bounds"][0]))


# ---------------------------------------------------------------------------


class TestPolygons:
    @pytest.mark.parametrize("make", [holed_plate, dcut_plate, capped_sphere])
    def test_polygons_and_rule_equal_the_library(self, make):
        kn = pytest.importorskip("kgeom.nurbs", reason="the geometry library is not installed")
        a = make()
        mine = T.trim_polygons(a, 0, a["patch_uv_bounds"][0])
        patch = _library_patch(kn, a)
        lib = kn.trim_polygons(patch)
        assert len(mine) == len(lib)
        for (pm, om), (pl, ol) in zip(mine, lib):
            assert om == ol and np.array_equal(pm, pl)
        rng = np.random.default_rng(7)
        uv = rng.uniform(0, 1, (20000, 2))
        assert np.array_equal(T.even_odd(T.polygon_edges(mine), uv[:, 0], uv[:, 1]),
                              kn.point_in_trim(patch, uv[:, 0], uv[:, 1]))

    def test_slabs_keep_every_crossing(self):
        # The parity over a piece's slab equals the parity over the whole patch.
        g = NurbsGeometry(holed_plate())
        lv = g.leaves
        edges = T.polygon_edges(lv.trim_polygons[0])
        rng = np.random.default_rng(3)
        uv = rng.uniform(0, 1, (20000, 2))
        k = np.clip(np.floor((uv[:, 1] - lv.piece_slab[0, 0]) * lv.piece_slab[0, 1]), 0,
                    lv.piece_edges.shape[1] - 1).astype(int)
        full = T.even_odd(edges, uv[:, 0], uv[:, 1])
        per = np.array([T.even_odd(lv.piece_edges[0, kk], uu, vv)[0] for kk, uu, vv in zip(k, uv[:, 0], uv[:, 1])])
        assert np.array_equal(full, per)


def _plate_rays(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-L_PLATE, L_PLATE, (n, 2))
    o = np.column_stack([xy, np.full(n, 5.0)])
    d = np.tile([0.0, 0.0, -1.0], (n, 1))
    return o, d, xy


def _band_xy(precision):
    # the polygon band (1e-6 of the unit square's diagonal) plus 64 units of the
    # dtype at the parameters' scale, mapped to (x, y) by the plate's 20 mm per unit
    return 2.0 * L_PLATE * (T.TRIM_TOLERANCE * math.sqrt(2.0) + 64 * _u(precision))


class TestHole:
    N = 40000

    @pytest.mark.parametrize(("backend", "precision"), CONFIGS)
    def test_hit_fraction_and_every_ray(self, backend, precision):
        _set(backend, precision)
        o, d, xy = _plate_rays(self.N, 11)
        g = NurbsGeometry(holed_plate())
        O, D = _cast(backend, precision, o, d)
        t, _, hit, _ = g.ray_intersect(O, D)
        hit = _np(hit)
        assert g.overflow_count() == 0
        cx, cy = 2 * L_PLATE * np.array(HOLE_C) - L_PLATE
        a = 2 * L_PLATE * HOLE_RHO
        r = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        # closed form: the hole removes pi a^2 of the 4 L^2 the beam covers
        p = 1.0 - math.pi * a * a / (4 * L_PLATE**2)
        se = math.sqrt(p * (1 - p) / self.N)
        assert abs(hit.mean() - p) <= 4 * se
        far = np.abs(r - a) > _band_xy(precision)
        assert np.array_equal(hit[far], r[far] > a)
        assert np.allclose(_np(t)[hit].astype(float), 5.0, rtol=0, atol=64 * _u(precision) * 10)

    def test_library_agrees_on_every_hit(self):
        kn = pytest.importorskip("kgeom.nurbs", reason="the geometry library is not installed")
        o, d, _ = _plate_rays(self.N, 12)
        a = holed_plate()
        g = NurbsGeometry(a)
        _, _, hit, _ = g.ray_intersect(o, d)
        patch = _library_patch(kn, a)
        assert kn.point_in_trim(patch, g.last_u[hit], g.last_v[hit]).all()
        # and every miss is a point the library puts outside (the exact (u, v) of the ray)
        uv = (o[~hit, :2] + L_PLATE) / (2 * L_PLATE)
        assert not kn.point_in_trim(patch, uv[:, 0], uv[:, 1]).any()


class TestDCut:
    N = 40000

    @pytest.mark.parametrize(("backend", "precision"), CONFIGS)
    def test_hit_fraction_and_every_ray(self, backend, precision):
        _set(backend, precision)
        o, d, xy = _plate_rays(self.N, 21)
        g = NurbsGeometry(dcut_plate())
        O, D = _cast(backend, precision, o, d)
        _, _, hit, _ = g.ray_intersect(O, D)
        hit = _np(hit)
        c = 2 * L_PLATE * np.array(DCUT_C) - L_PLATE
        a = 2 * L_PLATE * DCUT_RHO
        x_chord = c[0] + a * math.cos(DCUT_ALPHA)
        # closed form: the disc less the segment beyond the chord, alpha its half-angle
        area = math.pi * a * a - 0.5 * a * a * (2 * DCUT_ALPHA - math.sin(2 * DCUT_ALPHA))
        p = area / (4 * L_PLATE**2)
        se = math.sqrt(p * (1 - p) / self.N)
        assert abs(hit.mean() - p) <= 4 * se
        r = np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1])
        inside = (r < a) & (xy[:, 0] < x_chord)
        band = _band_xy(precision)
        far = (np.abs(r - a) > band) & (np.abs(xy[:, 0] - x_chord) > band)
        assert np.array_equal(hit[far], inside[far])


class TestSphereCap:
    N = 20000

    def test_leaves_inside_the_opening_are_not_built(self):
        full = NurbsGeometry(contract(*sphere_net(), []))
        capped = NurbsGeometry(capped_sphere())
        assert capped.leaves.n_dropped > 0
        assert capped.leaves.n + capped.leaves.n_dropped == full.leaves.n
        assert full.leaves.piece_edges is None

    @pytest.mark.parametrize(("backend", "precision"), CONFIGS)
    def test_the_next_root_is_found_through_the_opening(self, backend, precision):
        _set(backend, precision)
        R = R_SPHERE
        z_cut = sphere_profile_z(V_CUT)
        rho_c = math.sqrt(R * R - z_cut * z_cut)
        rng = np.random.default_rng(31)
        rho = R * np.sqrt(rng.uniform(0, 0.999, self.N))
        ph = rng.uniform(0, 2 * math.pi, self.N)
        o = np.column_stack([rho * np.cos(ph), rho * np.sin(ph), np.full(self.N, 30.0)])
        d = np.tile([0.0, 0.0, -1.0], (self.N, 1))
        g = NurbsGeometry(capped_sphere())
        O, D = _cast(backend, precision, o, d)
        t, _, hit, _ = g.ray_intersect(O, D)
        t, hit = _np(t).astype(float), _np(hit)
        assert g.overflow_count() == 0
        assert hit.all()
        zh = np.sqrt(R * R - rho * rho)
        through = rho < rho_c
        expect = np.where(through, 30.0 + zh, 30.0 - zh)
        # the band in rho: the polygon's (1e-6 of the domain's diagonal) plus 64 units
        # of the dtype, times the largest |S_v| on the sphere (2 pi R bounds it)
        band = (T.TRIM_TOLERANCE * math.sqrt(2.0) + 64 * _u(precision)) * 2 * math.pi * R
        far = np.abs(rho - rho_c) > band
        # roots conditioned by 1 / |cos| at the coordinate scale (30 mm), as the kind's sphere test
        cos = np.maximum(zh / R, 1e-3)
        bound = 64 * _u(precision) * 40.0 / cos
        assert np.all(np.abs(t[far] - expect[far]) <= bound[far])
        # closed form: the share of a uniform disc beam of radius R that passes the opening
        share = rho_c**2 / (0.999 * R * R)
        se = math.sqrt(share * (1 - share) / self.N)
        went = np.abs(t - (30.0 + zh)) < np.abs(t - (30.0 - zh))
        assert abs(went.mean() - share) <= 4 * se


class TestUntrimmedUnchanged:
    @pytest.mark.parametrize(("backend", "precision"), CONFIGS)
    def test_rectangle_loops_build_and_trace_as_no_loops(self, backend, precision):
        _set(backend, precision)
        bare = NurbsGeometry(contract(*sphere_net(), []))
        rect = NurbsGeometry(contract(*sphere_net(), [(RECT, True)]))
        assert rect.leaves.piece_edges is None and rect.leaves.n == bare.leaves.n
        rng = np.random.default_rng(5)
        w = rng.normal(size=(3000, 3))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        o = 30.0 * w
        d = -w + 0.3 * rng.normal(size=(3000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        O, D = _cast(backend, precision, o, d)
        out_b = bare.ray_intersect(O, D)
        out_r = rect.ray_intersect(O, D)
        for x, y in zip(out_b, out_r):
            assert _np(x).tobytes() == _np(y).tobytes()


class TestTracedTrimmed:
    @staticmethod
    def _scene(geometry, det):
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=15.0, rx=np.pi),
            CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=9.0),
        )
        scene.add_component("M", ReflectiveComponent(CoordinateSystem(), geometry, reflectance=0.9, name="M"))
        scene.add_detector("D", CoordinateSystem(z=30.0), det)
        return scene

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_emulated_replay_is_the_eager_trace(self, precision):
        from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: PLC0415

        _set("torch", precision)
        det = IrradianceDetectorConfig(width=60.0, height=60.0, num_pixels_x=16, num_pixels_y=16,
                                       splat="bilinear", absorb=False)

        def ledger(backend):
            res = self._scene(NurbsGeometry(capped_sphere()), det).trace(
                num_rays=4000, seed=11, max_depth=8, batch_size=2048, backend=backend
            )
            return _np(res.detectors["D"].irradiance).tobytes(), float(res.flux_conservation_error)

        eager = ledger(TorchBackend(seed=11, alive_check_every=0, compact_every=0))
        emulated = ledger(TorchBackend(seed=11, alive_check_every=0, graph_replay="emulate"))
        assert eager == emulated

    def test_adjoint_away_from_the_trim_curve_is_the_untrimmed_one(self):
        _set("torch", "float64")
        net = sphere_net()
        rng = np.random.default_rng(9)
        # rays from below onto the southern half, far from the cut at v = 0.875
        xy = rng.uniform(-5, 5, (500, 2))
        o = torch.tensor(np.column_stack([xy, np.full(500, -30.0)]))
        d = torch.tensor(np.tile([0.0, 0.0, 1.0], (500, 1)))
        grads = []
        for loops in ([], capped_sphere_loops()):
            a = contract(*net, loops)
            cp = torch.tensor(a["ctrl_points"], requires_grad=True)
            g = NurbsGeometry(a, control_points=cp)
            t, n, hit, _ = g.ray_intersect(o, d)
            (t[hit].sum() + n[hit].sum()).backward()
            grads.append((_np(t).tobytes(), cp.grad.numpy().tobytes()))
        assert grads[0] == grads[1]


def capped_sphere_loops():
    loop = [line((0, 0), (1, 0)), line((1, 0), (1, V_CUT)), line((1, V_CUT), (0, V_CUT)), line((0, V_CUT), (0, 0))]
    return [(loop, True)]


def test_flux_band_bound_of_the_hole():
    """The flux the band can move across the hole's edge, stated for this plate.

    The polygon's chords lie within tol of the circle in (u, v); the band's
    area on the surface is at most the loop's length in (u, v) times tol times
    the largest |S_u x S_v| (400 mm^2 per unit area here), and the flux it can
    misassign is that area times the irradiance. Asserted against the beam's
    total so a change of the tolerance or the rule shows here.
    """
    g = NurbsGeometry(holed_plate())
    poly = g.leaves.trim_polygons[0][1][0]
    length = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
    tol = T.trim_tolerance((0, 1, 0, 1))
    band_area = length * tol * (2 * L_PLATE) ** 2
    share = band_area / (2 * L_PLATE) ** 2
    assert length == pytest.approx(2 * math.pi * HOLE_RHO, rel=1e-6)
    assert share < 3e-6
