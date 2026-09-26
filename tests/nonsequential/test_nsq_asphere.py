"""The even and odd asphere geometry kinds (KronosNSRT issue 30).

What each class pins, and the route it uses:

- ``TestZeroCoefficients``: with every coefficient zero the kind is the conic
  kind, bit for bit, on every backend and precision, forward and in gradient
  mode (the first residual of a base-conic seed is the polynomial alone).
- ``TestAgainstSequentialEngine``: the independent route. The library's
  sequential ``EvenAsphere``/``OddAsphere`` (a different code path: host
  Newton from its own conic seed, early exit, float64) on the same surface and
  rays; agreement within the sum of the two engines' residual tolerances over
  the slope.
- ``TestFloat32``: the float32 hit points against float64 within the dtype
  rule (the float32 tolerance and input rounding, over the slope).
- ``TestTheoryChapter7``: T-07-5 (aimed rays, error and step count) and
  T-07-6 (the tangent guard fires at cos 1e-4 and not at 0.5) of
  ``docs/theory/07_geometry.md`` in the research repository.
- ``TestMissReasons``: every miss reason is reachable and recorded, and a
  miss never reports a distance.
- ``TestFirstCrossing``: on a strongly aspheric surface the nearest crossing
  is returned where the base-conic seed alone converges to a farther one;
  checked against a brute-force sign scan.
- ``TestAdjoint``: dt and the normal with respect to the curvature, the conic
  constant, every coefficient and the placement, against fourth-order central
  differences of the primal root.
- ``TestTraced``: a paraboloid mirror built as a flat base plus an r^2 term
  focuses a collimated beam to a point; the emulated graph replay of a scene
  with an asphere equals the eager fixed-width trace bit for bit and transfers
  nothing to or from the host.
- ``TestRegistration``: the kinds, their IR lowering and the volume helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    ConicGeometry,
    EvenAsphereGeometry,
    IrradianceDetectorConfig,
    NSQScene,
    OddAsphereGeometry,
    RayDatabaseConfig,
    ReflectiveComponent,
    Spectrum,
)
from optiland.nonsequential import kinds
from optiland.nonsequential.components.geometry.analytic import asphere as A
from optiland.nonsequential.ir.lower import _lower_geometry

torch = pytest.importorskip("torch", reason="Torch not available")

# The theory chapter's asphere (docs/theory/07_geometry.md section 7.6):
# R = 25 mm, K = -0.5, A4 = 1e-6, A6 = 1e-8, A8 = 1e-9 (coefficients[0]
# multiplies r^2, so the r^4 term is coefficients[1]).
THEORY = dict(radius=25.0, conic=-0.5, aperture_radius=12.5)
THEORY_COEFFS = [0.0, 1e-6, 1e-8, 1e-9]

U64 = 2.0**-53


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _set(backend: str, precision: str) -> None:
    be.set_backend(backend)
    be.set_precision(precision)


def _np(x):
    return be.to_numpy(x.detach() if torch.is_tensor(x) else x)


def _fan(n: int, half: float, z0: float = -5.0, seed: int = 3):
    """Random rays from a plane below the surface, hits and misses mixed."""
    rng = np.random.default_rng(seed)
    o = np.column_stack(
        [rng.uniform(-half, half, n), rng.uniform(-half, half, n), np.full(n, z0)]
    )
    d = rng.normal(size=(n, 3))
    d[:, 2] = np.abs(d[:, 2]) + 0.2
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o, d


def _slope_terms(g, o, d, t):
    """|f'| and |grad G| at the hit points (float64, numpy)."""
    x = o[:, 0] + t * d[:, 0]
    y = o[:, 1] + t * d[:, 1]
    z = o[:, 2] + t * d[:, 2]
    rmin = np.float64(4 * U64)
    _, fp, gn, _, _ = g._evaluate(x, y, z, d[:, 0], d[:, 1], d[:, 2], rmin)
    scale = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z), np.ones_like(x)])
    return np.abs(fp), gn, scale


SURFACES = {
    "prolate": (25.0, -0.5, 12.5),
    "hyperboloid_concave": (-40.0, -2.0, 15.0),
    "flat_base": (np.inf, 0.0, 10.0),
    "oblate": (30.0, 0.8, 12.0),
}


class TestZeroCoefficients:
    """Zero coefficients: the conic kind's numbers, to the bit."""

    @pytest.mark.parametrize(
        "backend,precision",
        [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")],
    )
    @pytest.mark.parametrize("surface", sorted(SURFACES))
    @pytest.mark.parametrize("cls", [EvenAsphereGeometry, OddAsphereGeometry])
    def test_bit_identical_to_conic(self, backend, precision, surface, cls):
        _set(backend, precision)
        radius, conic, ap = SURFACES[surface]
        o, d = _fan(4000, ap + 3.0)
        O, D = be.array(o), be.array(d)
        ref = ConicGeometry(radius, conic, ap).ray_intersect(O, D)
        out = cls(radius, conic, ap, [0.0, 0.0, 0.0]).ray_intersect(O, D)
        assert _np(ref[2]).any()  # the fan does hit
        for a, b in zip(ref, out):
            assert _np(a).tobytes() == _np(b).tobytes()

    def test_gradient_mode_forward_bits_and_gradient(self):
        _set("torch", "float64")
        o, d = _fan(4000, 15.0)
        O, D = be.array(o), be.array(d)
        r_c = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
        r_a = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
        ref = ConicGeometry(r_c, -0.5, 12.5).ray_intersect(O, D)
        out = EvenAsphereGeometry(r_a, -0.5, 12.5, [0.0, 0.0]).ray_intersect(O, D)
        for a, b in zip(ref, out):
            assert _np(a).tobytes() == _np(b).tobytes()
        hit = ref[2]
        (g_c,) = torch.autograd.grad(ref[0][hit].sum() + ref[3][hit][:, 0].sum(), r_c)
        (g_a,) = torch.autograd.grad(out[0][hit].sum() + out[3][hit][:, 0].sum(), r_a)
        # The conic kind differentiates its closed form, the asphere the
        # implicit root: the same derivative, rounded differently.
        assert abs(g_a.item() - g_c.item()) <= 1e-12 * abs(g_c.item())


class TestAgainstSequentialEngine:
    """The sequential engine's asphere trace on the same surface and rays."""

    @staticmethod
    def _sequential(cls, coeffs, o, d):
        from optiland.rays import RealRays  # noqa: PLC0415

        geom = cls(
            CoordinateSystem(), THEORY["radius"], THEORY["conic"],
            tol=0.0, max_iter=100, coefficients=coeffs,
        )
        n = o.shape[0]
        rays = RealRays(
            o[:, 0], o[:, 1], o[:, 2], d[:, 0], d[:, 1], d[:, 2],
            np.ones(n), np.full(n, 0.55),
        )
        return np.asarray(be.to_numpy(geom.distance(rays)), dtype=float)

    @pytest.mark.parametrize(
        "kind,coeffs",
        [("even", THEORY_COEFFS), ("odd", [2e-3, 1e-4, 1e-6, -1e-7])],
    )
    def test_hit_points_agree(self, kind, coeffs):
        from optiland.geometries import EvenAsphere, OddAsphere  # noqa: PLC0415

        _set("numpy", "float64")
        # A fan from a point 30 mm below the vertex, aimed at a 9 x 9 grid of
        # points inside the aperture.
        grid = np.linspace(-11.0, 11.0, 9)
        gx, gy = np.meshgrid(grid, grid)
        keep = gx**2 + gy**2 <= 12.0**2
        targets = np.column_stack([gx[keep], gy[keep], np.full(keep.sum(), 2.0)])
        src = np.array([0.5, 2.0, -30.0])
        d = targets - src
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = np.tile(src, (d.shape[0], 1))

        cls_new = EvenAsphereGeometry if kind == "even" else OddAsphereGeometry
        cls_seq = EvenAsphere if kind == "even" else OddAsphere
        g = cls_new(**THEORY, coefficients=coeffs)
        t, _, hit, _ = g.ray_intersect(o, d)
        t_seq = self._sequential(cls_seq, coeffs, o, d)
        assert hit.all()
        # Each engine stops within its own residual tolerance: this kind at
        # 32 ulp(scale) |grad G| (then one polishing step), the sequential one
        # at 8 eps max|t|. A residual r is a distance r / |f'| along the ray;
        # twice each tolerance covers the rounding of each residual itself.
        fp, gn, scale = _slope_terms(g, o, d, t)
        bound = (
            2 * 32 * np.spacing(scale) * gn + 2 * 8 * 2 * U64 * np.max(np.abs(t_seq))
        ) / fp
        diff = np.abs(t - t_seq)
        assert np.all(diff <= bound), (diff / bound).max()


class TestFloat32:
    """float32 against float64 within the dtype rule."""

    @pytest.mark.parametrize("cls,coeffs", [
        (EvenAsphereGeometry, THEORY_COEFFS),
        (OddAsphereGeometry, [2e-3, 1e-4, 1e-6, -1e-7]),
    ])
    def test_hit_points(self, cls, coeffs):
        o, d = _fan(4000, 15.0, z0=-5.0, seed=11)
        _set("torch", "float64")
        g64 = cls(**THEORY, coefficients=coeffs)
        t64, _, h64, n64 = (
            _np(v) for v in g64.ray_intersect(be.array(o), be.array(d))
        )
        _set("torch", "float32")
        o32 = o.astype(np.float32)
        d32 = d.astype(np.float32)
        g32 = cls(**THEORY, coefficients=coeffs)
        t32, _, h32, n32 = (
            _np(v) for v in g32.ray_intersect(be.array(o32), be.array(d32))
        )
        _set("numpy", "float64")
        both = h64 & h32
        assert both.sum() > 1000
        # The float32 root: its own residual tolerance (32 ulps, twice for the
        # residual's rounding) and the rounding of the float32 inputs (half an
        # ulp of each origin coordinate, u32 of each direction component over
        # the path t), all over the slope.
        o64, d64 = o[both], d[both]
        fp, gn, scale = _slope_terms(g64, o64, d64, t64[both])
        u32 = 2.0**-24
        ulp32 = np.spacing(scale.astype(np.float32)).astype(float)
        inputs = (
            np.spacing(np.abs(o64).max(axis=1).astype(np.float32)).astype(float)
            + 2 * u32 * t64[both]
        ) * gn
        bound = (2 * 32 * ulp32 * gn + inputs) / fp
        diff = np.abs(t32[both].astype(float) - t64[both])
        assert np.all(diff <= bound), (diff / bound).max()
        # Hit sets differ only where float64 finds the root within that same
        # bound of the aperture rim or of the tangent guard's threshold.
        differ = h64 ^ h32
        if differ.any():
            idx = np.where(differ & h64)[0]
            p = o[idx] + t64[idx, None] * d[idx]
            r = np.hypot(p[:, 0], p[:, 1])
            fp_i, gn_i, _ = _slope_terms(g64, o[idx], d[idx], t64[idx])
            near_rim = np.abs(r - THEORY["aperture_radius"]) <= 1e-4
            near_guard = np.abs(fp_i / gn_i - g64.guard_eta) <= 1e-4
            assert np.all(near_rim | near_guard)


def _aimed_ray(g, r_point: float, cos_i: float, sign: float = -1.0, length=8.0):
    """A ray aimed at the surface point (r_point, 0, sag) with incidence
    cosine ``cos_i``, tilted in the meridional plane; t* = length exactly by
    construction (the theory's check C7.6)."""
    x = np.array([r_point])
    y = np.array([0.0])
    s = np.array([r_point, 0.0, float(g.sag(x, y)[0])])
    n = g._normal_local(x, y)[0]
    n = n / np.linalg.norm(n)
    tang = np.cross(n, [0.0, 1.0, 0.0])
    tang /= np.linalg.norm(tang)
    d = cos_i * n + sign * np.sqrt(1.0 - cos_i * cos_i) * tang
    d /= np.linalg.norm(d)
    if d[2] < 0:
        d = -d
    return (s - length * d)[None, :], d[None, :], s


class TestTheoryChapter7:
    """T-07-5 and T-07-6 on the chapter's own surface (float64)."""

    @pytest.mark.parametrize("cos_i", [1.0, 0.5, 0.1])
    def test_t07_5_aimed_rays(self, cos_i):
        g = EvenAsphereGeometry(**THEORY, coefficients=THEORY_COEFFS)
        o, d, s = _aimed_ray(g, 12.0, cos_i)
        t, _, hit, _ = g.ray_intersect(o, d)
        assert hit[0]
        # T-07-5's bound, 8 u t*, and the conditioning floor of any root
        # evaluated in the working dtype: a few ulps of the coordinate scale
        # in the residual are ulp/cos along the ray. The chapter's bound
        # alone omits the second term; its own table measured 4e-14 mm (45 u
        # relative) at cos 0.5, above it.
        floor = 4 * np.spacing(12.0) / cos_i
        assert abs(t[0] - 8.0) <= max(8 * U64 * 8.0, floor)
        assert g.last_steps[0] <= 8

    def test_t07_6_guard_fires_at_1e_4(self):
        g = EvenAsphereGeometry(**THEORY, coefficients=THEORY_COEFFS)
        o, d, _ = _aimed_ray(g, 12.0, 1e-4)
        t, _, hit, _ = g.ray_intersect(o, d)
        assert not hit[0]
        assert np.isinf(t[0])
        assert int(g.last_status[0]) == A.GRAZING

    def test_t07_6_guard_quiet_at_0_5(self):
        g = EvenAsphereGeometry(**THEORY, coefficients=THEORY_COEFFS)
        o, d, _ = _aimed_ray(g, 12.0, 0.5)
        t, _, hit, _ = g.ray_intersect(o, d)
        assert hit[0] and int(g.last_status[0]) == A.HIT

    def test_domain_seed_is_a_miss_without_nan(self):
        """A seed with (1 + K) c^2 r^2 > 1 is never square-rooted."""
        g = EvenAsphereGeometry(**THEORY, coefficients=THEORY_COEFFS)
        r_t = 1.0 / ((1.0 / 25.0) * np.sqrt(0.5))  # 35.36 mm
        o = np.array([[r_t + 5.0, 0.0, -5.0]])
        d = np.array([[0.0, 0.0, 1.0]])
        with np.errstate(invalid="raise"):
            t, st, steps, fp, gn, tol = g._refine(
                o, d, np.array([5.0]), np.array([True]), np.array([False])
            )
        assert int(st[0]) == A.DOMAIN
        assert np.all(np.isfinite([t[0], fp[0], gn[0], tol[0]]))


class TestMissReasons:
    """Every reason is reachable; a miss never carries a distance."""

    def _one(self, o, d, **kw):
        g = EvenAsphereGeometry(**THEORY, coefficients=THEORY_COEFFS, **kw)
        t, n, hit, ng = g.ray_intersect(np.array([o], float), np.array([d], float))
        assert np.all(np.isfinite(n)) and np.all(np.isfinite(ng))
        return t[0], bool(hit[0]), A.MISS_REASONS[int(g.last_status[0])]

    def test_no_seed(self):
        # Parallel to the axis beyond the base ellipsoid's reach (r_t = 35 mm).
        t, hit, why = self._one([40.0, 0.0, -5.0], [0.0, 0.0, 1.0])
        assert (hit, why) == (False, "no_seed") and np.isinf(t)

    def test_behind(self):
        t, hit, why = self._one([5.0, 0.0, 5.0], [0.0, 0.0, 1.0])
        assert (hit, why) == (False, "behind") and np.isinf(t)

    def test_aperture(self):
        t, hit, why = self._one([20.0, 0.0, -5.0], [0.0, 0.0, 1.0])
        assert (hit, why) == (False, "aperture") and np.isinf(t)

    def test_not_converged_returns_no_iterate(self):
        g = EvenAsphereGeometry(**THEORY, coefficients=THEORY_COEFFS, max_iterations=1)
        o, d, _ = _aimed_ray(g, 6.0, 0.1)
        t, _, hit, _ = g.ray_intersect(o, d)
        assert not hit[0] and np.isinf(t[0])
        assert int(g.last_status[0]) == A.NOT_CONVERGED

    def test_counts_read_on_demand(self):
        g = EvenAsphereGeometry(**THEORY, coefficients=THEORY_COEFFS)
        o, d = _fan(500, 20.0)
        g.ray_intersect(o, d)
        counts = g.miss_reason_counts()
        assert sum(counts.values()) == 500
        assert counts["hit"] > 0


class TestFirstCrossing:
    """A strongly aspheric (non-monotone) surface: the nearest crossing."""

    def test_against_brute_force(self):
        g = EvenAsphereGeometry(25.0, 0.0, 12.0, [0.0, -3e-4, 2e-6])
        o, d = _fan(1500, 12.0, z0=-5.0, seed=2)
        t, _, hit, _ = g.ray_intersect(o, d)

        ts = np.linspace(1e-9, 80.0, 80001)
        t_ref = np.full(o.shape[0], np.inf)
        for i in range(o.shape[0]):
            p = o[i] + ts[:, None] * d[i]
            r2 = p[:, 0] ** 2 + p[:, 1] ** 2
            with np.errstate(invalid="ignore"):  # beyond the sag's domain: NaN
                f = p[:, 2] - g.sag(p[:, 0], p[:, 1])
            ch = np.where((np.sign(f[:-1]) != np.sign(f[1:])) & (r2[:-1] <= 144.0))[0]
            for j in ch:
                lo, hi = ts[j], ts[j + 1]
                for _ in range(60):
                    m = 0.5 * (lo + hi)
                    pm = o[i] + m * d[i]
                    fm = pm[2] - g.sag(pm[:1], pm[1:2])[0]
                    if np.sign(fm) == np.sign(f[j]):
                        lo = m
                    else:
                        hi = m
                pm = o[i] + lo * d[i]
                if pm[0] ** 2 + pm[1] ** 2 <= 144.0:
                    t_ref[i] = lo
                    break
        found = np.isfinite(t_ref)
        assert np.array_equal(found, hit)
        assert np.max(np.abs(t[hit] - t_ref[hit]) / t_ref[hit]) < 1e-12


def _fd4(fun, x0: float, h: float):
    return (-fun(x0 + 2 * h) + 8 * fun(x0 + h) - 8 * fun(x0 - h) + fun(x0 - 2 * h)) / (
        12 * h
    )


class TestAdjoint:
    """Implicit derivatives against fourth-order central differences."""

    PARAMS = ["radius", "conic", "a1", "a4", "a6", "a8", "shift_z", "shift_x"]

    @staticmethod
    def _outputs(name: str, value, o, d):
        """t and the normal's x and z at every ray, as functions of one
        parameter (a torch scalar), others at the theory's values."""
        radius, conic = 25.0, -0.5
        coeffs = [torch.tensor(v, dtype=torch.float64) for v in [2e-4, 1e-6, 1e-8, 1e-9]]
        shift = [torch.tensor(0.0, dtype=torch.float64) for _ in range(2)]
        if name == "radius":
            radius = value
        elif name == "conic":
            conic = value
        elif name in ("a1", "a4", "a6", "a8"):
            coeffs[["a1", "a4", "a6", "a8"].index(name)] = value
        else:
            shift[["shift_z", "shift_x"].index(name)] = value
        g = EvenAsphereGeometry(radius, conic, 12.5, coeffs)
        O = torch.tensor(o, dtype=torch.float64)
        offset = torch.stack(
            [shift[1], torch.zeros((), dtype=torch.float64), shift[0]]
        )
        t, _, hit, n = g.ray_intersect(O + offset, torch.tensor(d, dtype=torch.float64))
        return t, n, hit

    @pytest.mark.parametrize("name", PARAMS)
    def test_matches_finite_differences(self, name):
        _set("torch", "float64")
        base = {
            "radius": 25.0, "conic": -0.5, "a1": 2e-4, "a4": 1e-6, "a6": 1e-8,
            "a8": 1e-9, "shift_z": 0.0, "shift_x": 0.0,
        }[name]
        # Steps that move the sag at the rim by about 1e-3 mm: the primal root
        # is accurate to ~1e-14 mm (tolerance and polish), so the rounding
        # term of the difference is ~1e-14 / 1e-3 relative to the derivative,
        # and the h^4 truncation term is below it.
        h = {
            "radius": 25.0 * 1e-4, "conic": 1e-3, "a1": 1e-3 / 12.5**2,
            "a4": 1e-3 / 12.5**4, "a6": 1e-3 / 12.5**6, "a8": 1e-3 / 12.5**8,
            "shift_z": 1e-3, "shift_x": 1e-3,
        }[name]
        # Origins in a 16 mm square 8 mm below the vertex, directions within
        # about 17 degrees of the axis: most rays hit the 12.5 mm aperture.
        rng = np.random.default_rng(5)
        o = np.column_stack(
            [rng.uniform(-8, 8, 60), rng.uniform(-8, 8, 60), np.full(60, -8.0)]
        )
        d = np.column_stack(
            [rng.uniform(-0.3, 0.3, 60), rng.uniform(-0.3, 0.3, 60), np.ones(60)]
        )
        d /= np.linalg.norm(d, axis=1, keepdims=True)

        p = torch.tensor(base, dtype=torch.float64, requires_grad=True)
        t, n, hit = self._outputs(name, p, o, d)
        hit_np = _np(hit)
        assert hit_np.sum() >= 30
        rows = np.where(hit_np)[0]
        outs = [t, n[:, 0], n[:, 2]]
        ad = np.zeros((3, rows.size))
        for k, y in enumerate(outs):
            for j, i in enumerate(rows):
                (g_,) = torch.autograd.grad(y[i], p, retain_graph=True, allow_unused=True)
                ad[k, j] = 0.0 if g_ is None else g_.item()

        def value(x):
            with torch.no_grad():
                tt, nn, hh = self._outputs(name, torch.tensor(x, dtype=torch.float64), o, d)
                assert np.array_equal(_np(hh), hit_np)
                return np.stack([_np(tt)[rows], _np(nn[:, 0])[rows], _np(nn[:, 2])[rows]])

        fd = _fd4(value, base, h)
        scale = np.maximum(np.abs(fd).max(axis=1, keepdims=True), 1e-300)
        err = np.abs(ad - fd) / scale
        assert err.max() < 1e-7, err.max()
        # The gradient really flows (not a zero column).
        assert np.abs(ad[0]).max() > 0


class TestTraced:
    """The kind inside the bounce loop."""

    FOCAL = 50.0

    def _scene(self, detector):
        # A concave paraboloid opening toward +z, built as a flat base plus
        # r^2 / (4 f): exact, so every reflected ray passes through (0, 0, f).
        # The source sits below the focal plane and fires down, so the only
        # crossings of the focal plane are the reflected rays going up.
        f = self.FOCAL
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=0.5 * f, rx=np.pi),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55), total_flux=1.0,
                aperture_radius=8.0,
            ),
        )
        scene.add_component(
            "M",
            ReflectiveComponent(
                CoordinateSystem(),
                EvenAsphereGeometry(0.0, 0.0, 10.0, [1.0 / (4 * f)]),
                reflectance=1.0,
                name="M",
            ),
        )
        scene.add_detector("D", CoordinateSystem(z=f), detector)
        return scene

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_paraboloid_focuses_to_a_point(self, backend):
        _set(backend, "float64")
        scene = self._scene(RayDatabaseConfig(width=40.0, height=40.0, absorb=True))
        res = scene.trace(num_rays=2000, seed=3, max_depth=4)
        db = res.detectors["D"]
        x = np.asarray(_np(db.x), float)
        y = np.asarray(_np(db.y), float)
        # Every recorded crossing is a reflected ray at the focus. The spot is
        # a point to rounding: 64 ulps of the 50 mm scale.
        assert x.size > 1000
        assert np.max(np.hypot(x, y)) <= 64 * np.spacing(self.FOCAL)

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_emulated_replay_is_the_eager_trace(self, precision):
        from optiland.nonsequential.backends.torch_backend import (  # noqa: PLC0415
            TorchBackend,
        )

        _set("torch", precision)
        det = IrradianceDetectorConfig(
            width=4.0, height=4.0, num_pixels_x=16, num_pixels_y=16, splat="bilinear",
            absorb=False,
        )

        def ledger(backend):
            res = self._scene(det).trace(
                num_rays=6000, seed=11, max_depth=8, batch_size=2048, backend=backend
            )
            return (
                _np(res.detectors["D"].irradiance).tobytes(),
                float(res.flux_conservation_error),
            )

        eager = ledger(TorchBackend(seed=11, alive_check_every=0, compact_every=0))
        emulated = ledger(TorchBackend(seed=11, alive_check_every=0, graph_replay="emulate"))
        assert eager == emulated


class TestRegistration:
    def test_kinds_registered(self):
        names = kinds.registered_kinds()["geometry"]
        assert "even_asphere" in names and "odd_asphere" in names

    def test_lowering(self):
        g = OddAsphereGeometry(20.0, -1.0, 8.0, [1e-3, 2e-4])
        kind, params = _lower_geometry(g)
        assert kind == "odd_asphere"
        assert params["coefficients"] == [1e-3, 2e-4]
        assert params["radius"] == 20.0 and params["aperture_radius"] == 8.0
        assert params["max_iterations"] == A.DEFAULT_MAX_ITERATIONS

    def test_domain_check(self):
        with pytest.raises(ValueError, match="undefined inside the aperture"):
            EvenAsphereGeometry(10.0, 0.0, 10.5, [1e-4])

    def test_rim_sag_and_bounding_box(self):
        g = EvenAsphereGeometry(**THEORY, coefficients=THEORY_COEFFS)
        a = THEORY["aperture_radius"]
        assert g.rim_sag() == pytest.approx(float(g.sag(np.array([a]), np.array([0.0]))[0]))
        box = g.bounding_box((np.zeros(3), np.eye(3)))
        r = np.linspace(0, a, 1001)
        z = g.sag(r, np.zeros_like(r))
        assert box.zmin <= z.min() and box.zmax >= z.max()

    def test_detached_copy_keeps_coefficients(self):
        g = EvenAsphereGeometry(
            torch.tensor(25.0, requires_grad=True), -0.5, 12.5, THEORY_COEFFS
        )
        c = g.detached_copy()
        assert isinstance(c, EvenAsphereGeometry)
        assert c.coefficient_values() == THEORY_COEFFS
        assert c.radius == 25.0
