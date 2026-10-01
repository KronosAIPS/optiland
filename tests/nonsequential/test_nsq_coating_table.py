"""The tabulated coating kind, ``optiland.coatings.TabulatedCoating``.

A coating given as a table of R and T for s and p against wavelength and angle of
incidence, with the shared library's coating-table field names, interpolated
bilinearly inside the bounce loop. The tables here are sampled from an independent
characteristic matrix written in this file (Born and Wolf, ``exp(-i w t)``,
``N = n + i k``), never from the engine's own thin-film module, so a table test does
not inherit a thin-film defect.

What is pinned, each against its own route:

* the refusals (shapes, ranges, energy, a missing T, one phase grid, no substrate);
* the interpolation rule: a node returns its value; any point equals an independent
  bilinear interpolation of the same grids (1e-14 at float64; 64 u_32 relative at
  float32, the operation count of the lookup); a point outside is clamped;
* the error bound: at every cell centre of a lossless layer's table the error
  against the closed form is inside the table's own second-difference estimate;
* the far side: a ray from the substrate reads the table at the Snell angle
  (T exactly, R exactly for a lossless coating) and reflects totally beyond the
  critical angle; the relative phases follow the lossless Stokes relations;
* the Stokes mode: refused without phase grids; realizable (M22^2 + M23^2 =
  R_s R_p); the relative phase at a node equal to the closed form's;
* a traced face on a split tree reads exactly the table's value; a mirror takes a
  table as its reflectance; an unpolarized Stokes trace equals the scalar one bit
  for bit; a steady bounce uploads nothing;
* the JSON form round-trips; a library record lowers field for field.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import BaseCoating, TabulatedCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry
from optiland.nonsequential.components.reflective import ReflectiveComponent
from optiland.nonsequential.ir.scene_ir import SamplingPolicy

U32 = 2.0**-24
N_SUB = 1.52
N_FILM = 1.38
D_FILM = 0.55 / (4 * N_FILM)
WL_NM = np.arange(450.0, 650.0 + 1e-9, 10.0)
ANG = np.arange(0.0, 80.0 + 1e-9, 5.0)


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


# ---------------------------------------------------------------------------
# An independent characteristic matrix, exp(-i w t)
# ---------------------------------------------------------------------------


def charmat(n0, layers, ns, lam_um, theta_deg, pol):
    """(r, t, R, T): r_p in the Fresnel sign (section 4.2), t the tangential amplitude."""
    s0 = n0 * math.sin(math.radians(theta_deg))

    def ncos(N):
        c = cmath.sqrt(complex(N) ** 2 - s0 * s0)
        if c.imag < 0 or (c.imag == 0 and c.real < 0):
            c = -c
        return c

    def eta(N):
        return ncos(N) if pol == "s" else complex(N) ** 2 / ncos(N)

    M = np.eye(2, dtype=complex)
    for N, d in layers:
        dl = 2 * math.pi * d * ncos(N) / lam_um
        e = eta(N)
        M = M @ np.array(
            [[cmath.cos(dl), -1j * cmath.sin(dl) / e], [-1j * e * cmath.sin(dl), cmath.cos(dl)]]
        )
    B, C = M @ np.array([1.0, eta(ns)])
    e0 = eta(complex(n0))
    r = (e0 * B - C) / (e0 * B + C)
    t = 2 * e0 / (e0 * B + C)
    T = abs(t) ** 2 * eta(ns).real / e0.real
    return (r if pol == "s" else -r), t, abs(r) ** 2, T


def layer_terms(lam_um, theta_deg, n0=1.0, ns=N_SUB, layers=((N_FILM, D_FILM),)):
    rs, ts, Rs, Ts = charmat(n0, layers, ns, lam_um, theta_deg, "s")
    rp, tp, Rp, Tp = charmat(n0, layers, ns, lam_um, theta_deg, "p")
    pr = math.degrees(cmath.phase(rp * rs.conjugate()))
    pt = math.degrees(cmath.phase(tp * ts.conjugate())) if abs(tp * ts) > 0 else 0.0
    return Rs, Rp, Ts, Tp, pr, pt


def sampled(wl_nm=WL_NM, ang=ANG, **kw):
    out = np.zeros((6, wl_nm.size, ang.size))
    for i, w in enumerate(wl_nm):
        for j, a in enumerate(ang):
            out[:, i, j] = layer_terms(w / 1000.0, a, **kw)
    return out


GRIDS = sampled()


def table(phases=True, **kw):
    fields = dict(zip(("r_s", "r_p", "t_s", "t_p"), GRIDS[:4], strict=True))
    if phases:
        kw.setdefault("phase_r_deg", GRIDS[4])
        kw.setdefault("phase_t_deg", GRIDS[5])
    kw.setdefault("substrate_material", IdealMaterial(N_SUB))
    return TabulatedCoating(WL_NM, ANG, **fields, **kw)


def bilinear(grid, wl_nm, ang, w, a):
    """An independent bilinear interpolation with edge clamping."""
    w = min(max(w, wl_nm[0]), wl_nm[-1])
    a = min(max(a, ang[0]), ang[-1])
    i = min(int(np.searchsorted(wl_nm, w, side="right")) - 1, wl_nm.size - 2)
    j = min(int(np.searchsorted(ang, a, side="right")) - 1, ang.size - 2)
    tw = (w - wl_nm[i]) / (wl_nm[i + 1] - wl_nm[i])
    ta = (a - ang[j]) / (ang[j + 1] - ang[j])
    lo = grid[i, j] + ta * (grid[i, j + 1] - grid[i, j])
    hi = grid[i + 1, j] + ta * (grid[i + 1, j + 1] - grid[i + 1, j])
    return lo + tw * (hi - lo)


def _rays(points):
    w = np.array([p[0] for p in points]) / 1000.0
    c = np.cos(np.radians([p[1] for p in points]))
    return w, c


# ---------------------------------------------------------------------------


class TestRefusals:
    def test_shape(self):
        with pytest.raises(ValueError, match="shaped"):
            TabulatedCoating(WL_NM, ANG, GRIDS[0].T, GRIDS[1], GRIDS[2], GRIDS[3],
                             substrate_material=1.52)

    def test_axes(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            TabulatedCoating(WL_NM[::-1], ANG, *GRIDS[:4], substrate_material=1.52)
        with pytest.raises(ValueError, match="0..90"):
            TabulatedCoating(WL_NM, ANG + 15.0, *GRIDS[:4], substrate_material=1.52)

    def test_values_and_energy(self):
        bad = GRIDS[0].copy()
        bad[0, 0] = 1.2
        with pytest.raises(ValueError, match="outside 0..1"):
            TabulatedCoating(WL_NM, ANG, bad, *GRIDS[1:4], substrate_material=1.52)
        with pytest.raises(ValueError, match="exceeds 1"):
            TabulatedCoating(WL_NM, ANG, GRIDS[0] + 0.01, GRIDS[1], GRIDS[2], GRIDS[3],
                             substrate_material=1.52)

    def test_missing_transmittance(self):
        with pytest.raises(ValueError, match="no t_s and no t_p|no t_s"):
            TabulatedCoating(WL_NM, ANG, r_s=GRIDS[0], r_p=GRIDS[1], substrate_material=1.52)
        c = TabulatedCoating(WL_NM, ANG, r_s=GRIDS[0], r_p=GRIDS[1], lossless=True,
                             substrate_material=1.52)
        assert np.array_equal(c.grids["t_s"], 1.0 - GRIDS[0])

    def test_one_phase_grid_and_no_substrate(self):
        with pytest.raises(ValueError, match="both phase"):
            TabulatedCoating(WL_NM, ANG, *GRIDS[:4], phase_r_deg=GRIDS[4], substrate_material=1.52)
        with pytest.raises(ValueError, match="substrate_material"):
            TabulatedCoating(WL_NM, ANG, *GRIDS[:4])


class TestInterpolationRule:
    def test_node_returns_its_value(self):
        c = table()
        pts = [(w, a) for w in WL_NM[::4] for a in ANG[1::3]]
        R, T = c.evaluate(*_rays(pts))
        for k, (w, a) in enumerate(pts):
            i, j = int(np.where(WL_NM == w)[0][0]), int(np.where(ANG == a)[0][0])
            assert abs(R[k] - 0.5 * (GRIDS[0][i, j] + GRIDS[1][i, j])) <= 1e-13
            assert abs(T[k] - 0.5 * (GRIDS[2][i, j] + GRIDS[3][i, j])) <= 1e-13

    def test_equals_an_independent_bilinear_interpolation(self):
        rng = np.random.default_rng(5)
        pts = list(zip(rng.uniform(450, 650, 200), rng.uniform(1, 79, 200), strict=True))
        w, cos = _rays(pts)
        g = table().lookup(w, cos)
        for k, (wn, a) in enumerate(pts):
            a_eng = math.degrees(math.acos(cos[k]))  # the engine's own angle
            for f, name in enumerate(("r_s", "r_p", "t_s", "t_p")):
                assert abs(g[name][k] - bilinear(GRIDS[f], WL_NM, ANG, w[k] * 1000, a_eng)) <= 1e-14

    def test_outside_is_clamped(self):
        c = table()
        R, _ = c.evaluate(*_rays([(300.0, 20.0), (900.0, 20.0), (550.0, 88.0)]))
        Rin, _ = c.evaluate(*_rays([(450.0, 20.0), (650.0, 20.0), (550.0, 80.0)]))
        assert np.allclose(R, Rin, rtol=0, atol=1e-14)

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_torch_legs(self, precision):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(7)
        pts = list(zip(rng.uniform(440, 660, 64), rng.uniform(0, 82, 64), strict=True))
        w, cos = _rays(pts)
        c = table()
        Rn, Tn = c.evaluate(w, cos)
        be.set_backend("torch")
        be.set_device("cpu")
        be.grad_mode.disable()
        be.set_precision(precision)
        dt = torch.float64 if precision == "float64" else torch.float32
        Rt, Tt = c.evaluate(torch.tensor(w, dtype=dt), torch.tensor(cos, dtype=dt))
        assert Rt.dtype == dt
        # float32: 64 u32 relative, the lookup's operation count (the angle's
        # conditioning included); float64: 1e-14 absolute
        tol = 1e-14 if precision == "float64" else 64 * U32 * np.maximum(Rn, 1e-3)
        assert np.all(np.abs(Rt.numpy() - Rn) <= tol)
        tol_t = 1e-14 if precision == "float64" else 64 * U32
        assert np.all(np.abs(Tt.numpy() - Tn) <= tol_t)


class TestAgainstTheClosedForm:
    def test_cell_centres_within_the_second_difference_estimate(self):
        """Up to 60 degrees, where R's curvature changes little within a cell.

        The estimate is the largest second difference of the whole table over
        8; it bounds the error where the second derivative is nearly constant
        across a cell. Towards grazing it grows by more than a factor two
        within one 5-degree cell, and the estimate is then not a bound: the
        last cell (77.5 degrees) is shown to exceed it, as the docstring says.
        """
        c = table()
        est = c.interpolation_error_estimate()
        bound = 0.5 * (est["r_s"] + est["r_p"])
        centres = [(w + 5.0, a + 2.5) for w in WL_NM[:-1] for a in ANG[:-1] if a + 2.5 <= 60.0]
        R, _ = c.evaluate(*_rays(centres))
        exact = np.array([0.5 * sum(layer_terms(w / 1000, a)[:2]) for w, a in centres])
        assert np.max(np.abs(R - exact)) <= bound
        R77, _ = c.evaluate(*_rays([(645.0, 77.5)]))
        assert abs(R77[0] - 0.5 * sum(layer_terms(0.645, 77.5)[:2])) > bound


class TestFarSide:
    def test_reads_the_table_at_the_snell_angle(self):
        c = table()
        th_s = np.array([0.0, 10.0, 25.0, 40.0])
        th_0 = np.degrees(np.arcsin(N_SUB * np.sin(np.radians(th_s))))
        wl = np.full(4, 0.555)
        Rb, Tb = c.evaluate(wl, np.cos(np.radians(th_s)), from_substrate=np.ones(4, bool))
        Rf, Tf = c.evaluate(wl, np.cos(np.radians(th_0)))
        assert np.max(np.abs(Rb - Rf)) <= 1e-13 and np.max(np.abs(Tb - Tf)) <= 1e-13

    def test_lossless_far_side_against_the_reversed_layer(self):
        """Within the interpolation estimate of the reversed layer's closed form."""
        c = table()
        est = c.interpolation_error_estimate()
        th_s = np.array([5.0, 20.0, 35.0])
        wl = np.full(3, 0.555)
        Rb, Tb = c.evaluate(wl, np.cos(np.radians(th_s)), from_substrate=np.ones(3, bool))
        for k, a in enumerate(th_s):
            rb = layer_terms(0.555, a, n0=N_SUB, ns=1.0)
            assert abs(Rb[k] - 0.5 * (rb[0] + rb[1])) <= 0.5 * (est["r_s"] + est["r_p"])
            assert abs(Rb[k] + Tb[k] - 1.0) <= 1e-12

    def test_beyond_the_critical_angle_from_the_substrate(self):
        R, T = table().evaluate(np.full(2, 0.55), np.cos(np.radians([45.0, 70.0])),
                                from_substrate=np.ones(2, bool))
        assert np.array_equal(R, [1.0, 1.0]) and np.array_equal(T, [0.0, 0.0])

    def test_far_side_phase_is_the_stokes_relation(self):
        """phase_r' = 2 phase_t - phase_r, checked at nodes against the reversed layer."""
        c = table()
        for th_0 in (10.0, 30.0, 45.0):  # nodes, so no interpolation enters
            th_s = math.degrees(math.asin(math.sin(math.radians(th_0)) / N_SUB))
            sp = c.sp(np.array([0.55]), np.array([math.cos(math.radians(th_s))]),
                      np.array([True]))
            got = math.degrees(math.atan2(float(sp.xr_im[0]), float(sp.xr_re[0])))
            want = layer_terms(0.55, th_s, n0=N_SUB, ns=1.0)[4]
            assert abs(((got - want) + 180) % 360 - 180) < 1e-6
            gt = math.degrees(math.atan2(float(sp.xt_sin[0]), float(sp.xt_cos[0])))
            assert abs(((gt - layer_terms(0.55, th_s, n0=N_SUB, ns=1.0)[5]) + 180) % 360 - 180) < 1e-6


class TestStokes:
    def test_refused_without_phase_grids(self):
        c = table(phases=False)
        with pytest.raises(ValueError, match="phase_r_deg"):
            c.sp(np.array([0.55]), np.array([0.9]))
        # its scalar use is unaffected
        c.evaluate(np.array([0.55]), np.array([0.9]))

    def test_realizable_and_the_node_phase(self):
        c = table()
        pts = [(550.0, 45.0), (600.0, 30.0), (470.0, 60.0)]
        w, cos = _rays(pts)
        sp = c.sp(w, cos)
        lhs = np.asarray(sp.xr_re) ** 2 + np.asarray(sp.xr_im) ** 2
        assert np.allclose(lhs, np.asarray(sp.Rs) * np.asarray(sp.Rp), rtol=1e-13, atol=0)
        for k, (wn, a) in enumerate(pts):
            want = layer_terms(wn / 1000, a)[4]
            got = math.degrees(math.atan2(float(sp.xr_im[k]), float(sp.xr_re[k])))
            assert abs(((got - want) + 180) % 360 - 180) < 1e-9


def _face(coating, polarization=None):
    scene = NSQScene()
    t = math.radians(30.0)
    scene.add_source(
        "S",
        CoordinateSystem(y=-50.0 * math.sin(t), z=-50.0 * math.cos(t), rx=-t),
        CollimatedSourceConfig(spectrum=Spectrum.monochromatic(0.555), total_flux=1.0,
                               aperture_radius=0.5),
    )
    scene.add_component(
        "face",
        RefractiveComponent(CoordinateSystem(z=0.0), PlaneGeometry(), VACUUM,
                            NSQMaterial(optiland_material=IdealMaterial(N_SUB)),
                            coating=coating, name="face"),
    )
    scene.add_detector("reflected", CoordinateSystem(z=-100.0), IrradianceDetectorConfig(
        width=2000, height=2000, num_pixels_x=1, num_pixels_y=1, splat="hard"))
    scene.sampling_policy = SamplingPolicy(split_depth=2, split_budget=8.0, rr_start_flux=1e-16)
    return scene


class TestTraced:
    def test_split_tree_reads_the_table(self):
        c = table()
        result = _face(c).trace(num_rays=64, seed=1, max_depth=2)
        frac = result.detectors["reflected"].total_flux_float / result.total_flux_in
        R, _ = c.evaluate(np.array([0.555]), np.array([math.cos(math.radians(30.0))]))
        assert frac == pytest.approx(float(R[0]), rel=1e-12)

    def test_unpolarized_stokes_trace_equals_the_scalar_trace(self):
        from optiland.nonsequential.backends.numpy_backend import NumpyBackend

        c = table()
        a = _face(c).trace(num_rays=256, seed=2, max_depth=3,
                           backend=NumpyBackend(seed=2, polarization="off"))
        b = _face(c).trace(num_rays=256, seed=2, max_depth=3,
                           backend=NumpyBackend(seed=2, polarization="stokes"))
        assert a.detectors["reflected"].total_flux_float == b.detectors["reflected"].total_flux_float

    def test_stokes_trace_refuses_a_table_without_phases(self):
        from optiland.nonsequential.backends.numpy_backend import NumpyBackend

        with pytest.raises(ValueError, match="phase_r_deg"):
            _face(table(phases=False)).trace(num_rays=16, seed=1, max_depth=2,
                                             backend=NumpyBackend(seed=1, polarization="stokes"))

    def test_mirror_takes_a_table(self):
        c = table()
        scene = NSQScene()
        scene.add_source("S", CoordinateSystem(z=-50.0), CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.555), total_flux=1.0, aperture_radius=0.5))
        scene.add_component("m", ReflectiveComponent(CoordinateSystem(z=0.0), PlaneGeometry(), c,
                                                     material_front=VACUUM, name="m"))
        scene.add_detector("back", CoordinateSystem(z=-100.0), IrradianceDetectorConfig(
            width=200, height=200, num_pixels_x=1, num_pixels_y=1, splat="hard"))
        result = scene.trace(num_rays=64, seed=1, max_depth=2)
        frac = result.detectors["back"].total_flux_float / result.total_flux_in
        R, _ = c.evaluate(np.array([0.555]), np.array([1.0]))
        assert frac == pytest.approx(float(R[0]), rel=1e-12)

    def test_a_steady_bounce_uploads_nothing(self):
        pytest.importorskip("torch")
        from tests.nonsequential.test_nsq_host_uploads import _trace_and_count

        def build():
            scene = NSQScene()
            scene.add_source("S", CoordinateSystem(z=-50.0), CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.555), total_flux=1.0, aperture_radius=2.5))
            scene.add_component("front", RefractiveComponent(
                CoordinateSystem(z=0.0), PlaneGeometry(), VACUUM,
                NSQMaterial(optiland_material=IdealMaterial(N_SUB)), coating=table(), name="front"))
            scene.add_component("back", RefractiveComponent(
                CoordinateSystem(z=10.0), PlaneGeometry(),
                NSQMaterial(optiland_material=IdealMaterial(N_SUB)), VACUUM, name="back"))
            for name, z in (("reflected", -100.0), ("transmitted", 50.0)):
                scene.add_detector(name, CoordinateSystem(z=z), IrradianceDetectorConfig(
                    width=40, height=40, num_pixels_x=4, num_pixels_y=4))
            return scene

        counter = _trace_and_count(build, 12)
        assert counter.steady_bounces >= 6
        assert dict(counter.recurring) == {}


class TestRecords:
    def test_json_round_trip(self):
        c = table()
        d = c.to_dict()
        assert set(("wavelength_nm", "angle_deg", "r_s", "r_p", "t_s", "t_p",
                    "phase_r_deg", "phase_t_deg")) <= set(d)
        back = BaseCoating.from_dict(d)
        assert isinstance(back, TabulatedCoating)
        w, cos = _rays([(555.0, 33.0), (612.0, 51.0)])
        assert np.array_equal(back.evaluate(w, cos)[0], c.evaluate(w, cos)[0])

    def test_a_record_lowers_field_for_field(self):
        class FakeTable:
            def arrays(self):
                return {"wavelength_nm": WL_NM, "angle_deg": ANG, "r_s": GRIDS[0],
                        "r_p": GRIDS[1], "t_s": GRIDS[2], "t_p": GRIDS[3]}

        class FakeRecord:
            kind, name, table = "table", "vendor AR", FakeTable()

        c = TabulatedCoating.from_record(FakeRecord(), substrate_material=IdealMaterial(N_SUB))
        assert c.name == "vendor AR" and not c.has_phases
        with pytest.raises(ValueError, match="kind 'table'"):
            TabulatedCoating.from_record(type("R", (), {"kind": "stack", "table": None})())

    def test_the_library_record_type(self):
        so = pytest.importorskip("kmat.surface_optics")
        tab = so.CoatingTable(wavelength_nm=WL_NM.tolist(), angle_deg=ANG.tolist(),
                              r_s=GRIDS[0].tolist(), r_p=GRIDS[1].tolist(),
                              t_s=GRIDS[2].tolist(), t_p=GRIDS[3].tolist())
        c = TabulatedCoating.from_table(tab, substrate_material=IdealMaterial(N_SUB))
        assert np.array_equal(c.grids["r_s"], GRIDS[0])

    def test_sequential_engine_scales_intensity(self):
        from optiland.rays import RealRays

        c = table()
        rays = RealRays(x=[0.0], y=[0.0], z=[0.0], L=[0.0], M=[math.sin(0.5)],
                        N=[math.cos(0.5)], intensity=[1.0], wavelength=[0.555])
        # the incoming direction, which the sequential engine records before an interaction
        rays.L0, rays.M0, rays.N0 = be.copy(rays.L), be.copy(rays.M), be.copy(rays.N)
        c.reflect(rays, nx=be.array([0.0]), ny=be.array([0.0]), nz=be.array([1.0]))
        R, _ = c.evaluate(np.array([0.555]), np.array([math.cos(0.5)]))
        assert float(be.to_numpy(rays.i)[0]) == pytest.approx(float(R[0]), rel=1e-14)
