"""The tabulated coating's optional reverse-side table.

A table describes an interface as its incident medium sees it. Without a
second table a ray from the substrate reads the near table at the Snell angle:
exact for T always and for R of a lossless coating, an approximation for R
where the coating absorbs, and a forced total reflection beyond the critical
angle. A reverse-side table (the interface as the substrate sees it, at the
angle in the substrate) replaces that rule; the kind states which one it used
(``far_side``, ``far_side_exact``).

The stack here absorbs and is asymmetric, so its two sides differ: one 10 nm
metal-like layer (3.13 + 4.33i) and a quarter wave of 1.38 between vacuum and a
1.52 substrate. Both tables are sampled from a characteristic matrix written in
this file (``exp(-i w t)``, ``N = n + i k``), never from the engine's thin-film
module. The checks:

* the side the kind reports, with and without the reverse table, absorbing and
  lossless;
* at grid nodes, a ray from the substrate reads the reverse table's value (the
  closed form from the substrate, to 1e-14), where the reciprocity rule is off
  by more than 0.05 in R; T agrees between the two rules (reciprocity holds
  for T), to the interpolation of the Snell angle;
* the near side is the same with or without a reverse table, bit for bit;
* beyond the critical angle from the substrate the reverse table reads the
  frustrated reflectance (below one), and a traced face books the absorptance
  (issue 96);
* in Stokes mode the far side's relative phase is the reverse table's;
* refusals, the JSON form and lowering from the library's field names.
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

N_SUB = 1.52
METAL = complex(3.13, 4.33)
LAYERS = ((METAL, 0.010), (1.38, 0.55 / (4 * 1.38)))  # from the vacuum side
WL_NM = np.array([500.0, 550.0, 600.0])
ANG = np.arange(0.0, 90.0, 5.0)  # 0 to 85 degrees


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def charmat(n0, layers, ns, lam_um, theta_deg, pol):
    """(r, t, R, T) by the characteristic matrix; r_p in the Fresnel sign."""
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


def terms(lam_um, theta_deg, side):
    """(R_s, R_p, T_s, T_p, phase_r, phase_t) from the vacuum ("near") or the substrate ("far")."""
    if side == "near":
        n0, layers, ns = 1.0, LAYERS, N_SUB
    else:
        n0, layers, ns = N_SUB, tuple(reversed(LAYERS)), 1.0
    rs, ts, Rs, Ts = charmat(n0, layers, ns, lam_um, theta_deg, "s")
    rp, tp, Rp, Tp = charmat(n0, layers, ns, lam_um, theta_deg, "p")
    pr = math.degrees(cmath.phase(rp * rs.conjugate()))
    pt = math.degrees(cmath.phase(tp * ts.conjugate())) if abs(tp * ts) > 0 else 0.0
    return Rs, Rp, max(Ts, 0.0), max(Tp, 0.0), pr, pt


def grids(side):
    out = np.zeros((6, WL_NM.size, ANG.size))
    for i, w in enumerate(WL_NM):
        for j, a in enumerate(ANG):
            out[:, i, j] = terms(w / 1000.0, a, side)
    names = ("r_s", "r_p", "t_s", "t_p", "phase_r_deg", "phase_t_deg")
    return dict(zip(names, out, strict=True))


NEAR = grids("near")
FAR = grids("far")


def table(reverse=True, phases=True):
    g = dict(NEAR)
    rev = dict(FAR, wavelength_nm=WL_NM, angle_deg=ANG)
    if not phases:
        for f in ("phase_r_deg", "phase_t_deg"):
            g.pop(f)
            rev.pop(f)
    return TabulatedCoating(
        WL_NM, ANG, **g, substrate_material=IdealMaterial(N_SUB),
        reverse=rev if reverse else None,
    )


def far_eval(c, theta_s_deg, wl_um=0.55):
    th = np.atleast_1d(np.asarray(theta_s_deg, dtype=float))
    return c.evaluate(
        np.full(th.shape, wl_um), np.cos(np.radians(th)), from_substrate=np.ones(th.shape, bool)
    )


class TestWhichRule:
    def test_the_kind_states_its_far_side(self):
        assert table().far_side == "reverse table" and table().far_side_exact
        plain = table(reverse=False)
        assert plain.far_side == "reciprocity" and not plain.far_side_exact

    def test_a_lossless_table_without_reverse_is_exact(self):
        ang = np.array([0.0, 30.0, 60.0])
        r = np.array([[0.04, 0.05, 0.09]])
        c = TabulatedCoating([550.0], ang, r_s=r, r_p=r, lossless=True, substrate_material=1.5)
        assert c.far_side == "reciprocity" and c.far_side_exact


class TestFarSide:
    @pytest.mark.parametrize("theta_s", [0.0, 10.0, 25.0, 35.0])
    def test_reads_the_reverse_table_at_its_own_angle(self, theta_s):
        """At a node: the closed form from the substrate, where reciprocity is off."""
        R, T = far_eval(table(), theta_s)
        Rs, Rp, Ts, Tp, _, _ = terms(0.55, theta_s, "far")
        assert R[0] == pytest.approx(0.5 * (Rs + Rp), rel=1e-14)
        assert T[0] == pytest.approx(0.5 * (Ts + Tp), rel=1e-14, abs=1e-16)
        R_rec, T_rec = far_eval(table(reverse=False), theta_s)
        assert abs(R_rec[0] - R[0]) > 0.05  # the two sides differ
        # T is reciprocal: the two rules agree to the Snell angle's interpolation
        assert T_rec[0] == pytest.approx(T[0], abs=5e-3)

    def test_the_two_sides_of_the_closed_form_differ(self):
        near = terms(0.55, 20.0, "near")
        s0 = math.sin(math.radians(20.0)) / N_SUB
        far = terms(0.55, math.degrees(math.asin(s0)), "far")
        assert abs(0.5 * (near[0] + near[1]) - 0.5 * (far[0] + far[1])) > 0.05
        assert 0.5 * (near[2] + near[3]) == pytest.approx(0.5 * (far[2] + far[3]), rel=1e-12)

    def test_near_side_is_unchanged_bit_for_bit(self):
        th = np.radians([0.0, 12.5, 30.0, 47.5, 70.0])
        wl = np.array([0.5, 0.52, 0.55, 0.575, 0.6])
        mask = np.array([False, True, False, True, False])
        a = table().lookup(wl, np.cos(th), from_substrate=mask)
        b = table(reverse=False).lookup(wl, np.cos(th))
        for name, v in b.items():
            assert np.array_equal(a[name][~mask], v[~mask]), name

    @pytest.mark.parametrize("theta_s", [45.0, 60.0, 75.0])
    def test_beyond_the_critical_angle_from_the_substrate(self, theta_s):
        """asin(1 / 1.52) = 41.14 deg: the reverse table reads the frustrated reflectance."""
        R, T = far_eval(table(), theta_s)
        Rs, Rp, _, _, _, _ = terms(0.55, theta_s, "far")
        assert R[0] == pytest.approx(0.5 * (Rs + Rp), rel=1e-14)
        assert R[0] < 0.95 and T[0] == 0.0
        R1, T1 = far_eval(table(reverse=False), theta_s)
        assert R1[0] == 1.0 and T1[0] == 0.0

    def test_torch_float32_selects_the_same_sides(self):
        torch = pytest.importorskip("torch")
        be.set_backend("torch")
        be.set_device("cpu")
        be.grad_mode.disable()
        be.set_precision("float32")
        th = torch.deg2rad(be.array([10.0, 25.0, 60.0]))
        wl = be.array([0.55, 0.55, 0.55])
        mask = torch.tensor([True, False, True])
        R, _ = table().evaluate(wl, torch.cos(th), from_substrate=mask)
        R_near, _ = table().evaluate(wl, torch.cos(th))
        assert torch.equal(R[1], R_near[1])
        for k, a in ((0, 10.0), (2, 60.0)):
            Rs, Rp, _, _, _, _ = terms(0.55, a, "far")
            assert float(R[k]) == pytest.approx(0.5 * (Rs + Rp), rel=1e-5)


class TestTracedFromTheGlass:
    @pytest.mark.parametrize("theta_s", [25.0, 60.0])
    def test_split_tree_reads_the_reverse_table(self, theta_s):
        """The beam inside the glass: below the cutoff the split tree carries R and T;
        beyond it the ray reflects R and the absorptance is booked (issue 96)."""
        from optiland.nonsequential.ir.scene_ir import SamplingPolicy  # noqa: PLC0415

        t = math.radians(theta_s)
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(y=50.0 * math.sin(t), z=50.0 * math.cos(t), rx=math.pi - t),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=0.5
            ),
        )
        face = RefractiveComponent(
            CoordinateSystem(z=0.0), PlaneGeometry(), VACUUM,
            NSQMaterial(optiland_material=IdealMaterial(N_SUB)), coating=table(), name="face",
        )
        scene.add_component("face", face)
        for name, z in (("reflected", 100.0), ("transmitted", -100.0)):
            scene.add_detector(name, CoordinateSystem(z=z), IrradianceDetectorConfig(
                width=4000, height=4000, num_pixels_x=1, num_pixels_y=1, splat="hard"))
        scene.sampling_policy = SamplingPolicy(split_depth=2, split_budget=8.0, rr_start_flux=1e-16)
        result = scene.trace(num_rays=32, seed=1, max_depth=2)
        phi = float(result.total_flux_in)
        refl = float(result.detectors["reflected"].total_flux_float) / phi
        trans = float(result.detectors["transmitted"].total_flux_float) / phi
        Rs, Rp, Ts, Tp, _, _ = terms(0.55, theta_s, "far")
        assert refl == pytest.approx(0.5 * (Rs + Rp), rel=1e-13)
        assert trans == pytest.approx(0.5 * (Ts + Tp), rel=1e-13, abs=1e-16)
        loss = float(face.coating_loss) / phi
        assert loss == pytest.approx(1.0 - refl - trans, rel=1e-12)


class TestStokesFarSide:
    def test_far_phase_is_the_reverse_tables(self):
        c = table()
        for th in (10.0, 30.0, 60.0):  # nodes
            sp = c.sp(np.array([0.55]), np.array([math.cos(math.radians(th))]), np.array([True]))
            got = math.degrees(math.atan2(float(sp.xr_im[0]), float(sp.xr_re[0])))
            want = terms(0.55, th, "far")[4]
            assert abs(((got - want) + 180) % 360 - 180) < 1e-9

    def test_phases_on_one_side_only_are_refused(self):
        rev = dict(FAR, wavelength_nm=WL_NM, angle_deg=ANG)
        rev.pop("phase_r_deg")
        rev.pop("phase_t_deg")
        with pytest.raises(ValueError, match="both sides"):
            TabulatedCoating(WL_NM, ANG, **NEAR, substrate_material=N_SUB, reverse=rev)


class TestFormsAndRefusals:
    def test_unknown_field_and_nested_reverse_are_refused(self):
        rev = dict(FAR, wavelength_nm=WL_NM, angle_deg=ANG, colour=[1])
        with pytest.raises(ValueError, match="unknown fields"):
            TabulatedCoating(WL_NM, ANG, **NEAR, substrate_material=N_SUB, reverse=rev)
        inner = table()
        with pytest.raises(ValueError, match="no reverse table of its own"):
            TabulatedCoating(WL_NM, ANG, **NEAR, substrate_material=N_SUB, reverse=inner)

    def test_a_coating_object_as_the_reverse_table(self):
        rev = TabulatedCoating(WL_NM, ANG, **FAR, substrate_material=1.0,
                               incident_material=N_SUB)
        c = TabulatedCoating(WL_NM, ANG, **NEAR, substrate_material=N_SUB, reverse=rev)
        R, _ = far_eval(c, 25.0)
        R2, _ = far_eval(table(), 25.0)
        assert np.array_equal(R, R2)

    def test_json_round_trip(self):
        c = table()
        d = c.to_dict()
        assert "reverse" in d and "reverse" not in table(reverse=False).to_dict()
        back = BaseCoating.from_dict(d)
        assert back.far_side == "reverse table"
        th = np.radians([5.0, 25.0, 65.0])
        wl = np.full(3, 0.55)
        m = np.array([True, False, True])
        for x, y in zip(c.evaluate(wl, np.cos(th), m), back.evaluate(wl, np.cos(th), m),
                        strict=True):
            assert np.array_equal(x, y)

    def test_from_table_takes_the_reverse_as_library_arrays(self):
        class Rec:  # the library's CoatingTable shape: arrays() gives the fields
            def __init__(self, d):
                self._d = d

            def arrays(self):
                return dict(self._d)

        near = Rec(dict(NEAR, wavelength_nm=WL_NM, angle_deg=ANG))
        far = Rec(dict(FAR, wavelength_nm=WL_NM, angle_deg=ANG))
        c = TabulatedCoating.from_table(near, substrate_material=N_SUB, reverse=far)
        R, _ = far_eval(c, 25.0)
        R2, _ = far_eval(table(), 25.0)
        assert np.array_equal(R, R2)
