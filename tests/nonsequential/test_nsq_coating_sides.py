"""A coating met from its substrate side (the research repository's issue 83).

A thin-film stack describes an interface as its incident medium sees it. A ray
arriving from the substrate sees the layers in the opposite order between the
two media swapped, at its own angle in the substrate. Before this fix the
engine evaluated every ray from the incident medium at the ray's own cosine,
which is wrong whenever the stack absorbs (an asymmetric stack reflects
differently from its two sides) and whenever the angle in the substrate is not
the angle in the incident medium.

Each test has an independent route:

* the reversed stack against a characteristic matrix written here (Born and
  Wolf, ``exp(-i w t)``, ``N = n + i k``), for an asymmetric absorbing layer from
  each side;
* a lossless stack from both sides against energy conservation (R + T = 1) and
  reciprocity (T from the substrate at theta_s equals T from the incident
  medium at the Snell angle theta_0, and so does R for a lossless stack);
* the mask: ``from_substrate`` all False is the side-blind evaluation bit for
  bit, all True is an explicitly reversed ``ThinFilmStack`` bit for bit;
* a traced face, the beam arriving inside the glass, on a deterministic split
  tree: the reflected and transmitted fractions are the reversed stack's;
* the side decision of a lens's back face, whose front medium is the glass.

Tolerances: 1e-12 against the independent matrix (a product of at most three
2x2 complex matrices and a quotient, about 60 rounded operations on values of
order one: 60 u = 6.7e-15, and 1e-12 leaves room for the arccos of the cosine
the adapter takes as input, whose error grows as 1 / sin theta). The traced
fractions to 1e-12 relative: the split tree hands each child exactly R or T.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
    SurfaceConfig,
)
from optiland.nonsequential.components.coating_support import (
    UnpolarizedThinFilmCoating,
    coating_incident_is_front,
)
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.thin_film import ThinFilmStack

WL = 0.55
N_SUB = 1.52
CR = complex(3.13, 4.33)  # a chromium-like layer, chapter 04 section 4.4
CR_NM = 10.0


def _cr_stack() -> ThinFilmStack:
    st = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(N_SUB))
    st.add_layer_nm(IdealMaterial(CR.real, CR.imag), CR_NM)
    return st


def _lossless_stack() -> ThinFilmStack:
    """Two quarter waves, H then L, at 0.55 um: asymmetric in its layer order."""
    st = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(N_SUB), reference_wl_um=WL)
    st.add_layer_qwot(IdealMaterial(2.32))
    st.add_layer_qwot(IdealMaterial(1.38))
    return st


def _reversed(st: ThinFilmStack) -> ThinFilmStack:
    rev = ThinFilmStack(st.substrate_material, st.incident_material)
    rev.layers.extend(reversed(st.layers))
    return rev


def charmat_RT(n0: float, layers, ns: complex, theta_deg: float, pol: str):
    """Independent characteristic matrix: (R, T) of ``layers`` [(N, d_um)] between n0 and ns."""
    s0 = n0 * math.sin(math.radians(theta_deg))

    def ncos(N):
        c = cmath.sqrt(N * N - s0 * s0)
        if c.imag < 0 or (c.imag == 0 and c.real < 0):
            c = -c
        return c

    def eta(N):
        nc = ncos(N)
        return nc if pol == "s" else N * N / nc

    M = np.eye(2, dtype=complex)
    for N, d in layers:
        dl = 2 * math.pi * d * ncos(N) / WL
        e = eta(N)
        M = M @ np.array(
            [[cmath.cos(dl), -1j * cmath.sin(dl) / e], [-1j * e * cmath.sin(dl), cmath.cos(dl)]]
        )
    B, C = M @ np.array([1.0, eta(ns)])
    e0 = eta(complex(n0))
    r = (e0 * B - C) / (e0 * B + C)
    T = 4 * e0.real * eta(ns).real / abs(e0 * B + C) ** 2
    return abs(r) ** 2, T


def charmat_unpol(n0, layers, ns, theta_deg):
    Rs, Ts = charmat_RT(n0, layers, ns, theta_deg, "s")
    Rp, Tp = charmat_RT(n0, layers, ns, theta_deg, "p")
    return 0.5 * (Rs + Rp), 0.5 * (Ts + Tp)


def _eval(coating, thetas_deg, from_substrate=None):
    th = np.radians(np.asarray(thetas_deg, dtype=float))
    wl = np.full(th.shape, WL)
    kw = {} if from_substrate is None else {"from_substrate": np.asarray(from_substrate)}
    R, T = coating.evaluate(wl, np.cos(th), **kw)
    return np.asarray(R, dtype=float), np.asarray(T, dtype=float)


THETAS = [0.0, 20.0, 35.0, 40.0]


class TestTheMask:
    def test_all_false_is_the_side_blind_evaluation_bit_for_bit(self):
        for st in (_cr_stack(), _lossless_stack()):
            c = UnpolarizedThinFilmCoating(st)
            R0, T0 = _eval(c, THETAS)
            R1, T1 = _eval(c, THETAS, from_substrate=[False] * len(THETAS))
            assert np.array_equal(R0, R1) and np.array_equal(T0, T1)

    def test_all_true_is_the_reversed_stack_bit_for_bit(self):
        for st in (_cr_stack(), _lossless_stack()):
            R1, T1 = _eval(UnpolarizedThinFilmCoating(st), THETAS, [True] * len(THETAS))
            R2, T2 = _eval(UnpolarizedThinFilmCoating(_reversed(st)), THETAS)
            assert np.array_equal(R1, R2) and np.array_equal(T1, T2)

    def test_mixed_mask_selects_per_ray(self):
        st = _cr_stack()
        c = UnpolarizedThinFilmCoating(st)
        mask = [True, False, True, False]
        R, T = _eval(c, THETAS, mask)
        Rf, Tf = _eval(c, THETAS)
        Rb, Tb = _eval(c, THETAS, [True] * 4)
        want_R = np.where(mask, Rb, Rf)
        want_T = np.where(mask, Tb, Tf)
        assert np.array_equal(R, want_R) and np.array_equal(T, want_T)


class TestAgainstTheReversedClosedForm:
    def test_absorbing_layer_from_each_side(self):
        """10 nm of 3.13 + 4.33i on 1.52: R from air and R from the glass differ."""
        c = UnpolarizedThinFilmCoating(_cr_stack())
        d = CR_NM / 1000.0
        Rf, Tf = _eval(c, THETAS)
        Rb, Tb = _eval(c, THETAS, [True] * len(THETAS))
        for i, th in enumerate(THETAS):
            rf, tf = charmat_unpol(1.0, [(CR, d)], complex(N_SUB), th)
            rb, tb = charmat_unpol(N_SUB, [(CR, d)], complex(1.0), th)
            assert abs(Rf[i] - rf) <= 1e-12 and abs(Tf[i] - tf) <= 1e-12
            assert abs(Rb[i] - rb) <= 1e-12 and abs(Tb[i] - tb) <= 1e-12
        # the defect this fixes: the incident-side value read for a ray from the glass
        assert abs(Rb[0] - Rf[0]) > 0.05

    def test_absorbing_layer_reciprocity_of_T(self):
        """T is the same both ways at Snell-corresponding angles, absorbing or not."""
        c = UnpolarizedThinFilmCoating(_cr_stack())
        theta_s = np.array([0.0, 10.0, 25.0, 40.0])
        theta_0 = np.degrees(np.arcsin(N_SUB * np.sin(np.radians(theta_s))))
        _, Tb = _eval(c, theta_s, [True] * 4)
        _, Tf = _eval(c, theta_0)
        assert np.max(np.abs(Tb - Tf)) <= 1e-12


class TestLosslessFromBothSides:
    def test_energy_conservation(self):
        c = UnpolarizedThinFilmCoating(_lossless_stack())
        for mask in ([False] * 4, [True] * 4):
            R, T = _eval(c, THETAS, mask)
            assert np.max(np.abs(R + T - 1.0)) <= 1e-12

    def test_reciprocity(self):
        """A lossless stack: R and T from the glass at theta_s equal those from air at theta_0."""
        c = UnpolarizedThinFilmCoating(_lossless_stack())
        theta_s = np.array([0.0, 10.0, 25.0, 40.0])
        theta_0 = np.degrees(np.arcsin(N_SUB * np.sin(np.radians(theta_s))))
        Rb, Tb = _eval(c, theta_s, [True] * 4)
        Rf, Tf = _eval(c, theta_0)
        assert np.max(np.abs(Rb - Rf)) <= 1e-12
        assert np.max(np.abs(Tb - Tf)) <= 1e-12

    def test_beyond_the_critical_angle_from_the_glass(self):
        """From the glass beyond asin(1/1.52) = 41.14 deg the lossless stack reflects all."""
        c = UnpolarizedThinFilmCoating(_lossless_stack())
        R, T = _eval(c, [45.0, 60.0], [True, True])
        assert np.max(np.abs(R - 1.0)) <= 1e-12 and np.max(np.abs(T)) <= 1e-12


class TestSideDecision:
    def test_plane_face_with_the_glass_behind(self):
        c = UnpolarizedThinFilmCoating(_cr_stack())
        glass = NSQMaterial(optiland_material=IdealMaterial(N_SUB))
        assert coating_incident_is_front(c, VACUUM, glass) is True
        assert coating_incident_is_front(c, glass, VACUUM) is False

    def test_explicit_side_overrides(self):
        glass = NSQMaterial(optiland_material=IdealMaterial(N_SUB))
        c = UnpolarizedThinFilmCoating(_cr_stack(), incident_side="back")
        assert coating_incident_is_front(c, VACUUM, glass) is False
        c = UnpolarizedThinFilmCoating(_cr_stack(), incident_side="front")
        assert coating_incident_is_front(c, glass, VACUUM) is True
        with pytest.raises(ValueError, match="incident_side"):
            UnpolarizedThinFilmCoating(_cr_stack(), incident_side="inside")

    def test_side_blind_coating_has_no_side(self):
        from optiland.coatings import SimpleCoating

        assert coating_incident_is_front(SimpleCoating(0.9, 0.1), VACUUM, VACUUM) is None

    def test_lens_back_face_written_from_the_air(self):
        """A lens's back face has the glass in front; a stack written air -> glass sits behind it."""
        scene = NSQScene()
        scene.add_lens(
            "L",
            CoordinateSystem(z=0.0),
            LensConfig(
                r1=1.0e9, r2=1.0e9, thickness=5.0,
                material=NSQMaterial(optiland_material=IdealMaterial(N_SUB)),
                front_aperture_radius=10.0,
                front=SurfaceConfig(coating=UnpolarizedThinFilmCoating(_cr_stack())),
                back=SurfaceConfig(coating=UnpolarizedThinFilmCoating(_cr_stack())),
            ),
        )
        faces = {s.name: s for s in scene.surfaces}
        assert faces["L.front"]._coating_incident_front() is True
        assert faces["L.back"]._coating_incident_front() is False


def _trace_from_the_glass(theta_deg: float):
    """A beam inside the glass meets the coated face (front: vacuum, back: glass)."""
    t = math.radians(theta_deg)
    scene = NSQScene()
    # the source sits in the glass half-space (z > 0) and fires towards -z, tilted by theta
    scene.add_source(
        "S",
        CoordinateSystem(y=50.0 * math.sin(t), z=50.0 * math.cos(t), rx=math.pi - t),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(WL), total_flux=1.0, aperture_radius=0.5
        ),
    )
    scene.add_component(
        "face",
        RefractiveComponent(
            CoordinateSystem(z=0.0),
            PlaneGeometry(),
            VACUUM,
            NSQMaterial(optiland_material=IdealMaterial(N_SUB)),
            coating=UnpolarizedThinFilmCoating(_cr_stack()),
            name="face",
        ),
    )
    for name, z in (("reflected", 100.0), ("transmitted", -100.0)):
        scene.add_detector(
            name,
            CoordinateSystem(z=z),
            IrradianceDetectorConfig(
                width=2000, height=2000, num_pixels_x=1, num_pixels_y=1, splat="hard"
            ),
        )
    scene.sampling_policy = SamplingPolicy(split_depth=2, split_budget=8.0, rr_start_flux=1e-16)
    result = scene.trace(num_rays=64, seed=1, max_depth=2)
    phi = result.total_flux_in
    return (
        result.detectors["reflected"].total_flux_float / phi,
        result.detectors["transmitted"].total_flux_float / phi,
    )


class TestTracedFromTheGlass:
    @pytest.mark.parametrize("theta_deg", [0.0, 30.0])
    def test_split_tree_reads_the_reversed_stack(self, theta_deg):
        refl, trans = _trace_from_the_glass(theta_deg)
        rb, tb = charmat_unpol(N_SUB, [(CR, CR_NM / 1000.0)], complex(1.0), theta_deg)
        rf, _ = charmat_unpol(1.0, [(CR, CR_NM / 1000.0)], complex(N_SUB), theta_deg)
        assert refl == pytest.approx(rb, rel=1e-12)
        assert trans == pytest.approx(tb, rel=1e-12)
        assert abs(refl - rf) > 0.05


class TestTorchLegs:
    """The mask on the torch backend at both precisions: the same selections, bit for bit."""

    @pytest.fixture(autouse=True)
    def _restore(self):
        yield
        import optiland.backend as be

        be.set_backend("numpy")
        be.set_precision("float64")

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_mask_on_torch(self, precision):
        torch = pytest.importorskip("torch")
        import optiland.backend as be

        be.set_backend("torch")
        be.set_device("cpu")
        be.grad_mode.disable()
        be.set_precision(precision)
        th = torch.deg2rad(be.array(THETAS))
        wl = be.array([WL] * len(THETAS))
        ci = torch.cos(th)
        for st in (_cr_stack(), _lossless_stack()):
            c = UnpolarizedThinFilmCoating(st)
            R0, T0 = c.evaluate(wl, ci)
            R1, T1 = c.evaluate(wl, ci, from_substrate=torch.zeros(len(THETAS), dtype=torch.bool))
            assert torch.equal(R0, R1) and torch.equal(T0, T1)
            R2, T2 = c.evaluate(wl, ci, from_substrate=torch.ones(len(THETAS), dtype=torch.bool))
            R3, T3 = UnpolarizedThinFilmCoating(_reversed(st)).evaluate(wl, ci)
            assert torch.equal(R2, R3) and torch.equal(T2, T3)


class TestStokesAdapter:
    def test_reverse_mask_is_the_reversed_stack(self):
        from optiland.nonsequential import polarization as P

        st = _cr_stack()
        th = np.radians(np.asarray(THETAS))
        wl = np.full(th.shape, WL)
        a = P.thin_film_sp(st, wl, np.cos(th), reverse=np.ones(th.shape, dtype=bool))
        b = P.thin_film_sp(_reversed(st), wl, np.cos(th))
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(np.asarray(x), np.asarray(y))
