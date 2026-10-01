"""One time convention for the thin-film module (the research repository's issues 78 and 74).

The module now states ``exp(-i omega t)``: ``N = n + i k``, layer matrices
``[[cos d, -i sin d / eta], [-i eta sin d, cos d]]``, the root of ``N cos theta``
with a non-negative imaginary part. Each test has an independent route, the
characteristic matrix and the Fresnel amplitudes written here in the same
convention (Born and Wolf, Principles of Optics, section 1.6.2; chapter 04 sections
4.2 and 4.4 of the research repository):

* an empty stack equals the bare interface to the last place, reflection and
  transmission, below and beyond the critical angle, and ``JonesThinFilm`` of an
  empty stack equals ``JonesFresnel`` there too;
* a coated face beyond its critical angle (frustrated total internal reflection)
  has the characteristic matrix's relative phase; before the fix it matched neither
  convention (-34.00 against -29.22 degrees, issue 78);
* an absorbing film and a lossless stack carry the reference's phases directly,
  without the adapter's former conjugation;
* the powers are the module's as before (bit for bit on lossless stacks, where
  every intermediate is the exact conjugate of the former one).

Tolerances: 1e-14 at float64 on amplitudes and phase elements of magnitude at most
two (a product of at most ten 2x2 complex matrices and a quotient); float32 legs are
compared at 64 u_32, the bound the existing adapter tests use for an eleven-layer
chain.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import JonesThinFilm
from optiland.jones import JonesFresnel
from optiland.materials import IdealMaterial
from optiland.nonsequential import polarization as P
from optiland.rays import RealRays
from optiland.thin_film import ThinFilmStack

U32 = 2.0**-24
WL = 0.55


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


LEGS = [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")]
LEG_IDS = [f"{b}-{p}" for b, p in LEGS]


def _configure(backend, precision):
    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
        be.grad_mode.disable()
    be.set_precision(precision)
    return 1e-14 if precision == "float64" else 64 * U32


def _c(x):
    x = x.resolve_conj() if hasattr(x, "resolve_conj") else x
    return np.asarray(be.to_numpy(x))


def fresnel(n1, n2, theta_deg):
    """(r_s, r_p, t_s, t_p) Fresnel field amplitudes, exp(-i w t): Im(n2 cos_t) >= 0."""
    ci, si = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    nct = cmath.sqrt(complex(n2) ** 2 - (n1 * si) ** 2)
    if nct.imag < 0 or (nct.imag == 0 and nct.real < 0):
        nct = -nct
    ct = nct / n2
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    ts = 2 * n1 * ci / (n1 * ci + n2 * ct)
    tp = 2 * n1 * ci / (n2 * ci + n1 * ct)
    return rs, rp, ts, tp


def charmat(n0, layers, ns, theta_deg, pol):
    """Admittance-form (r, t) of a stack; r_p returned in the Fresnel sign."""
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
        dl = 2 * math.pi * d * ncos(N) / WL
        e = eta(N)
        M = M @ np.array(
            [[cmath.cos(dl), -1j * cmath.sin(dl) / e], [-1j * e * cmath.sin(dl), cmath.cos(dl)]]
        )
    B, C = M @ np.array([1.0, eta(ns)])
    e0 = eta(complex(n0))
    r = (e0 * B - C) / (e0 * B + C)
    t = 2 * e0 / (e0 * B + C)
    return (r if pol == "s" else -r), t


def _module(stack, thetas_deg, pol):
    th = be.array(np.radians(np.asarray(thetas_deg, dtype=float)))
    wl = be.array(np.full(len(thetas_deg), WL))
    out = stack.compute_rtRTA_elementwise(wl, th, pol)
    return _c(out["r"]), _c(out["t"])


def _qw(n0, ns, indices):
    st = ThinFilmStack(IdealMaterial(n0), IdealMaterial(ns), reference_wl_um=WL)
    layers = []
    for n in indices:
        st.add_layer_qwot(IdealMaterial(n))
        layers.append((complex(n), WL / 4 / n))
    return st, layers


class TestEmptyStackIsTheBareInterface:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    @pytest.mark.parametrize("media", [(1.0, 1.5), (1.5, 1.0)], ids=["external", "internal"])
    def test_reflection_and_transmission(self, leg, media):
        """r_s, r_p (Fresnel sign), t_s, and t_p as a field amplitude, at 0 to 85 degrees.

        1.5 -> 1.0 crosses the critical angle (41.81 degrees): beyond it r has
        unit modulus and the reference phase, and t the evanescent amplitude.
        """
        window = _configure(*leg)
        n1, n2 = media
        thetas = [0.0, 20.0, 40.0, 45.0, 60.0, 85.0]
        st = ThinFilmStack(IdealMaterial(n1), IdealMaterial(n2))
        rs, ts = _module(st, thetas, "s")
        rp, tp = _module(st, thetas, "p")
        for i, th in enumerate(thetas):
            frs, frp, fts, ftp = fresnel(n1, n2, th)
            ci = math.cos(math.radians(th))
            ct = cmath.sqrt(1 - (n1 * math.sin(math.radians(th)) / n2) ** 2)
            if ct.imag < 0:
                ct = -ct
            w = window * 2
            assert abs(rs[i] - frs) <= w
            assert abs(-rp[i] - frp) <= w
            assert abs(ts[i] - fts) <= w
            # the module's t_p is tangential: field amplitude = t_p cos_i / cos_t
            assert abs(tp[i] * ci / ct - ftp) <= w * max(1.0, abs(ci / ct))

    def test_tir_phase_is_the_analytic_reference(self):
        _configure("numpy", "float64")
        st = ThinFilmStack(IdealMaterial(1.5), IdealMaterial(1.0))
        rs, _ = _module(st, [45.0], "s")
        rp, _ = _module(st, [45.0], "p")
        rel = math.degrees(cmath.phase(-rp[0] * np.conj(rs[0])))
        assert abs(rel + 36.86989764584402) < 1e-12
        assert abs(math.degrees(cmath.phase(rs[0])) + 36.86989764584402) < 1e-12

    def test_jones_thin_film_equals_jones_fresnel_beyond_the_critical_angle(self):
        """Reflection and transmission, including the evanescent amplitude (issue 74)."""
        _configure("numpy", "float64")
        thetas = [0.0, 30.0, 45.0, 60.0, 80.0]
        n = len(thetas)
        z = [0.0] * n
        rays = RealRays(x=z, y=z, z=z, L=z, M=z, N=[1.0] * n, intensity=[1.0] * n,
                        wavelength=[WL] * n)
        aoi = np.radians(thetas)
        st = ThinFilmStack(IdealMaterial(1.5), IdealMaterial(1.0))
        for reflect in (True, False):
            a = JonesThinFilm(st).calculate_matrix(rays, reflect, aoi)
            b = JonesFresnel(IdealMaterial(1.5), IdealMaterial(1.0)).calculate_matrix(
                rays, reflect, aoi)
            assert np.max(np.abs(a - b)) <= 10 * 2.0**-53 * 4


class TestStacksCarryTheReferencePhase:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_frustrated_total_internal_reflection(self, leg):
        """A quarter wave of 1.38 between 1.5 and 1.0, below and beyond 41.81 degrees.

        At 60 degrees the reference relative phase is -29.22 degrees; the module
        gave -34.00 before (issue 78).
        """
        window = _configure(*leg)
        st, layers = _qw(1.5, 1.0, [1.38])
        thetas = [30.0, 45.0, 60.0, 75.0]
        th = be.array(np.radians(thetas))
        sp = P.thin_film_sp(st, be.array(np.full(4, WL)), be.cos(th))
        for i, a in enumerate(thetas):
            rs, _ = charmat(1.5, layers, 1.0, a, "s")
            rp, _ = charmat(1.5, layers, 1.0, a, "p")
            x = rp * np.conj(rs)
            assert abs(float(_c(sp.xr_re)[i]) - x.real) <= window
            assert abs(float(_c(sp.xr_im)[i]) - x.imag) <= window
        assert bool(np.all(_c(sp.phase_valid)))
        _configure("numpy", "float64")
        rs, _ = charmat(1.5, layers, 1.0, 60.0, "s")
        rp, _ = charmat(1.5, layers, 1.0, 60.0, "p")
        assert abs(math.degrees(cmath.phase(rp * np.conj(rs))) + 29.22) < 5e-3

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_absorbing_film_and_lossless_stack(self, leg):
        window = _configure(*leg)
        cases = []
        st = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(1.52))
        st.add_layer_nm(IdealMaterial(3.13, 4.33), 10.0)
        cases.append((st, [(complex(3.13, 4.33), 0.010)], 1.0, 1.52))
        st2, layers2 = _qw(1.0, 1.5, [2.32, 1.38, 2.32, 1.38])
        cases.append((st2, layers2, 1.0, 1.5))
        thetas = [0.0, 30.0, 45.0, 70.0]
        for stack, layers, n0, ns in cases:
            rs_m, ts_m = _module(stack, thetas, "s")
            rp_m, tp_m = _module(stack, thetas, "p")
            for i, a in enumerate(thetas):
                rs, ts = charmat(n0, layers, ns, a, "s")
                rp, tp = charmat(n0, layers, ns, a, "p")
                assert abs(rs_m[i] - rs) <= window * 4
                assert abs(-rp_m[i] - rp) <= window * 4
                assert abs(ts_m[i] - ts) <= window * 4
                assert abs(tp_m[i] - tp) <= window * 4

    def test_the_metal_mirror_relative_phase(self):
        """The bare metal's arg(r_p r_s*) at 45 degrees, -168.23 degrees, read directly."""
        _configure("numpy", "float64")
        st = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(0.96, 6.69))
        rs, _ = _module(st, [45.0], "s")
        rp, _ = _module(st, [45.0], "p")
        assert abs(math.degrees(cmath.phase(-rp[0] * np.conj(rs[0]))) + 168.23) < 5e-3
