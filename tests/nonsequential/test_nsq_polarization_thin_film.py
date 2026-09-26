"""The thin-film adapter of the polarization module (the research repository's issue 5, item 2).

``polarization.thin_film_sp`` takes the fork's thin-film module's s and p results
and returns them in the polarization module's convention (``exp(-i w t)``, the
catalogue's analytic reference). Each correction it applies is pinned by its own
test against an independent route:

* the power terms, on a bare interface, beyond the critical angle, on a metal
  and on the r1_23 quarter-wave stack, against closed forms;
* the ``r_p`` sign flip, on a bare interface against the Fresnel amplitudes;
* the conjugation, on a metal and on lossless stacks with layers, against an
  independent characteristic-matrix calculation written below (Born and Wolf's
  ``exp(-i w t)`` matrix ``[[cos d, -i sin d / eta], [-i eta sin d, cos d]]``);
* no conjugation beyond the critical angle of a bare interface, against the
  analytic reference's TIR phase;
* the one case outside both conventions (a coated face beyond its critical
  angle) is flagged, not corrected.

Tolerances: 1e-14 on a power term or a phase element at float64 (the module
itself is a product of a few 2x2 complex matrices; the closed forms agree to
1e-15 on bare interfaces). At float32, ``64 u_32`` = 3.8e-6 on an element of
magnitude at most one, a bound on a 2x2 characteristic-matrix chain of
eleven layers; the measured agreement is reported in ulps by the build log.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.nonsequential import polarization as P
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


def _configure(backend: str, precision: str) -> float:
    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
        be.grad_mode.disable()
    be.set_precision(precision)
    return 1e-14 if precision == "float64" else 64 * U32


def _n(x):
    return np.asarray(be.to_numpy(x), dtype=np.float64)


def _grazing(window: float, theta_deg: float) -> float:
    """The window near grazing incidence: at least ``1e-15 / cos theta_i``.

    The adapter's input is ``cos theta_i`` and the module recovers the angle
    with ``arccos``; the digits that loses grow as ``1 / cos theta_i`` (57 at
    89 degrees), which the float64 window of 1e-14 does not cover there.
    """
    c = math.cos(math.radians(theta_deg))
    return max(window, 1e-15 / c) if c > 0 else window


def _sp(stack, thetas_deg):
    th = np.radians(np.asarray(thetas_deg, dtype=np.float64))
    wl = be.array(np.full(th.shape, WL))
    ci = be.array(np.cos(th))
    return P.thin_film_sp(stack, wl, ci)


# ---------------------------------------------------------------------------
# Independent references, exp(-i w t), N = n + i k
# ---------------------------------------------------------------------------


def fresnel(n1: complex, n2: complex, theta_deg: float):
    """Complex (r_s, r_p) of a bare interface; the decaying root Im(n2 cos_t) >= 0."""
    ci = math.cos(math.radians(theta_deg))
    si = math.sin(math.radians(theta_deg))
    ct = cmath.sqrt(1 - (n1 * si / n2) ** 2)
    if (n2 * ct).imag < 0:
        ct = -ct
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    return rs, rp


def characteristic_matrix(n0: float, layers, ns: complex, theta_deg: float, pol: str):
    """Born and Wolf's characteristic-matrix r and t; r_p returned in the Fresnel sign."""
    s0 = n0 * math.sin(math.radians(theta_deg))

    def cos_in(N):
        c = cmath.sqrt(1 - (s0 / N) ** 2)
        if (N * c).imag < 0 or ((N * c).imag == 0 and (N * c).real < 0):
            c = -c
        return c

    def eta(N):
        c = cos_in(N)
        return N * c if pol == "s" else N / c

    M = np.eye(2, dtype=complex)
    for N, d in layers:
        dl = 2 * math.pi * N * d * cos_in(N) / WL
        e = eta(N)
        M = M @ np.array(
            [[cmath.cos(dl), -1j * cmath.sin(dl) / e], [-1j * e * cmath.sin(dl), cmath.cos(dl)]]
        )
    B, C = M @ np.array([1.0, eta(ns)])
    e0 = eta(complex(n0))
    r = (e0 * B - C) / (e0 * B + C)
    t = 2 * e0 / (e0 * B + C)
    return (r, t) if pol == "s" else (-r, t)


def qw_stack(n0, pairs, nh, nl, ns, extra=()):
    st = ThinFilmStack(IdealMaterial(n0), IdealMaterial(ns), reference_wl_um=WL)
    layers = []
    for _ in range(pairs):
        for n in (nh, nl):
            st.add_layer_qwot(IdealMaterial(n))
            layers.append((complex(n), WL / 4 / n))
    for n in extra:
        st.add_layer_qwot(IdealMaterial(n))
        layers.append((complex(n), WL / 4 / n))
    return st, layers


# ---------------------------------------------------------------------------


class TestPowerTerms:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_bare_interface(self, leg):
        window = _configure(*leg)
        thetas = [0.0, 15.0, 30.0, 45.0, 56.309932474020215, 60.0, 75.0, 89.0]
        sp = _sp(ThinFilmStack(IdealMaterial(1.0), IdealMaterial(1.5)), thetas)
        for i, th in enumerate(thetas):
            rs, rp = fresnel(1.0, 1.5, th)
            w = _grazing(window, th)
            assert abs(_n(sp.Rs)[i] - abs(rs) ** 2) <= w
            assert abs(_n(sp.Rp)[i] - abs(rp) ** 2) <= w
            assert abs(_n(sp.Ts)[i] - (1 - abs(rs) ** 2)) <= w
            assert abs(_n(sp.Tp)[i] - (1 - abs(rp) ** 2)) <= w

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_total_internal_reflection(self, leg):
        window = _configure(*leg)
        sp = _sp(ThinFilmStack(IdealMaterial(1.5), IdealMaterial(1.0)), [45.0, 50.2294, 60.0])
        assert np.max(np.abs(_n(sp.Rs) - 1.0)) <= window
        assert np.max(np.abs(_n(sp.Rp) - 1.0)) <= window

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_metal(self, leg):
        window = _configure(*leg)
        n_metal = complex(0.96, 6.69)
        thetas = [0.0, 45.0, 70.0]
        sp = _sp(ThinFilmStack(IdealMaterial(1.0), IdealMaterial(0.96, 6.69)), thetas)
        for i, th in enumerate(thetas):
            rs, rp = fresnel(1.0, n_metal, th)
            assert abs(_n(sp.Rs)[i] / abs(rs) ** 2 - 1.0) <= window
            assert abs(_n(sp.Rp)[i] / abs(rp) ** 2 - 1.0) <= window

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_r1_23_stack(self, leg):
        """(HL)^5, nH 2.32, nL 1.38 on 1.5 at 0.55 um: the closed form at 0 deg, the case value at 45."""
        window = _configure(*leg)
        st, _ = qw_stack(1.0, 5, 2.32, 1.38, 1.5)
        sp = _sp(st, [0.0, 45.0])
        y = (2.32 / 1.38) ** 10 * 1.5
        closed = ((1.0 - y) / (1.0 + y)) ** 2
        r_u = 0.5 * (_n(sp.Rs) + _n(sp.Rp))
        assert abs(r_u[0] - closed) <= window
        assert abs(r_u[1] - 0.9503515751153687) <= window


class TestPhaseCorrections:
    def test_the_module_r_p_has_the_opposite_sign(self):
        """The raw module r_p is minus the Fresnel r_p (bare 1 -> 1.5); the adapter negates it."""
        _configure("numpy", "float64")
        st = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(1.5))
        for th in (0.0, 30.0, 45.0, 75.0):
            raw = st.compute_rtRTA_elementwise(
                be.array([WL]), be.array([math.radians(th)]), polarization="p"
            )["r"]
            rs, rp = fresnel(1.0, 1.5, th)
            assert abs(complex(np.asarray(raw)[0]) + rp) < 1e-15
        sp = _sp(st, [45.0])
        assert abs(_n(sp.xr_re)[0] - -0.027911062) < 1e-9  # T-06-6's M22, negative

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_bare_interface_phase(self, leg):
        window = _configure(*leg)
        thetas = [0.0, 30.0, 45.0, 60.0, 89.0]
        sp = _sp(ThinFilmStack(IdealMaterial(1.0), IdealMaterial(1.5)), thetas)
        for i, th in enumerate(thetas):
            rs, rp = fresnel(1.0, 1.5, th)
            x = rp * np.conj(rs)
            w = _grazing(window, th)
            assert abs(_n(sp.xr_re)[i] - x.real) <= w
            assert abs(_n(sp.xr_im)[i] - x.imag) <= w
            assert abs(_n(sp.xt_cos)[i] - 1.0) <= w and abs(_n(sp.xt_sin)[i]) <= w

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_tir_is_not_conjugated(self, leg):
        """Beyond the critical angle of a bare interface the module's phase is already the reference's."""
        window = _configure(*leg)
        thetas = [45.0, 50.2294, 60.0]
        sp = _sp(ThinFilmStack(IdealMaterial(1.5), IdealMaterial(1.0)), thetas)
        for i, th in enumerate(thetas):
            rs, rp = fresnel(1.5, 1.0, th)
            x = rp * np.conj(rs)
            assert abs(_n(sp.xr_re)[i] - x.real) <= window
            assert abs(_n(sp.xr_im)[i] - x.imag) <= window
        assert abs(math.degrees(math.atan2(_n(sp.xr_im)[0], _n(sp.xr_re)[0])) + 36.8699) < 1e-4
        assert bool(np.all(_n(sp.phase_valid)))

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_metal_is_conjugated(self, leg):
        window = _configure(*leg)
        n_metal = complex(0.96, 6.69)
        thetas = [0.0, 45.0, 70.0]
        sp = _sp(ThinFilmStack(IdealMaterial(1.0), IdealMaterial(0.96, 6.69)), thetas)
        for i, th in enumerate(thetas):
            rs, rp = fresnel(1.0, n_metal, th)
            x = rp * np.conj(rs)
            assert abs(_n(sp.xr_re)[i] - x.real) <= window
            assert abs(_n(sp.xr_im)[i] - x.imag) <= window
        # the closed form's relative phase at 45 degrees, -168.23 degrees
        assert abs(math.degrees(math.atan2(_n(sp.xr_im)[1], _n(sp.xr_re)[1])) + 168.23) < 5e-3

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    @pytest.mark.parametrize(
        "name,pairs,extra", [("one quarter-wave layer", 0, (1.38,)), ("r1_23 (HL)^5", 5, ())]
    )
    def test_lossless_stack_is_conjugated(self, leg, name, pairs, extra):
        """A lossless stack with layers: reflection conjugated, transmission phase as it is.

        This extends the research card of 2026-09-26, which found the
        conjugation on an absorbing substrate only: the module's layer
        matrices are in the exp(+i w t) convention whether or not anything
        absorbs.
        """
        window = _configure(*leg)
        st, layers = qw_stack(1.0, pairs, 2.32, 1.38, 1.5, extra=extra)
        thetas = [0.0, 30.0, 45.0, 60.0]
        sp = _sp(st, thetas)
        for i, th in enumerate(thetas):
            rs, ts = characteristic_matrix(1.0, layers, complex(1.5), th, "s")
            rp, tp = characteristic_matrix(1.0, layers, complex(1.5), th, "p")
            x = rp * np.conj(rs)
            assert abs(_n(sp.xr_re)[i] - x.real) <= window
            assert abs(_n(sp.xr_im)[i] - x.imag) <= window
            xt = tp * np.conj(ts)
            assert abs(_n(sp.xt_cos)[i] - xt.real / abs(xt)) <= window
            assert abs(_n(sp.xt_sin)[i] - xt.imag / abs(xt)) <= window

    def test_coated_face_beyond_its_critical_angle_is_flagged(self):
        """The module's phase there matches neither convention; the adapter says so."""
        _configure("numpy", "float64")
        st, layers = qw_stack(1.5, 0, 2.32, 1.38, 1.0, extra=(1.38,))
        thetas = [30.0, 60.0]
        sp = _sp(st, thetas)
        assert list(_n(sp.phase_valid)) == [1.0, 0.0]
        rs, _ = characteristic_matrix(1.5, layers, complex(1.0), 60.0, "s")
        rp, _ = characteristic_matrix(1.5, layers, complex(1.0), 60.0, "p")
        ref = math.degrees(cmath.phase(rp * np.conj(rs)))
        got = math.degrees(math.atan2(_n(sp.xr_im)[1], _n(sp.xr_re)[1]))
        assert abs(abs(got) - abs(ref)) > 1.0  # not a conjugate, not equal
        # the power terms stay right: |r| = 1
        assert abs(_n(sp.Rs)[1] - 1.0) < 1e-14 and abs(_n(sp.Rp)[1] - 1.0) < 1e-14
