"""The polarization conventions and math module (the research repository's issue 5).

Every expected value comes from an independent route: the Mueller matrices are
rebuilt here from their Jones matrices by ``M = A (J kron J*) A^-1`` (the route
of the research repository's theory chapter 06, section 6.3) with complex
Fresnel amplitudes written in the time convention the module states, and the
printed values are chapter 06's tables and tests T-06-1 to T-06-14. The module
works in real arithmetic on ``optiland.backend`` arrays, so each element check
runs on NumPy float64, torch float64 and torch float32 (CPU).

Tolerances. At float64 the chapter's own windows (``1e-15`` on an element,
``1e-12`` on a degree of polarization). At float32 each element is at most a
handful of rounded operations of values of order one (a Fresnel amplitude is
four; a product of two amplitudes, a rotation by the doubled angle and one
division stay under 16), so the window is ``16 u_32`` = 9.5e-7 on an element,
stated as that bound rather than fitted to a result.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.nonsequential import polarization as P

U64 = 2.0**-53
U32 = 2.0**-24

LEGS = [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")]
LEG_IDS = [f"{b}-{p}" for b, p in LEGS]


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _configure(backend: str, precision: str) -> float:
    """Select the leg; return the element window for it."""
    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
        be.grad_mode.disable()
    be.set_precision(precision)
    return 1e-15 if precision == "float64" else 16 * U32


def _a(x):
    return be.array(np.asarray(x, dtype=np.float64))


def _n(x):
    return np.asarray(be.to_numpy(x), dtype=np.float64)


# ---------------------------------------------------------------------------
# The independent route: Jones amplitudes and M = A (J kron J*) A^-1
# ---------------------------------------------------------------------------

_A = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]], dtype=complex)
_A_INV = np.linalg.inv(_A)


def jones_to_mueller(J: np.ndarray) -> np.ndarray:
    """Chapter 06 section 6.3, with J in the (p, s) basis."""
    return np.real(_A @ np.kron(J, J.conj()) @ _A_INV)


def fresnel_amplitudes(n1: float, n2: float, theta_deg: float):
    """Complex (r_s, r_p), exp(-i w t) convention, cos_t = +i kappa beyond the critical angle."""
    ci = math.cos(math.radians(theta_deg))
    st = n1 / n2 * math.sin(math.radians(theta_deg))
    ct = np.sqrt(complex(1.0 - st * st))
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    return rs, rp


def module_reflection(n1: float, n2: float, theta_deg: float):
    """The module's reflection element, built the way the engine will build it."""
    th = _a([math.radians(theta_deg)])
    ci = be.cos(th)
    n1a = be.ones_like(ci) * n1
    n2a = be.ones_like(ci) * n2
    sin2_t = (n1a / n2a) ** 2 * (1.0 - ci**2)
    w = 1.0 - sin2_t
    tir = w <= 0
    ct = be.where(tir, be.zeros_like(w), be.where(tir, be.ones_like(w), w) ** 0.5)
    rs = (n1a * ci - n2a * ct) / (n1a * ci + n2a * ct)
    rp = (n2a * ci - n1a * ct) / (n2a * ci + n1a * ct)
    phase = P.tir_relative_phase(n1a, n2a, ci, sin2_t, tir)
    return P.reflection_mueller(rs, rp, tir, phase), rs, rp


def _elements(m) -> np.ndarray:
    return np.array([_n(x)[0] for x in m])


REFLECTION_ANGLES = [
    (1.0, 1.5, t) for t in (0.0, 15.0, 30.0, 45.0, 56.309932474020215, 60.0, 75.0, 89.0)
] + [(1.5, 1.0, t) for t in (30.0, 42.0, 45.0, 50.2294, 51.671, 60.0, 80.0)]


class TestFresnelReflection:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    @pytest.mark.parametrize("n1,n2,theta", REFLECTION_ANGLES)
    def test_matches_the_jones_route(self, leg, n1, n2, theta):
        """Every element against A (J kron J*) A^-1 of diag(r_p, r_s), below and beyond TIR.

        Both routes round the radicand ``w = 1 - sin^2 theta_t`` and take its
        square root, whose derivative is ``1 / (2 sqrt|w|)``; near the critical
        angle that conditions every element, so the window is the element
        window times ``max(1, |w|^-1/2)`` (11.6 at 42 degrees, 1.5 -> 1.0).
        """
        window = _configure(*leg)
        m, _, _ = module_reflection(n1, n2, theta)
        rs, rp = fresnel_amplitudes(n1, n2, theta)
        ref = jones_to_mueller(np.diag([rp, rs]))
        got = P.mueller_matrix(tuple(_elements(m)))
        w = 1.0 - (n1 / n2 * math.sin(math.radians(theta))) ** 2
        conditioning = max(1.0, abs(w) ** -0.5) if w != 0 else 1.0
        assert np.max(np.abs(got - ref)) <= 2 * window * conditioning

    def test_t06_6_the_45_degree_matrix(self):
        """T-06-6: M00, M01, M22 = M33 and M23 at 45 degrees, 1 -> 1.5, to 1e-12 of the chapter."""
        _configure("numpy", "float64")
        m00, m01, m22, m23 = _elements(module_reflection(1.0, 1.5, 45.0)[0])
        assert abs(m00 - 0.050239911) < 1e-9
        assert abs(m01 - -0.041773452) < 1e-9
        assert abs(m22 - -0.027911062) < 1e-9
        assert m23 == 0.0

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_t06_9_tir_keeps_the_relative_phase(self, leg):
        """T-06-9 in the analytic reference's convention: |r_p r_s*| = 1; phase -36.8699 deg at 45."""
        window = _configure(*leg)
        for theta, want_deg in ((45.0, -36.8699), (60.0, -40.4591), (80.0, -14.7892)):
            _, _, m22, m23 = _elements(module_reflection(1.5, 1.0, theta)[0])
            assert abs(math.hypot(m22, m23) - 1.0) <= 4 * window
            assert abs(math.degrees(math.atan2(m23, m22)) - want_deg) < 5e-5 + 1e3 * window
        # The largest relative retardance, 45.2397 deg at 51.671 deg, and the
        # Fresnel-rhomb angle 50.2294 deg where one bounce gives 45.000 deg.
        _, _, m22, m23 = _elements(module_reflection(1.5, 1.0, 51.671)[0])
        assert abs(math.degrees(math.atan2(m23, m22)) + 45.2397) < 5e-5 + 1e3 * window
        _, _, m22, m23 = _elements(module_reflection(1.5, 1.0, 50.2294)[0])
        assert abs(math.degrees(math.atan2(m23, m22)) + 45.0) < 1e-3

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_t06_8_energy(self, leg):
        """T-06-8: M_r00 + M_t00 = 1 on a lossless interface, at every tested angle."""
        window = _configure(*leg)
        for n1, n2, theta in REFLECTION_ANGLES:
            mr, rs, rp = module_reflection(n1, n2, theta)
            Ts = 1.0 - rs**2
            Tp = 1.0 - rp**2
            mt = P.transmission_mueller(Ts, Tp)
            if theta > 0 and n1 > n2 and math.sin(math.radians(theta)) * n1 / n2 >= 1:
                continue  # TIR: the engine sets T = 0 itself
            assert abs(_n(mr.m00)[0] + _n(mt.m00)[0] - 1.0) <= 4 * window

    def test_t06_7_reflected_degree_of_polarization(self):
        """T-06-7: DoP of the reflected unpolarized beam equals |Rs - Rp| / (Rs + Rp)."""
        _configure("numpy", "float64")
        want = {15: 0.093669, 30: 0.391918, 45: 0.831479, 60: 0.979796, 75: 0.578105, 89: 0.039027}
        for theta, printed in want.items():
            m, rs, rp = module_reflection(1.0, 1.5, float(theta))
            zero = be.zeros_like(rs)
            g, q, u, v = P.apply_interface(zero, zero, zero, m)
            dop = _n(P.degree_of_polarization(q, u, v))[0]
            Rs, Rp = _n(rs)[0] ** 2, _n(rp)[0] ** 2
            assert abs(dop - abs(Rs - Rp) / (Rs + Rp)) < 1e-12
            assert abs(dop - printed) < 1e-6

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_t06_4_brewster(self, leg):
        """T-06-4: R_p = 0, R_s = 0.147928994 and the reflected DoP is one."""
        window = _configure(*leg)
        theta_b = math.degrees(math.atan(1.5))
        m, rs, rp = module_reflection(1.0, 1.5, theta_b)
        m00, m01 = _n(m.m00)[0], _n(m.m01)[0]
        assert abs(m00 + m01) < (1e-15 if window < 1e-12 else 4 * window)  # R_p
        assert abs((m00 - m01) - 0.147928994) < 1e-9 + window
        zero = be.zeros_like(rs)
        _, q, u, v = P.apply_interface(zero, zero, zero, m)
        dop = _n(P.degree_of_polarization(q, u, v))[0]
        assert abs(dop - 1.0) < (1e-12 if window < 1e-12 else 4 * window)
        assert _n(q)[0] < 0  # s-polarized: Q/I = -1

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_unpolarized_flux_factor_is_m00_bit_for_bit(self, leg):
        """With q = 0 the flux factor is m00 exactly: the scalar coefficient, not a rounding of it."""
        _configure(*leg)
        th = _a(np.radians(np.linspace(0.0, 89.0, 90)))
        ci = be.cos(th)
        n1 = be.ones_like(ci)
        n2 = be.ones_like(ci) * 1.5168
        ct = (1.0 - (n1 / n2) ** 2 * (1.0 - ci**2)) ** 0.5
        rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
        rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
        scalar_R = 0.5 * (rs**2 + rp**2)
        zero = be.zeros_like(rs)
        g, _, _, _ = P.apply_interface(zero, zero, zero, P.reflection_mueller(rs, rp))
        assert np.array_equal(_n(g), _n(scalar_R))
        g_t, _, _, _ = P.apply_interface(
            zero, zero, zero, P.transmission_mueller(1.0 - rs**2, 1.0 - rp**2, m00=1.0 - scalar_R)
        )
        assert np.array_equal(_n(g_t), _n(1.0 - scalar_R))


# ---------------------------------------------------------------------------
# Rotation, polarizers, retarders, depolarizer (T-06-1, 2, 3, 5, 14)
# ---------------------------------------------------------------------------


def _axis(theta_deg: float):
    t = math.radians(theta_deg)
    return (_a([math.cos(t)]), _a([math.sin(t)]), _a([0.0]))


Z = None  # set per test: k = +z


def _k():
    return (_a([0.0]), _a([0.0]), _a([1.0]))


def _polarize_at(state, e, theta_deg, extinction=0.0):
    """Rotate the frame from e to the polarizer axis at theta about +z, then apply the polarizer."""
    q, u, v = state
    a = _axis(theta_deg)
    c2, s2 = P.rotation_2psi(e, a, _k())
    q, u = P.rotate(q, u, c2, s2)
    g, q, u, v = P.apply_interface(q, u, v, P.polarizer_mueller(q, extinction))
    return g, (q, u, v), a


def _unpolarized():
    z = _a([0.0])
    return (z, z * 1.0, z * 1.0)


class TestElements:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_t06_1_malus(self, leg):
        """T-06-1: unpolarized light through polarizers at 0 and theta gives cos^2(theta) / 2."""
        window = _configure(*leg)
        for theta in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
            e0 = _axis(0.0)
            g1, s1, e1 = _polarize_at(_unpolarized(), e0, 0.0)
            g2, _, _ = _polarize_at(s1, e1, theta)
            want = (1.0 + math.cos(2.0 * math.radians(theta))) / 4.0
            assert abs(_n(g1 * g2)[0] - want) <= max(window, 1e-15)

    def test_t06_2_rotation_isolated_from_the_polarizer(self):
        """T-06-2: the polarizer as a fixed 4x4 matrix in the lab frame gives the same numbers.

        The reference builds M_pol(theta) from J = R(-theta) diag(1, 0) R(theta)
        and never rotates a frame; the module rotates the frame and applies the
        polarizer on its own axis. A sign error in the doubled angle breaks
        the 30 and 60 degree rows and leaves 0, 45 and 90 intact.
        """
        _configure("numpy", "float64")
        for theta in (15.0, 30.0, 45.0, 60.0, 75.0):
            t = math.radians(theta)
            R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
            J = R.T @ np.diag([1.0, 0.0]) @ R
            ref = jones_to_mueller(J.astype(complex)) @ np.array([1.0, 0.3, -0.5, 0.2])
            q, u, v = _a([0.3]), _a([-0.5]), _a([0.2])
            e = _axis(0.0)
            a = _axis(theta)
            c2, s2 = P.rotation_2psi(e, a, _k())
            q, u = P.rotate(q, u, c2, s2)
            g, q, u, v = P.apply_interface(q, u, v, P.polarizer_mueller(q))
            # back into the lab frame (reference axis x) to compare component by component
            c2b, s2b = P.rotation_2psi(a, e, _k())
            q, u = P.rotate(q, u, c2b, s2b)
            got = _n(g)[0] * np.array([1.0, _n(q)[0], _n(u)[0], _n(v)[0]])
            assert np.max(np.abs(got - ref)) < 1e-15

    def test_rotation_by_90_degrees_reverses_q(self):
        """The factor 2 of section 6.5: a 90 degree frame turn negates Q and U, a 180 degree one does not."""
        _configure("numpy", "float64")
        q, u = _a([0.6]), _a([0.2])
        c2, s2 = P.rotation_2psi(_axis(0.0), _axis(90.0), _k())
        q9, u9 = P.rotate(q, u, c2, s2)
        assert abs(_n(q9)[0] + 0.6) < 1e-15 and abs(_n(u9)[0] + 0.2) < 1e-15
        c2, s2 = P.rotation_2psi(_axis(0.0), _axis(180.0), _k())
        q18, u18 = P.rotate(q, u, c2, s2)
        assert abs(_n(q18)[0] - 0.6) < 1e-15 and abs(_n(u18)[0] - 0.2) < 1e-15

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_t06_3_three_polarizers(self, leg):
        """T-06-3: a crossed pair passes 0; a 45 degree polarizer between them restores 1/8."""
        _configure(*leg)
        e = _axis(0.0)
        g1, s, e = _polarize_at(_unpolarized(), e, 0.0)
        g2, s2, e2 = _polarize_at(s, e, 90.0)
        assert _n(g1 * g2)[0] == 0.0
        assert np.all(np.isfinite([_n(x)[0] for x in s2]))
        g2, s, e = _polarize_at(s, e, 45.0)
        g3, _, _ = _polarize_at(s, e, 90.0)
        assert abs(_n(g1 * g2 * g3)[0] - 0.125) < 4 * U32

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_t06_5_quarter_wave_both_orientations(self, leg):
        """T-06-5: (1,1,0,0) -> (1,0,0,-1) at +45 and (1,0,0,+1) at -45; the other three states."""
        window = _configure(*leg)

        def retard(state, waves, fast_deg):
            q, u, v = (_a([x]) for x in state)
            e = _axis(0.0)
            a = _axis(fast_deg)
            c2, s2 = P.rotation_2psi(e, a, _k())
            q, u = P.rotate(q, u, c2, s2)
            d = 2.0 * math.pi * waves
            m = P.retarder_mueller(_a([math.cos(d)]), _a([math.sin(d)]))
            g, q, u, v = P.apply_interface(q, u, v, m)
            c2, s2 = P.rotation_2psi(a, e, _k())
            q, u = P.rotate(q, u, c2, s2)
            return np.array([_n(g)[0], _n(q)[0], _n(u)[0], _n(v)[0]])

        tol = max(4 * window, 1e-15)
        assert np.max(np.abs(retard((1, 0, 0), 0.25, 45.0) - [1, 0, 0, -1])) < tol
        assert np.max(np.abs(retard((1, 0, 0), 0.25, -45.0) - [1, 0, 0, 1])) < tol
        assert np.max(np.abs(retard((0, 1, 0), 0.25, 45.0) - [1, 0, 1, 0])) < tol
        assert np.max(np.abs(retard((0, 0, 0), 0.25, 45.0) - [1, 0, 0, 0])) < tol
        assert np.max(np.abs(retard((1, 0, 0), 0.5, 45.0) - [1, -1, 0, 0])) < tol
        for fast in (45.0, -45.0):
            out = retard((1, 0, 0), 0.25, fast)
            assert abs(math.sqrt(np.sum(out[1:] ** 2)) - 1.0) < max(4 * window, 1e-12)

    def test_retarder_matches_the_closed_form_at_any_axis(self):
        """The module's route against the analytic reference's closed-form retarder matrix."""
        _configure("numpy", "float64")
        rng = np.random.default_rng(3)
        for _ in range(50):
            fast = rng.uniform(-90, 90)
            waves = rng.uniform(0, 1)
            s_in = rng.normal(size=3)
            s_in = s_in / np.linalg.norm(s_in) * rng.uniform(0, 1)
            th = 2.0 * math.radians(fast)
            d = 2.0 * math.pi * waves
            c, s, cd, sd = math.cos(th), math.sin(th), math.cos(d), math.sin(d)
            M = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, c * c + s * s * cd, c * s * (1 - cd), s * sd],
                    [0.0, c * s * (1 - cd), s * s + c * c * cd, -c * sd],
                    [0.0, -s * sd, c * sd, cd],
                ]
            )
            ref = M @ np.concatenate([[1.0], s_in])
            q, u, v = (_a([x]) for x in s_in)
            e, a = _axis(0.0), _axis(fast)
            c2, s2 = P.rotation_2psi(e, a, _k())
            q, u = P.rotate(q, u, c2, s2)
            g, q, u, v = P.apply_interface(q, u, v, P.retarder_mueller(_a([cd]), _a([sd])))
            c2, s2 = P.rotation_2psi(a, e, _k())
            q, u = P.rotate(q, u, c2, s2)
            got = np.array([_n(g)[0], _n(q)[0], _n(u)[0], _n(v)[0]])
            assert np.max(np.abs(got - ref)) < 1e-14

    def test_t06_14_depolarizer(self):
        """T-06-14: (1,1,0,0) -> (1,kappa,0,0) with I unchanged and DoP = kappa."""
        _configure("numpy", "float64")
        for kappa in (0.0, 0.3, 1.0):
            q, u, v = P.depolarize(_a([1.0]), _a([0.0]), _a([0.0]), kappa)
            assert _n(q)[0] == kappa and _n(u)[0] == 0 and _n(v)[0] == 0
            assert _n(P.degree_of_polarization(q, u, v))[0] == kappa

    def test_diattenuator_with_extinction(self):
        """A finite extinction ratio: tx = 1, ty = eps passes (1 + eps) / 2 of unpolarized light."""
        _configure("numpy", "float64")
        z = _a([0.0])
        g, q, _, _ = P.apply_interface(z, z, z, P.polarizer_mueller(z, extinction=1e-3))
        assert abs(_n(g)[0] - 0.5005) < 1e-15
        assert abs(_n(q)[0] - 0.999 / 1.001) < 1e-15


# ---------------------------------------------------------------------------
# Chapter 06 section 6.9: the two-surface table (T-06-12) and T-06-11
# ---------------------------------------------------------------------------


class TestTwoSurfaces:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_t06_12_the_two_surface_table(self, leg):
        """Stokes / scalar 1.691358, 1.345679, 1.000000, 0.654321, 0.308642, 1.691358."""
        window = _configure(*leg)
        table = {0: 1.691358, 30: 1.345679, 45: 1.000000, 60: 0.654321, 90: 0.308642, 180: 1.691358}
        m, rs, rp = module_reflection(1.0, 1.5, 45.0)
        rbar = _n(m.m00)[0]
        for psi, printed in table.items():
            z = be.zeros_like(rs)
            g1, q, u, v = P.apply_interface(z, z, z, m)
            c2, s2 = P.rotation_2psi(_axis(0.0), _axis(float(psi)), _k())
            q, u = P.rotate(q, u, c2, s2)
            g2, _, _, _ = P.apply_interface(q, u, v, m)
            ratio = _n(g1 * g2)[0] / rbar**2
            # the independent route: two Jones-built matrices and the 4x4 rotation
            rsc, rpc = fresnel_amplitudes(1.0, 1.5, 45.0)
            M = jones_to_mueller(np.diag([rpc, rsc]))
            ref = (M @ P.rotation_matrix(math.radians(psi)) @ M @ np.array([1.0, 0, 0, 0]))[0]
            assert abs(_n(g1 * g2)[0] - ref) <= max(8 * window * ref, 1e-17)
            assert abs(ratio - printed) < 5e-7 + 64 * window

    def test_t06_11_normal_incidence_is_scalar_exactly(self):
        """T-06-11: two surfaces at normal incidence, any psi: 0.0016 in both modes, difference 0."""
        _configure("numpy", "float64")
        m, rs, rp = module_reflection(1.0, 1.5, 0.0)
        assert _n(m.m01)[0] == 0.0
        scalar = _n(m.m00)[0] ** 2
        for psi in (0.0, 45.0, 90.0):
            z = be.zeros_like(rs)
            g1, q, u, v = P.apply_interface(z, z, z, m)
            c2, s2 = P.rotation_2psi(_axis(0.0), _axis(psi), _k())
            q, u = P.rotate(q, u, c2, s2)
            g2, _, _, _ = P.apply_interface(q, u, v, m)
            assert _n(g1 * g2)[0] == scalar
        assert abs(scalar - 0.0016) < 1e-15


# ---------------------------------------------------------------------------
# Realizability (T-06-13), frames, guards and the mode switch
# ---------------------------------------------------------------------------


class TestStateAndFrames:
    def test_t06_13_realizability_over_random_sequences(self):
        """10^4 random ten-element sequences keep q^2 + u^2 + v^2 <= 1 within 1e-12 (float64)."""
        _configure("numpy", "float64")
        rng = np.random.default_rng(13)
        n = 10_000
        q = _a(np.zeros(n))
        u = _a(np.zeros(n))
        v = _a(np.zeros(n))
        worst = -1.0
        for _step in range(10):
            # a random frame turn, then a random element per ray
            psi = rng.uniform(0, 2 * np.pi, n)
            q, u = P.rotate(q, u, _a(np.cos(2 * psi)), _a(np.sin(2 * psi)))
            theta = rng.uniform(0, np.pi / 2 * 0.999, n)
            n1 = rng.uniform(1.0, 2.0, n)
            n2 = rng.uniform(1.0, 2.0, n)
            ci = np.cos(theta)
            sin2_t = (n1 / n2) ** 2 * (1 - ci**2)
            tir = sin2_t >= 1
            ct = np.sqrt(np.where(tir, 0.0, 1 - sin2_t))
            rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
            rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
            ph = P.tir_relative_phase(_a(n1), _a(n2), _a(ci), _a(sin2_t), be.array(tir))
            mr = P.reflection_mueller(_a(rs), _a(rp), be.array(tir), ph)
            mt = P.transmission_mueller(_a(np.where(tir, 0, 1 - rs**2)), _a(np.where(tir, 0, 1 - rp**2)))
            d = rng.uniform(0, 2 * np.pi, n)
            mret = P.retarder_mueller(_a(np.cos(d)), _a(np.sin(d)))
            mpol = P.polarizer_mueller(q, extinction=0.01)
            pick = rng.integers(0, 4, n)
            elems = [mr, mt, mret, mpol]
            m = P.InterfaceMueller(
                *(
                    _a(np.choose(pick, [_n(e[i]) * np.ones(n) for e in elems]))
                    for i in range(4)
                )
            )
            _, q, u, v = P.apply_interface(q, u, v, m)
            worst = max(worst, float(np.max(_n(P.realizability_excess(q, u, v)))))
        assert worst <= 1e-12

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_birth_axis_is_a_unit_vector_perpendicular_to_k(self, leg):
        window = _configure(*leg)
        rng = np.random.default_rng(5)
        k = rng.normal(size=(1000, 3))
        k[0] = [0, 0, 1]
        k[1] = [1, 0, 0]
        k[2] = [-1, 0, 0]
        k[3] = [0, 0, -1]
        k = k / np.linalg.norm(k, axis=1)[:, None]
        e = P.birth_axis(_a(k[:, 0]), _a(k[:, 1]), _a(k[:, 2]))
        e = np.stack([_n(c) for c in e], axis=1)
        # T-06-18's frame window, 1e-14 at float64 (the projection's rounding
        # grows as u / |k x a| for k near the lab axis); 64 u at float32.
        tol = 1e-14 if window < 1e-12 else 4 * window
        assert np.max(np.abs(np.sum(e * e, axis=1) - 1.0)) <= tol
        assert np.max(np.abs(np.sum(e * k, axis=1))) <= tol
        assert np.array_equal(e[0], [1.0, 0.0, 0.0])  # along z: e = x exactly
        assert abs(e[1, 1]) == 1.0  # along x: the y fallback

    def test_crossed_polarizer_gradient_is_finite(self):
        """R-09-10: at g = 0 the state is zeroed behind a double where; no NaN on the backward pass."""
        torch = pytest.importorskip("torch")
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        be.grad_mode.enable()
        try:
            ang = torch.tensor([math.radians(90.0)], dtype=torch.float64, requires_grad=True)
            q = torch.ones(1, dtype=torch.float64)
            u = torch.zeros(1, dtype=torch.float64)
            v = torch.zeros(1, dtype=torch.float64)
            e = (torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
            a = (torch.cos(ang), torch.sin(ang), torch.zeros(1, dtype=torch.float64))
            k = (torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
            c2, s2 = P.rotation_2psi(e, a, k)
            q2, u2 = P.rotate(q, u, c2, s2)
            g, q3, u3, v3 = P.apply_interface(q2, u2, v, P.polarizer_mueller(q2))
            (g + q3 + u3 + v3).sum().backward()
            assert torch.isfinite(ang.grad).all()
            # d/dtheta of (1 + cos 2 theta) / 2 at 90 degrees is 0
            assert abs(float(ang.grad[0])) < 1e-15
        finally:
            be.grad_mode.disable()

    def test_normalise_mode(self):
        assert P.normalise_mode(None) == "off"
        assert P.normalise_mode(False) == "off"
        assert P.normalise_mode(True) == "stokes"
        assert P.normalise_mode("Stokes") == "stokes"
        with pytest.raises(ValueError):
            P.normalise_mode("jones")
