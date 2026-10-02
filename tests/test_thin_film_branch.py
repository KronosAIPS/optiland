"""The branch of the thin-film root at a rounding-level negative extinction.

The research repository's issue 106 (its issue 98 for the analytic module): the
root of ``N^2 - s0^2`` is the limit of the principal root of ``w + i 0`` (first
quadrant), and a negative ``Im w`` no larger than
``ROOT_NOISE_RELATIVE * (|N|^2 + |s0|^2)`` is rounding noise of a lossless
medium. Before the rule, a record with ``k = -2.6e-11`` flipped the root to a
backward wave and the stack below reflected R = 4.045.

The table: one 0.1 um layer of index 1.444 at 1.55 um on a substrate of index
3.47501, unpolarized. The clean values are checked against an independent
route (the Fresnel amplitudes of the two interfaces folded with the layer
phase, in real arithmetic, no matrix); the noisy ones against the clean ones
to the order of the perturbation (``|k| / n`` about 1e-11).
"""

from __future__ import annotations

import cmath
import math

import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.thin_film import ThinFilmStack
from optiland.thin_film.core import ROOT_NOISE_RELATIVE, _snell_cos

LAM_UM = 1.55
LAYER_N, LAYER_D_UM = 1.444, 0.1
SUBSTRATE = 3.47501


def _airy_unpolarized(n0, n1, d_um, n2, lam_um, theta_deg):
    """R of one lossless layer between real media, by Fresnel amplitudes."""
    s0 = n0 * math.sin(math.radians(theta_deg))
    c0, c1, c2 = (math.sqrt(n * n - s0 * s0) for n in (n0, n1, n2))
    delta = 2.0 * math.pi * d_um * c1 / lam_um
    out = 0.0
    for pol in ("s", "p"):
        if pol == "s":
            r01, r12 = (c0 - c1) / (c0 + c1), (c1 - c2) / (c1 + c2)
        else:
            r01 = (n1 * n1 * c0 - n0 * n0 * c1) / (n1 * n1 * c0 + n0 * n0 * c1)
            r12 = (n2 * n2 * c1 - n1 * n1 * c2) / (n2 * n2 * c1 + n1 * n1 * c2)
        ph = cmath.exp(2j * delta)
        out += 0.5 * abs((r01 + r12 * ph) / (1.0 + r01 * r12 * ph)) ** 2
    return out


def _stack_r(n0: complex, ns: complex, theta_deg: float) -> float:
    st = ThinFilmStack(
        incident_material=IdealMaterial(n=n0.real, k=n0.imag),
        substrate_material=IdealMaterial(n=ns.real, k=ns.imag),
    )
    st.add_layer(IdealMaterial(n=LAYER_N), LAYER_D_UM)
    out = st.compute_rtRTA(LAM_UM, math.radians(theta_deg), "u")
    return float(be.to_numpy(out["R"]).ravel()[0])


def test_threshold_is_the_analytic_modules():
    # mirrored from knsrt.analytic.ROOT_NOISE_RELATIVE (the library's 1e-9 band)
    assert ROOT_NOISE_RELATIVE == 1.0e-9


def test_issue_table_substrate_at_30_degrees(set_test_backend):
    clean = _stack_r(1 + 0j, SUBSTRATE + 0j, 30.0)
    noisy = _stack_r(1 + 0j, complex(SUBSTRATE, -2.6e-11), 30.0)
    ref = _airy_unpolarized(1.0, LAYER_N, LAYER_D_UM, SUBSTRATE, LAM_UM, 30.0)
    assert clean == pytest.approx(ref, abs=1e-13)
    assert clean == pytest.approx(0.2541161994686, abs=1e-12)
    # before the rule: 4.0453973473520435
    assert noisy == pytest.approx(clean, abs=1e-10)


def test_issue_table_ambient_at_10_degrees(set_test_backend):
    clean = _stack_r(SUBSTRATE + 0j, 1 + 0j, 10.0)
    noisy = _stack_r(complex(SUBSTRATE, -2.6e-11), 1 + 0j, 10.0)
    ref = _airy_unpolarized(SUBSTRATE, LAYER_N, LAYER_D_UM, 1.0, LAM_UM, 10.0)
    assert clean == pytest.approx(ref, abs=1e-13)
    assert clean == pytest.approx(0.2588745384764, abs=1e-12)
    # before the rule: 4.137682607988687
    assert noisy == pytest.approx(clean, abs=1e-10)


@pytest.mark.parametrize("k", [-2.6e-11, 2.6e-11, -1e-14, 0.0, -1e-10])
def test_sign_of_a_rounding_level_k_does_not_matter(set_test_backend, k):
    base = _stack_r(1 + 0j, SUBSTRATE + 0j, 30.0)
    assert _stack_r(1 + 0j, complex(SUBSTRATE, k), 30.0) == pytest.approx(
        base, abs=1e-9
    )


def _cos(n0: complex, theta_deg: float, n: complex) -> complex:
    th = be.atleast_1d(be.array([math.radians(theta_deg)]))
    c = _snell_cos(
        be.to_complex(be.array([n0.real])) + 1j * n0.imag,
        th,
        be.to_complex(be.array([n.real])) + 1j * n.imag,
    )
    return complex(be.to_numpy(c).ravel()[0])


def test_root_is_forward_below_critical_and_evanescent_beyond(set_test_backend):
    assert _cos(1 + 0j, 30.0, complex(1.5, -3e-11)).real > 0.9
    # beyond the critical angle of glass to air: s0 = 1.5 sin 60 = 1.299
    for k in (0.0, -0.0, -3e-11, 3e-11):
        nc = _cos(1.5 + 0j, 60.0, complex(1.0, k)) * complex(1.0, k)
        assert nc.imag > 0.0
        assert abs(nc.real) < 1e-9


def test_absorbing_substrate_keeps_the_decaying_root(set_test_backend):
    nc = _cos(1 + 0j, 30.0, complex(SUBSTRATE, 0.1)) * complex(SUBSTRATE, 0.1)
    assert nc.real > 0.0
    assert nc.imag > 0.0
    assert _stack_r(1 + 0j, complex(SUBSTRATE, 0.1), 30.0) == pytest.approx(
        0.2518186952905619, abs=1e-12
    )


def test_gain_beyond_the_noise_level_keeps_the_earlier_rule(set_test_backend):
    # k = -1e-3: Im w = -2 n k is far above 1e-9 (|N|^2 + |s0|^2); the root is
    # flipped to Im >= 0 as before (a gain medium is outside the passive theory)
    nc = _cos(1 + 0j, 30.0, complex(SUBSTRATE, -1e-3)) * complex(SUBSTRATE, -1e-3)
    assert nc.imag >= 0.0
    assert nc.real < 0.0
