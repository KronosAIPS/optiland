"""JonesThinFilm with an empty stack reproduces JonesFresnel (the research repository's issue 74).

The thin-film module returns the admittance form's coefficients: its r_p is the
negative of the Fresnel r_p and its t is the ratio of tangential fields. Before the
fix ``JonesThinFilm`` put ``-r_p(module) = +r_p(Fresnel)`` where ``JonesFresnel`` puts
``-r_p(Fresnel)``, and the tangential t_p where ``JonesFresnel`` puts the field
amplitude: at 45 degrees, 1.0 -> 1.5, the reflected p entry was +0.0920 against
-0.0920 and the transmitted p entry 0.9080 against 0.7280.

Tolerance: 2.2e-15 absolute, 10 u on entries of magnitude at most 2: each route is
about ten rounded operations and the two round differently (the module through
admittances scaled by sqrt(eps0 / mu0), JonesFresnel through the index ratio).
Measured: at most 4.4e-16.

Beyond the critical angle the module's evanescent root is the exp(-i omega t) one,
as JonesFresnel's is, so the reflection matches there too; the transmitted
(evanescent) amplitude carries the module's conjugation of t and is compared only
below the critical angle here (the research repository's issue 78).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import JonesThinFilm
from optiland.jones import JonesFresnel
from optiland.materials import IdealMaterial
from optiland.rays import RealRays
from optiland.thin_film import ThinFilmStack

TOL = 10 * 2.0**-53 * 2


def _rays(n: int) -> RealRays:
    z = [0.0] * n
    return RealRays(x=z, y=z, z=z, L=z, M=z, N=[1.0] * n, intensity=[1.0] * n,
                    wavelength=[0.55] * n)


def _pair(n1: float, n2: float, thetas_deg, reflect: bool):
    aoi = be.array(np.radians(thetas_deg))
    rays = _rays(len(thetas_deg))
    stack = ThinFilmStack(IdealMaterial(n1), IdealMaterial(n2))
    a = np.asarray(be.to_numpy(JonesThinFilm(stack).calculate_matrix(rays, reflect, aoi)))
    b = np.asarray(be.to_numpy(JonesFresnel(IdealMaterial(n1), IdealMaterial(n2))
                               .calculate_matrix(rays, reflect, aoi)))
    return a, b


@pytest.mark.parametrize("reflect", [True, False], ids=["reflect", "transmit"])
def test_empty_stack_is_jones_fresnel(set_test_backend, reflect):
    a, b = _pair(1.0, 1.5, [0.0, 15.0, 30.0, 45.0, 56.3, 60.0, 75.0, 85.0], reflect)
    assert np.max(np.abs(a - b)) <= TOL


def test_the_issue_values(set_test_backend):
    """The numbers the issue quotes, now on the JonesFresnel side."""
    r, _ = _pair(1.0, 1.5, [0.0, 45.0], True)
    assert np.allclose(np.diag(r[0]).real, [-0.2, -0.2, -1.0], atol=1e-15)
    assert np.allclose(np.diag(r[1]).real[:2], [-0.3033, -0.0920], atol=5e-5)
    t, _ = _pair(1.0, 1.5, [45.0], False)
    assert abs(t[0, 1, 1].real - 0.7280) < 5e-5


def test_internal_reflection_below_and_beyond_the_critical_angle(set_test_backend):
    """1.5 -> 1.0: reflection below and beyond 41.8 degrees, transmission below it."""
    a, b = _pair(1.5, 1.0, [0.0, 20.0, 40.0, 45.0, 60.0], True)
    assert np.max(np.abs(a - b)) <= TOL
    a, b = _pair(1.5, 1.0, [0.0, 20.0, 40.0], False)
    assert np.max(np.abs(a - b)) <= TOL


def test_powers_unchanged(set_test_backend):
    """The fix moves no power: |t_p|^2 n2 cos_t / (n1 cos_i) is still the module's T_p."""
    stack = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(1.5))
    th = np.radians([10.0, 45.0, 70.0])
    rays = _rays(3)
    j = np.asarray(be.to_numpy(JonesThinFilm(stack).calculate_matrix(rays, False, be.array(th))))
    tp = j[:, 1, 1]
    cos_t = np.sqrt(1 - (np.sin(th) / 1.5) ** 2)
    Tp = np.abs(tp) ** 2 * 1.5 * cos_t / np.cos(th)
    want = np.asarray(be.to_numpy(stack.compute_rtRTA_elementwise(be.array([0.55] * 3),
                                                                  be.array(th), "p")["T"]))
    assert np.allclose(Tp, want, rtol=1e-14, atol=0)
    assert math.isfinite(float(Tp.sum()))
