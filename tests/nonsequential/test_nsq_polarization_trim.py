"""The Stokes Fresnel event's operation trim (2026-10-02): every value bit-identical.

Two trims, each pinned against the composition it replaced:

* a coated face in Stokes mode evaluates its coating once: the scalar R and T are the means
  of the s and p terms (``polarization.coating_sp``), which equal the coating's own
  ``evaluate`` bit for bit, for a thin-film stack from either side and for a table;
* the bare face forms only the elements the event uses, each squared amplitude once: equal
  to ``reflection_mueller`` and ``transmission_mueller`` composed as before, bit for bit.

Both on numpy float64 and torch float32 (the precisions the catalogue grades).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import TabulatedCoating
from optiland.materials import IdealMaterial
from optiland.nonsequential import polarization as P
from optiland.nonsequential.components.coating_support import UnpolarizedThinFilmCoating
from optiland.thin_film import ThinFilmStack

LEGS = [("numpy", "float64"), ("torch", "float32")]


@pytest.fixture(autouse=True)
def _restore():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _leg(backend, precision):
    if backend == "torch":
        pytest.importorskip("torch")
        be.set_backend("torch")
        be.set_device("cpu")
        be.grad_mode.disable()
    else:
        be.set_backend("numpy")
    be.set_precision(precision)


def _bits(x):
    return np.asarray(be.to_numpy(x)).tobytes()


def _stack():
    st = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(1.52))
    st.add_layer(IdealMaterial(3.13, 4.33), 0.01)
    st.add_layer(IdealMaterial(1.38), 0.0996)
    return st


def _table():
    wl = np.array([500.0, 600.0])
    ang = np.array([0.0, 30.0, 60.0, 85.0])
    g = np.linspace(0.02, 0.3, 8).reshape(2, 4)
    return TabulatedCoating(wl, ang, r_s=g, r_p=0.5 * g, t_s=0.9 - g, t_p=0.95 - g,
                            phase_r_deg=10 * g, phase_t_deg=-5 * g, substrate_material=1.52)


@pytest.mark.parametrize("leg", LEGS)
@pytest.mark.parametrize("kind", ["thin_film", "table"])
def test_coating_sp_means_are_evaluate_bit_for_bit(leg, kind):
    _leg(*leg)
    coating = UnpolarizedThinFilmCoating(_stack()) if kind == "thin_film" else _table()
    th = np.radians([0.0, 10.0, 33.0, 47.0, 61.0, 80.0])
    cos_i = be.array(np.cos(th))
    wl = be.array(np.full(th.shape, 0.55))
    for mask in (None, np.array([False, True, False, True, True, False])):
        m = None if mask is None else be.array(mask)
        if m is not None and leg[0] == "torch":
            m = m.bool()
        sp = P.coating_sp(coating, wl, cos_i, m)
        R, T = (coating.evaluate(wl, cos_i) if m is None
                else coating.evaluate(wl, cos_i, from_substrate=m))
        assert _bits(0.5 * (sp.Rs + sp.Rp)) == _bits(R)
        assert _bits(0.5 * (sp.Ts + sp.Tp)) == _bits(T)


@pytest.mark.parametrize("leg", LEGS)
def test_bare_event_equals_the_old_composition(leg):
    _leg(*leg)
    n = 7
    th = np.linspace(0.05, 1.45, n)
    L = be.array(np.zeros(n))
    M = be.array(np.sin(th))
    N = be.array(np.cos(th))
    dirs = be.stack([L, M, N], axis=1)
    normals = be.stack([L * 0.0, L * 0.0, L * 0.0 - 1.0], axis=1)
    cos_i = be.abs((dirs * normals).sum(axis=1))

    class Rays:
        pass

    for n1v, n2v in ((1.0, 1.5), (1.5, 1.0)):
        rays = Rays()
        rays.L, rays.M, rays.N = L, M, N
        rays.pol_q = be.array(np.full(n, 0.3))
        rays.pol_u = be.array(np.full(n, -0.2))
        rays.pol_v = be.array(np.full(n, 0.1))
        rays.pol_ex, rays.pol_ey, rays.pol_ez = L * 0.0 + 1.0, L * 0.0, L * 0.0
        n1 = L * 0.0 + n1v
        n2 = L * 0.0 + n2v
        sin2_t = (n1 / n2) ** 2 * (1.0 - cos_i**2)
        w = 1.0 - sin2_t
        tir = w < 1e-6
        cos_t = be.where(tir, w * 0.0, be.where(tir, w * 0.0 + 1.0, w) ** 0.5)
        rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
        rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
        R = be.where(tir, rs * 0.0 + 1.0, 0.5 * (rs**2 + rp**2))
        T = 1.0 - R
        got = P.fresnel_stokes(rays, dirs, normals, n1, n2, cos_i, sin2_t, tir, rs, rp, R, T)
        # the composition before the trim
        zero = be.zeros_like(R)
        phase = P.tir_relative_phase(n1, n2, cos_i, sin2_t, tir)
        m_r = P.reflection_mueller(rs, rp, tir, phase)
        m_r = P.InterfaceMueller(R, m_r.m01, m_r.m22, m_r.m23)
        m_t = P.transmission_mueller(1.0 - rs**2, 1.0 - rp**2, m00=T)
        m_t = P.InterfaceMueller(T, be.where(tir, zero, -m_r.m01), be.where(tir, zero, m_t.m22), zero)
        for a, b in zip(got.m_r, m_r, strict=True):
            assert _bits(a) == _bits(b)
        for a, b in zip(got.m_t, m_t, strict=True):
            assert _bits(a) == _bits(b)
        assert math.isfinite(float(np.asarray(be.to_numpy(got.R_eff)).sum()))
