"""refractiveindex.info's formula 4 in the record material (zinc sulfide's default page).

Research repository issue 87; the shared library's issue 68. The library's
integration head of 2026-10-02 resolves ``kmat.get("ZnS")`` to the Debenham page
(Applied Optics 23, 2238, 1984), a formula-4 page whose two resonant terms have
no Sellmeier numerator (C3 = C7 = 0). What is pinned here:

1. **The library's own evaluator, to the bit.** On numpy float64 the engine's
   n at a wavelength in micrometres equals ``kmat.rii.evaluate_formula(4, ...)``
   at the same micrometres: the same terms in the same order on the same
   doubles.
2. **Against ``kmat.n`` over the page's whole band.** At float64 (numpy) and at
   float32 (torch on the CPU), within a first-order rounding bound computed
   from the two evaluation chains before any comparison (the record-material
   test's :class:`_R`, extended by a power), never from the measured gap.
3. **The form.** A list that leaves a term incomplete is refused by name; a
   family the engine does not evaluate is refused by name; the mapping round
   trips; the coefficients are registered and carry a gradient.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.nonsequential import RecordMaterial, RecordRefused
from optiland.nonsequential.parameter_register import INTERIOR_BOUNDARY

from .test_nsq_record_material import U32, U64, _R

kmat = pytest.importorskip("kmat", reason="the shared material library (kmat) is not importable")


def _zns():
    try:
        record = kmat.get("ZnS")
    except LookupError as exc:  # a library older than the integration head of 2026-10-02 (issue 68)
        pytest.skip(f"this library does not resolve ZnS: {exc}")
    model = record.optical.epsilon.model
    if model.kind != "rii_formula" or model.family != 4:
        pytest.skip(f"ZnS's default page is not a formula-4 page in this library ({model.kind})")
    return record


@pytest.fixture(autouse=True)
def _numpy_float64():
    be.set_backend("numpy")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _pow(x: _R, p: float) -> _R:
    """``x ** p``: exact for p = 0, one product for p = 2, else one rounding plus the input's error times p."""
    if p == 0.0:
        return _R.exact(1.0, x.u)
    if p == 2.0:
        return x.sq()
    v = x.v**p
    return _R(v, abs(p) * abs(v) * x.e / abs(x.v) + x.u * abs(v), x.u)


def _formula4_chain(c, lam: _R, u: float, stored: bool) -> _R:
    """n of formula 4, term for term (the engine's ``_formula4_eps`` and ``kmat.rii``'s formula 4)."""
    k = _R.stored if stored else _R.exact
    cst = [k(v, u) for v in c]
    # C4^C5 is a Python double in both evaluations (one rounding of the pow)
    def resonance(r, q):
        v = float(r) ** float(q)
        return _R(v, 0.5 * U64 * abs(v) + (0.5 * u * abs(v) if stored else 0.0), u)

    eps = cst[0]
    if len(c) > 4:
        eps = eps + (cst[1] * _pow(lam, c[2])) / (lam.sq() - resonance(c[3], c[4]))
    if len(c) > 8:
        eps = eps + (cst[5] * _pow(lam, c[6])) / (lam.sq() - resonance(c[7], c[8]))
    for i in range(9, len(c) - 1, 2):
        eps = eps + cst[i] * _pow(lam, c[i + 1])
    return eps.sqrt()


def _band(material: RecordMaterial, count: int = 1201) -> np.ndarray:
    lo, hi = material.range_m
    w = np.linspace(lo * 1e6, hi * 1e6, count)
    return w[(w * 1e-6 >= lo) & (w * 1e-6 <= hi)]


def test_zns_is_the_library_s_formula_4_to_the_bit():
    from kmat import rii

    record = _zns()
    material = RecordMaterial.from_kmat(record)
    assert material.n_kind == "rii_formula_4"
    assert material.identity["key"] == ["main", "ZnS", "Debenham"]
    w = _band(material)
    engine = np.asarray(material.n(w), dtype=np.float64)
    library = rii.evaluate_formula(4, record.optical.epsilon.model.coefficients, w)
    assert np.array_equal(engine, library)


def test_zns_against_kmat_n_at_float64():
    record = _zns()
    material = RecordMaterial.from_kmat(record)
    c = material.to_mapping()["n_model"]["coefficients"]
    w = _band(material)
    engine = np.asarray(material.n(w), dtype=np.float64)
    library = np.asarray(kmat.n(record, w * 1e-6), dtype=np.float64)
    for wi, a, b in zip(w, engine, library):
        mine = _formula4_chain(c, _R.exact(float(wi), U64), U64, stored=False)
        theirs = _formula4_chain(c, _R(float(wi), 2.5 * U64 * float(wi), U64), U64, stored=False)
        bound = mine.e + theirs.sq().sqrt().e  # kmat: metres back to micrometres, n, n^2, sqrt
        assert abs(a - b) <= bound, f"ZnS at {wi} um: {a!r} vs kmat {b!r}, bound {bound:.3e}"


def test_zns_against_kmat_n_at_float32():
    torch = pytest.importorskip("torch")
    record = _zns()
    material = RecordMaterial.from_kmat(record)
    c = material.to_mapping()["n_model"]["coefficients"]
    w = _band(material)
    library = np.asarray(kmat.n(record, w * 1e-6), dtype=np.float64)
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float32")
    engine = material.n(torch.tensor(w, dtype=torch.float32)).detach().double().cpu().numpy()
    for wi, a, b in zip(w, engine, library):
        mine = _formula4_chain(c, _R.stored(float(wi), U32), U32, stored=False)
        theirs = _formula4_chain(c, _R(float(wi), 2.5 * U64 * float(wi), U64), U64, stored=False)
        bound = mine.e + theirs.sq().sqrt().e
        assert abs(a - b) <= bound, f"ZnS at {wi} um: {a!r} vs kmat {b!r}, bound {bound:.3e}"


def test_zns_d_line_index_is_the_page_s():
    """The Debenham page at the helium d line, the number the catalogue's thin-film cases do not use
    (they keep their stated 2.32 as case data, the maintainer's ruling of 2026-10-01)."""
    record = _zns()
    material = RecordMaterial.from_kmat(record)
    n_d = float(np.asarray(material.n(np.array([0.5875618])))[0])
    assert n_d == float(kmat.n(record, 0.5875618e-6))
    assert round(n_d, 4) == 2.3677


@pytest.mark.parametrize("count", [2, 3, 4, 6, 7, 8])
def test_an_incomplete_formula_4_list_is_refused_by_name(count):
    from types import SimpleNamespace as NS

    record = NS(
        identity=NS(name="X", variant="default", key=None, full_name="X"),
        optical=NS(
            epsilon=NS(model=NS(kind="rii_formula", family=4, coefficients=[1.0] * count), validity={}),
            other={},
        ),
    )
    with pytest.raises(RecordRefused, match="formula 4"):
        RecordMaterial.from_kmat(record)


@pytest.mark.parametrize("family", [3, 5, 6, 7, 8, 9])
def test_a_family_the_engine_does_not_evaluate_is_refused_by_name(family):
    from types import SimpleNamespace as NS

    record = NS(
        identity=NS(name="X", variant="default", key=None, full_name="X"),
        optical=NS(
            epsilon=NS(model=NS(kind="rii_formula", family=family, coefficients=[1.0] * 5), validity={}),
            other={},
        ),
    )
    with pytest.raises(RecordRefused, match=f"formula {family}"):
        RecordMaterial.from_kmat(record)


def test_a_tail_term_is_evaluated():
    """A formula-4 list with a polynomial tail (C10 w^C11): the engine's sum against an independent one."""
    c = [2.0, 0.5, 2.0, 0.1, 2.0, 0.25, 0.0, 6.0, 2.0, -0.01, 2.0]
    material = RecordMaterial({"n_model": {"kind": "rii_formula_4", "coefficients": c}})
    for w in (0.5, 1.0, 2.0):
        eps = c[0] + c[1] * w**2 / (w**2 - 0.1**2) + c[5] / (w**2 - 36.0) + c[9] * w**2
        assert float(np.asarray(material.n(np.array([w])))[0]) == pytest.approx(math.sqrt(eps), rel=4e-16)


def test_the_mapping_round_trips_and_the_coefficients_are_registered():
    record = _zns()
    material = RecordMaterial.from_kmat(record)
    mapping = material.to_mapping()
    again = RecordMaterial(mapping)
    assert again.to_mapping() == mapping
    w = np.array([0.5, 1.0, 5.0])
    assert np.array_equal(np.asarray(again.n(w)), np.asarray(material.n(w)))
    from optiland.nonsequential.parameter_register import CONTRACT

    assert CONTRACT["formula4_coefficients"][0] == INTERIOR_BOUNDARY


def test_a_coefficient_carries_its_gradient():
    """d n / d C1 = 1 / (2 n): C1 enters n^2 additively."""
    torch = pytest.importorskip("torch")
    record = _zns()
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    material = RecordMaterial.from_kmat(record)
    c = torch.tensor(material.to_mapping()["n_model"]["coefficients"], dtype=torch.float64, requires_grad=True)
    material.formula4_coefficients = c
    n = material.n(torch.tensor([0.5875618], dtype=torch.float64))
    n.sum().backward()
    assert float(c.grad[0]) == pytest.approx(1.0 / (2.0 * float(n.detach()[0])), rel=1e-14)
