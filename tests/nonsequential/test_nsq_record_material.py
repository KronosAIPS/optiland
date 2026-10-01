"""The engine takes a material from a record of the shared material library (kmat).

Research repository issue 87; the shared library's issue 348 (the maintainer's
ruling of 2026-09-30: every material a run traces comes from a library record
handed to the engine). What is pinned here:

1. **n(lambda) against the library.** For N-BK7, fused silica, CaF2, MgF2 and
   Schott F2 (and an Abbe glass and a stated index), the engine's index over
   each record's whole stated band against ``kmat.n``: at float64 on numpy and
   at float32 on torch, each within a bound computed *before* the comparison
   from the two evaluation chains (a running first-order rounding bound,
   :class:`_R`: every stored coefficient and input rounded once, every
   operation one rounding of its result), never from the measured gap.
2. **k(lambda).** A linear table evaluated on numpy float64 is the library's
   own ``np.interp`` on the same breakpoints at the same query, so it is
   bit-identical to ``kmat.nk``.
3. **The band.** A wavelength outside the record's stated band is refused by
   name unless the material is built to extrapolate (the library's rule).
4. **The singlet, catalogue page against record.** The quickstart singlet
   traced with the engine's own ``"N-BK7"`` page and with the library record:
   bit-identical detector maps and totals on numpy and torch at float64. The
   mechanism: the two pages carry the same Sellmeier coefficients, evaluated in
   the same order of operations (n equal at every wavelength), and the same k
   table, whose breakpoints differ only by the metre-micrometre conversion (one
   unit in the last place) -- so k is equal at the scene's wavelengths, which
   the test asserts first. At float32 the record's k query ``w * 1e-6`` is one
   more float32 rounding than the catalogue's query in micrometres, so k, and
   with it the bulk absorption, differs in the last bits; with k removed from
   both the float32 traces are bit-identical, which places the whole
   difference in k.
5. **Identity.** The record's key and content hash survive the scene's JSON,
   the scene IR (strict lowering) and the trace's ``environment["materials"]``.
6. **The gradient.** A Sellmeier strength placed as a tensor is registered
   (``interior+boundary``) and its autograd derivative of the transmitted flux
   equals a fourth-order central difference.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MaterialOutOfRange,
    NSQMaterial,
    NSQScene,
    RecordMaterial,
    RecordRefused,
    Spectrum,
)
from optiland.nonsequential.ir.lower import lower
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.nonsequential.materials.record_material import record_mapping

kmat = pytest.importorskip("kmat", reason="the shared material library (kmat) is not importable")

U64 = 2.0**-53
U32 = 2.0**-24

#: The five records of the brief, by the call that names each one.
RECORDS = {
    "N-BK7": lambda: kmat.get("N-BK7"),
    "fused silica": lambda: kmat.get("fused silica"),
    "CaF2": lambda: kmat.get("CaF2"),
    "MgF2": lambda: kmat.get("MgF2"),
    "F2 (Schott)": lambda: kmat.get("F2", maker="schott"),
    "Abbe 1.5168/64.17": lambda: kmat.abbe(1.5168, 64.17),
    "ideal 1.5168": lambda: kmat.ideal(1.5168),
}

#: The hydrogen F, helium d and hydrogen C lines [um].
FDC = (0.4861327, 0.5875618, 0.6562725)


@pytest.fixture(autouse=True)
def _numpy_float64():
    be.set_backend("numpy")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


# -- the running rounding bound --------------------------------------------------


class _R:
    """A value with a first-order bound on its absolute rounding error.

    Each operation adds one rounding of its result (``u |result|``) to the
    propagated input errors; a stored constant carries half a unit
    (:meth:`stored`). The bound depends only on the chain of operations and the
    values, not on any engine output.
    """

    def __init__(self, v: float, e: float, u: float):
        self.v, self.e, self.u = float(v), float(e), u

    @classmethod
    def exact(cls, v, u):
        return cls(v, 0.0, u)

    @classmethod
    def stored(cls, v, u):
        return cls(v, 0.5 * u * abs(v), u)

    def _new(self, v, e):
        return _R(v, e + self.u * abs(v), self.u)

    def __add__(self, o):
        return self._new(self.v + o.v, self.e + o.e)

    def __sub__(self, o):
        return self._new(self.v - o.v, self.e + o.e)

    def __mul__(self, o):
        return self._new(self.v * o.v, abs(self.v) * o.e + abs(o.v) * self.e)

    def __truediv__(self, o):
        v = self.v / o.v
        return self._new(v, (self.e + abs(v) * o.e) / abs(o.v))

    def sq(self):
        return self * self

    def cube(self):
        # x**3: modelled as two roundings (x * x * x), which covers a pow.
        return self * self * self

    def sqrt(self):
        v = math.sqrt(self.v)
        return self._new(v, self.e / (2.0 * v))


def _engine_chain(mapping, w: float, u: float, w_stored: bool) -> _R:
    """The engine's evaluation of n at ``w`` [um], step for step (RecordMaterial._calculate_n)."""
    n_model = mapping["n_model"]
    lam = _R.stored(w, u) if w_stored else _R.exact(w, u)
    kind = n_model["kind"]
    if kind == "constant":
        return _R.stored(n_model["n"], u)
    if kind == "sellmeier":
        n = _R.stored(n_model["eps_inf"], u)
        for i, b in enumerate(n_model["strengths"]):
            if "resonance_um" in n_model:
                c = _R.stored(n_model["resonance_um"][i], u).sq()
            else:
                c = _R.stored(n_model["resonance_um2"][i], u)
            n = n + (_R.stored(b, u) * lam.sq()) / (lam.sq() - c)
        return n.sqrt()
    if kind == "buchdahl":
        nu = [_R.stored(v, u) for v in n_model["nu"]]
        d = lam - _R.stored(n_model["lambda0_um"], u)
        x = d / (_R.exact(1.0, u) + _R.exact(n_model["alpha"], u) * d)
        return _R.stored(n_model["n0"], u) + nu[0] * x + nu[1] * x.sq() + nu[2] * x.cube()
    q = (lam * _R.stored(1e-6, u)) if w_stored else _R(w * 1e-6, (1.5 * u) * w * 1e-6, u)
    return _interp_chain(n_model["wavelength_m"], n_model["values"], q, u, torch_order=w_stored)


def _interp_chain(xs, ys, q: _R, u: float, torch_order: bool) -> _R:
    j = int(np.clip(np.searchsorted(xs, q.v, side="right"), 1, len(xs) - 1))
    x0, x1 = _R.stored(xs[j - 1], u), _R.stored(xs[j], u)
    y0, y1 = _R.stored(ys[j - 1], u), _R.stored(ys[j], u)
    if torch_order:  # the engine's torch interp: y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return y0 + ((y1 - y0) * (q - x0)) / (x1 - x0)
    return ((y1 - y0) / (x1 - x0)) * (q - x0) + y0  # np.interp


def _kmat_chain(mapping, w: float) -> _R:
    """kmat.n at ``w * 1e-6`` m, float64: metres to micrometres, the page's formula, n^2, sqrt."""
    u = U64
    n_model = mapping["n_model"]
    # w * 1e-6 then * 1e6: two roundings and the constant's own representation error.
    lam = _R(w, 2.5 * u * w, u)
    kind = n_model["kind"]
    if kind == "constant":
        return _R.exact(n_model["n"], u)
    if kind == "sellmeier":
        n = _R.exact(n_model["eps_inf"], u)
        for i, b in enumerate(n_model["strengths"]):
            if "resonance_um" in n_model:
                c = _R.exact(n_model["resonance_um"][i], u).sq()
            else:
                c = _R.exact(n_model["resonance_um2"][i], u)
            n = n + (_R.exact(b, u) * lam.sq()) / (lam.sq() - c)
        return n.sqrt().sq().sqrt()
    if kind == "buchdahl":
        nu = [_R.exact(v, u) for v in n_model["nu"]]
        d = lam - _R.exact(n_model["lambda0_um"], u)
        x = d / (_R.exact(1.0, u) + _R.exact(n_model["alpha"], u) * d)
        n = _R.exact(n_model["n0"], u) + nu[0] * x + nu[1] * x.sq() + nu[2] * x.cube()
        return n.sq().sqrt()
    q = _R(w * 1e-6, 1.5 * u * w * 1e-6, u)
    return _interp_chain(n_model["wavelength_m"], n_model["values"], q, u, torch_order=False)


def _band_grid(material: RecordMaterial, count: int = 1201) -> np.ndarray:
    lo, hi = material.range_m if material.range_m is not None else (0.4e-6, 0.8e-6)
    w = np.linspace(lo * 1e6, hi * 1e6, count)
    return w[(w * 1e-6 >= lo) & (w * 1e-6 <= hi)]


# -- 1, 2: n and k against the library --------------------------------------------


@pytest.mark.parametrize("name", list(RECORDS))
def test_index_against_the_library_at_float64(name):
    """numpy float64: within the two chains' rounding bound at every point of the band."""
    record = RECORDS[name]()
    material = RecordMaterial.from_kmat(record)
    mapping = material.to_mapping()
    w = _band_grid(material)
    engine = np.asarray(material.n(w), dtype=np.float64) * np.ones_like(w)
    library = np.asarray(kmat.n(record, w * 1e-6), dtype=np.float64)
    for wi, a, b in zip(w, engine, library):
        bound = _engine_chain(mapping, float(wi), U64, w_stored=False).e + _kmat_chain(mapping, float(wi)).e
        assert abs(a - b) <= bound, f"{name} at {wi} um: {a!r} vs kmat {b!r}, bound {bound:.3e}"


@pytest.mark.parametrize("name", list(RECORDS))
def test_index_against_the_library_at_float32(name):
    """torch float32 on the CPU: within the float32 chain's bound plus the library's."""
    torch = pytest.importorskip("torch")
    record = RECORDS[name]()
    material = RecordMaterial.from_kmat(record)
    mapping = material.to_mapping()
    w = _band_grid(material)
    library = np.asarray(kmat.n(record, w * 1e-6), dtype=np.float64)
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float32")
    out = material.n(torch.tensor(w, dtype=torch.float32))
    engine = out.detach().double().cpu().numpy() * np.ones_like(w)
    for wi, a, b in zip(w, engine, library):
        bound = _engine_chain(mapping, float(wi), U32, w_stored=True).e + _kmat_chain(mapping, float(wi)).e
        assert abs(a - b) <= bound, f"{name} at {wi} um: {a!r} vs kmat {b!r}, bound {bound:.3e}"


@pytest.mark.parametrize("name", ["N-BK7", "F2 (Schott)", "CaF2"])
def test_tabulated_values_are_the_library_s_own(name):
    """A linear table on numpy float64 is np.interp on the record's breakpoints: equal to the bit."""
    record = RECORDS[name]()
    material = RecordMaterial.from_kmat(record)
    w = _band_grid(material)
    n_lib, k_lib = kmat.nk(record, w * 1e-6)
    if material.k_kind == "table":
        assert np.array_equal(np.asarray(material.k(w)), np.asarray(k_lib))
    if material.n_kind == "table":
        assert np.array_equal(np.asarray(material.n(w)), np.asarray(n_lib))


def test_a_stated_index_is_the_stated_number():
    """kmat.ideal(1.5168) evaluates to 1.5168 exactly, as the engine's ideal material does."""
    from optiland.materials import IdealMaterial

    material = RecordMaterial.from_kmat(kmat.ideal(1.5168))
    w = np.array([0.4861327, 0.5876, 0.6562725])
    assert np.array_equal(np.asarray(material.n(w)), np.full(3, 1.5168))
    assert np.array_equal(np.asarray(material.n(w)), np.asarray(IdealMaterial(n=1.5168, k=0.0).n(w)))
    assert material.range_m is None and material.n_kind == "constant"


# -- 3: the band ----------------------------------------------------------------------


def test_outside_the_band_is_refused_by_name_and_read_when_asked():
    record = kmat.get("N-BK7")
    material = RecordMaterial.from_kmat(record)
    with pytest.raises(MaterialOutOfRange, match="N-BK7"):
        material.n(np.array([0.25]))
    with pytest.raises(kmat.records.OutOfValidity):  # the library refuses the same wavelength
        kmat.n(record, 0.25e-6)
    wide = RecordMaterial.from_kmat(record, extrapolate=True)
    value = float(np.asarray(wide.n(np.array([0.25])))[0])
    assert abs(value - float(kmat.n(record, 0.25e-6, extrapolate=True))) < 1e-12


def test_a_form_the_engine_cannot_evaluate_exactly_is_refused():
    from types import SimpleNamespace as NS

    record = NS(
        identity=NS(name="X", variant="default", key=None, full_name="X"),
        optical=NS(
            epsilon=NS(model=NS(kind="rii_formula", family=4, coefficients=[1.0, 2.0, 3.0]), validity={}),
            other={},
        ),
    )
    with pytest.raises(RecordRefused, match="formula 4"):
        RecordMaterial.from_kmat(record)


def test_a_pole_set_reads_as_the_same_sellmeier_terms():
    """kmat's pole variant of N-BK7 (lossless Lorentz poles) against the formula page."""
    poles = RecordMaterial.from_kmat(kmat.get("N-BK7", variant="N-BK7-poles"))
    page = RecordMaterial.from_kmat(kmat.get("N-BK7"))
    w = np.linspace(0.31, 2.49, 501)
    assert poles.n_kind == "sellmeier"
    # One rounding in each resonance wavelength through omega_0 = 2 pi c / lambda_0,
    # amplified by |d n / d ln C_i| <= 1e3 near the band's ends: far below 1e-12.
    assert np.max(np.abs(np.asarray(poles.n(w)) - np.asarray(page.n(w)))) < 1e-12


# -- 4: the singlet --------------------------------------------------------------------


def _singlet(material, spectrum, k_free=False):
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(z=0.0),
        CollimatedSourceConfig(spectrum=spectrum, total_flux=1.0, aperture_radius=5.0),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50),
        LensConfig(r1=100, r2=-100, thickness=5, material=material, front_aperture_radius=12.5),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=150),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=64, num_pixels_y=64),
    )
    return scene


def _trace_pair(k_free=False):
    from optiland.materials import Material

    spectrum = Spectrum(wavelengths=np.array(FDC), weights=np.ones(3))
    catalogue = NSQMaterial(optiland_material=Material("N-BK7"))
    mapping = RecordMaterial.from_kmat(kmat.get("N-BK7")).to_mapping()
    if k_free:
        catalogue.optiland_material._k = None
        catalogue.optiland_material._k_wavelength = None
        catalogue.optiland_material._k_warning_printed = True
        mapping["k_model"] = {"kind": "constant", "k": 0.0}
    record = NSQMaterial.from_record(mapping)
    out = []
    for material in (catalogue, record):
        result = _singlet(material, spectrum).trace(num_rays=4000, seed=7, max_depth=16)
        data = np.asarray(be.to_numpy(result.detectors["D1"].data), dtype=np.float64)
        out.append((data, float(result.total_flux_detected), float(result.total_flux_bulk_absorbed)))
    return out


def _set(backend, precision):
    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
    be.set_precision(precision)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_singlet_catalogue_page_and_record_bit_identical_at_float64(backend):
    from optiland.materials import Material

    if backend == "torch":
        pytest.importorskip("torch")
    _set(backend, "float64")
    w = np.array(FDC)
    page, rec = Material("N-BK7"), RecordMaterial.from_kmat(kmat.get("N-BK7"))
    # The mechanism's two preconditions, at the scene's wavelengths.
    assert np.array_equal(be.to_numpy(page.n(be.asarray(w))), be.to_numpy(rec.n(be.asarray(w))))
    assert np.array_equal(be.to_numpy(page.k(be.asarray(w))), be.to_numpy(rec.k(be.asarray(w))))
    (a, fa, ba), (b, fb, bb) = _trace_pair()
    assert np.array_equal(a, b) and fa == fb and ba == bb


def test_singlet_at_float32_differs_only_through_k():
    pytest.importorskip("torch")
    _set("torch", "float32")
    (a, fa, ba), (b, fb, bb) = _trace_pair(k_free=True)
    assert np.array_equal(a, b) and fa == fb and ba == bb == 0.0
    (a, fa, ba), (b, fb, bb) = _trace_pair()
    # With k present the totals agree to float32 rounding of a 1e-3 W bulk loss.
    assert abs(fa - fb) <= 64 * U32 * fa and abs(ba - bb) <= 64 * U32 * ba


# -- 5: identity ------------------------------------------------------------------------


def test_identity_survives_json_ir_and_the_run_record(tmp_path):
    record = kmat.get("F2", maker="schott")
    material = NSQMaterial.from_record(record)
    scene = _singlet(material, Spectrum.monochromatic(0.5875618))
    path = tmp_path / "scene.json"
    scene.to_json(path)
    text = json.loads(path.read_text())
    blob = json.dumps(text)
    assert record.content_hash() in blob and "SCHOTT-optical" in blob
    again = NSQScene.from_json(path)
    r1 = scene.trace(num_rays=500, seed=3, max_depth=8)
    r2 = again.trace(num_rays=500, seed=3, max_depth=8)
    assert np.array_equal(np.asarray(r1.detectors["D1"].data), np.asarray(r2.detectors["D1"].data))
    ir = lower(scene, strict=True)
    media = [m for m in ir.media if m.n_model.get("kind") == "record"]
    assert len(media) == 1 and media[0].n_model["identity"]["content_hash"] == record.content_hash()
    rows = r1.environment["materials"]
    assert rows == r2.environment["materials"]
    assert rows[0]["key"] == ["specs", "SCHOTT-optical", "F2"]
    assert rows[0]["content_hash"] == record.content_hash()
    assert rows[0]["library"] == "kmat" and rows[0]["library_version"] == kmat.__version__


def test_no_materials_key_without_a_record():
    scene = _singlet("N-BK7", Spectrum.monochromatic(0.55))
    assert "materials" not in scene.trace(num_rays=200, seed=1, max_depth=8).environment


def test_mapping_round_trip_is_lossless():
    for make in RECORDS.values():
        mapping = record_mapping(make())
        again = RecordMaterial(json.loads(json.dumps(mapping))).to_mapping()
        assert again == json.loads(json.dumps(mapping))


# -- 6: the gradient ------------------------------------------------------------------


def test_a_sellmeier_strength_is_registered_and_its_gradient_is_right():
    torch = pytest.importorskip("torch")
    from optiland.nonsequential.parameter_register import INTERIOR_BOUNDARY

    _set("torch", "float64")
    mapping = RecordMaterial.from_kmat(kmat.get("N-BK7")).to_mapping()

    def loss(b0):
        m = RecordMaterial(mapping)
        m.sellmeier_strengths = torch.stack(
            [b0, torch.tensor(mapping["n_model"]["strengths"][1], dtype=torch.float64),
             torch.tensor(mapping["n_model"]["strengths"][2], dtype=torch.float64)]
        )
        scene = _singlet(NSQMaterial(optiland_material=m), Spectrum.monochromatic(0.5875618))
        scene.sampling_policy = SamplingPolicy(reflect_prob=0.0)
        result = scene.trace(num_rays=500, seed=5, max_depth=8)
        return result, result.detectors["D1"].data.sum()

    b = torch.tensor(mapping["n_model"]["strengths"][0], dtype=torch.float64, requires_grad=True)
    result, f = loss(b)
    entries = [e for e in result.parameter_register if e.name.endswith("optiland_material.sellmeier_strengths")]
    assert len(entries) == 1 and entries[0].gradient_class == INTERIOR_BOUNDARY
    (ad,) = torch.autograd.grad(f, b)
    h = 1e-4
    with torch.no_grad():
        fp2, fp1, fm1, fm2 = (
            float(loss(torch.tensor(float(b) + m * h, dtype=torch.float64))[1]) for m in (2, 1, -1, -2)
        )
    fd = (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)
    assert fd != 0.0
    # The fourth-order difference's truncation (h^4 f^(5), f smooth in B) and its
    # rounding (about 1e4 u |f| / h) are both below 1e-7 of the slope here.
    assert abs(float(ad) - fd) <= 1e-7 * abs(fd) + 1.5e4 * U64 * abs(float(f.detach())) / h
