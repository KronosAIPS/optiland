"""A material taken from a record of the shared material library, as plain data.

The research repository's issue 87 and the shared library's issue 348 (the
maintainer's ruling of 2026-09-30): every material a run traces comes from a
record of the family's material library (``kmat``) handed to the engine; the
engine keeps no catalogue of its own. This module is the engine's side of
that contract, built the way the NURBS kind reads the geometry library: a
**mapping contract** the engine evaluates itself, plus one thin adapter
(:meth:`RecordMaterial.from_kmat`) that reads a library record by duck typing.
The engine never imports the library.

The mapping (schema ``nsq-material-record/1``)::

    {
      "schema": "nsq-material-record/1",
      "identity": {"library": "kmat", "library_version": "0.5.5",
                   "name": "N-BK7", "variant": "N-BK7",
                   "key": ["specs", "SCHOTT-optical", "N-BK7"],
                   "content_hash": "579bad42a0ea963b"},
      "range_m": [3e-07, 2.5e-06] or null,
      "extrapolate": false,
      "n_model": one of
        {"kind": "constant", "n": 1.5168}
        {"kind": "sellmeier", "eps_inf": 1.0, "strengths": [B_i],
         "resonance_um": [lambda_i]}          # or "resonance_um2": [C_i]
        {"kind": "buchdahl", "n0": n_d, "nu": [nu1, nu2, nu3],
         "lambda0_um": 0.5875618, "alpha": 2.5}
        {"kind": "table", "wavelength_m": [...], "values": [...]},
      "k_model": {"kind": "constant", "k": 0.0} or
                 {"kind": "table", "wavelength_m": [...], "values": [...]}
    }

How each model is evaluated, per ray wavelength ``w`` in micrometres, with the
active backend's operations at its working precision (numpy or torch, float64
or float32, on the device the wavelengths live on):

- ``constant``: the stated number, broadcast (``kmat.ideal``'s record).
- ``sellmeier``: ``n = sqrt(eps_inf + sum_i B_i w^2 / (w^2 - C_i))`` with
  ``C_i = lambda_i^2`` (squared at evaluation, refractiveindex.info formula 1)
  or ``C_i`` given (formula 2). This is the exact pole set of
  ``kmat.sellmeier_poles`` written in wavelength: ``eps_inf`` the
  high-frequency term, ``B_i`` the strengths, ``lambda_i`` the resonance
  wavelengths. The order of operations is the existing engine's own
  formula-1 and formula-2 evaluator's, so a page the engine's catalogue also
  carries evaluates to the same bits.
- ``buchdahl``: ``n = n0 + nu1 x + nu2 x^2 + nu3 x^3`` with
  ``x = d / (1 + alpha d)``, ``d = w - lambda0`` (the three-term Buchdahl form
  of Robb and Mercado, Applied Optics 22(8), 1198, 1983, which ``kmat.abbe``
  produces from a d-line index and an Abbe number).
- ``table``: linear interpolation in wavelength on the record's own breakpoints
  (metres, untouched), queried at ``w * 1e-6``; the rule ``kmat.nk`` applies
  to a ``table_1d`` property with ``interpolation="linear"`` (``np.interp``).
  On the torch backend the engine's own ``interp`` is used, which forms the
  same straight line with a different rounding order.

**The range.** ``range_m`` is the band the record states (the intersection of
the n and k properties' validity and of any table's breakpoints). A wavelength
outside it is refused with :class:`MaterialOutOfRange` naming the material and
the band, unless the mapping asks to extrapolate -- the library's own rule
(``kmat.nk(..., extrapolate=False)`` raises ``OutOfValidity``). A table read
past its ends with ``extrapolate`` holds its end value (``np.interp``'s rule,
and the library's).

**Identity.** The record's identity key and content hash travel with the
material: in :meth:`to_mapping`, in the scene's JSON
(:mod:`optiland.nonsequential.serialization`), in the scene IR's medium list
(:mod:`optiland.nonsequential.ir.lower`) and in every trace's
``SimulationResult.environment["materials"]``, so a result says which page it
traced.

**Gradients.** The model's coefficients are public attributes
(``sellmeier_eps_inf``, ``sellmeier_strengths``, ``sellmeier_resonance_um`` or
``sellmeier_resonance_um2``, ``buchdahl_n0``, ``buchdahl_nu``, ``constant_n``,
``constant_k``, ``table_n``, ``table_k``). A torch tensor with
``requires_grad=True`` placed there is evaluated by torch operations on the
graph, and the parameter register of chapter 09 lists it under the material's
surfaces with the class ``interior+boundary`` (Fresnel, Snell and
Beer-Lambert through n(lambda); the onset of total internal reflection is the
boundary term, absent like every other one).
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

import optiland.backend as be
from optiland.materials.base import BaseMaterial

#: The mapping's schema tag.
RECORD_SCHEMA = "nsq-material-record/1"

#: 2 pi c in metres per second, for a pole's angular frequency (kmat.poles).
_TWO_PI_C = 2.0 * math.pi * 299792458.0

_N_KINDS = ("constant", "sellmeier", "buchdahl", "table")
_K_KINDS = ("constant", "table")


class MaterialOutOfRange(ValueError):
    """A wavelength outside the band the material's record states."""


class RecordRefused(ValueError):
    """A library record the engine cannot evaluate exactly, refused by name."""


def _floats(values) -> list[float]:
    return [float(v) for v in values]


def _stored_constant(value):
    """A stated constant held as the engine's ideal material holds it: a one-element array.

    Made with the backend active at construction, as ``IdealMaterial`` makes
    its ``index``, and read back with the stored dtype preserved, so a stated
    index built from a record evaluates to the same bits as the ideal material
    it replaces at every precision. A tensor (a design variable) is kept as a
    one-element view of itself, on its graph.
    """
    if hasattr(value, "requires_grad"):
        return value.reshape(1)
    return be.array([float(value)])


def _plain(value):
    """A coefficient as plain data: a tensor detached to Python floats."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim else float(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return float(value)


class RecordMaterial(BaseMaterial):
    """A material evaluated from a library record's mapping (module docstring).

    Args:
        mapping: The ``nsq-material-record/1`` mapping.
        propagation_model: As for every optiland material.

    Raises:
        ValueError: If the mapping is not of the schema, or names a model kind
            the engine does not evaluate.
    """

    def __init__(self, mapping: dict, propagation_model=None):
        super().__init__(propagation_model)
        if mapping.get("schema", RECORD_SCHEMA) != RECORD_SCHEMA:
            raise ValueError(
                f"material mapping schema {mapping.get('schema')!r}; this engine reads {RECORD_SCHEMA!r}"
            )
        self.identity: dict = dict(mapping.get("identity") or {})
        rng = mapping.get("range_m")
        self.range_m: tuple[float, float] | None = (
            None if rng is None else (float(rng[0]), float(rng[1]))
        )
        self.extrapolate: bool = bool(mapping.get("extrapolate", False))
        n_model = dict(mapping["n_model"])
        k_model = dict(mapping.get("k_model") or {"kind": "constant", "k": 0.0})
        self.n_kind: str = n_model["kind"]
        self.k_kind: str = k_model["kind"]
        if self.n_kind not in _N_KINDS:
            raise ValueError(f"n_model kind {self.n_kind!r} is not one of {_N_KINDS}")
        if self.k_kind not in _K_KINDS:
            raise ValueError(f"k_model kind {self.k_kind!r} is not one of {_K_KINDS}")

        if self.n_kind == "constant":
            self.constant_n = _stored_constant(n_model["n"])
        elif self.n_kind == "sellmeier":
            self.sellmeier_eps_inf = n_model["eps_inf"]
            self.sellmeier_strengths = n_model["strengths"]
            if ("resonance_um" in n_model) == ("resonance_um2" in n_model):
                raise ValueError("a sellmeier n_model states resonance_um or resonance_um2, exactly one")
            if "resonance_um" in n_model:
                self.sellmeier_resonance_um = n_model["resonance_um"]
                count = len(self.sellmeier_resonance_um)
            else:
                self.sellmeier_resonance_um2 = n_model["resonance_um2"]
                count = len(self.sellmeier_resonance_um2)
            if count != len(self.sellmeier_strengths):
                raise ValueError("a sellmeier n_model needs one resonance per strength")
        elif self.n_kind == "buchdahl":
            self.buchdahl_n0 = n_model["n0"]
            self.buchdahl_nu = n_model["nu"]
            self.buchdahl_lambda0_um = float(n_model["lambda0_um"])
            self.buchdahl_alpha = float(n_model["alpha"])
        else:
            self.table_n_wavelength_m = _floats(n_model["wavelength_m"])
            self.table_n = n_model["values"]

        if self.k_kind == "constant":
            self.constant_k = _stored_constant(k_model["k"])
        else:
            self.table_k_wavelength_m = _floats(k_model["wavelength_m"])
            self.table_k = k_model["values"]

    # -- the library adapter ------------------------------------------------

    @classmethod
    def from_kmat(cls, record, *, extrapolate: bool = False) -> RecordMaterial:
        """The material of a ``kmat.MaterialRecord``, read without importing kmat.

        Reads ``record.identity``, ``record.content_hash()`` and
        ``record.optical``: the permittivity model (``rii_formula`` families 1
        and 2, ``sellmeier``, ``buchdahl``, or ``poles`` whose every pole is a
        lossless Lorentz term) or a stated ``n`` (``constant`` or a linear
        ``table_1d``), and the ``k`` property (absent: 0, the library's rule
        for an index-only page; ``constant``; a linear ``table_1d``). Any other
        form is refused with :class:`RecordRefused` naming it: the engine
        evaluates only what it can evaluate exactly as the library does.

        Args:
            record: A ``kmat.MaterialRecord`` (``kmat.get``, ``kmat.ideal``,
                ``kmat.abbe``).
            extrapolate: Read past the record's stated band, as
                ``kmat.nk(..., extrapolate=True)`` does.

        Returns:
            The material.
        """
        return cls(record_mapping(record, extrapolate=extrapolate))

    @classmethod
    def from_pole_set(cls, pole_set, *, identity: dict | None = None,
                      extrapolate: bool = False) -> RecordMaterial:
        """The material of a ``kmat.PoleSet`` (``kmat.sellmeier_poles(page)``).

        Each lossless Lorentz pole ``(delta_epsilon, omega_0)`` becomes the
        Sellmeier term ``delta_epsilon w^2 / (w^2 - lambda_0^2)`` with
        ``lambda_0 = 2 pi c / omega_0`` in micrometres. The round trip through
        the angular frequency costs about one rounding in ``lambda_0``; a
        record read by :meth:`from_kmat` takes the page's own coefficients
        instead and costs none.

        Raises:
            RecordRefused: If a pole is damped (``gamma != 0``) or not Lorentz.
        """
        strengths, resonance_um = [], []
        for pole in pole_set.poles:
            if pole.get("kind", "lorentz") != "lorentz" or float(pole.get("gamma", 0.0)) != 0.0:
                raise RecordRefused(
                    f"a pole set with a damped or non-Lorentz pole ({pole!r}) is lossy; "
                    "this engine evaluates a real index from lossless poles only"
                )
            strengths.append(float(pole["delta_epsilon"]))
            resonance_um.append(_TWO_PI_C / float(pole["omega_0"]) * 1e6)
        rng = getattr(pole_set, "range_m", None)
        return cls(
            {
                "schema": RECORD_SCHEMA,
                "identity": dict(identity or {"name": getattr(pole_set, "source", "") or "pole set"}),
                "range_m": None if rng is None else [float(rng[0]), float(rng[1])],
                "extrapolate": extrapolate,
                "n_model": {
                    "kind": "sellmeier",
                    "eps_inf": float(pole_set.eps_inf),
                    "strengths": strengths,
                    "resonance_um": resonance_um,
                },
                "k_model": {"kind": "constant", "k": 0.0},
            }
        )

    # -- the mapping ----------------------------------------------------------

    def to_mapping(self) -> dict:
        """The ``nsq-material-record/1`` mapping, as plain JSON-safe data."""
        if self.n_kind == "constant":
            n_model = {"kind": "constant", "n": _plain(self.constant_n)[0]}
        elif self.n_kind == "sellmeier":
            n_model = {
                "kind": "sellmeier",
                "eps_inf": _plain(self.sellmeier_eps_inf),
                "strengths": _plain(self.sellmeier_strengths),
            }
            if hasattr(self, "sellmeier_resonance_um"):
                n_model["resonance_um"] = _plain(self.sellmeier_resonance_um)
            else:
                n_model["resonance_um2"] = _plain(self.sellmeier_resonance_um2)
        elif self.n_kind == "buchdahl":
            n_model = {
                "kind": "buchdahl",
                "n0": _plain(self.buchdahl_n0),
                "nu": _plain(self.buchdahl_nu),
                "lambda0_um": self.buchdahl_lambda0_um,
                "alpha": self.buchdahl_alpha,
            }
        else:
            n_model = {
                "kind": "table",
                "wavelength_m": list(self.table_n_wavelength_m),
                "values": _plain(self.table_n),
            }
        if self.k_kind == "constant":
            k_model = {"kind": "constant", "k": _plain(self.constant_k)[0]}
        else:
            k_model = {
                "kind": "table",
                "wavelength_m": list(self.table_k_wavelength_m),
                "values": _plain(self.table_k),
            }
        return {
            "schema": RECORD_SCHEMA,
            "identity": dict(self.identity),
            "range_m": None if self.range_m is None else list(self.range_m),
            "extrapolate": self.extrapolate,
            "n_model": n_model,
            "k_model": k_model,
        }

    @property
    def label(self) -> str:
        """A short name for messages and the scene IR: the record's name and variant."""
        name = self.identity.get("name") or "record"
        variant = self.identity.get("variant")
        return f"{name}:{variant}" if variant and variant not in ("default", name) else str(name)

    # -- BaseMaterial ---------------------------------------------------------

    def _parameters(self) -> tuple:
        names = [
            "constant_n", "sellmeier_eps_inf", "sellmeier_strengths", "sellmeier_resonance_um",
            "sellmeier_resonance_um2", "buchdahl_n0", "buchdahl_nu", "table_n", "constant_k", "table_k",
        ]
        return tuple(getattr(self, n) for n in names if hasattr(self, n))

    def _cache_state(self) -> tuple | None:
        """Every coefficient, so an in-place change or a gradient disables reuse."""
        return self._state_key(self._parameters())

    def _check_range(self, wavelength) -> None:
        if self.range_m is None or self.extrapolate:
            return
        lo, hi = self.range_m
        if hasattr(wavelength, "detach"):
            w = wavelength.detach()
            w_min, w_max = float(w.min()), float(w.max())
            narrow = str(w.dtype) in ("torch.float32", "torch.float16", "torch.bfloat16")
        else:
            w = np.asarray(wavelength)
            narrow = w.dtype.itemsize < 8
            w = w.astype(np.float64)
            w_min, w_max = float(w.min()), float(w.max())
        # The comparison is the library's, in metres, on the same product w * 1e-6.
        # A float32 wavelength is the stated one rounded to float32, which can fall
        # half a unit outside a band end the caller asked for exactly (0.334 um is
        # 0.33399999 in float32); the band is widened by that one rounding there.
        slack = 2.0**-24 if narrow else 0.0
        if w_min * 1e-6 < lo * (1.0 - slack) or w_max * 1e-6 > hi * (1.0 + slack):
            asked = f"{w_min:g} um" if w_min == w_max else f"{w_min:g} to {w_max:g} um"
            raise MaterialOutOfRange(
                f"material {self.label} (record {self.identity.get('content_hash')}) is stated for "
                f"{lo * 1e6:g} to {hi * 1e6:g} um and the trace asks {asked}; the record's "
                "band is the library's (kmat raises OutOfValidity there). Build the material with "
                "extrapolate=True to read the model past it"
            )

    def _calculate_n(self, wavelength, **kwargs):
        self._check_range(wavelength)
        if self.n_kind == "constant":
            n = self._as_backend_array(self.constant_n, preserve_dtype=True)[0]
            if be.is_array_like(wavelength) and be.size(wavelength) > 1:
                return self._broadcast_like(n, wavelength)
            return n
        if self.n_kind == "sellmeier":
            b = self._as_backend_array(self.sellmeier_strengths)
            if hasattr(self, "sellmeier_resonance_um"):
                lam = self._as_backend_array(self.sellmeier_resonance_um)
                c = [lam[i] ** 2 for i in range(len(self.sellmeier_strengths))]
            else:
                c2 = self._as_backend_array(self.sellmeier_resonance_um2)
                c = [c2[i] for i in range(len(self.sellmeier_strengths))]
            # The existing engine's formula-1/2 order: n = 1 + c0, then each term.
            n = self._as_backend_array(be.atleast_1d(self.sellmeier_eps_inf))[0]
            for i in range(len(self.sellmeier_strengths)):
                n = n + b[i] * wavelength**2 / (wavelength**2 - c[i])
            return be.sqrt(n)
        if self.n_kind == "buchdahl":
            nu = self._as_backend_array(self.buchdahl_nu)
            n0 = self._as_backend_array(be.atleast_1d(self.buchdahl_n0))[0]
            d = wavelength - self.buchdahl_lambda0_um
            x = d / (1.0 + self.buchdahl_alpha * d)
            return n0 + nu[0] * x + nu[1] * x**2 + nu[2] * x**3
        return self._interp(self.table_n_wavelength_m, self.table_n, wavelength)

    def _calculate_k(self, wavelength, **kwargs):
        self._check_range(wavelength)
        if self.k_kind == "constant":
            k = self._as_backend_array(self.constant_k, preserve_dtype=True)[0]
            if be.is_array_like(wavelength) and be.size(wavelength) > 1:
                return self._broadcast_like(k, wavelength)
            return k
        return self._interp(self.table_k_wavelength_m, self.table_k, wavelength)

    def _interp(self, grid_m, values, wavelength):
        query = wavelength * 1e-6
        return be.interp(query, self._as_backend_array(grid_m), self._as_backend_array(values))

    def to_dict(self):
        data = super().to_dict()
        data["record"] = self.to_mapping()
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(data["record"])


def _library_version(record) -> str | None:
    module = sys.modules.get((type(record).__module__ or "").split(".")[0])
    return getattr(module, "__version__", None)


def _validity(prop) -> tuple[float, float] | None:
    band = (getattr(prop, "validity", None) or {}).get("wavelength")
    return None if not band else (float(band[0]), float(band[1]))


def _intersect(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (max(a[0], b[0]), min(a[1], b[1]))


def _table(prop, what: str, name: str) -> tuple[dict, tuple[float, float]]:
    model = prop.model
    if model.kind != "table_1d" or getattr(model, "variable", "wavelength") != "wavelength":
        raise RecordRefused(f"{name}: its {what} is a {model.kind!r} model; the engine reads a "
                            "constant or a table in wavelength")
    if getattr(model, "interpolation", "linear") != "linear":
        raise RecordRefused(f"{name}: its {what} table interpolates {model.interpolation!r}; the "
                            "engine evaluates the library's linear rule only")
    x, y = _floats(model.x), _floats(model.y)
    return {"kind": "table", "wavelength_m": x, "values": y}, (x[0], x[-1])


def record_mapping(record, *, extrapolate: bool = False) -> dict:
    """The ``nsq-material-record/1`` mapping of a ``kmat.MaterialRecord`` (duck-typed).

    See :meth:`RecordMaterial.from_kmat` for what is read and what is refused.
    """
    ident = record.identity
    name = getattr(ident, "full_name", None) or ident.name
    optical = getattr(record, "optical", None)
    if optical is None:
        raise RecordRefused(f"{name}: the record has no optical group")
    band = None
    eps = optical.epsilon
    other = dict(getattr(optical, "other", {}) or {})
    if eps is not None:
        model = eps.model
        band = _validity(eps)
        if model.kind == "rii_formula":
            c = _floats(model.coefficients)
            if model.family not in (1, 2):
                raise RecordRefused(f"{name}: refractiveindex.info formula {model.family} is not a "
                                    "Sellmeier form; the engine evaluates families 1 and 2")
            key = "resonance_um" if model.family == 1 else "resonance_um2"
            n_model = {"kind": "sellmeier", "eps_inf": 1.0 + c[0],
                       "strengths": c[1::2], key: c[2::2]}
        elif model.kind == "sellmeier":
            n_model = {"kind": "sellmeier", "eps_inf": 1.0 + float(model.constant),
                       "strengths": _floats(model.B), "resonance_um2": _floats(model.C)}
        elif model.kind == "buchdahl":
            n_model = {"kind": "buchdahl", "n0": float(model.n0),
                       "nu": [float(model.nu1), float(model.nu2), float(model.nu3)],
                       "lambda0_um": float(model.lambda0), "alpha": float(model.alpha)}
        elif model.kind == "poles":
            named = model.named or []
            if not named or any(p.get("kind") != "lorentz" or float(p.get("gamma", 0.0)) != 0.0 for p in named):
                raise RecordRefused(f"{name}: a pole model with damped or unnamed poles is lossy; "
                                    "the engine evaluates lossless Lorentz poles only")
            n_model = {"kind": "sellmeier", "eps_inf": float(model.eps_inf),
                       "strengths": [float(p["delta_epsilon"]) for p in named],
                       "resonance_um": [_TWO_PI_C / float(p["omega_0"]) * 1e6 for p in named]}
        else:
            raise RecordRefused(f"{name}: a permittivity model of kind {model.kind!r} is not one "
                                "the engine evaluates")
    elif "n" in other:
        prop = other["n"]
        band = _validity(prop)
        if prop.model.kind == "constant":
            n_model = {"kind": "constant", "n": float(prop.model.value)}
        else:
            n_model, span = _table(prop, "index", name)
            band = _intersect(band, span)
    else:
        raise RecordRefused(f"{name}: the record states no refractive index")

    if "k" in other:
        prop = other["k"]
        band = _intersect(band, _validity(prop))
        if prop.model.kind == "constant":
            k_model = {"kind": "constant", "k": float(prop.model.value)}
        else:
            k_model, span = _table(prop, "extinction coefficient", name)
            band = _intersect(band, span)
    else:
        k_model = {"kind": "constant", "k": 0.0}

    key = getattr(ident, "key", None)
    return {
        "schema": RECORD_SCHEMA,
        "identity": {
            "library": (type(record).__module__ or "").split(".")[0] or None,
            "library_version": _library_version(record),
            "name": ident.name,
            "variant": getattr(ident, "variant", None),
            "key": None if key is None else [str(v) for v in key],
            "content_hash": record.content_hash() if hasattr(record, "content_hash") else None,
        },
        "range_m": None if band is None else [band[0], band[1]],
        "extrapolate": bool(extrapolate),
        "n_model": n_model,
        "k_model": k_model,
    }
