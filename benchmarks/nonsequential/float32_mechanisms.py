"""Ray-by-ray reproduction of the two float32 failures of the Fresnel catalogue cases.

Two cases of the rung-1 validation catalogue fail at float32 and pass at
float64 on the same engine and the same seed:

* the Fresnel reflectance of a vacuum/N-BK7 interface at 89 degrees
  incidence (the catalogue reads reflectance 1.0 against 0.9046, identical in
  every batch), and
* the transmittance of an N-BK7/vacuum interface one millidegree inside the
  critical angle (the catalogue reads 0.1259 against 0.0358).

This tool rebuilds the two scenes the way the catalogue runner builds them (a
0.02 mm collimated pencil along +z, a plane interface at z = 10 mm tilted by
the incidence angle about x, a one-pixel 500 mm absorbing detector at
z = 10.001 mm with the same tilt, ``max_depth = 2``) and traces them with the
interface's interaction and the detector intersection wrapped, so every hit
ray's state is recorded: the incident direction and normal, the Fresnel
quantities (``cos_i``, ``sin^2 t``, the radicand ``w = 1 - sin^2 t``, the
``cos_t`` the engine used, R), the branch taken, the outgoing direction and
origin, and the next bounce's intersection distances.

The Fresnel quantities are recomputed inside the wrapper from the very arrays
the engine is handed, with the same expressions and in the working dtype, and
the recomputation is checked bitwise against the direction the engine then
writes: ``replica_matches`` in the output is the fraction of hit rays whose
recomputed outgoing direction equals the engine's to the last bit. Anything
below 1.0 means the tool no longer mirrors the engine and its numbers must not
be read.

Usage::

    python -m benchmarks.nonsequential.float32_mechanisms            # both cases
    python -m benchmarks.nonsequential.float32_mechanisms --case critical --rays 20000
    python -m benchmarks.nonsequential.float32_mechanisms --json out.json

Not part of the pytest suite; the tests that pin the mechanisms are in
``tests/nonsequential/test_nsq_float32_fresnel.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
    _tol,
)
from optiland.nonsequential.backends import array_backend as _ab
from optiland.nonsequential.components import refractive as _refractive
from optiland.nonsequential.components.base import coordinate_magnitude

#: Catalogue index of N-BK7 at the d-line, pinned exactly as the catalogue does.
NBK7_N = 1.5168
#: The d-line wavelength every catalogue case uses [um].
LAMBDA_UM = 0.5876
#: The pencil the catalogue runner uses [mm].
APERTURE_MM = 0.02
#: The interface's and the detector's positions along z [mm], as the runner has them.
Z_INTERFACE = 10.0
Z_DETECTOR = 10.001

CRITICAL_DEG = math.degrees(math.asin(1.0 / NBK7_N))


def configure(backend: str, precision: str) -> None:
    """Switch the array backend before a scene is built (torch runs on the CPU)."""
    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
    be.set_precision(precision)


def build_scene(case: str, theta_deg: float, z_detector: float = Z_DETECTOR) -> NSQScene:
    """The catalogue runner's scene for one angle.

    Args:
        case: ``"external"`` (vacuum to N-BK7, the 89-degree reflectance
            case) or ``"critical"`` (N-BK7 to vacuum, the critical-angle
            case).
        theta_deg: Incidence angle, the tilt of the interface about x.
        z_detector: Where the detector plane crosses the z axis [mm].
    """
    glass = NSQMaterial(optiland_material=IdealMaterial(n=NBK7_N, k=0.0))
    front, back = (VACUUM, glass) if case == "external" else (glass, VACUUM)
    rx = math.radians(theta_deg)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(LAMBDA_UM),
            total_flux=1.0,
            aperture_radius=APERTURE_MM,
        ),
    )
    scene.add_component(
        "IF",
        RefractiveComponent(
            CoordinateSystem(z=Z_INTERFACE, rx=rx),
            _plane(),
            material_front=front,
            material_back=back,
        ),
    )
    scene.add_detector(
        "T",
        CoordinateSystem(z=z_detector, rx=rx),
        IrradianceDetectorConfig(
            width=500, height=500, num_pixels_x=1, num_pixels_y=1, splat="hard"
        ),
    )
    return scene


def _plane():
    from optiland.nonsequential.components.geometry.analytic.plane import (  # noqa: PLC0415
        PlaneGeometry,
    )

    return PlaneGeometry()


def _np(x: Any) -> np.ndarray:
    return np.asarray(to_numpy(x))


def _f64(x: Any) -> np.ndarray:
    return _np(x).astype(np.float64)


def clamped_refraction_cosine(sin2_t):
    """The refraction cosine as the engine formed it before the cure.

    TIR only for ``sin^2 t > 1``, and the radicand clamped to
    ``_tol.radicand_floor`` (1e-12 at float64, 5.37e-4 at float32). Kept so
    the tool reproduces the failure on the engine it was measured on, and so
    a test can put the old rule back as a control.
    """
    tir = sin2_t > 1.0
    cos_t = be.where(
        tir,
        be.zeros_like(sin2_t),
        be.maximum(1.0 - sin2_t, _tol.radicand_floor(be.ones_like(sin2_t))) ** 0.5,
    )
    return tir, cos_t


def _refraction_cosine(sin2_t):
    """Whichever rule the engine in use applies (the module attribute, so a patch is seen)."""
    rule = getattr(_refractive, "refraction_cosine", clamped_refraction_cosine)
    return rule(sin2_t)


def fresnel_replica(dirs, normals, n_geom, n_front, n_back):
    """The interface's Fresnel arithmetic, the same expressions in the working dtype.

    A copy of ``RefractiveComponent.interact`` from the incidence cosine to
    the refracted direction, kept expression for expression so that it rounds
    exactly as the engine does. Returns backend arrays.
    """
    dot = (dirs * normals).sum(axis=1)
    cos_i = be.abs(dot)
    dot_geom = (dirs * n_geom).sum(axis=1)
    entering_back = dot_geom > 0.0
    n1 = be.where(entering_back, n_front, n_back)
    n2 = be.where(entering_back, n_back, n_front)
    n_ratio = n1 / (n2 + _tol.tiny_for(n2))
    sin2_t = n_ratio**2 * (1.0 - cos_i**2)
    w = 1.0 - sin2_t
    tir, cos_t = _refraction_cosine(sin2_t)
    rs_denom = n1 * cos_i + n2 * cos_t
    rs = (n1 * cos_i - n2 * cos_t) / (rs_denom + _tol.tiny_for(rs_denom))
    rp_denom = n2 * cos_i + n1 * cos_t
    rp = (n2 * cos_i - n1 * cos_t) / (rp_denom + _tol.tiny_for(rp_denom))
    R = be.where(tir, be.ones_like(rs), 0.5 * (rs**2 + rp**2))
    raw_dot = (dirs * normals).sum(axis=1, keepdims=True)
    reflected = dirs - 2.0 * raw_dot * normals
    norms_r = (reflected * reflected).sum(axis=1, keepdims=True) ** 0.5
    reflected = reflected / (norms_r + _tol.tiny_for(norms_r))
    n_facing = be.where(raw_dot < 0, normals, -normals)
    cos_i_pos = be.abs(raw_dot)
    nr = n_ratio[:, None]
    refracted = nr * dirs + (nr * cos_i_pos - cos_t[:, None]) * n_facing
    norms_t = (refracted * refracted).sum(axis=1, keepdims=True) ** 0.5
    refracted = refracted / (norms_t + _tol.tiny_for(norms_t))
    return {
        "cos_i": cos_i,
        "sin2_t": sin2_t,
        "w": w,
        "tir": tir,
        "cos_t": cos_t,
        "R": R,
        "reflected": reflected,
        "refracted": refracted,
    }


@dataclass
class Record:
    """Everything the instrumented trace saw, as float64/int NumPy arrays."""

    interactions: list[dict[str, np.ndarray]] = field(default_factory=list)
    components: list[dict[str, np.ndarray]] = field(default_factory=list)
    detectors: list[dict[str, np.ndarray]] = field(default_factory=list)


@contextlib.contextmanager
def instrument(record: Record):
    """Wrap the interface interaction and the two intersection passes."""
    orig_interact = RefractiveComponent.interact
    orig_scene = _ab.ArrayBackend.intersect_scene
    orig_det = _ab.intersect_detectors

    def interact(self, rays, t, normals, hit_mask, rng, bsdf_ir, n_geom, sampling=None, forced_branch=None):
        ids = _np(rays.ray_id)
        bounce = _np(rays.bounce)
        dirs = be.stack([rays.L, rays.M, rays.N], axis=1)
        wl = rays.wavelength
        orig_interact(self, rays, t, normals, hit_mask, rng, bsdf_ir, n_geom, sampling, forced_branch)
        rep = fresnel_replica(dirs, normals, n_geom, self.material_front.n(wl), self.material_back.n(wl))
        out = be.stack([rays.L, rays.M, rays.N], axis=1)
        hm = _np(hit_mask).astype(bool)
        out_np = _np(out)
        refl_np = _np(rep["reflected"])
        refr_np = _np(rep["refracted"])
        is_refl = np.all(out_np == refl_np, axis=1)
        is_refr = np.all(out_np == refr_np, axis=1)
        pos = np.stack([_np(rays.x), _np(rays.y), _np(rays.z)], axis=1)
        entry = {
            "ray_id": ids[hm],
            "bounce": bounce[hm],
            "dir_in": _f64(dirs)[hm],
            "normal": _f64(normals)[hm],
            "n_geom": _f64(n_geom)[hm],
            "dir_out": out_np.astype(np.float64)[hm],
            "pos_out": pos.astype(np.float64)[hm],
            "mag_out": _f64(coordinate_magnitude(rays))[hm],
            "reflected": is_refl[hm],
            "transmitted": (is_refr & ~is_refl)[hm],
            "replica_ok": (is_refl | is_refr)[hm],
        }
        for key in ("cos_i", "sin2_t", "w", "cos_t", "R"):
            entry[key] = _f64(rep[key])[hm]
        entry["tir"] = _np(rep["tir"]).astype(bool)[hm]
        entry["dtype"] = str(_np(rays.x).dtype)
        record.interactions.append(entry)

    def intersect_scene(self, rays, surfaces):
        res = orig_scene(self, rays, surfaces)
        record.components.append(
            {"ray_id": _np(rays.ray_id), "bounce": _np(rays.bounce), "alive": _np(rays.alive).astype(bool),
             "t": _f64(res[0]), "idx": _np(res[2])}
        )
        return res

    def intersect_detectors(rays, detectors):
        res = orig_det(rays, detectors)
        record.detectors.append(
            {"ray_id": _np(rays.ray_id), "bounce": _np(rays.bounce), "alive": _np(rays.alive).astype(bool),
             "t": _f64(res[0]), "idx": _np(res[1])}
        )
        return res

    RefractiveComponent.interact = interact
    _ab.ArrayBackend.intersect_scene = intersect_scene
    _ab.intersect_detectors = intersect_detectors
    try:
        yield record
    finally:
        RefractiveComponent.interact = orig_interact
        _ab.ArrayBackend.intersect_scene = orig_scene
        _ab.intersect_detectors = orig_det


def trace(case: str, theta_deg: float, backend: str, precision: str, rays: int, seed: int,
          z_detector: float = Z_DETECTOR) -> tuple[Record, Any]:
    """One instrumented trace of the catalogue scene."""
    configure(backend, precision)
    scene = build_scene(case, theta_deg, z_detector)
    record = Record()
    with instrument(record):
        result = scene.trace(num_rays=rays, seed=seed, max_depth=2)
    return record, result


def _concat(entries: list[dict[str, Any]], keep) -> dict[str, Any]:
    """Concatenate per-call records (one per batch) after masking each with ``keep``."""
    out: dict[str, Any] = {}
    for e in entries:
        m = keep(e)
        for k, v in e.items():
            if isinstance(v, np.ndarray):
                out.setdefault(k, []).append(v[m])
            else:
                out[k] = v
    return {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in out.items()}


def _events(record: Record, bounce: int) -> dict[str, Any]:
    """Every interface interaction at the given bounce, sorted by ray id."""
    ev = _concat(record.interactions, lambda e: e["bounce"] == bounce)
    if not ev:
        return {}
    order = np.argsort(ev["ray_id"])
    return {k: (v[order] if isinstance(v, np.ndarray) else v) for k, v in ev.items()}


def _next_hits(record: Record, bounce: int) -> dict[str, np.ndarray]:
    """The intersection distances of the pass whose live rays have ``bounce`` hits, per ray id."""
    comp = _concat(record.components, lambda e: e["alive"] & (e["bounce"] == bounce))
    det = _concat(record.detectors, lambda e: e["alive"] & (e["bounce"] == bounce))
    if not comp or comp["ray_id"].size == 0:
        return {}
    oc = np.argsort(comp["ray_id"])
    od = np.argsort(det["ray_id"])
    if not np.array_equal(comp["ray_id"][oc], det["ray_id"][od]):
        raise RuntimeError("component and detector passes saw different rays")
    return {"ray_id": comp["ray_id"][oc], "t_comp": comp["t"][oc], "t_det": det["t"][od]}


def summarise(case: str, theta_deg: float, rays: int, seed: int, sample: int = 6,
              z_detector: float = Z_DETECTOR) -> dict[str, Any]:
    """The same rays at float64 (numpy and torch) and float32 (torch), side by side."""
    legs = {}
    for backend, precision in (("numpy", "float64"), ("torch", "float64"), ("torch", "float32")):
        record, result = trace(case, theta_deg, backend, precision, rays, seed, z_detector)
        ev = _events(record, 0)
        nxt = _next_hits(record, 1)
        legs[f"{backend}/{precision}"] = (record, result, ev, nxt)

    out: dict[str, Any] = {"case": case, "theta_deg": theta_deg, "rays": rays, "seed": seed,
                           "z_detector": z_detector, "legs": {}}
    for name, (record, result, ev, nxt) in legs.items():
        n_hit = ev["ray_id"].size
        detected = float(result.total_flux_detected / result.total_flux_in)
        leg = {
            "detected_fraction": detected,
            "hit_rays": int(n_hit),
            "replica_matches": float(ev["replica_ok"].mean()) if n_hit else float("nan"),
            "transmitted_fraction": float(ev["transmitted"].mean()) if n_hit else float("nan"),
            "tir_fraction": float(ev["tir"].mean()) if n_hit else float("nan"),
            "mean_R": float(ev["R"].mean()) if n_hit else float("nan"),
            "w_min": float(ev["w"].min()), "w_max": float(ev["w"].max()),
            "cos_t_min": float(ev["cos_t"].min()), "cos_t_max": float(ev["cos_t"].max()),
            "dtype": ev["dtype"],
        }
        if nxt:
            tr_ids = ev["ray_id"][ev["transmitted"]]
            sel = np.isin(nxt["ray_id"], tr_ids)
            t_det = nxt["t_det"][sel]
            t_comp = nxt["t_comp"][sel]
            leg["transmitted_seen_by_detector"] = float(np.isfinite(t_det).mean()) if sel.any() else float("nan")
            leg["transmitted_rehit_interface"] = float(np.isfinite(t_comp).mean()) if sel.any() else float("nan")
            rf_sel = np.isin(nxt["ray_id"], ev["ray_id"][ev["reflected"]])
            leg["reflected_rehit_interface"] = (
                float(np.isfinite(nxt["t_comp"][rf_sel]).mean()) if rf_sel.any() else float("nan")
            )
        out["legs"][name] = leg
    # Per-ray table: the first `sample` rays by id, float64 and float32 side by side.
    ev64 = legs["torch/float64"][2]
    ev32 = legs["torch/float32"][2]
    common = np.intersect1d(ev64["ray_id"], ev32["ray_id"])[:sample]
    rows = []
    for rid in common:
        a = int(np.searchsorted(ev64["ray_id"], rid))
        b = int(np.searchsorted(ev32["ray_id"], rid))
        rows.append({
            "ray_id": int(rid),
            "dir_in_f64": ev64["dir_in"][a].tolist(), "dir_in_f32": ev32["dir_in"][b].tolist(),
            "cos_i_f64": float(ev64["cos_i"][a]), "cos_i_f32": float(ev32["cos_i"][b]),
            "w_f64": float(ev64["w"][a]), "w_f32": float(ev32["w"][b]),
            "cos_t_f64": float(ev64["cos_t"][a]), "cos_t_f32": float(ev32["cos_t"][b]),
            "R_f64": float(ev64["R"][a]), "R_f32": float(ev32["R"][b]),
            "branch_f64": "T" if ev64["transmitted"][a] else "R",
            "branch_f32": "T" if ev32["transmitted"][b] else "R",
            "dir_out_f64": ev64["dir_out"][a].tolist(), "dir_out_f32": ev32["dir_out"][b].tolist(),
        })
    out["rays_side_by_side"] = rows
    # The detector gap at the hit, in the working dtype's own step.
    n = np.array([0.0, -math.sin(math.radians(theta_deg)), math.cos(math.radians(theta_deg))])
    for name in ("torch/float64", "torch/float32"):
        ev = legs[name][2]
        dt = np.float32 if name.endswith("float32") else np.float64
        tr = ev["transmitted"]
        if not tr.any():
            continue
        p = ev["pos_out"][tr]
        d = ev["dir_out"][tr]
        gap = (np.array([0.0, 0.0, z_detector]) - p) @ n  # perpendicular, origin to detector
        t_det = gap / (d @ n)
        ulp = np.spacing(np.maximum(ev["mag_out"][tr], 1.0).astype(dt)).astype(np.float64)
        thr = _tol.DEFAULT_ACCEPT_K * ulp
        out["legs"][name]["detector_gap_perp_ulps_median"] = float(np.median(gap / ulp))
        out["legs"][name]["detector_t_over_threshold_median"] = float(np.median(t_det / thr))
        out["legs"][name]["detector_t_below_threshold_fraction"] = float((t_det <= thr).mean())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--case", choices=["external", "critical", "both"], default="both")
    ap.add_argument("--rays", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=10_001)
    ap.add_argument("--json", default=None, help="write every summary to this file")
    args = ap.parse_args(argv)

    plan = []
    if args.case in ("critical", "both"):
        plan += [("critical", CRITICAL_DEG - 1e-3), ("critical", CRITICAL_DEG)]
    if args.case in ("external", "both"):
        plan += [("external", a) for a in (80.0, 85.0, 88.0, 88.5, 89.0)]
    results = []
    for case, theta in plan:
        s = summarise(case, theta, args.rays, args.seed)
        results.append(s)
        print(f"\n== {case} {theta:.9f} deg, {args.rays} rays, seed {args.seed}")
        for name, leg in s["legs"].items():
            print(f"  {name:14s} " + ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in leg.items()))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
