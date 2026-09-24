"""Documented OSLO surface equations mapped to existing Optiland geometries."""

from __future__ import annotations

import math
import re
from typing import Any


def surface_geometry(data: dict[str, Any], scale: float) -> dict[str, Any]:
    """Return surface kwargs in millimeters (Program Reference pp. 70–74)."""
    radius = data.get("RD", math.inf) or math.inf
    params = {
        "surface_type": "standard",
        "radius": radius * scale,
        "conic": data.get("CC", 0.0),
    }
    kind = data.get("ASP", "ADO")
    general = {int(k[2:]): v for k, v in data.items() if re.fullmatch(r"AS\d+", k)}
    if general and max(general) > 256:
        raise ValueError("OSLO asphere coefficient index exceeds supported limit 256")
    if kind == "ASX":
        order = int((math.sqrt(8 * max(general, default=0) + 1) - 1) // 2)
        coefficients = [[0.0] * (order + 1) for _ in range(order + 1)]
        for i, value in general.items():
            degree = int((math.sqrt(8 * i + 1) - 1) // 2)
            y_power = i - degree * (degree + 1) // 2
            coefficients[degree - y_power][y_power] = value * scale ** (1 - degree)
        params.update(surface_type="polynomial", coefficients=coefficients)
    elif kind in {"ASR", "ARA"}:
        if general.get(0, 0.0):
            raise ValueError(f"OSLO {kind} with nonzero AS0 requires an offset sag")
        step = 2 if kind == "ASR" else 1
        coefficients = [
            general.get(i, 0.0) * scale ** (1 - step * i)
            for i in range(1, max(general, default=1) + 1)
        ]
        params.update(
            surface_type="even_asphere" if step == 2 else "odd_asphere",
            coefficients=coefficients,
        )
    elif any(k in data for k in ("AD", "AE", "AF", "AG")):
        coefficients = [0.0] + [
            data.get(k, 0.0) * scale ** (1 - power)
            for k, power in zip(("AD", "AE", "AF", "AG"), (4, 6, 8, 10), strict=True)
        ]
        params.update(surface_type="even_asphere", coefficients=coefficients)
    if "CVX" in data:
        if kind not in {"ADO", "ASR"}:
            raise ValueError("OSLO toric requires an even YZ asphere profile")
        cvx = data["CVX"]
        params.update(
            surface_type="toroidal",
            radius_y=radius * scale,
            radius_x=scale / cvx if cvx else math.inf,
            toroidal_coeffs_poly_y=params.pop("coefficients", []),
        )
    return params
