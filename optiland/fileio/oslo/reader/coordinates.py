"""OSLO intrinsic rotations and local/global coordinate references."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from optiland.fileio.oslo.constants import (
    OBJECT_INFINITY_THRESHOLD,
    THICKNESS_INFINITY_THRESHOLD,
)


def reference_index(value: int, current: int) -> int:
    """Resolve an OSLO negative relative or nonnegative absolute surface index."""
    return current + value if value < 0 else value


def surface_coordinates(
    surfaces: dict[int, dict[str, Any]], scale: float
) -> dict[int, dict[str, float]]:
    """Build absolute poses without passing incompatible thickness/z arguments.

    OSLO rotations are intrinsic, about -x, -y and +z. DT=1 translates then
    rotates XYZ; DT=-1 rotates ZYX then translates. See Optics Reference pp.
    142–145 and Program Reference pp. 65–68.
    """
    frames = {}
    bases = {}
    result = {}
    last_index = max(surfaces, default=0)
    next_position, next_rotation = np.zeros(3), np.eye(3)
    for index, data in sorted(surfaces.items()):
        base_position, base_rotation = next_position.copy(), next_rotation.copy()
        if index == 0:
            distance = data.get("TH", 0.0)
            base_position[2] = (
                -distance * scale
                if abs(distance) < OBJECT_INFINITY_THRESHOLD
                else -math.copysign(math.inf, distance)
            )
        elif index == 1:
            base_position, base_rotation = np.zeros(3), np.eye(3)
        if "GC" in data:
            ref = reference_index(int(data["GC"]), index)
            if ref not in frames:
                raise ValueError(
                    f"OSLO GC at surface {index} requires a preceding surface"
                )
            base_position, base_rotation = (v.copy() for v in frames[ref])
        if index > 0 and not np.all(np.isfinite(base_position)):
            raise ValueError("OSLO coordinate reference must have a finite position")
        bases[index] = (base_position, base_rotation)
        order = data.get("DT", 1)
        if order not in {1, -1}:
            raise ValueError("OSLO DT must be 1 (decenter/tilt) or -1 (tilt/decenter)")
        angles = [-data.get("TLA", 0.0), -data.get("TLB", 0.0), data.get("TLC", 0.0)]
        rotation = Rotation.from_euler(
            "XYZ" if order == 1 else "ZYX",
            angles if order == 1 else angles[::-1],
            degrees=True,
        ).as_matrix()
        shift = np.array([data.get(k, 0.0) for k in ("DCX", "DCY", "DCZ")]) * scale
        pivot = np.array([data.get(k, 0.0) for k in ("TOX", "TOY", "TOZ")]) * scale
        shift = (shift if order == 1 else rotation @ shift) + pivot - rotation @ pivot
        position = base_position + base_rotation @ shift
        rotation = base_rotation @ rotation
        frames[index] = (position, rotation)
        result[index] = dict(zip(("x", "y", "z"), position.tolist(), strict=True))
        result[index].update(
            zip(
                ("rx", "ry", "rz"),
                Rotation.from_matrix(rotation).as_euler("xyz").tolist(),
                strict=True,
            )
        )
        next_position, next_rotation = position, rotation
        if "RCO" in data:
            ref = reference_index(int(data["RCO"]), index)
            if ref not in frames:
                raise ValueError(
                    f"OSLO RCO at surface {index} references an unavailable surface"
                )
            next_position, next_rotation = bases[index] if ref == index else frames[ref]
        if data.get("BEN"):
            if "GC" in data or "RCO" in data or (angles[0] and angles[1]) or angles[2]:
                raise ValueError(
                    "OSLO BEN currently supports single-axis local mirror tilts"
                )
            if data.get("material") != "RFL":
                raise ValueError(
                    "OSLO BEN requires a mapped reflecting surface (RFL/RFH)"
                )
            extra = Rotation.from_euler("XYZ", angles, degrees=True).as_matrix()
            next_rotation = rotation @ extra
        distance = data.get("TH", 0.0)
        if 0 < index < last_index and abs(distance) >= THICKNESS_INFINITY_THRESHOLD:
            raise ValueError(
                "OSLO infinite thickness with coordinate transforms is not mapped"
            )
        if index != 0 and abs(distance) < THICKNESS_INFINITY_THRESHOLD:
            next_position = next_position + next_rotation[:, 2] * distance * scale
    return result
