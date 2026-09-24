"""Apply OSLO's static sequential solves through Optiland's solve APIs."""

from __future__ import annotations

import math

from scipy.optimize import root_scalar

import optiland.backend as be
from optiland.paraxial_path import require_global_z_geometry
from optiland.solves.factory import SolveFactory

SOLVES = {
    "PU": "marginal_ray_angle_curvature",
    "PUC": "chief_ray_angle_curvature",
    "PY": "marginal_ray_height_thickness",
    "PYC": "chief_ray_height_thickness",
    "EC": None,
}


def apply_solve(optic, index: int, command: str, value: float, scale: float) -> None:
    """Solve one surface, checking that native no-op cases did not hide failure."""
    if not 1 <= index < len(optic.surfaces) - 1:
        raise ValueError(f"{command} requires an interior optical surface")
    if optic.obj_space_telecentric and command in {"PYC", "PUC"}:
        # Native paraxial chief rays target the stop center, whereas TELE's
        # real chief rays launch parallel to the object-space axis.
        raise ValueError("OSLO telecentric chief-ray solves are not mapped")
    require_global_z_geometry(optic.surfaces, f"OSLO {command}")
    if command == "EC":
        height = value * scale
        if height < 0:
            raise ValueError("EC requires a nonnegative edge height")
        x, y = be.array([0.0]), be.array([height])
        sag1 = optic.surfaces[index].geometry.sag(x, y)
        sag2 = optic.surfaces[index + 1].geometry.sag(x, y)
        thickness = float((sag1 - sag2).item())
        if not math.isfinite(thickness):
            raise ValueError("EC edge is outside the surface's sag domain")
        optic.updater.set_thickness(thickness, index)
        return
    # OSLO paraxial analysis deliberately ignores tilts/decenters (Program
    # Reference p. 251), as native scalar traces do on straight paths.
    # https://lambdares.com/hubfs/Support/support/oslo/oslo_releases/OSLOProgramReference.pdf#page=265
    is_height = command in {"PY", "PYC"}
    target_index = index + 1 if is_height else index
    target = value * scale if is_height else value
    solve = SolveFactory.create_solve(optic, SOLVES[command], target_index, target)
    solve.apply()
    ray = (
        optic.paraxial.chief_ray
        if command.endswith("C")
        else optic.paraxial.marginal_ray
    )
    if command == "PUC":
        # The native chief-angle solve stops at 1e-5. Refine its solution to
        # preserve the precision of saved OSLO targets without changing that API.
        curvature = 1 / float(optic.surfaces[index].geometry.radius)

        def residual(c):
            optic.updater.set_radius(1 / c if c else math.inf, index)
            return float(ray()[1][index].item()) - target

        if abs(residual(curvature)) > 1e-9:
            result = root_scalar(
                residual,
                x0=curvature,
                x1=curvature + max(abs(curvature) * 1e-4, 1e-8),
                xtol=1e-12,
            )
            if not result.converged:
                raise ValueError("PUC curvature refinement did not converge")
            residual(result.root)
    check_solve(optic, index, command, value, scale)
    if is_height:
        # Native height solves move coordinates directly. Keep the prescription
        # thickness consistent for downstream pickups, updater calls and export.
        optic.surfaces[index].thickness = float(
            (
                optic.surfaces[index + 1].geometry.cs.z
                - optic.surfaces[index].geometry.cs.z
            ).item()
        )


def check_solve(optic, index: int, command: str, value: float, scale: float) -> None:
    """Verify a solve target against the current, fully rebuilt prescription."""
    absolute_tolerance = 1e-9
    if command == "EC":
        x, y = be.array([0.0]), be.array([value * scale])
        first = optic.surfaces[index].geometry
        second = optic.surfaces[index + 1].geometry
        # EC requires physical contact (Program Reference p. 45), not just
        # a TH equal to the difference of two unpositioned local sags. Express
        # the first surface's edge point in the second surface's coordinates
        # and check that it lies on that surface after all pickups/transforms.
        # https://lambdares.com/hubfs/Support/support/oslo/oslo_releases/OSLOProgramReference.pdf#page=59
        edge = be.array([0.0, value * scale, float(first.sag(x, y).item())])
        origin1, rotation1 = first.cs.get_effective_transform()
        origin2, rotation2 = second.cs.get_effective_transform()
        local_edge = rotation2.T @ (origin1 + rotation1 @ edge - origin2)
        target = 0.0
        actual = float(
            (local_edge[2] - second.sag(local_edge[0], local_edge[1])).item()
        )
        # Transforming and subtracting positioned points incurs roundoff in
        # the active precision. Allow a small error budget at that coordinate
        # scale, including cancellation away from the global origin.
        epsilon = be.finfo(be.to_numpy(local_edge).dtype).eps
        magnitude = max(
            abs(float(component.item()))
            for vector in (origin1, origin2, edge)
            for component in vector
        )
        absolute_tolerance = max(absolute_tolerance, 8 * epsilon * magnitude)
    else:
        is_height = command in {"PY", "PYC"}
        ray = (
            optic.paraxial.chief_ray
            if command.endswith("C")
            else optic.paraxial.marginal_ray
        )
        target = value * scale if is_height else value
        actual = float(
            ray()[0 if is_height else 1][index + 1 if is_height else index].item()
        )
    if not math.isfinite(actual) or not math.isclose(
        actual, target, rel_tol=1e-7, abs_tol=absolute_tolerance
    ):
        raise ValueError(
            f"{command} at surface {index}: target {target:g} could not be reached "
            f"(got {actual:g})"
        )
