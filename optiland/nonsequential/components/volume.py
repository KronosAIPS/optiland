"""Volume -- a closed, outward-oriented solid built from boundary surfaces.

Medium sidedness (which material a ray is entering) is fixed geometrically
per-surface -- see ``RefractiveComponent.interact`` -- and stays the sole
source of truth for n1/n2. ``Volume`` is a separate, independent check: a
compound component's boundary surfaces are supposed to form a genuinely
closed solid, and nothing else checks that. It validates a boundary list at
construction time and raises loudly (``NonWatertightVolumeError``) if the
surfaces do not actually close up or are inconsistently oriented, rather
than letting a silent gap leak flux at trace time.

A ray-level medium stack (``NSQRayBundle.medium_stack``/``medium_depth``,
pushed/popped by ``RefractiveComponent.interact`` on every transmitted ray)
runs alongside this as a runtime cross-check: it does not feed back into
n1/n2 either, but a pop on an empty stack is counted in
``Diagnostics.medium_stack_underflows`` as a likely geometry defect. This
``Volume`` boundary list is not yet wired into that stack via ids (there is
no ``SceneIR``-level ``VolumeIR`` population), so the stack's push/pop
identity currently comes from ``NSQMaterial`` object identity
(``optiland.nonsequential.materials.nsq_material.medium_stack_id``), not
from a volume registry.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential._utils import as_float
from optiland.nonsequential.components.base import BaseComponent, _get_transform

if TYPE_CHECKING:
    from optiland.nonsequential.materials.nsq_material import NSQMaterial

# The rim-coincidence floor [mm]: two rims closer than one nanometre are one
# rim. A physical construction tolerance (the original specification's
# number), dtype-independent by intent (chapter 08, R-08-7's exception for a
# physical threshold). The numerical part of the tolerance is not this
# constant: it is the rounding bound of the working precision at the rims'
# coordinate scale, derived in :func:`_placement_error_bound` and
# :func:`_check_watertight` (KronosNSRT issue 80). The tolerance used is the
# larger of the two.
WATERTIGHT_TOL = 1e-6

_RIM_SAMPLES = 64
_PARITY_DIRECTIONS = 8
_PARITY_SEED = 0
_PARITY_MAX_BOUNCES = 64
_PARITY_EPSILON = 1e-6


class NonWatertightVolumeError(Exception):
    """A Volume's boundary surfaces do not form a closed, consistently
    outward-oriented solid.

    Raised at :class:`Volume` construction, never as a warning: a leak in
    the boundary lets rays enter or exit a solid without the medium stack
    (or, in this revamp, the per-surface geometric sidedness check)
    noticing, which is exactly the class of silent-wrong-answer failure
    this validation exists to prevent.
    """


def _rectangle_perimeter(half_width: float, half_height: float, n: int) -> np.ndarray:
    """Sample ``n`` points roughly evenly around a rectangle's perimeter.

    Args:
        half_width: Half-width along local x [mm].
        half_height: Half-height along local y [mm].
        n: Number of points to sample.

    Returns:
        (n, 3) array of local-frame points at z=0.
    """
    perim = 4.0 * (half_width + half_height)
    if perim <= 0.0:
        return np.zeros((n, 3))
    s = (np.arange(n) / n) * perim
    x = np.zeros(n)
    y = np.zeros(n)
    # Walk the perimeter starting at (+hw, -hh), going counter-clockwise.
    edges = [
        (2 * half_width, (1.0, 0.0), (-half_width, -half_height)),
        (2 * half_height, (0.0, 1.0), (half_width, -half_height)),
        (2 * half_width, (-1.0, 0.0), (half_width, half_height)),
        (2 * half_height, (0.0, -1.0), (-half_width, half_height)),
    ]
    remaining = s.copy()
    start = 0.0
    for length, (dx, dy), (ox, oy) in edges:
        on_edge = (remaining >= start) & (remaining < start + length)
        local_s = remaining[on_edge] - start
        x[on_edge] = ox + dx * local_s
        y[on_edge] = oy + dy * local_s
        start += length
    return np.stack([x, y, np.zeros(n)], axis=1)


def _rim_points(
    component: BaseComponent, n_samples: int = _RIM_SAMPLES
) -> np.ndarray | None:
    """Sample points along a component's aperture rim, in global coordinates.

    The global points are :func:`_rim_local_points` placed by the
    component's effective transform. ``None`` when the geometry has no
    finite rim.
    """
    local_pts = _rim_local_points(component, n_samples)
    if local_pts is None:
        return None
    translation, rotation = _get_transform(component.cs)
    return local_pts @ rotation.T + translation


def _rim_local_points(
    component: BaseComponent, n_samples: int = _RIM_SAMPLES
) -> np.ndarray | None:
    """Sample points along a component's aperture rim, in its local frame.

    Supports the analytic geometries the compound builders actually use
    (conic, finite plane, annulus, frustum). Geometries with no finite open
    edge (an infinite plane, a full sphere, a mesh) return ``None`` -- there
    is nothing for a neighbouring surface to meet, so watertightness
    contributes nothing to check for them.

    Args:
        component: The boundary surface to sample.
        n_samples: Points per rim loop.

    Returns:
        (n_samples * num_loops, 3) local-frame points, or ``None``.
    """
    from optiland.nonsequential.components.geometry.analytic.annulus import (  # noqa: PLC0415
        AnnularPlaneGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.conic import (  # noqa: PLC0415
        ConicGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.frustum import (  # noqa: PLC0415
        CylindricalFrustumGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.plane import (  # noqa: PLC0415
        FinitePlaneGeometry,
    )
    from optiland.nonsequential.components.lens import _sag_at_rim  # noqa: PLC0415

    from optiland.nonsequential.components.geometry.analytic.asphere import (  # noqa: PLC0415
        _AsphereGeometry,
    )

    geom = component.geometry
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    loops: list[np.ndarray] = []

    if isinstance(geom, _AsphereGeometry):
        # The rim of an asphere sits at its full sag (conic plus polynomial).
        r = as_float(geom.aperture_radius)
        z = geom.rim_sag()
        loops.append(
            np.stack(
                [r * np.cos(theta), r * np.sin(theta), np.full(n_samples, z)], axis=1
            )
        )
    elif isinstance(geom, ConicGeometry):
        r = as_float(geom.aperture_radius)
        z = _sag_at_rim(as_float(geom.radius), as_float(geom.conic), r)
        loops.append(
            np.stack(
                [r * np.cos(theta), r * np.sin(theta), np.full(n_samples, z)], axis=1
            )
        )
    elif isinstance(geom, FinitePlaneGeometry):
        if geom.aperture_radius is not None:
            r = as_float(geom.aperture_radius)
            loops.append(
                np.stack(
                    [r * np.cos(theta), r * np.sin(theta), np.zeros(n_samples)], axis=1
                )
            )
        else:
            hw = as_float(geom.width) / 2.0
            hh = as_float(geom.height) / 2.0
            loops.append(_rectangle_perimeter(hw, hh, n_samples))
    elif isinstance(geom, AnnularPlaneGeometry):
        ri = as_float(geom.inner_radius)
        ro = as_float(geom.outer_radius)
        z = as_float(geom.z_offset)
        loops.append(
            np.stack(
                [ri * np.cos(theta), ri * np.sin(theta), np.full(n_samples, z)], axis=1
            )
        )
        loops.append(
            np.stack(
                [ro * np.cos(theta), ro * np.sin(theta), np.full(n_samples, z)], axis=1
            )
        )
    elif isinstance(geom, CylindricalFrustumGeometry):
        rf, zf = as_float(geom.r_front), as_float(geom.z_front)
        rb, zb = as_float(geom.r_back), as_float(geom.z_back)
        loops.append(
            np.stack(
                [rf * np.cos(theta), rf * np.sin(theta), np.full(n_samples, zf)], axis=1
            )
        )
        loops.append(
            np.stack(
                [rb * np.cos(theta), rb * np.sin(theta), np.full(n_samples, zb)], axis=1
            )
        )
    else:
        # Infinite plane, sphere, mesh: no finite rim supported yet.
        return None

    return np.concatenate(loops, axis=0)


def _unit_roundoff(cs: object) -> float:
    """The unit roundoff u of the precision a placement is computed in.

    Read from the live dtype of the effective transform the coordinate
    system produces (float32 under ``be.set_precision("float32")`` on the
    torch backend, float64 otherwise): u = 2**-24 or 2**-53.
    """
    from optiland.backend.utils import to_numpy  # noqa: PLC0415

    t_be, _ = cs.get_effective_transform()
    dtype = np.asarray(to_numpy(t_be)).dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)
    return float(np.finfo(dtype).eps) / 2.0


def _placement_error_bound(cs: object, u: float) -> tuple[float, float]:
    """First-order bounds on the rounding of a placement computed in precision u.

    ``CoordinateSystem.get_effective_transform`` builds the rotation R and
    translation t of a surface in the working precision: the six values are
    stored in it, R = Rz @ Ry @ Rx from their sines and cosines, and each
    ``reference_cs`` link composes t = t_ref + R_ref @ tau and
    R = R_ref @ R_local. The rim points themselves are float64 arithmetic on
    float64 parameters; only R and t carry the working precision. Counted
    under the standard model fl(a op b) = (a op b)(1 + d), |d| <= u, an inner
    product of length 3 within gamma_3 = 3u of the sum of the absolute terms,
    and sine and cosine within one ulp (2u) of the stored angle's value:

    * Local rotation, any angle nonzero: each entry of Rz @ Ry @ Rx is at most
      two terms, each a product of at most three sines or cosines (the other
      factors are exact zeros and ones, whose products and sums are exact);
      the absolute terms sum to at most 1 (Cauchy-Schwarz on unit rows and
      columns). Three function values at 2u each (6u), two multiplications
      (2u), one addition (1u), and the angles' storage, at most u|theta| per
      factor, summed over two terms (2 Theta u, Theta = |rx| + |ry| + |rz|):
      e_R,local = (9 + 2 Theta) u per entry. All angles zero: R is exactly the
      identity, e_R,local = 0.
    * Root placement: t is the stored offset, e_t = u max|tau_i|.
    * A link: R = R_ref @ R_local has entry error at most
      sqrt(3) (e_R,ref + e_R,local) + 3u (a column or row of a rotation has a
      1-norm at most sqrt(3)), or exactly e_R,ref when R_local is the identity.
      t_i = t_ref,i + sum_j R_ref,ij tau_j has error at most
      e_t,ref + (e_R,ref + u + 3u) ||tau||_1 + u (|t_ref|_inf + ||tau||_1):
      the reference rotation's error, the offset's storage (u), the inner
      product (3u) and the final addition (u at the sum's magnitude); a zero
      offset adds nothing.

    Args:
        cs: The surface's coordinate system (a reference chain allowed).
        u: Unit roundoff of the working precision.

    Returns:
        ``(e_R, e_t)``: the bound on every entry of R (dimensionless) and on
        every component of t [mm].
    """
    angles = [abs(as_float(cs.rx)), abs(as_float(cs.ry)), abs(as_float(cs.rz))]
    tau = np.array([as_float(cs.x), as_float(cs.y), as_float(cs.z)])
    theta = sum(angles)
    e_r_local = 0.0 if theta == 0.0 else (9.0 + 2.0 * theta) * u
    if cs.reference_cs is None:
        return e_r_local, u * float(np.max(np.abs(tau)))
    e_r_ref, e_t_ref = _placement_error_bound(cs.reference_cs, u)
    if theta == 0.0:
        e_r = e_r_ref
    else:
        e_r = np.sqrt(3.0) * (e_r_ref + e_r_local) + 3.0 * u
    tau_1 = float(np.sum(np.abs(tau)))
    if tau_1 == 0.0:
        return e_r, e_t_ref
    t_ref = _get_transform(cs.reference_cs)[0]
    t_ref_inf = float(np.max(np.abs(t_ref)))
    e_t = e_t_ref + (e_r_ref + 4.0 * u) * tau_1 + u * (t_ref_inf + tau_1)
    return e_r, e_t


def _rim_error_bound(component: BaseComponent, local_pts: np.ndarray) -> float:
    """Bound on each global coordinate's rounding of a component's rim points [mm].

    A global rim coordinate is g_i = sum_j R_ij p_j + t_i with p the float64
    local rim point, so its error is at most e_R ||p||_1 + e_t
    (:func:`_placement_error_bound`), taken at the rim's largest ||p||_1. The
    float64 arithmetic of p and of the product itself rounds at 2**-53 of
    the scale, below the floor ``WATERTIGHT_TOL`` by orders of magnitude at
    any scene size under 10**6 mm, and is not counted.
    """
    u = _unit_roundoff(component.cs)
    e_r, e_t = _placement_error_bound(component.cs, u)
    p_1 = float(np.max(np.sum(np.abs(local_pts), axis=1))) if len(local_pts) else 0.0
    return e_r * p_1 + e_t


def _check_watertight(
    boundary: list[BaseComponent], tol: float = WATERTIGHT_TOL
) -> np.ndarray | None:
    """Verify every boundary surface's rim is met by a neighbour's rim.

    The tolerance for surface a's rim is the larger of the floor ``tol`` and
    sqrt(3) (E_a + max_b E_b), with E the per-coordinate rounding bound of
    each surface's rim points (:func:`_rim_error_bound`) and b over the other
    surfaces: two rims that coincide in exact arithmetic are at most that far
    apart once both are placed in the working precision (sqrt(3) turns the
    per-coordinate bound into a Euclidean distance). In float64 the second
    term is about 1e-13 mm at a 100 mm scale, so the floor decides and the
    verdicts are those of the floor alone; in float32 it is about 6e-5 mm for
    a lens 50 mm from the origin, where float32's own spacing is 3.8e-6 mm
    (KronosNSRT issue 80).

    Args:
        boundary: The volume's boundary surfaces.
        tol: The floor of the allowed gap [mm].

    Returns:
        All sampled rim points (for reuse as a centroid estimate), or
        ``None`` if no surface in ``boundary`` has a finite rim.

    Raises:
        NonWatertightVolumeError: If any rim point is farther than its
            tolerance from every other surface's rim.
    """
    rims = []
    for comp in boundary:
        local_pts = _rim_local_points(comp)
        if local_pts is None:
            continue
        translation, rotation = _get_transform(comp.cs)
        pts = local_pts @ rotation.T + translation
        rims.append((comp, pts, _rim_error_bound(comp, local_pts)))
    if not rims:
        return None
    if len(rims) == 1:
        return rims[0][1]

    for i, (comp_i, pts_i, err_i) in enumerate(rims):
        other_pts = np.concatenate(
            [p for j, (_, p, _) in enumerate(rims) if j != i], axis=0
        )
        err_other = max(e for j, (_, _, e) in enumerate(rims) if j != i)
        tol_i = max(tol, float(np.sqrt(3.0)) * (err_i + err_other))
        # (n_i, n_other) pairwise distances -- rim samples are small (a few
        # hundred points across a handful of surfaces), so this is cheap.
        diff = pts_i[:, None, :] - other_pts[None, :, :]
        dists = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
        worst = float(dists.max())
        if worst > tol_i:
            raise NonWatertightVolumeError(
                f"Volume boundary is not watertight: surface "
                f"'{comp_i.name or type(comp_i).__name__}' has a rim point "
                f"{worst:.3g} mm from the nearest point on any other boundary "
                f"surface (tolerance {tol_i:.2e} mm). Check that neighbouring "
                f"surfaces' aperture radii and rim geometry agree."
            )
    return np.concatenate([p for _, p, _ in rims], axis=0)


class _DetachedProxy:
    """A minimal (cs, geometry) pair usable with ``BaseComponent.intersect``.

    Not a real component -- just enough duck-typed surface for
    ``intersect()`` (which only reads ``self.cs``/``self.geometry``) to
    work against a fully detached, plain-float geometry clone.
    """

    def __init__(self, cs: object, geometry: object) -> None:
        self.cs = cs
        self.geometry = geometry

    intersect = BaseComponent.intersect


def _detached_cs(cs: object) -> object:
    """Return a plain-float clone of a ``CoordinateSystem``.

    A differentiable scene may give x/y/z/rx/ry/rz live ``torch.Tensor``
    values (position/tilt are not currently differentiable NSQ parameters,
    but the ``CoordinateSystem`` type itself does not forbid it). Even a
    numpy-backend computation cannot touch a tensor that requires grad
    without detaching it first -- ``be.cos(rx)`` fails exactly like any
    other numpy ufunc would. Recurses through ``reference_cs`` chains, the
    same nesting :mod:`optiland.nonsequential.serialization` already
    detaches for JSON export.

    Args:
        cs: A live ``CoordinateSystem``.

    Returns:
        A new ``CoordinateSystem`` with every field a plain float.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415

    return CoordinateSystem(
        x=as_float(cs.x),
        y=as_float(cs.y),
        z=as_float(cs.z),
        rx=as_float(cs.rx),
        ry=as_float(cs.ry),
        rz=as_float(cs.rz),
        reference_cs=_detached_cs(cs.reference_cs) if cs.reference_cs else None,
    )


def _detached_geometry(geometry: object) -> object:
    """Return a plain-float clone of ``geometry`` for construction-time checks.

    The watertightness/ray-parity checks never need gradients (they run
    once, at construction, on discrete pass/fail geometry) but a
    differentiable scene may attach a ``torch.Tensor`` radius/conic/etc to
    the live geometry. Cloning with :func:`as_float` keeps that live
    component's tensor untouched while giving this check plain numpy
    arithmetic to work with, regardless of which backend is currently
    active.

    Args:
        geometry: A live ``ComponentGeometry`` instance.

    Returns:
        A new instance of the same class with every numeric parameter
        detached to a plain float, or ``geometry`` itself if its class is
        not one of the parametrized analytic geometries (nothing to
        detach).
    """
    from optiland.nonsequential.components.geometry.analytic.annulus import (  # noqa: PLC0415
        AnnularPlaneGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.conic import (  # noqa: PLC0415
        ConicGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.frustum import (  # noqa: PLC0415
        CylindricalFrustumGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.plane import (  # noqa: PLC0415
        FinitePlaneGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.sphere import (  # noqa: PLC0415
        SphereGeometry,
    )

    from optiland.nonsequential.components.geometry.analytic.asphere import (  # noqa: PLC0415
        _AsphereGeometry,
    )

    if isinstance(geometry, _AsphereGeometry):
        return geometry.detached_copy()
    if isinstance(geometry, ConicGeometry):
        return ConicGeometry(
            as_float(geometry.radius),
            as_float(geometry.conic),
            as_float(geometry.aperture_radius),
        )
    if isinstance(geometry, FinitePlaneGeometry):
        ap = geometry.aperture_radius
        return FinitePlaneGeometry(
            as_float(geometry.width),
            as_float(geometry.height),
            as_float(ap) if ap is not None else None,
        )
    if isinstance(geometry, AnnularPlaneGeometry):
        return AnnularPlaneGeometry(
            as_float(geometry.inner_radius),
            as_float(geometry.outer_radius),
            as_float(geometry.z_offset),
        )
    if isinstance(geometry, CylindricalFrustumGeometry):
        return CylindricalFrustumGeometry(
            as_float(geometry.r_front),
            as_float(geometry.r_back),
            as_float(geometry.z_front),
            as_float(geometry.z_back),
        )
    if isinstance(geometry, SphereGeometry):
        ap = geometry.aperture_radius
        return SphereGeometry(
            as_float(geometry.radius), as_float(ap) if ap is not None else None
        )
    return geometry


def _count_crossings(
    boundary: list[BaseComponent],
    origin: np.ndarray,
    direction: np.ndarray,
    max_bounces: int = _PARITY_MAX_BOUNCES,
) -> int:
    """Count how many times a ray crosses the boundary before escaping.

    Reuses each component's own ``intersect()`` -- no separate "all hits
    along a ray" geometry API is needed: the ray is walked hit-by-hit,
    nudged past each crossing by a small epsilon, and re-intersected against
    every boundary surface, up to ``max_bounces`` (a bound, not an
    unbounded loop).

    Args:
        boundary: The volume's boundary surfaces.
        origin: Ray start point, shape (3,).
        direction: Unit ray direction, shape (3,).
        max_bounces: Maximum crossings to count before giving up.

    Returns:
        Number of boundary crossings before the ray escapes to infinity.
    """
    from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

    proxies = [
        _DetachedProxy(_detached_cs(comp.cs), _detached_geometry(comp.geometry))
        for comp in boundary
    ]

    o = origin.astype(np.float64).copy()
    d = direction.astype(np.float64).copy()
    count = 0
    for _ in range(max_bounces):
        t_min = np.inf
        for comp in proxies:
            rays = NSQRayBundle(
                x=np.array([o[0]]),
                y=np.array([o[1]]),
                z=np.array([o[2]]),
                L=np.array([d[0]]),
                M=np.array([d[1]]),
                N=np.array([d[2]]),
                flux=np.array([1.0]),
                wavelength=np.array([0.55]),
                n_current=np.array([1.0]),
                bounce=np.array([0], dtype=np.int32),
                alive=np.array([True]),
                ray_id=np.array([0], dtype=np.int64),
            )
            t, _normals, hit, _n_geom = comp.intersect(rays)
            t0 = float(t[0])
            if bool(hit[0]) and t0 < t_min:
                t_min = t0
        if not np.isfinite(t_min):
            break
        o = o + (t_min + _PARITY_EPSILON) * d
        count += 1
    return count


def _vertex_axis_candidate(boundary: list[BaseComponent]) -> np.ndarray | None:
    """The mean of every boundary surface's own on-axis vertex, in global
    coordinates -- the fix for issue #14's sibling gap (issue #13, item 1):
    the rim-point mean used to be the sole interior-point estimate, and for
    a deep meniscus (front and back both curving the same way, closely
    spaced) that mean falls outside the glass on the axis, so the volume's
    own ray-parity check refused any element beyond a semi-diameter far
    inside the beam the element actually needs to pass.

    A :class:`~optiland.nonsequential.components.geometry.analytic.conic.
    ConicGeometry` face -- the only "vertex" surface ``Lens``/``Doublet``
    build -- sits at its own coordinate system's origin by construction (a
    conic's local vertex is local (0, 0, 0)), so its global vertex position
    is exactly that surface's own translation: no sag evaluation needed.
    For the two (or, for a cemented group folded into one volume, more)
    conic faces bounding one glass volume, the mean of their vertices is a
    point on the shared optical axis, strictly between the front-most and
    back-most vertex -- inside the glass by construction whenever the
    centre thickness between them is positive, regardless of how deep the
    meniscus is, because it never depends on the rim at all.

    Args:
        boundary: The volume's boundary surfaces.

    Returns:
        The mean vertex position, or ``None`` if fewer than two boundary
        surfaces are conic (nothing on-axis to average -- the caller falls
        back to its other candidates).
    """
    from optiland.nonsequential.components.geometry.analytic.conic import (  # noqa: PLC0415
        ConicGeometry,
    )

    from optiland.nonsequential.components.geometry.analytic.asphere import (  # noqa: PLC0415
        _AsphereGeometry,
    )

    # An asphere's vertex is its local origin too.
    vertices = [
        np.asarray(_get_transform(comp.cs)[0], dtype=float)
        for comp in boundary
        if isinstance(comp.geometry, (ConicGeometry, _AsphereGeometry))
    ]
    if len(vertices) < 2:
        return None
    return np.mean(vertices, axis=0)


def _check_normals_outward(
    boundary: list[BaseComponent],
    centroid: np.ndarray,
    num_directions: int = _PARITY_DIRECTIONS,
    seed: int = _PARITY_SEED,
) -> None:
    """Ray-parity check: every direction from ``centroid`` must exit an odd
    number of times.

    A point inside a closed, consistently outward-oriented solid crosses
    its boundary an odd number of times along any ray to infinity; an even
    count means either a gap (the ray slipped through unnoticed) or a
    surface with an inconsistent orientation.

    Args:
        boundary: The volume's boundary surfaces.
        centroid: An interior point estimate, shape (3,).
        num_directions: Number of random directions to test.
        seed: RNG seed, for a deterministic (reproducible) check.

    Raises:
        NonWatertightVolumeError: If any tested direction crosses an even
            number of times.
    """
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(num_directions, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    for d in dirs:
        count = _count_crossings(boundary, centroid, d)
        if count == 0 or count % 2 == 0:
            raise NonWatertightVolumeError(
                f"Volume boundary failed the inside/outside ray-parity check: "
                f"a ray cast from the estimated interior point "
                f"{centroid.tolist()} in direction {d.tolist()} crossed the "
                f"boundary {count} times (expected an odd, nonzero count for "
                f"a point inside a closed, consistently outward-oriented "
                f"solid). This usually means a gap in the boundary, a "
                f"surface with an unexpectedly flipped normal, or that the "
                f"estimated interior point is not actually inside the solid."
            )


@dataclass
class Volume:
    """A closed, outward-oriented solid built from boundary surfaces.

    Validated at construction: every boundary surface's rim must be met by
    a neighbour's rim (watertightness), and the boundary must enclose its
    own estimated interior point consistently from every direction
    (orientation). Both checks raise :class:`NonWatertightVolumeError`
    rather than warning -- a leaky or misoriented boundary produces
    silently wrong flux accounting, exactly the failure class this
    validation exists to catch at construction time instead of at trace
    time.

    Attributes:
        name: Human-readable label.
        boundary: Closed, outward-oriented list of boundary surfaces.
        interior: The medium inside this volume.
    """

    name: str
    boundary: list[BaseComponent]
    interior: NSQMaterial
    _skip_validation: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._skip_validation:
            return
        if not self.boundary:
            raise NonWatertightVolumeError(
                f"Volume '{self.name}' has no boundary surfaces."
            )
        rim_points = _check_watertight(self.boundary)

        # Interior-point candidates, tried in order until the ray-parity
        # check accepts one (issue #13, item 1). The on-axis vertex mean
        # goes first: unlike the rim-point mean, it is inside the glass by
        # construction whenever the centre thickness is positive, so it
        # does not fail for a deep meniscus. The rim-point mean and the
        # per-surface-origin mean stay as fallbacks for a boundary with
        # fewer than two conic vertex surfaces (an edge-only union, a
        # light pipe, a single closed sphere), where they already worked.
        candidates: list[np.ndarray] = []
        vertex_mid = _vertex_axis_candidate(self.boundary)
        if vertex_mid is not None:
            candidates.append(vertex_mid)
        if rim_points is not None and len(rim_points) > 0:
            candidates.append(rim_points.mean(axis=0))
        candidates.append(
            np.mean([_get_transform(c.cs)[0] for c in self.boundary], axis=0)
        )

        # This check calls component.intersect(), which dispatches through
        # optiland.backend (be.*). The check is purely discrete geometry --
        # never differentiated -- so it always runs on the numpy backend,
        # regardless of which backend is active for the surrounding scene
        # (e.g. a Lens built while be.set_backend("torch") is active for a
        # gradient trace).
        import optiland.backend as be  # noqa: PLC0415

        previous_backend = be.get_backend()
        try:
            be.set_backend("numpy")
            last_error: NonWatertightVolumeError | None = None
            for centroid in candidates:
                try:
                    _check_normals_outward(self.boundary, centroid)
                    break
                except NonWatertightVolumeError as exc:
                    last_error = exc
            else:
                assert last_error is not None
                raise NonWatertightVolumeError(
                    f"Volume '{self.name}' failed the inside/outside "
                    f"ray-parity check from every one of "
                    f"{len(candidates)} candidate interior points tried "
                    f"(the on-axis vertex mean, the rim-point mean, and "
                    f"the per-surface-origin mean, whichever applied). "
                    f"Last candidate's error: {last_error}"
                ) from last_error
        finally:
            be.set_backend(previous_backend)

    @staticmethod
    def union(
        *parts: Volume | list[BaseComponent] | BaseComponent,
    ) -> list[BaseComponent]:
        """Concatenate already-disjoint boundary surfaces into one list.

        This is the CSG operation this revamp implements: gluing separately
        constructed, non-overlapping boundary pieces together (the stated
        use cases -- a lens with a flat, a light pipe with a chamfer) are
        boundary concatenation, not boolean surface evaluation. The result
        is not itself validated; pass it to :class:`Volume` to check it.

        Args:
            *parts: Any mix of ``Volume`` instances, lists of components, or
                single components.

        Returns:
            The concatenated boundary list, in argument order.
        """
        boundary: list[BaseComponent] = []
        for part in parts:
            if isinstance(part, Volume):
                boundary.extend(part.boundary)
            elif isinstance(part, list):
                boundary.extend(part)
            else:
                boundary.append(part)
        return boundary

    @staticmethod
    def intersection(*parts: object) -> None:
        """Not implemented: true CSG intersection needs a boolean surface
        evaluator.

        Raises:
            NotImplementedError: Always. Construct the intersected geometry
                directly with analytic primitives, or use :meth:`union` for
                concatenating already-disjoint boundary surfaces.
        """
        raise NotImplementedError(
            "Volume.intersection() requires a true boolean surface evaluator, "
            "which this revamp does not implement (D16, measured and specced "
            "only). Use Volume.union() for concatenating already-disjoint "
            "boundary surfaces, or construct the intersected geometry "
            "directly with analytic primitives."
        )

    @staticmethod
    def difference(*parts: object) -> None:
        """Not implemented: true CSG difference needs a boolean surface
        evaluator.

        Raises:
            NotImplementedError: Always. See :meth:`intersection`.
        """
        raise NotImplementedError(
            "Volume.difference() requires a true boolean surface evaluator, "
            "which this revamp does not implement (D16, measured and specced "
            "only). Use Volume.union() for concatenating already-disjoint "
            "boundary surfaces, or construct the differenced geometry "
            "directly with analytic primitives."
        )
