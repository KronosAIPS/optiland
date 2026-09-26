"""Base component for Non-Sequential Raytracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential._utils import as_param

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.components.geometry.base import AABB, ComponentGeometry
    from optiland.nonsequential.ir.bsdf_ir import BsdfIR
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng


class BaseComponent(ABC):
    """Abstract base class for all non-sequential optical components.

    Components define the geometry and optical interaction (reflection,
    refraction, absorption) for a surface in the NSQ scene.

    Attributes:
        cs: Coordinate system defining position and orientation in global frame.
        geometry: Shape of the component surface.
        material_front: Medium on the front side (normal-facing side).
        material_back: Medium on the back side.
        bsdf: Optional scatter model. None means specular-only.
        name: Optional human-readable label.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        material_front: NSQMaterial,
        material_back: NSQMaterial,
        bsdf: BaseBSDF | None = None,
        name: str = "",
        scatter_fraction: float = 1.0,
    ) -> None:
        """Initialize BaseComponent.

        Args:
            cs: Coordinate system for this component.
            geometry: Surface geometry.
            material_front: Medium on the front (normal-facing) side.
            material_back: Medium on the back side.
            bsdf: Optional BSDF scatter model.
            name: Optional label for this component.
            scatter_fraction: Probability that a ray striking this surface is
                routed through ``bsdf`` instead of the specular path.
                Differentiable: a ``torch.Tensor`` with
                ``requires_grad=True`` stays attached to the autograd graph
                -- see ``RefractiveComponent.interact``/
                ``ReflectiveComponent.interact`` for the detached-sample /
                attached-weight estimator that makes
                ``d(flux)/d(scatter_fraction)`` correct rather than zero.
        """
        self.cs = cs
        self.geometry = geometry
        self.material_front = material_front
        self.material_back = material_back
        self.bsdf = bsdf
        self.name = name
        self.scatter_fraction = as_param(scatter_fraction)
        # The two parts of the last solved hit distance -- see intersect()
        # and advance_to_hit(). Not scene state: transient per-bounce
        # scratch, rewritten by every intersect() call and validated
        # per-ray before use.
        self._local_root: tuple[np.ndarray, np.ndarray] | None = None

    def refresh_backend_transform(self) -> None:
        """Upload this component's placement to the array backend once, for the trace about to run.

        The trace loop calls it after lowering the scene, so a component moved between two traces is
        re-read; inside the loop the transform is then a resident pair of arrays and no host round trip
        is made per bounce (the earlier per-call read and re-upload was the last synchronisation left).

        A placement that carries a gradient is attached here, once per trace
        (:func:`~optiland.nonsequential.parameter_register.attach_placement`):
        the value stays the host build's, the derivative flows to the tensors.
        """
        from optiland.nonsequential.parameter_register import (  # noqa: PLC0415
            attach_placement,
        )

        translation, rot = _get_transform(self.cs)
        self._be_transform = attach_placement(
            self.cs, be.array(translation), be.array(rot)
        )
        self._be_transform_key = _backend_key()

    def backend_transform(self):
        """The resident (translation, rotation) pair, refreshed if the backend configuration changed."""
        cached = getattr(self, "_be_transform", None)
        if cached is None or getattr(self, "_be_transform_key", None) != _backend_key():
            self.refresh_backend_transform()
            cached = self._be_transform
        return cached

    def intersect(
        self, rays: NSQRayBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find the nearest intersection of alive rays with this component.

        Transforms rays to local frame, delegates to geometry, then
        transforms normals back to global frame.

        Args:
            rays: The ray bundle in global coordinates.

        Returns:
            Tuple (t, normals, hit_mask, n_geom) in global frame:
                - t: Per-ray distances [mm], shape (N,). inf if no hit.
                - normals: Surface normals in global frame, shape (N, 3).
                - hit_mask: Boolean hit mask, shape (N,).
                - n_geom: Geometric (unflipped, direction-independent)
                    surface normal in global frame, shape (N, 3). See
                    :meth:`ComponentGeometry.ray_intersect`.
        """
        t_be, R_be = _resident_transform(self)

        # Global ray data as (N, 3) arrays
        positions_g = be.stack([rays.x, rays.y, rays.z], axis=1)
        directions_g = be.stack([rays.L, rays.M, rays.N], axis=1)

        # Transform to local frame
        positions_l = (positions_g - t_be) @ R_be
        directions_l = directions_g @ R_be

        # Origin advance (docs/theory/07_geometry.md sec 7.7, cure 3): before
        # solving the ray-surface quadratic, advance the local origin along
        # the ray to the point closest to the surface's vertex (the local
        # coordinate origin, which every analytic geometry here is centred
        # or based on), so the quadratic is solved with a small residual
        # rather than the ray's full, possibly scene-scale, local coordinate.
        # The removed distance is carried separately (t_adv) and added back
        # once the small local solve returns. By docs/theory/07_geometry.md
        # sec 7.5, the ray-conic discriminant loses accuracy as (L/R)^2 for
        # a throw of length L onto a surface of radius R; this replaces L
        # with an O(part size) residual, removing the term rather than
        # reducing it. No-op in effect when the origin is already near the
        # vertex (t_adv is then already small), which is the common case of
        # a ray freshly leaving the surface it is about to test again.
        t_adv = -(positions_l * directions_l).sum(axis=1)
        positions_adv = positions_l + t_adv[:, None] * directions_l

        # Self-intersection accept threshold: k ulps of the ray's own
        # *global*-frame coordinate magnitude (docs/theory/07_geometry.md
        # sec 7.7, docs/theory/08_precision.md sec 8.7), not a fixed
        # absolute length -- a bare 1e-9 mm is below the float32 step at 50
        # mm and below the float64 step once the scene is ~5e4 mm across.
        # Computed from the *global* position, not the local one, and
        # passed down to the geometry: a component's local (vertex-relative)
        # frame is often small even far into a scene -- a lens edge's own
        # extent is a few mm regardless of where the lens sits -- and a
        # threshold sized to that small local coordinate would be far
        # tighter than the rounding error actually carried in the ray's
        # tracked position, reintroducing self-hit instability rather than
        # removing it.
        #
        # Per ray, not one scalar for the whole bundle: the rounding a ray's
        # own position carries is set by that ray's own coordinate, and one
        # distant ray must not coarsen the threshold for every other ray
        # (at float32 a bundle reaching 1e4 mm would put the threshold at
        # 1e-2 mm for all of them, which skips a thin plate or a cemented
        # interface). It also removes a full-bundle reduction to a scalar
        # from the inner loop, which on a GPU is a device-to-host sync.
        t_min = _tol.accept_t_min(coordinate_magnitude(rays))
        # A geometry's own root-validity tests compare the LOCAL parameter
        # (t_local) against eps, e.g. "t_local > eps"; what must actually
        # hold is "t_local + t_adv > t_min" (a genuine forward hit in the
        # ray's real, unshifted parametrization -- the advance is a re-
        # parametrization, not a change of which points are ahead of the
        # ray). Shifting the threshold by t_adv makes the geometry's
        # existing "t_local > eps_shifted" tests exactly equivalent, with no
        # change to any geometry's internal comparisons: the true surface
        # point can land on either side of the advanced origin (a curved
        # surface's sag is not zero at the advance point in general), so a
        # plain "t_local > 0" requirement would wrongly reject a genuine hit
        # the advance happened to step past.
        eps_shifted = t_min - t_adv
        t_local, normals_l, hit_mask, n_geom_l = self.geometry.ray_intersect(
            positions_adv, directions_l, eps=eps_shifted
        )
        t_hit = t_local + t_adv

        # Keep the advance and the residual as two numbers. Their sum is
        # what the scene's nearest-hit comparison needs, but the sum alone
        # cannot place the ray back on the surface: rounding it costs
        # u*|t_hit|, so a ray that has just travelled a long leg lands that
        # far off the surface it just hit, and the next bounce's
        # intersection test finds a root there. advance_to_hit() rebuilds
        # the hit point from these two parts instead. See its docstring.
        self._local_root = (t_adv, t_local)

        # Note the accept/reject decision before overwriting t_hit: checking
        # the *post*-overwrite value here would always read back either the
        # original t_hit (t_hit > t_min already true) or +inf (which is also
        # > t_min), so it could never actually reject anything -- a
        # pre-existing latent bug that a threshold tight enough to matter
        # (the one this module replaces) never used to trigger.
        accepted = t_hit > t_min
        inf_like = be.ones_like(t_hit) * be.inf
        t_hit = be.where(accepted, t_hit, inf_like)
        hit_mask = hit_mask & accepted

        # Dead rays can't hit
        t_hit = be.where(rays.alive, t_hit, inf_like)
        hit_mask = hit_mask & rays.alive

        # Transform normals back to global: n_global_row = n_local_row @ R^T
        normals_g = normals_l @ R_be.T
        n_geom_g = n_geom_l @ R_be.T

        return t_hit, normals_g, hit_mask, n_geom_g

    def advance_to_hit(
        self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray
    ) -> None:
        """Move every ray in ``hit_mask`` onto its intersection point.

        Not ``p + t*d``. A hit distance is a single number of the size of
        the whole leg just travelled, so it can only be written down to
        ``u*|t|``: a ray that comes 1e6 mm to a surface 50 mm from the
        origin arrives with its position 1e-10 mm off that surface in
        float64 (measured), even though every coordinate involved is
        representable a thousand times more finely than that. The next
        bounce's intersection test then finds a real root at 1e-10 mm and
        the ray sticks to the surface it has just left. Raising the
        self-intersection threshold to cover it (the previous cure) makes
        the engine blind to any surface nearer than that -- at float32 and
        a 50 mm coordinate it reached 0.06 mm, which skips a thin plate, a
        cemented interface, or a coating modelled as a surface.

        The distance is therefore never composed before it is used. The
        intersection was solved from an origin already advanced into the
        surface's neighbourhood (see :meth:`intersect`), so the residual
        ``t_local`` is of the order of the part, not of the leg. Rebuilding
        the hit point in the surface's own frame,

            p = T + (o_adv + t_local * d_local) @ R^T,

        adds one large number, the surface's own position ``T``, and adds
        it last: the hit point lands on the surface to one ulp of ``T``,
        whatever the leg was. ``docs/theory/07_geometry.md`` section 7.7
        (cure 3) and R-07-4.

        The two parts come from this component's own last
        :meth:`intersect` call, and each ray checks for itself that they
        still compose to the ``t`` it is being advanced by; a ray whose
        check fails (a bundle this component has not just intersected --
        a bounded-splitting snapshot, a direct call in a test) falls back
        to the plain global update, per ray and without a host sync.

        Args:
            rays: Ray bundle, updated in place.
            t: Per-ray hit distance [mm], shape (N,). ``inf`` outside
                ``hit_mask``.
            hit_mask: Rays to advance, shape (N,).
        """
        advance_to_hit_in_frame(
            rays, t, hit_mask, self._local_root, _resident_transform(self)
        )

    def offset_from_surface(
        self, rays: NSQRayBundle, n_geom: np.ndarray, hit_mask: np.ndarray
    ) -> None:
        """Push an outgoing ray's origin off the surface it has just left.

        Cure 2 of ``docs/theory/07_geometry.md`` section 7.7, required by
        R-07-6 and not previously implemented. Called at the end of an
        interaction, after the outgoing direction is known: the origin moves
        by ``delta`` (:func:`optiland.nonsequential._tol.origin_offset`)
        along the geometric normal, signed into the hemisphere the outgoing
        ray leaves into.

        Why the accept threshold does not cover this on its own. The
        rebuilt hit point lands within about half an ulp of the surface,
        but on either side of it -- measured over a plane interface, 8119
        of 16384 rays landed on the side they were leaving, at up to 0.38
        ulp of their 10 mm coordinate. A ray leaving at ``alpha`` above the
        surface plane then re-crosses the surface after a *path length*
        ``delta_perp / sin(alpha)``. The threshold is a path length, so the
        ``1 / sin(alpha)`` is free amplification: at 0.36 degrees from
        grazing (an N-BK7/air exit one millidegree inside the critical
        angle) a 4e-16 mm residual becomes a 6.5e-14 mm root, above the
        2.8e-14 mm threshold, and the surface accepts the ray it just
        released. With the origin offset the ray starts ``delta`` clear of
        the surface on the correct side, so no root forms at all.

        The offset is applied for every hit ray, reflected or transmitted,
        and is signed per ray: it therefore also covers a second surface
        that shares this one's plane, and any later grazing exit, not only
        the case that exposed it.

        Args:
            rays: Ray bundle, updated in place. Positions must already be
                at the hit point and directions must already be the
                outgoing ones.
            n_geom: Geometric (unflipped) surface normal in the global
                frame, shape (N, 3) -- R-07-9: offsets use the geometric
                normal, never an interpolated one.
            hit_mask: Rays that interacted with this component, shape (N,).
        """
        offset_origin_from_surface(rays, n_geom, hit_mask)

    @abstractmethod
    def interact(
        self,
        rays: NSQRayBundle,
        t: np.ndarray,
        normals: np.ndarray,
        hit_mask: np.ndarray,
        rng: NSQRng,
        bsdf_ir: BsdfIR,
        n_geom: np.ndarray,
        sampling: SamplingPolicy | None = None,
        forced_branch: str | None = None,
    ) -> None:
        """Apply optical interaction at hit points (in-place).

        Updates ray positions, directions, flux, n_current, bounce, and
        alive status for rays that hit this component.

        This is a private implementation detail of the reference NumPy/Torch
        interpreters (``optiland.nonsequential.ir.interpreter
        .apply_primitive_interactions``), not the engine's public dispatch
        contract -- a non-Python backend never calls it. ``bsdf_ir`` is what
        makes the *dispatch* IR-driven: whether to route a hit ray through
        ``self.bsdf`` is decided from ``bsdf_ir.kind`` (verified to match
        ``self.bsdf``'s actual type by the caller), not from a bare
        ``self.bsdf is not None`` check.

        Args:
            rays: Ray bundle to update in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays that hit this component, shape (N,).
            rng: Keyed PCG32 RNG for stochastic interactions.
            bsdf_ir: This surface's lowered BSDF descriptor (``BsdfIR(kind=
                "none")`` when no scatter model is attached), matching
                ``self.bsdf``.
            n_geom: Geometric (unflipped) surface normal in global frame,
                shape (N, 3): points from ``material_front`` toward
                ``material_back``. ``RefractiveComponent`` uses this,
                not index proximity, to determine which material a ray is
                entering.
            sampling: The scene's rare-path sampling policy.
                Only ``RefractiveComponent`` consults it, to resolve the
                Fresnel reflect/transmit branch probability; ``None`` is
                treated as the default (unbiased, ``reflect_prob="fresnel"``)
                policy.
            forced_branch: ``"reflect"`` or ``"transmit"`` to deterministically
                force the branch instead of drawing it, or ``None`` for the
                normal stochastic draw. Used only by the NumPy forward
                engine's bounded-splitting orchestration (PR11;
                :mod:`optiland.nonsequential.ir.interpreter`) to build both
                children of a split ray; ignored by every component except
                ``RefractiveComponent``.
        """

    @property
    def bounding_box(self) -> AABB:
        """Axis-aligned bounding box in global coordinates.

        Returns:
            AABB for this component.
        """
        transform = _get_transform(self.cs)
        return self.geometry.bounding_box(transform)


def advance_to_hit_in_frame(rays, t, hit_mask, local_root, transform) -> None:
    """Move every ray in ``hit_mask`` onto its hit point, rebuilt in a frame.

    The body of :meth:`BaseComponent.advance_to_hit`, as a function, because
    a *transmissive* detector needs exactly the same treatment and is not a
    component: a ray that a detector lets through carries on from the point
    this puts it at, and the next bounce intersects that same plane again
    (``docs/build/X1_threshold_arithmetic.md`` section 7, note 4).

    Args:
        rays: Ray bundle, updated in place.
        t: Per-ray hit distance [mm], shape (N,). ``inf`` outside
            ``hit_mask``.
        hit_mask: Rays to advance, shape (N,).
        local_root: The ``(t_adv, t_local)`` pair the surface's own last
            intersect left behind, or ``None``. Each ray checks for itself
            that the two still compose to the ``t`` it is being advanced by,
            and a ray whose check fails falls back to the plain global
            update -- per ray, and without a host synchronisation.
        transform: The surface's ``(translation, rotation)`` on the array
            backend.
    """
    # Missed rays carry t = inf; zero it for the differentiable update
    # so a masked-out be.where branch cannot inject 0 * inf = NaN into
    # the backward pass.
    t_safe = be.where(hit_mask, t, be.zeros_like(t))
    x_g = rays.x + t_safe * rays.L
    y_g = rays.y + t_safe * rays.M
    z_g = rays.z + t_safe * rays.N

    # Same array type as well as same shape: a cache left over from a
    # trace on the other backend would otherwise reach a mixed
    # NumPy/Torch comparison below.
    if (
        local_root is not None
        and type(local_root[0]) is type(t)
        and local_root[0].shape == t.shape
    ):
        t_adv, t_local = local_root
        # Per-ray, elementwise: no reduction, so no device-to-host sync.
        usable = hit_mask & (t_adv + t_local == t)
        adv_safe = be.where(usable, t_adv, be.zeros_like(t_adv))
        loc_safe = be.where(usable, t_local, be.zeros_like(t_local))

        t_be, R_be = transform
        positions_g = be.stack([rays.x, rays.y, rays.z], axis=1)
        directions_l = be.stack([rays.L, rays.M, rays.N], axis=1) @ R_be
        # Bitwise the advanced origin intersect() solved from: same
        # inputs, same operations, and this ray's position has not been
        # touched since (one bounce moves each ray at exactly one
        # component).
        positions_adv = (
            (positions_g - t_be) @ R_be + adv_safe[:, None] * directions_l
        )
        hit_l = positions_adv + loc_safe[:, None] * directions_l
        hit_g = hit_l @ R_be.T + t_be
        x_g = be.where(usable, hit_g[:, 0], x_g)
        y_g = be.where(usable, hit_g[:, 1], y_g)
        z_g = be.where(usable, hit_g[:, 2], z_g)

    rays.x = be.where(hit_mask, x_g, rays.x)
    rays.y = be.where(hit_mask, y_g, rays.y)
    rays.z = be.where(hit_mask, z_g, rays.z)


def offset_origin_from_surface(rays, n_geom, hit_mask) -> None:
    """Push an outgoing ray's origin off the surface it has just left.

    The body of :meth:`BaseComponent.offset_from_surface`, as a function,
    for the same reason :func:`advance_to_hit_in_frame` is one: a
    transmissive detector leaves a ray on a plane the ray is about to be
    tested against again, and a grazing crossing turns the half-ulp it lands
    off that plane by into a root above the accept threshold.

    Args:
        rays: Ray bundle, updated in place. Positions must already be at the
            hit point and directions must already be the outgoing ones.
        n_geom: Geometric (unflipped) surface normal in the global frame,
            shape (N, 3).
        hit_mask: Rays that interacted with this surface, shape (N,).
    """
    dot = rays.L * n_geom[:, 0] + rays.M * n_geom[:, 1] + rays.N * n_geom[:, 2]
    delta = _tol.origin_offset(coordinate_magnitude(rays))
    # Sign from the outgoing hemisphere. A direction exactly in the
    # surface plane (dot == 0) never re-crosses it -- the geometries
    # reject a parallel ray on the denominator test -- so either sign
    # is safe there; +1 keeps the expression branch-free.
    signed = be.where(dot < 0, -delta, delta)
    step = be.where(hit_mask, signed, be.zeros_like(signed))
    rays.x = rays.x + step * n_geom[:, 0]
    rays.y = rays.y + step * n_geom[:, 1]
    rays.z = rays.z + step * n_geom[:, 2]


def coordinate_magnitude(rays: NSQRayBundle) -> np.ndarray:
    """Per-ray infinity norm of the global position, shape (N,).

    The scale the ray's own tracked position is resolved at, and so the
    scale every self-intersection threshold is measured in
    (docs/theory/07_geometry.md R-07-5). Built from the three coordinate
    arrays rather than from a stacked (N, 3) array so it costs no extra
    allocation, and reduced per ray rather than over the bundle so one
    distant ray cannot coarsen every other ray's threshold.

    Args:
        rays: The ray bundle, in global coordinates.

    Returns:
        ``max(|x|, |y|, |z|)`` per ray, in the working backend and dtype.
    """
    return be.maximum(be.maximum(be.abs(rays.x), be.abs(rays.y)), be.abs(rays.z))


def _get_transform(cs: CoordinateSystem) -> tuple[np.ndarray, np.ndarray]:
    """Extract (translation, rotation_matrix) from a CoordinateSystem.

    Returns plain numpy float64 arrays regardless of the current
    optiland backend.

    Args:
        cs: The coordinate system.

    Returns:
        Tuple (translation [mm], rotation_matrix) as numpy float64 arrays.
        rotation_matrix is (3, 3) and transforms column vectors local->global.
    """
    from optiland.backend.utils import to_numpy  # noqa: PLC0415

    t_be, R_be = cs.get_effective_transform()
    translation = to_numpy(t_be).astype(np.float64)
    rotation = to_numpy(R_be).astype(np.float64)
    return translation, rotation


def _resident_transform(component) -> tuple:
    """The component's (translation, rotation) on the array backend: the per-trace cache where the object
    carries one (a real component), else a direct upload (the detached proxies of the volume checks)."""
    cached = getattr(component, "backend_transform", None)
    if cached is not None:
        return cached()
    translation, rot = _get_transform(component.cs)
    return be.array(translation), be.array(rot)


def resident_table(owner, name: str, values):
    """A constant lookup table of ``owner``'s, uploaded to the backend once.

    Bin edges, a tabulated inverse CDF, a measured BSDF grid: all of them
    are configuration, not ray data. They are built once from the object's
    own parameters and never change during a trace, so the arithmetic that
    reads them has no reason to leave the device -- but only if the table is
    there too, which is what this uploads and keeps
    (``docs/theory/12_gpu_mapping.md`` R-12-8, the rule
    :func:`_resident_transform` follows for a placement).

    The cache is keyed on the backend configuration, so an object reused
    across a NumPy trace and a Torch one gets the right array each time.

    Args:
        owner: The object the table belongs to; the cache lives on it.
        name: Key for this table on this object.
        values: The table, as NumPy values.

    Returns:
        The table as an array of the active backend, in its working dtype
        and on its device.
    """
    key = _backend_key()
    if getattr(owner, "_be_tables_key", None) != key:
        owner._be_tables = {}
        owner._be_tables_key = key
    cached = owner._be_tables.get(name)
    if cached is None:
        cached = be.array(np.asarray(values, dtype=np.float64))
        owner._be_tables[name] = cached
    return cached


def _backend_key() -> tuple:
    """What a cached backend array depends on: the backend, its precision and its device."""
    name = be.get_backend()
    precision = be.get_precision()
    try:
        device = be.get_device()
    except Exception:  # noqa: BLE001 - the numpy backend has no device
        device = None
    return (str(name), str(precision), str(device))
