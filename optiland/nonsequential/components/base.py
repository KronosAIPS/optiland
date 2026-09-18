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
        """
        translation, rot = _get_transform(self.cs)
        self._be_transform = (be.array(translation), be.array(rot))
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
        # Missed rays carry t = inf; zero it for the differentiable update
        # so a masked-out be.where branch cannot inject 0 * inf = NaN into
        # the backward pass.
        t_safe = be.where(hit_mask, t, be.zeros_like(t))
        x_g = rays.x + t_safe * rays.L
        y_g = rays.y + t_safe * rays.M
        z_g = rays.z + t_safe * rays.N

        cached = self._local_root
        # Same array type as well as same shape: a cache left over from a
        # trace on the other backend would otherwise reach a mixed
        # NumPy/Torch comparison below.
        if (
            cached is not None
            and type(cached[0]) is type(t)
            and cached[0].shape == t.shape
        ):
            t_adv, t_local = cached
            # Per-ray, elementwise: no reduction, so no device-to-host sync.
            usable = hit_mask & (t_adv + t_local == t)
            adv_safe = be.where(usable, t_adv, be.zeros_like(t_adv))
            loc_safe = be.where(usable, t_local, be.zeros_like(t_local))

            t_be, R_be = _resident_transform(self)
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


def _backend_key() -> tuple:
    """What a cached backend array depends on: the backend, its precision and its device."""
    name = be.get_backend()
    precision = be.get_precision()
    try:
        device = be.get_device()
    except Exception:  # noqa: BLE001 - the numpy backend has no device
        device = None
    return (str(name), str(precision), str(device))
