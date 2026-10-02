"""Mesh geometry for Non-Sequential Raytracing.

Triangulated surface backed by trimesh. Uses trimesh's built-in BVH for
CPU-side ray intersection.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np

from optiland.nonsequential import _tol
from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry


class MeshGeometry(AnalyticGeometry):
    """Triangulated surface backed by a trimesh.Trimesh object.

    Intersection uses trimesh's ray caster (pyembree if installed, else
    the pure-Python fallback). GPU transfer is required on each intersection
    step for CuPy arrays; analytic geometry is preferred for GPU performance.

    Attributes:
        mesh: The underlying trimesh.Trimesh object.
    """

    def __init__(self, mesh: object) -> None:  # trimesh.Trimesh
        """Initialize MeshGeometry.

        Args:
            mesh: A trimesh.Trimesh instance defining the surface.

        Raises:
            ImportError: If trimesh is not installed.
        """
        try:
            import trimesh  # noqa: F401  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "trimesh is required for MeshGeometry. "
                "Install with: pip install trimesh"
            ) from e
        self.mesh = mesh

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray, eps: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the mesh using trimesh BVH.

        Arrays are converted to NumPy for trimesh, then results are
        converted back to the original array type (CuPy if needed).

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3).
            eps: See :meth:`ComponentGeometry.ray_intersect`.

        Returns:
            (t, normals, hit_mask, n_geom). n_geom is trimesh's raw
            ``face_normals`` value, i.e. the ``material_back`` side by
            contract (see :meth:`ComponentGeometry.ray_intersect`) is
            whichever side the mesh's face winding points away from --
            consistently outward for a properly wound (CCW, right-hand
            rule) closed mesh.
        """
        xp = _get_xp(origins)
        # Bring to CPU for trimesh: a torch tensor (any device) is copied to the
        # host in its own dtype and widened there to float64 (the Apple GPU's
        # one-call copy to a float64 host array writes zeros, the research
        # repository's issue 54). Before this the torch backend raised here.
        like = origins if _is_torch(origins) else None
        o_np = _to_numpy(origins)
        d_np = _to_numpy(directions)
        N = o_np.shape[0]

        # trimesh ray casting (KronosNSRT issue 29). trimesh keeps only the hits
        # ahead of the origin it is given (down to 1e-6 behind it), and the
        # threshold this kind is handed may lie behind the origin: the
        # component advances the origin along the ray to its closest approach
        # to the local origin and passes the threshold shifted by that advance,
        # so a face the advance stepped past (a box's entry face) was dropped by
        # trimesh before the threshold was ever applied. The query therefore
        # starts behind every point of the mesh along the ray (the bounding
        # box's half-diagonal beyond the projection of its centre), and the
        # hits are measured from the true origin below. Every hit, not only the
        # first: asked for one hit per ray, trimesh returns the nearest, which
        # for a ray leaving a face is that face itself; the threshold refused
        # it and the genuine hit beyond it was never returned. The loop below
        # keeps the nearest hit beyond the threshold.
        verts = np.asarray(self.mesh.vertices, dtype=np.float64)
        centre = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
        half_diag = 0.5 * float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        back = np.maximum(((o_np - centre) * d_np).sum(axis=1) + half_diag, 0.0) if N else np.zeros(0)
        locations, ray_indices, triangle_indices = self.mesh.ray.intersects_location(
            ray_origins=o_np - back[:, None] * d_np,
            ray_directions=d_np,
            multiple_hits=True,
        )

        t_out = np.full(N, np.inf, dtype=np.float64)
        normals_out = np.zeros((N, 3), dtype=np.float64)
        n_geom_out = np.zeros((N, 3), dtype=np.float64)

        # Self-intersection accept threshold, dtype-aware -- this class is
        # numpy float64 only (trimesh requirement), so the coordinate
        # magnitude below is always evaluated at float64 resolution. A
        # caller-supplied eps may be a scalar or one value per ray (the loop
        # passes ``(N,)``), and may be a backend array/tensor (from a torch
        # scene); it is brought to host float64, one value per ray, since this
        # loop is host Python (a per-ray eps raised here before, issue 29).
        if eps is None:
            origin_scale = np.abs(o_np).max() if N else 1.0
            eps = _tol.accept_t_min(origin_scale)
        if hasattr(eps, "detach"):
            eps = eps.detach().cpu().numpy()
        t_min = np.broadcast_to(np.asarray(_to_numpy(eps), dtype=np.float64), (N,))

        if len(ray_indices) > 0:
            # Compute t for each hit
            hit_vecs = locations - o_np[ray_indices]
            # t = dot(hit_vec, direction) / |direction|^2 ~ dot for unit vectors
            t_vals = (hit_vecs * d_np[ray_indices]).sum(axis=1)
            # Keep nearest positive hit per ray
            order = np.argsort(ray_indices)
            for idx, ri in enumerate(ray_indices[order]):
                tv = t_vals[order[idx]]
                if tv > t_min[ri] and tv < t_out[ri]:
                    t_out[ri] = tv
                    tri_idx = triangle_indices[order[idx]]
                    face_normal = self.mesh.face_normals[tri_idx]
                    n_geom_out[ri] = face_normal
                    # Flip to face incoming ray
                    if np.dot(d_np[ri], face_normal) > 0:
                        face_normal = -face_normal
                    normals_out[ri] = face_normal

        hit_mask_np = t_out < np.inf

        if like is not None:
            # Back to the rays' dtype and device, rounded on the host and then
            # moved; the mesh kind carries no gradient (trimesh is host code).
            from optiland.backend.torch_backend.capabilities import (  # noqa: PLC0415
                to_device_dtype,
            )

            import torch  # noqa: PLC0415

            def back(a, dtype):
                return to_device_dtype(torch.from_numpy(np.ascontiguousarray(a)), like.device, dtype)

            return (
                back(t_out, like.dtype),
                back(normals_out, like.dtype),
                back(hit_mask_np, torch.bool),
                back(n_geom_out, like.dtype),
            )

        if xp is not np:
            t_out = xp.array(t_out)
            normals_out = xp.array(normals_out)
            hit_mask_np = xp.array(hit_mask_np)
            n_geom_out = xp.array(n_geom_out)

        return t_out, normals_out, hit_mask_np, n_geom_out

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB for the mesh in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        t_vec = np.array(transform[0], dtype=float)
        R = np.array(transform[1], dtype=float)
        verts_local = np.array(self.mesh.vertices, dtype=float)
        verts_global = verts_local @ R.T + t_vec
        return AABB(verts_global.min(axis=0), verts_global.max(axis=0))


def _is_torch(arr) -> bool:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    return isinstance(arr, torch.Tensor)


def _to_numpy(arr: np.ndarray) -> np.ndarray:
    if _is_torch(arr):
        return arr.detach().cpu().numpy().astype(np.float64)
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    except ImportError:
        pass
    return arr


def _get_xp(arr: np.ndarray):
    try:
        import cupy  # type: ignore[import]

        if isinstance(arr, cupy.ndarray):
            return cupy
    except ImportError:
        pass
    return np
