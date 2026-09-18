"""Closed spherical cavity geometry for Non-Sequential Raytracing.

``SphericalCavityGeometry`` -- a watertight sphere of radius ``R`` centred on
the local origin, with any number of circular openings ("ports") cut through
its wall. A ray passes freely through a port and reflects (or is absorbed,
or refracted) everywhere else, so an integrating sphere, a light-trap cavity
and a closed hemispherical collector are all one primitive with a different
port list.

Against :class:`~optiland.nonsequential.components.geometry.analytic.sphere.SphereGeometry`
-- which is the same quadric limited by a *transverse* aperture radius, i.e.
a cap seen from the axis -- two things differ, and both are what the closed
cavity needs:

1. **The wall is a cap complement, not an aperture.** A port is given by a
   direction in the local frame and an angular radius about it, so its area
   is the spherical cap ``2 pi R^2 (1 - cos alpha)`` and its *area fraction*
   of the whole sphere is ``(1 - cos alpha) / 2`` exactly -- the ``f`` the
   integrating-sphere multiplier is written in. A transverse aperture radius
   cannot express that: it cuts the same cap only for a port on the local
   ``z`` axis, and mis-states the area by the flat-disc-against-cap
   difference (``docs/theory/11_validation_catalogue.md`` 11.4.17 lists it
   among the defects the port-fraction sweep exists to catch).

2. **Both roots are tested, not just the nearest.** A ray entering through a
   port crosses the sphere twice: the near root lands in the port opening and
   is *not* a hit, the far root is the wall it should reflect off. Choosing
   the nearest positive root and then rejecting it -- what an aperture check
   does -- loses that hit and leaks the ray out of the cavity. Root validity
   is therefore evaluated per root, and the nearest *valid* one is returned.

Watertight means what it says: away from the ports every ray direction from
any interior point meets the wall exactly once, with no gap at the poles, no
seam, and no direction-dependent aperture test. The unit tests assert it by
firing an isotropic bundle from the centre.

All operations are in LOCAL coordinates: the sphere centre is the local
origin, which is also what ``BaseComponent.intersect``'s origin advance is
built around (it advances to the point of closest approach to the local
origin, which for this quadric is the well-conditioned form of the solve).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential._utils import as_float, as_param
from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry


@dataclass(frozen=True)
class SphericalPort:
    """One circular opening in a spherical wall.

    A port is the spherical cap of angular radius ``half_angle_deg`` about
    ``axis``: the set of wall points whose direction from the sphere centre
    lies within that angle of ``axis``. Rays pass through it unaffected.

    Attributes:
        axis: Direction of the port centre in the geometry's local frame,
            ``(x, y, z)``. Need not be normalized; it is used as a
            direction only.
        half_angle_deg: Angular radius of the port [deg], in ``(0, 180]``.
            ``90`` removes a whole hemisphere.
    """

    axis: tuple[float, float, float]
    half_angle_deg: float

    def __post_init__(self) -> None:
        """Validate the axis and the angular radius.

        Raises:
            ValueError: If the axis is the zero vector, or the angular
                radius is outside ``(0, 180]``.
        """
        norm = math.sqrt(sum(float(a) ** 2 for a in self.axis))
        if norm == 0.0:
            raise ValueError("A port axis cannot be the zero vector.")
        if not 0.0 < float(self.half_angle_deg) <= 180.0:
            raise ValueError(
                "A port's half_angle_deg must lie in (0, 180]; got "
                f"{self.half_angle_deg}."
            )

    @property
    def unit_axis(self) -> tuple[float, float, float]:
        """The port axis as a unit vector."""
        ax, ay, az = (float(a) for a in self.axis)
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        return (ax / norm, ay / norm, az / norm)

    @property
    def cos_half_angle(self) -> float:
        """Cosine of the angular radius -- the wall/port test threshold."""
        return math.cos(math.radians(float(self.half_angle_deg)))

    @property
    def area_fraction(self) -> float:
        """This port's area as a fraction of the whole sphere's area.

        The cap of angular radius ``alpha`` has area
        ``2 pi R^2 (1 - cos alpha)`` against the sphere's ``4 pi R^2``.
        """
        return 0.5 * (1.0 - self.cos_half_angle)

    @classmethod
    def from_area_fraction(
        cls, axis: tuple[float, float, float], area_fraction: float
    ) -> SphericalPort:
        """Build a port covering a given fraction of the sphere's area.

        Args:
            axis: Port centre direction in the local frame.
            area_fraction: Fraction of the total sphere area, in ``(0, 1]``.

        Returns:
            The port whose cap has exactly this area fraction.

        Raises:
            ValueError: If ``area_fraction`` is outside ``(0, 1]``.
        """
        f = float(area_fraction)
        if not 0.0 < f <= 1.0:
            raise ValueError(f"A port area fraction must lie in (0, 1]; got {f}.")
        return cls(axis=tuple(float(a) for a in axis), half_angle_deg=math.degrees(
            math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * f)))
        ))


class SphericalCavityGeometry(AnalyticGeometry):
    """Watertight sphere with circular ports cut through its wall.

    Attributes:
        radius: Sphere radius [mm]. The centre is the local origin.
        ports: The openings, as a tuple of :class:`SphericalPort`.
    """

    def __init__(
        self,
        radius: float,
        ports: tuple[SphericalPort, ...] | list[SphericalPort] | None = None,
    ) -> None:
        """Initialize SphericalCavityGeometry.

        Args:
            radius: Sphere radius [mm].
            ports: Openings in the wall. ``None`` or empty gives a closed
                sphere.
        """
        self.radius = as_param(radius)
        self.ports: tuple[SphericalPort, ...] = tuple(ports or ())

    @classmethod
    def hemisphere(
        cls, radius: float, axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    ) -> SphericalCavityGeometry:
        """A closed hemisphere: the half of the sphere ``axis`` points into.

        The opposite half is removed as a single 90-degree port, so the
        surface is exactly ``{p : |p| = R, p . axis >= 0}`` and every ray
        leaving a point inside it with a positive ``axis`` component meets
        it once.

        Args:
            radius: Sphere radius [mm].
            axis: Local-frame direction of the retained half. Defaults to
                local +z.

        Returns:
            The hemisphere as a one-port cavity.
        """
        opposite = tuple(-float(a) for a in axis)
        return cls(radius, [SphericalPort(axis=opposite, half_angle_deg=90.0)])

    @property
    def port_area_fraction(self) -> float:
        """Total port area as a fraction of the sphere's area.

        The sum of the per-port cap fractions. It is the ``f`` of the
        integrating-sphere multiplier only while the ports do not overlap;
        overlapping ports are counted twice here, exactly as the closed form
        would (11.4.17's "port overlap at large f left uncorrected").
        """
        return float(sum(p.area_fraction for p in self.ports))

    def _on_wall(self, hx, hy, hz):
        """True where a point on the sphere is wall rather than port.

        Args:
            hx: Hit x in the local frame, shape (N,).
            hy: Hit y in the local frame, shape (N,).
            hz: Hit z in the local frame, shape (N,).

        Returns:
            Boolean array, shape (N,): True outside every port cap.
        """
        # All-False in the caller's own backend and device, with no import
        # of the ray-state helpers into a geometry module.
        in_port = (hx * 0.0) > 1.0
        for port in self.ports:
            ax, ay, az = port.unit_axis
            # |hit| == radius by construction, so the projection onto the
            # unit port axis divided by the radius is the cosine of the
            # angle between them -- no arccos, and no normalisation of a
            # vector that is already of known length.
            cos_angle = (hx * ax + hy * ay + hz * az) / self.radius
            in_port = in_port | (cos_angle >= port.cos_half_angle)
        return ~in_port

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray, eps: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the ported sphere.

        Both roots of the ray-sphere quadratic are tested against the accept
        threshold *and* against the port cut, and the nearest root that
        survives both is returned -- so a ray entering through a port keeps
        travelling and hits the far wall, instead of being lost at the near
        crossing.

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3), unit.
            eps: Minimum accepted ray parameter; see
                :meth:`ComponentGeometry.ray_intersect`.

        Returns:
            ``(t, normals, hit_mask, n_geom)``. ``n_geom`` points *inward*,
            toward the sphere centre, matching ``SphereGeometry``'s
            ``material_back``-is-the-interior convention;  ``normals`` is
            flipped to face the incoming ray, so a ray inside the cavity
            reflects off an inward-facing normal and a ray outside it off an
            outward-facing one, with no side flag to set.
        """
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

        # |o + t d|^2 = R^2, with |d| = 1 so the leading coefficient is 1.
        b = 2.0 * (ox * dx + oy * dy + oz * dz)
        c = ox**2 + oy**2 + oz**2 - self.radius**2

        discriminant = b**2 - 4.0 * c
        disc_ok = discriminant >= 0.0
        # Same radicand clamp as SphereGeometry: sqrt has an infinite
        # derivative at 0, and a masked-out branch still reaches the
        # backward pass as 0 * inf = NaN without the floor.
        disc_floor = _tol.radicand_floor(be.ones_like(discriminant))
        sqrt_disc = be.where(
            disc_ok,
            be.maximum(discriminant, disc_floor) ** 0.5,
            be.zeros_like(discriminant),
        )

        inf_arr = be.ones_like(discriminant) * be.inf
        if eps is None:
            eps = _tol.accept_t_min(be.abs(origins).max())

        t_near = (-b - sqrt_disc) / 2.0
        t_far = (-b + sqrt_disc) / 2.0

        use_near = self._root_valid(t_near, disc_ok, eps, origins, directions)
        use_far = self._root_valid(t_far, disc_ok, eps, origins, directions)
        use_far = use_far & ~use_near

        hit_mask = use_near | use_far
        t = be.where(use_near, t_near, be.where(use_far, t_far, inf_arr))

        # Rebuild the hit point from a finite t: a rejected root can be
        # +/-inf, and inf * 0 is NaN, which no later masking undoes.
        safe_t = be.where(hit_mask, t, be.zeros_like(t))
        hx = ox + safe_t * dx
        hy = oy + safe_t * dy
        hz = oz + safe_t * dz

        zero = be.zeros_like(hx)
        nx = be.where(hit_mask, hx / self.radius, zero)
        ny = be.where(hit_mask, hy / self.radius, zero)
        nz = be.where(hit_mask, hz / self.radius, zero)
        n_geom = be.stack([-nx, -ny, -nz], axis=1)

        # Flip the shading normal to face the incoming ray.
        dot = dx * nx + dy * ny + dz * nz
        flip = be.where(dot > 0, -1.0, 1.0)
        normals = be.stack([nx * flip, ny * flip, nz * flip], axis=1)

        return t, normals, hit_mask, n_geom

    def _root_valid(self, t, disc_ok, eps, origins, directions):
        """Whether one quadratic root is a genuine forward hit on the wall.

        Args:
            t: The root, shape (N,). May be non-finite where ``disc_ok`` is
                False.
            disc_ok: Whether the quadratic had a real solution, shape (N,).
            eps: Accept threshold for the ray parameter.
            origins: Local ray origins, shape (N, 3).
            directions: Local ray directions, shape (N, 3).

        Returns:
            Boolean array, shape (N,).
        """
        forward = disc_ok & (t > eps)
        safe_t = be.where(forward, t, be.zeros_like(t))
        hx = origins[:, 0] + safe_t * directions[:, 0]
        hy = origins[:, 1] + safe_t * directions[:, 1]
        hz = origins[:, 2] + safe_t * directions[:, 2]
        return forward & self._on_wall(hx, hy, hz)

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return the AABB of the sphere in global coordinates.

        The box is the whole sphere's regardless of the ports: a port
        removes wall, never extent, and a bounding box that shrank with the
        ports would exclude rays that legitimately enter through one.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        t = np.array(transform[0], dtype=float)
        r = as_float(self.radius)
        return AABB(t - r, t + r)
