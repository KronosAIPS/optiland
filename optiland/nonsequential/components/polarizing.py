"""The ideal linear polarizer and the ideal linear retarder (the research repository's issue 5).

Both are thin elements on a disc: a ray crossing the disc keeps its position,
its direction and its medium, and only its polarization and its flux change.
They are the minimal polarization's two component kinds (the research card's
item 7, chapter 06 section 6.5 of the theory):

* the **polarizer** transmits its axis fully and the orthogonal axis with the
  power transmittance ``extinction`` (0 is ideal): the linear diattenuator
  ``diag(1, extinction)`` in Jones form, Mueller ``m00 = (1 + e) / 2``,
  ``m01 = (1 - e) / 2``, ``m22 = sqrt(e)``;
* the **retarder** delays the component orthogonal to its fast axis by
  ``delta = 2 pi W lambda_d / lambda`` (``W`` the retardance in waves at the
  design wavelength ``lambda_d``, scaled as ``1 / lambda`` at every other
  wavelength, R-06-11; with no design wavelength ``delta = 2 pi W``
  everywhere), with no loss: Mueller ``m00 = 1``, ``m22 = cos delta``,
  ``m23 = -sin delta`` in its fast-axis frame, the convention in which a
  quarter wave with its fast axis at +45 degrees takes ``(1, 1, 0, 0)`` to
  ``(1, 0, 0, -1)``.

The element's axis is the placement's local ``x`` axis turned by
``axis_deg`` about its local ``+z`` (the disc normal), towards local ``y``.
For a ray that does not cross along the normal the axis is projected
perpendicular to the ray, the usual model of a thin ideal element.

In a Stokes trace the ray's state is rotated into the element's frame (the
angle from its reference axis to the projected element axis, measured about
the ray's direction), the element applied (:func:`~optiland.nonsequential
.polarization.apply_interface`), and the element axis becomes the ray's
reference axis. The flux factor ``g = m00 + m01 q`` is deterministic: there
is no branch, and ``1 - g`` of the incoming flux is booked as a surface loss
(chapter 10's coating bin). In a scalar trace the rays carry no state and
the element passes its unpolarized transmittance, ``(1 + extinction) / 2``
for a polarizer and 1 for a retarder: exactly the Stokes result for an
unpolarized ray, which is the scalar-equivalence condition this relies on
(chapter 06 section 6.9); a chain of polarizers is not scalar-equivalent,
and a scalar trace of one is not a model of it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.nonsequential import polarization as _pol
from optiland.nonsequential._utils import as_float, as_param, is_tensor
from optiland.nonsequential.components.base import (
    BaseComponent,
    _resident_transform,
)
from optiland.nonsequential.components.compound import CompoundComponent
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.components.ledger import LedgerBooking
from optiland.nonsequential.materials.nsq_material import VACUUM

if TYPE_CHECKING:
    import numpy as np

    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.configs import (
        PolarizerConfig,
        RetarderConfig,
    )
    from optiland.nonsequential.ir.bsdf_ir import BsdfIR
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng

#: The two elements a polarizing component can be.
ELEMENTS = ("polarizer", "retarder")


def _cos_sin_deg(angle):
    """``(cos, sin)`` of an angle in degrees: Python floats for a number, backend
    values (attached) for a tensor."""
    if is_tensor(angle):
        rad = angle * (math.pi / 180.0)
        return be.cos(rad), be.sin(rad)
    rad = math.radians(float(angle))
    return math.cos(rad), math.sin(rad)


class PolarizingComponent(BaseComponent, LedgerBooking):
    """An ideal linear polarizer or retarder on a disc at local z = 0.

    Attributes:
        element: ``"polarizer"`` or ``"retarder"``.
        axis_deg: The transmission axis (polarizer) or the fast axis
            (retarder), degrees from local x towards local y; may be a
            differentiable tensor.
        extinction: The polarizer's power transmittance of the blocked axis.
        retardance_waves: The retarder's retardance in waves at
            ``design_wavelength_um``.
        design_wavelength_um: The wavelength ``retardance_waves`` is given at
            [um], or ``None`` for a retardance that does not scale.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        element: str,
        axis_deg: float,
        aperture_radius: float,
        extinction: float = 0.0,
        retardance_waves: float = 0.25,
        design_wavelength_um: float | None = None,
        name: str = "",
    ) -> None:
        """Initialize PolarizingComponent.

        Args:
            cs: Coordinate system; the disc is its local z = 0.
            element: ``"polarizer"`` or ``"retarder"``.
            axis_deg: The element's axis, degrees from local x towards y.
            aperture_radius: Clear semi-diameter [mm].
            extinction: Polarizer only: the blocked axis's power
                transmittance, in [0, 1].
            retardance_waves: Retarder only: the retardance in waves.
            design_wavelength_um: Retarder only: the wavelength the
                retardance is given at [um], or ``None``.
            name: Optional label.

        Raises:
            ValueError: An unknown element, an extinction outside [0, 1] or
                a non-positive design wavelength.
        """
        if element not in ELEMENTS:
            raise ValueError(f"element must be one of {ELEMENTS}, got {element!r}")
        if not 0.0 <= as_float(extinction) <= 1.0:
            raise ValueError(f"extinction must be in [0, 1], got {extinction!r}")
        if design_wavelength_um is not None and not as_float(design_wavelength_um) > 0:
            raise ValueError(
                f"design_wavelength_um must be positive, got {design_wavelength_um!r}"
            )
        self.reset_ledger()
        super().__init__(
            cs,
            FinitePlaneGeometry(aperture_radius=aperture_radius),
            VACUUM,
            VACUUM,
            None,
            name,
        )
        self.element = element
        self.axis_deg = as_param(axis_deg)
        self.extinction = as_param(extinction)
        self.retardance_waves = as_param(retardance_waves)
        self.design_wavelength_um = (
            None if design_wavelength_um is None else as_float(design_wavelength_um)
        )

    # -- the element --------------------------------------------------------

    def _mueller(self, like, wavelength) -> _pol.InterfaceMueller:
        """The element's interface-form Mueller matrix in its own frame, per ray."""
        one = be.ones_like(like)
        if self.element == "polarizer":
            return _pol.diattenuator_mueller(one, one * self.extinction)
        # delta = 2 pi W (lambda_d / lambda): the ratio first, so a ray at the
        # design wavelength takes the ratio 1 exactly.
        if self.design_wavelength_um is None:
            delta = one * (2.0 * math.pi) * self.retardance_waves
        else:
            ratio = self.design_wavelength_um / wavelength
            delta = ratio * (2.0 * math.pi) * self.retardance_waves
        return _pol.retarder_mueller(be.cos(delta), be.sin(delta))

    def unpolarized_transmittance(self, like):
        """What the element passes of an unpolarized ray: its ``m00``."""
        one = be.ones_like(like)
        if self.element == "polarizer":
            return 0.5 * (one + one * self.extinction)
        return one

    def _axis_global(self, like):
        """The element axis in the global frame, as per-ray component arrays."""
        _, R = _resident_transform(self)
        c, s = _cos_sin_deg(self.axis_deg)
        # local (c, s, 0) -> global: row vector times R^T (R's columns are
        # the local axes in global coordinates)
        one = be.ones_like(like)
        return tuple(one * (c * R[i, 0] + s * R[i, 1]) for i in range(3))

    # -- the interaction ----------------------------------------------------

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
        """Apply the element to every hit ray (in place).

        Args:
            rays: Ray bundle updated in place.
            t: Hit distances [mm], shape (N,).
            normals: Unused -- the element acts in its own axis frame.
            hit_mask: True for rays crossing the element, shape (N,).
            rng: Unused -- nothing here is drawn.
            bsdf_ir: Unused -- an ideal element carries no scatter model.
            n_geom: Geometric normal, for the outgoing origin offset.
            sampling: Unused.
            forced_branch: Unused -- there is one branch.
        """
        self.advance_to_hit(rays, t, hit_mask)

        if rays.pol_q is None:
            g = self.unpolarized_transmittance(rays.flux)
        else:
            k = (rays.L, rays.M, rays.N)
            e = _pol.transport_axis((rays.pol_ex, rays.pol_ey, rays.pol_ez), k)
            a, a2 = _pol.perpendicular_axis(self._axis_global(rays.L), k)
            # A ray along the element axis has no projection of it: it keeps
            # its own frame (and is a ray grazing the disc, which the plane
            # geometry does not report as a hit in any case).
            tol = _pol.degeneracy_tolerance(rays.L)
            ok = a2 > tol * tol
            a = tuple(be.where(ok, x, y) for x, y in zip(a, e, strict=True))
            c2, s2 = _pol.rotation_2psi(e, a, k)
            q, u = _pol.rotate(rays.pol_q, rays.pol_u, c2, s2)
            m = self._mueller(rays.flux, rays.wavelength)
            g, q, u, v = _pol.apply_interface(q, u, rays.pol_v, m)
            rays.pol_q = be.where(hit_mask, q, rays.pol_q)
            rays.pol_u = be.where(hit_mask, u, rays.pol_u)
            rays.pol_v = be.where(hit_mask, v, rays.pol_v)
            rays.pol_ex = be.where(hit_mask, a[0], rays.pol_ex)
            rays.pol_ey = be.where(hit_mask, a[1], rays.pol_ey)
            rays.pol_ez = be.where(hit_mask, a[2], rays.pol_ez)

        # Chapter 10 (10.1): what the element does not pass is a surface
        # loss, booked before the multiply; the event is weight-preserving
        # (w = wg + w(1 - g)), so it leaves no residual.
        self.book_loss(rays.flux, 1.0 - g, hit_mask)
        rays.flux = rays.flux * be.where(hit_mask, g, be.ones_like(g))
        rays.bounce = be.where(hit_mask, rays.bounce + 1, rays.bounce)
        # R-07-6: leave the disc on the side the ray is heading for.
        self.offset_from_surface(rays, n_geom, hit_mask)


class _PolarizingElement(CompoundComponent):
    """One polarizing disc, placed as a compound so the scene registers it by name."""

    element = ""

    def __init__(self, name: str, cs: CoordinateSystem, config) -> None:
        self._name = name
        self._cs = cs
        self._config = config
        self._surfaces = [self._build()]

    def _build(self) -> PolarizingComponent:
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Registry name of this element."""
        return self._name

    @property
    def surfaces(self) -> list[BaseComponent]:
        """The one polarizing disc."""
        return self._surfaces

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """The disc's coordinate system."""
        return self._cs


class Polarizer(_PolarizingElement):
    """An ideal linear polarizer (see :class:`PolarizingComponent`)."""

    element = "polarizer"

    def __init__(self, name: str, cs: CoordinateSystem, config: PolarizerConfig) -> None:
        super().__init__(name, cs, config)

    def _build(self) -> PolarizingComponent:
        cfg = self._config
        return PolarizingComponent(
            self._cs,
            "polarizer",
            cfg.axis_deg,
            cfg.aperture_radius,
            extinction=cfg.extinction,
            name=f"{self._name}.surface",
        )


class Retarder(_PolarizingElement):
    """An ideal linear retarder (see :class:`PolarizingComponent`)."""

    element = "retarder"

    def __init__(self, name: str, cs: CoordinateSystem, config: RetarderConfig) -> None:
        super().__init__(name, cs, config)

    def _build(self) -> PolarizingComponent:
        cfg = self._config
        return PolarizingComponent(
            self._cs,
            "retarder",
            cfg.fast_axis_deg,
            cfg.aperture_radius,
            retardance_waves=cfg.retardance_waves,
            design_wavelength_um=cfg.design_wavelength_um,
            name=f"{self._name}.surface",
        )
