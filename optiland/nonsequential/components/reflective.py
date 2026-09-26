"""Reflective component for Non-Sequential Raytracing.

Mirrors and baffles with coating. Specular reflection (or BSDF scatter).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential.components.base import BaseComponent, _resident_transform
from optiland.nonsequential.components.coating_support import (
    reject_polarized_coating,
    resolve_reflectance,
)
from optiland.nonsequential.components.ledger import LedgerBooking
from optiland.nonsequential.components.sampling_support import scatter_branch
from optiland.nonsequential.materials.nsq_material import VACUUM

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.coatings import BaseCoating
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.components.geometry.base import ComponentGeometry
    from optiland.nonsequential.ir.bsdf_ir import BsdfIR
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng


class ReflectiveComponent(BaseComponent, LedgerBooking):
    """Purely reflective optical element (mirror, baffle).

    Reflects rays specularly (or via BSDF). Does not transmit.

    Attributes:
        cs: Coordinate system.
        geometry: Surface geometry.
        reflectance: Constant, callable(wavelength_um), or unpolarized
            BaseCoating giving the fraction of flux reflected.
        bsdf: Optional BSDF for scatter. None = specular mirror.
        name: Optional label.
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        geometry: ComponentGeometry,
        reflectance: float | Callable[[be.ndarray], be.ndarray] | BaseCoating,
        bsdf: BaseBSDF | None = None,
        material_front: NSQMaterial = VACUUM,
        name: str = "",
        scatter_fraction: float = 1.0,
    ) -> None:
        """Initialize ReflectiveComponent.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            reflectance: Fraction of incident flux reflected. Required --
                unlike ``bsdf``, there is no "perfect mirror" default:
                a mirror built without saying how reflective it is would
                otherwise silently reflect 100% of incoming flux. Accepts a
                constant in [0, 1], a ``callable(wavelength_um) ->
                reflectance``, or an unpolarized ``optiland.coatings
                .BaseCoating`` (e.g. ``SimpleCoating``); a
                ``BaseCoatingPolarized`` instance raises
                ``NotImplementedError`` at construction.
            bsdf: Optional BSDF scatter model. None = perfect mirror.
            material_front: Medium on the front side (default: vacuum).
            name: Optional label.
            scatter_fraction: Probability that a hit ray is routed through
                ``bsdf`` rather than specularly reflected.
        """
        reject_polarized_coating(reflectance, surface_name=name)
        self.reflectance = reflectance
        self.reset_ledger()
        super().__init__(
            cs,
            geometry,
            material_front,
            material_front,
            bsdf,
            name,
            scatter_fraction=scatter_fraction,
        )

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
        """Apply specular (or BSDF) reflection at hit points (in-place).

        Args:
            rays: Ray bundle updated in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays hitting this component, shape (N,).
            rng: Keyed PCG32 RNG. Draws are keyed by this ray's own id and
                its bounce count as of this interaction.
            bsdf_ir: This surface's lowered BSDF descriptor. Whether the
                scatter branch below runs at all is decided from
                ``bsdf_ir.kind != "none"`` (verified by the caller to match
                ``self.bsdf``), not from ``self.bsdf is not None``.
            n_geom: Geometric (unflipped) surface normal in the global
                frame, shape (N, 3). Not used to choose a medium -- a
                mirror never transmits -- but it is the direction the
                outgoing origin is offset along (R-07-6; see
                :meth:`BaseComponent.offset_from_surface`).
            sampling: Unused -- a mirror has no Fresnel reflect/transmit
                branch to importance-bias; its reflectance is
                applied as a deterministic flux weight, not a stochastic
                draw.
            forced_branch: Unused -- bounded splitting only applies
                to ``RefractiveComponent``'s Fresnel branch.
        """
        # Captured before any mutation below (including the bounce
        # increment at the end of this method) rebinds rays.bounce. Handed
        # to the RNG as they are: the generator works in the active
        # backend, so keying a draw copies nothing to the host.
        ray_id_key = rays.ray_id
        bounce_key = rays.bounce

        # Advance to the hit point, rebuilt in this surface's own frame --
        # see BaseComponent.advance_to_hit.
        self.advance_to_hit(rays, t, hit_mask)

        dirs = be.stack([rays.L, rays.M, rays.N], axis=1)

        # Specular reflection: d - 2*(d.n)*n. Always computed, because with a
        # scatter_fraction below 1 it is the fallback for rays that do not
        # enter the BSDF lobe.
        raw_dot = (dirs * normals).sum(axis=1, keepdims=True)
        reflected = dirs - 2.0 * raw_dot * normals
        norms_r = (reflected * reflected).sum(axis=1, keepdims=True) ** 0.5
        reflected = reflected / (norms_r + _tol.tiny_for(norms_r))
        hit_col = hit_mask[:, None]
        new_dirs = be.where(hit_col, reflected, dirs)

        # D-3: reflectance is mandatory, applied to every hit ray regardless
        # of whether it also scatters through a BSDF below.
        R = resolve_reflectance(self.reflectance, rays.wavelength)
        # Ch. 10 (10.1): the (1 - R) a mirror below unit reflectance removes
        # is a destination, not a leak. Booked before the multiply, while
        # the incoming weight is still in hand. The event is exactly
        # weight-preserving -- w = wR + w(1-R) -- so it leaves no residual.
        self.book_loss(rays.flux, 1.0 - R, hit_mask)
        rays.flux = rays.flux * be.where(hit_mask, R, be.ones_like(R))

        if bsdf_ir.kind != "none":
            # Compute BSDF for all N rays; where-select only scattering rays.
            # A mirror has no far side to transmit into (material_front ==
            # material_back), so the lobe's reflect/transmit side is unused
            # here -- unlike RefractiveComponent, there is no second medium
            # for a "transmitted" scattered ray to have entered.
            bsdf_dirs, bsdf_weights, _bsdf_transmitted = self.bsdf.sample(
                rays.num_rays,
                dirs,
                normals,
                rays.wavelength,
                rng,
                ray_id_key,
                bounce_key,
                frame=_resident_transform(self)[1],
            )
            # Route only a scatter_fraction of the hit rays into the lobe; the
            # rest reflect specularly.
            scatters, sf_gate = scatter_branch(
                self.scatter_fraction, hit_mask, rng, ray_id_key, bounce_key,
                owner=self,
            )
            # The scatter branch is a detached decision with a compensating
            # weight: unbiased, but not weight-preserving per realisation.
            self.book_residual(rays.flux, 1.0 - sf_gate, hit_mask)
            rays.flux = rays.flux * be.where(hit_mask, sf_gate, be.ones_like(sf_gate))

            new_dirs = be.where(scatters[:, None], bsdf_dirs, new_dirs)
            # What the lobe did not return is a surface loss when the weight
            # is a physical fraction of the incident flux, and a surface
            # loss plus a zero-mean event residual when it is a sampling
            # weight -- BaseBSDF.weight_is_albedo says which.
            bsdf_gate = be.where(
                scatters, bsdf_weights, be.ones_like(bsdf_weights)
            )
            albedo_gate = bsdf_gate
            if not self.bsdf.weight_is_albedo:
                albedo = self.bsdf.reflectance(dirs, normals, rays.wavelength)
                albedo_gate = be.where(scatters, albedo, be.ones_like(albedo))
            self.book_lobe(rays.flux, bsdf_gate, albedo_gate, hit_mask)
            rays.flux = rays.flux * bsdf_gate

        rays.L = new_dirs[:, 0]
        rays.M = new_dirs[:, 1]
        rays.N = new_dirs[:, 2]

        # n_current unchanged (reflection stays in same medium)
        rays.bounce = be.where(hit_mask, rays.bounce + 1, rays.bounce)

        # R-07-6: push the reflected (or scattered) origin clear of the
        # mirror on the side it leaves into -- see
        # BaseComponent.offset_from_surface. Signed per ray from the final
        # direction, so a BSDF lobe's direction is the one that decides it.
        self.offset_from_surface(rays, n_geom, hit_mask)
