"""Refractive component for Non-Sequential Raytracing.

Transmits and reflects (lenses, prisms, windows). Uses detached-sample /
attached-weight Fresnel splitting for differentiable Monte Carlo.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential._utils import resident_scalar
from optiland.nonsequential.components.base import BaseComponent, _resident_transform
from optiland.nonsequential.components.coating_support import (
    evaluate_transmissive_coating,
    reject_polarized_coating,
)
from optiland.nonsequential.components.ledger import LedgerBooking
from optiland.nonsequential.components.sampling_support import (
    detached as _detached,
)
from optiland.nonsequential.components.sampling_support import (
    scatter_branch,
)
from optiland.nonsequential.materials.nsq_material import medium_stack_id_value
from optiland.nonsequential.ray_bundle import (
    MEDIUM_STACK_EMPTY,
    MEDIUM_STACK_MAX_DEPTH,
    MEDIUM_STACK_OVERFLOW_RAISES,
    MediumStackOverflowError,
    backend_bool_full,
    backend_gather_slot,
    backend_int_full,
    backend_masked_fill,
    backend_scatter_slot,
)
from optiland.nonsequential.rng import EventSlot
from optiland.nonsequential.sampling import resolve_reflect_prob

if TYPE_CHECKING:
    from optiland.coatings import BaseCoating
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.components.geometry.base import ComponentGeometry
    from optiland.nonsequential.ir.bsdf_ir import BsdfIR
    from optiland.nonsequential.ir.scene_ir import SamplingPolicy
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng


def refraction_cosine(sin2_t):
    """Cosine of the refraction angle, and the total-internal-reflection mask.

    ``cos(theta_t) = sqrt(w)`` with ``w = 1 - sin^2(theta_t)``. Two rules,
    both from docs/theory/08_precision.md:

    * **The domain test is in the dtype's own units** (section 8.7, the
      radicand row; R-08-7). A radicand below ``k u_T``
      (:func:`optiland.nonsequential._tol.radicand_min`, ``k = 4``) cannot be
      told from zero in the working dtype, so no transmitted wave is asserted
      there: the ray is totally reflected. The Fresnel transmittance is
      continuous at the critical angle (it goes to zero with ``cos(theta_t)``),
      so this moves the transmittance by at most its value at
      ``w = k u_T`` -- about 5.7 ``sqrt(4 u_T)``, 2.8e-3 at float32 and 1.2e-7
      at float64 for N-BK7 against vacuum -- and only for incidence within
      ``k u_T / (r^2 sin 2 theta_i)`` of the critical angle (1.0e-7 rad at
      float32, 2.0e-16 rad at float64).
    * **The input is masked, not the output** (section 8.8; R-08-8). Lanes
      that are totally reflected take the square root of 1 and discard it,
      so the backward pass never meets ``sqrt'(0)``; every other lane has
      ``w >= k u_T`` and a finite derivative.

    What this replaces. The radicand used to be clamped to
    ``_tol.radicand_floor``, a float64 budget of 1e-12 scaled by the dtype's
    ulp ratio: 1e-12 at float64, but 5.37e-4 at float32, so ``cos(theta_t)``
    never fell below 0.0232 and every refraction within 1.33 degrees of
    grazing was bent and weighted as if it were 1.33 degrees from grazing.
    One millidegree inside the critical angle of N-BK7 against vacuum that
    gave a transmittance of 0.124 where float64 reads 0.0373 on the same
    rays; the float32 radicand itself was right to 0.1 u_T. [measured,
    benchmarks/nonsequential/float32_mechanisms.py]

    At float64 the two rules agree bit for bit wherever ``w >= 1e-12``: the
    clamp returned ``w`` there and ``sqrt(w)`` is formed from the same value.
    They differ only within ``4.4e-13`` rad of the critical angle, where the
    old clamp asserted a transmitted wave with ``cos(theta_t) = 1e-6`` (a
    transmittance of 5.8e-6 at N-BK7's critical angle) and this one
    reflects totally when ``w`` is below ``4 u_64``.

    Args:
        sin2_t: ``(n1 / n2)^2 sin^2(theta_i)`` per ray, in the working
            backend and dtype.

    Returns:
        ``(tir, cos_t)``: the boolean total-internal-reflection mask and the
        cosine of the refraction angle (0 where ``tir``).
    """
    w = 1.0 - sin2_t
    tir = w < _tol.radicand_min(w)
    w_safe = be.where(tir, be.ones_like(w), w)
    cos_t = be.where(tir, be.zeros_like(w), w_safe**0.5)
    return tir, cos_t


class RefractiveComponent(BaseComponent, LedgerBooking):
    """Refractive optical element (lens, prism, window).

    At each interface, Fresnel splitting uses the detached-sample /
    attached-weight scheme: the branch decision (reflect vs transmit) is
    drawn from a detached probability, while the throughput weight carries
    the attached reflectance so gradients flow through material parameters.

    The two materials name the media on either side of the surface, and the
    component works out which one a ray is leaving by comparing the ray
    direction against the surface's geometric normal (``n_geom``, fixed per
    surface point, pointing from ``material_front`` toward ``material_back``)
    -- never by comparing refractive index values. Crossing direction
    therefore does not matter: the same surface refracts correctly for a ray
    on its way in, for a ghost or retro-reflection coming back through, and
    for the far side of a closed solid modelled as a single geometry, even
    when the two adjacent media have nearly identical indices (a cemented
    doublet, oil immersion).

    Attributes:
        cs: Coordinate system.
        geometry: Surface geometry.
        material_front: Medium on the front (normal-facing) side.
        material_back: Medium on the back side.
        bsdf: Optional BSDF for scatter. None = specular.
        name: Optional label.
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
        coating: BaseCoating | None = None,
    ) -> None:
        """Initialize RefractiveComponent.

        Args:
            cs: Coordinate system.
            geometry: Surface geometry.
            material_front: Front-side medium. By contract (see
                ``ComponentGeometry.ray_intersect``), this is the medium on
                the side the geometry's *unflipped* normal points away
                from -- for the analytic geometries, the local -z side.
            material_back: Back-side medium -- the side the geometry's
                unflipped normal points toward (local +z, for the analytic
                geometries).
            bsdf: Optional BSDF scatter model.
            name: Optional label.
            scatter_fraction: Probability that a hit ray is routed through
                ``bsdf`` rather than refracted.
            coating: Optional ``optiland.coatings.BaseCoating`` (e.g.
                ``SimpleCoating``) or a
                ``coating_support.UnpolarizedThinFilmCoating``. When set, its
                R/T (via ``evaluate_transmissive_coating`` --
                wavelength-only for ``SimpleCoating``, wavelength- and
                angle-of-incidence-dependent for
                ``UnpolarizedThinFilmCoating``) replace the bare Fresnel R/T
                so NSQ agrees with the sequential engine's coating model.
                Must be unpolarized -- a ``BaseCoatingPolarized`` instance
                raises ``NotImplementedError`` immediately, since NSQ rays
                carry no polarization state.
        """
        reject_polarized_coating(coating, surface_name=name)
        self.coating = coating
        self.reset_ledger()
        super().__init__(
            cs,
            geometry,
            material_front,
            material_back,
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
        """Apply Fresnel refraction/reflection at hit points (in-place).

        Uses detached-sample / attached-weight Fresnel: the reflect/transmit
        branch decision is drawn from a detached probability so stochastic
        choices do not block gradients; the throughput weight multiplier
        carries the attached reflectance so ∂flux/∂R is non-zero. When
        ``self.coating`` is set, its R/T replace the bare Fresnel values
        (still forced to R=1/T=0 under TIR, where no coating can restore a
        transmitted wave).

        Args:
            rays: Ray bundle updated in-place.
            t: Hit distances [mm], shape (N,).
            normals: Surface normals in global frame, shape (N, 3).
            hit_mask: True for rays hitting this component, shape (N,).
            rng: Keyed PCG32 RNG (used for detached sampling only). Draws
                are keyed by this ray's own id and its bounce count as of
                this interaction, so they are independent of batch_size,
                compaction, and every other ray in the bundle.
            bsdf_ir: This surface's lowered BSDF descriptor. Whether the
                scatter branch below runs at all is decided from
                ``bsdf_ir.kind != "none"`` (verified by the caller to match
                ``self.bsdf``), not from ``self.bsdf is not None``.
            n_geom: Geometric surface normal in global frame, shape (N, 3),
                fixed per surface point and pointing from ``material_front``
                toward ``material_back``. Used, not ``rays.n_current``,
                to determine which material a ray is entering.
            sampling: The scene's rare-path sampling policy.
                Resolves the reflect-branch sampling probability -- see
                :func:`optiland.nonsequential.sampling.resolve_reflect_prob`.
                ``None`` defaults to ``reflect_prob="fresnel"`` (today's
                behaviour). Ignored when ``forced_branch`` is set.
            forced_branch: ``"reflect"`` or ``"transmit"`` to deterministically
                force the branch (weight = R or T exactly, no importance
                division) instead of drawing it stochastically. Used only by
                the NumPy forward engine's bounded-splitting orchestration
                to build both children of a split ray.
        """
        # Every RNG draw in this call is keyed to the ray's identity as of
        # this specific interaction event -- captured before any of the
        # mutations below (including the bounce increment) rebind
        # rays.bounce out from under us. The keys are handed to the RNG in
        # whatever array type the bundle holds: the generator works in the
        # active backend, so nothing is copied to the host to key a draw.
        ray_id_key = rays.ray_id
        bounce_key = rays.bounce

        # Advance hit rays to the intersection point. Rebuilt in this
        # surface's own frame from the advance and the residual the
        # intersection was solved with, not p + t*d -- see
        # BaseComponent.advance_to_hit.
        self.advance_to_hit(rays, t, hit_mask)

        dirs = be.stack([rays.L, rays.M, rays.N], axis=1)
        wl = rays.wavelength  # µm

        # Determine n1 and n2 for each ray (based on side of the surface)
        dot = (dirs * normals).sum(axis=1)  # signed cos_theta
        cos_theta_i = be.abs(dot)

        # Evaluate the front/back indices at each wavelength -- attached
        # (differentiable).
        n_front = self.material_front.n(wl)
        n_back = self.material_back.n(wl)
        # Extinction coefficients: tracked the same way as n1/n2 below
        # so rays.k_current always reflects the medium a ray is currently
        # travelling through, for Beer-Lambert attenuation on its next hop.
        k_front = self.material_front.k(wl)
        k_back = self.material_back.k(wl)

        # n_geom is fixed per surface point and points from material_front
        # toward material_back (D-1; see ComponentGeometry.ray_intersect),
        # independent of which side the ray approaches from. This replaces
        # the old index-proximity heuristic
        # (`abs(n1 - n2_back) < abs(n1 - n2_front)` against rays.n_current),
        # which silently mis-resolved whenever the two adjacent media had
        # similar indices (a cemented doublet, oil immersion) or the ray
        # took an unexpected path (a ghost re-entering a solid). Comparing
        # ray direction against n_geom is direction-agnostic *and*
        # index-value-agnostic: correct for a ray on its way in, a
        # retro-reflection, or the far side of a closed solid.
        dot_geom = (dirs * n_geom).sum(axis=1)
        entering_back = dot_geom > 0.0
        n1 = be.where(entering_back, n_front, n_back)
        n2 = be.where(entering_back, n_back, n_front)
        k1 = be.where(entering_back, k_front, k_back)
        k2 = be.where(entering_back, k_back, k_front)

        # Fresnel reflectance (unpolarized, attached). Denominator guards are
        # sqrt(smallest normal) of the working dtype, not a bare 1e-30 --
        # docs/theory/08_precision.md sec 8.8.
        n_ratio = n1 / (n2 + _tol.tiny_for(n2))
        sin2_t = n_ratio**2 * (1.0 - cos_theta_i**2)
        # TIR is decided on the radicand the dtype can resolve, and every
        # radicand it does resolve is square-rooted as it is -- see
        # refraction_cosine below for the float32 failure the old clamp made.
        tir, cos_theta_t = refraction_cosine(sin2_t)

        rs_denom = n1 * cos_theta_i + n2 * cos_theta_t
        rs = (n1 * cos_theta_i - n2 * cos_theta_t) / (
            rs_denom + _tol.tiny_for(rs_denom)
        )
        rp_denom = n2 * cos_theta_i + n1 * cos_theta_t
        rp = (n2 * cos_theta_i - n1 * cos_theta_t) / (
            rp_denom + _tol.tiny_for(rp_denom)
        )
        R_fresnel = be.where(tir, be.ones_like(rs), 0.5 * (rs**2 + rp**2))

        # A coating overrides the bare Fresnel R/T with its own (possibly
        # wavelength- and angle-dependent, possibly lossy: R + T < 1) values
        # -- except under TIR, where there is no real transmitted wave
        # regardless of what the coating claims, so reflection stays forced
        # to R=1, T=0. evaluate_transmissive_coating dispatches on what the
        # coating exposes -- see coating_support.py -- so a scalar
        # SimpleCoating and an angle-dependent UnpolarizedThinFilmCoating
        # both flow through this one call.
        if self.coating is not None:
            R_used, T_used = evaluate_transmissive_coating(
                self.coating, wl, cos_theta_i
            )
        else:
            R_used = R_fresnel
            T_used = 1.0 - R_fresnel
        R_used = be.where(tir, be.ones_like(R_used), R_used)
        T_used = be.where(tir, be.zeros_like(T_used), T_used)

        # --- Detached-sample / attached-weight ---
        # The decision is detached with detach(), not by copying the value
        # to the host: a branch probability is a number the sampler must
        # not differentiate through, which is a graph property, not a
        # question of which memory it lives in.
        R_det = _detached(R_used)

        if forced_branch is not None:
            # Bounded-splitting orchestration (PR11, NumPy forward engine
            # only): the branch is fixed, not drawn, and the weight is the
            # exact deterministic R or T -- no importance division, since
            # there is no probability being compensated for. TIR rays are
            # unaffected: T_used is already forced to 0 there, so a forced
            # "transmit" branch on a TIR ray correctly carries zero flux
            # rather than raising or fabricating a wave that cannot exist.
            do_reflect = backend_bool_full(
                tir.shape, forced_branch == "reflect", like=tir
            )
            weight = be.where(do_reflect, R_used, T_used)
        else:
            # Importance-biased branch probability: generalises
            # the plain-Fresnel estimator (p == R) to any detached
            # probability p, dividing by p rather than R_det so the
            # estimator stays unbiased for any p in (0, 1) -- only the
            # variance changes. reflect_prob="fresnel" reproduces the
            # original weight formula exactly.
            p_be = resolve_reflect_prob(sampling, R_det) if sampling else R_det
            # [1e-12, 1 - 1e-12] at float64, [1e-12, 1 - 4 u] where the upper
            # literal would round to 1 (float32): see _tol.branch_probability_bounds.
            p_lo, p_hi = _tol.branch_probability_bounds(p_be)
            p_det = be.clip(_detached(p_be), p_lo, p_hi)
            u = rng.uniform(ray_id_key, bounce_key, EventSlot.FRESNEL_BRANCH)
            do_reflect = (u < p_det) | tir

            # Throughput weight: forward value is 1.0 in expectation; carries
            # gradients through R/T. Generalizes the plain-Fresnel
            # weight_transmit = (1-R)/(1-R_det) to allow T != 1-R (coating
            # absorption) *and* p != R (importance biasing): the branch is
            # still a single reflect-vs-transmit draw with P(reflect) = p,
            # so E[weight] = R on the reflect branch and T on the transmit
            # branch regardless of p -- exact flux conservation in
            # expectation, with the shortfall R+T<1 taken up by the
            # deterministic T weight rather than a separate absorption draw.
            # For TIR rays weight stays 1.0 (full reflection is deterministic).
            weight_reflect = R_used / (p_det + _tol.tiny_for(p_det))
            weight_transmit = T_used / (1.0 - p_det + _tol.tiny_for(p_det))
            weight = be.where(do_reflect, weight_reflect, weight_transmit)
            # TIR: weight is exactly 1
            weight = be.where(tir, be.ones_like(weight), weight)

        # Ch. 10 (10.1) and (10.2), booked together because they share the
        # same incoming weight. What the surface absorbs is w(1 - R - T),
        # which a lossy coating makes non-zero and a bare Fresnel interface
        # leaves at zero; what the estimator neither passed on nor booked is
        # w(R + T - weight), zero for the plain-Fresnel choice p = R and for
        # TIR, non-zero under importance biasing. The two sum to
        # w(1 - weight), the whole change in flux, so nothing is counted
        # twice and nothing is left over.
        #
        # A forced branch is one half of an exhaustive split: this call and
        # its sibling start from the same incoming weight w and hand on wR
        # and wT, so together they preserve w(R + T) exactly and the event
        # residual is zero term by term (docs/theory/10_ledger_and_
        # diagnostics.md 10.1: an exhaustive split with physical weights has
        # delta = 0). The split's one loss, w(1 - R - T), is booked once, on
        # the reflect child; the transmit child books nothing. Booking each
        # child as if it were a single draw -- what this did before -- put
        # w into the residual per split, so every split trace reported a
        # conservation error of order the split count.
        if forced_branch is None:
            self.book_loss(rays.flux, 1.0 - R_used - T_used, hit_mask)
            self.book_residual(rays.flux, R_used + T_used - weight, hit_mask)
        elif forced_branch == "reflect":
            self.book_loss(rays.flux, 1.0 - R_used - T_used, hit_mask)

        # Apply weight to flux for hit rays
        rays.flux = rays.flux * be.where(hit_mask, weight, be.ones_like(weight))

        # Compute reflected direction: d - 2*(d.n)*n
        raw_dot = (dirs * normals).sum(axis=1, keepdims=True)
        reflected = dirs - 2.0 * raw_dot * normals
        norms_r = (reflected * reflected).sum(axis=1, keepdims=True) ** 0.5
        reflected = reflected / (norms_r + _tol.tiny_for(norms_r))

        # Compute refracted direction (Snell's law, vector form)
        n_facing = be.where(raw_dot < 0, normals, -normals)
        cos_i_pos = be.abs(raw_dot)
        n_ratio_col = n_ratio[:, None]
        cos_t_col = cos_theta_t[:, None]
        refracted = (
            n_ratio_col * dirs + (n_ratio_col * cos_i_pos - cos_t_col) * n_facing
        )
        norms_t = (refracted * refracted).sum(axis=1, keepdims=True) ** 0.5
        refracted = refracted / (norms_t + _tol.tiny_for(norms_t))

        # Select direction based on branch decision
        do_reflect_col = do_reflect[:, None]
        new_d = be.where(do_reflect_col, reflected, refracted)

        # Apply only to hit rays
        hit_col = hit_mask[:, None]
        rays.L = be.where(hit_col[:, 0], new_d[:, 0], rays.L)
        rays.M = be.where(hit_col[:, 0], new_d[:, 1], rays.M)
        rays.N = be.where(hit_col[:, 0], new_d[:, 2], rays.N)

        # Update n_current/k_current: stays medium 1 on reflect, becomes
        # medium 2 on refract.
        rays.n_current = be.where(
            hit_mask, be.where(do_reflect, n1, n2), rays.n_current
        )
        rays.k_current = be.where(
            hit_mask, be.where(do_reflect, k1, k2), rays.k_current
        )

        # D1: medium stack push/pop -- a diagnostic cross-check, never fed
        # back into n1/n2 above. Direction is decided by medium identity,
        # not entering_back (a Lens's two faces share one +n_geom
        # convention but opposite interior sides). Reaching ambient (id 0)
        # always unwinds the whole stack, since abutting media (e.g. a
        # cemented doublet) push sequentially without true nesting; a pop
        # at depth 0 is counted as a leak. Reaching a non-ambient medium
        # that matches one level below the top is a true nesting exit
        # (pop); anything else pushes.
        #
        # The state is device-resident and so is the update: masks and
        # integer arithmetic over every ray, one gather and one scatter per
        # stack operation, with no row indexing, no compaction to the hit
        # rows and no Python branch on ray data. Rays that do not transmit
        # take every lane too and write their own value back.
        stack = rays.medium_stack
        depth = rays.medium_depth
        transmit = hit_mask & ~do_reflect
        zero = backend_int_full(depth.shape, 0, like=depth, bits=32)

        # The ids as Python ints; inside the compiled bounce step, as device
        # values (medium_stack_id_value), the same numbers.
        front_id = medium_stack_id_value(self.material_front, like=depth)
        back_id = medium_stack_id_value(self.material_back, like=depth)
        # entering_back is a device bool mask; one operand must carry the
        # integer dtype so the ids are not cast to the working float type.
        # Every integer constant below is held on the device beside the
        # stack (resident_scalar) rather than uploaded at every call.
        front_arr = backend_int_full(depth.shape, front_id, like=depth, bits=64)
        # Inside the compiled bounce step the id is already a 0-d int64
        # value on the device (the one resident_scalar would build).
        back_arr = (
            back_id
            if be.is_torch_tensor(back_id)
            else resident_scalar(self, "medium", back_id, front_arr)
        )
        mat2 = be.where(entering_back, back_arr, front_arr)

        ambient = transmit & (mat2 == 0)
        underflow = ambient & (depth == 0)

        # "Below the top" is the entry one level under the stack pointer;
        # a stack shallower than two levels has ambient (0) below it. The
        # gather index is clamped so every lane reads a valid slot, and the
        # mask is applied to the value, not to the index.
        deep = depth >= 2
        below = backend_gather_slot(stack, be.where(deep, depth - 2, zero))
        below = be.where(deep, below, resident_scalar(self, "stack", 0, below))

        non_ambient = transmit & (mat2 != 0)
        pop = non_ambient & (depth >= 1) & (mat2 == below)
        push = non_ambient & ~pop
        overflow = push & (depth >= MEDIUM_STACK_MAX_DEPTH)

        # Turning an overflow into a raise means reducing a per-ray mask to
        # a Python bool. That is free while the stack is a host array, and a
        # device synchronisation once it is not -- so on a device the push
        # saturates (the ray keeps the medium it had) and the event is
        # counted below with the other stack inconsistencies, which is what
        # MEDIUM_STACK_OVERFLOW_RAISES = False already asked for. A device
        # kernel cannot raise; it can only count.
        raises = MEDIUM_STACK_OVERFLOW_RAISES and not be.is_torch_tensor(stack)
        if raises and be.any(overflow):
            raise MediumStackOverflowError(
                f"Medium stack exceeded MEDIUM_STACK_MAX_DEPTH="
                f"{MEDIUM_STACK_MAX_DEPTH} at surface "
                f"{self.name or type(self).__name__!r}. This "
                "indicates either pathologically deep volume "
                "nesting or a geometry defect that pushes "
                "without popping."
            )
        push_ok = push & ~overflow

        # Unwinding to ambient empties the whole row -- the one pass over the
        # full table. In place, like the push and pop below: a bundle owns its
        # own stack table (compact/select/concat all copy it).
        stack = backend_masked_fill(stack, ambient[:, None], MEDIUM_STACK_EMPTY)

        # Pop: the vacated top slot becomes empty. Push: the slot the stack
        # pointer names takes the new medium id. Both write every row, the
        # unaffected ones writing back what they already held.
        slot = be.where(pop, depth - 1, zero)
        held = backend_gather_slot(stack, slot)
        empty = resident_scalar(self, "stack", MEDIUM_STACK_EMPTY, held)
        stack = backend_scatter_slot(stack, slot, be.where(pop, empty, held))
        slot = be.where(push_ok, depth, zero)
        held = backend_gather_slot(stack, slot)
        rays.medium_stack = backend_scatter_slot(
            stack, slot, be.where(push_ok, mat2, held)
        )

        delta = be.where(
            pop,
            resident_scalar(self, "depth", -1, zero),
            be.where(push_ok, resident_scalar(self, "depth", 1, zero), zero),
        )
        next_depth = depth + delta
        rays.medium_depth = be.where(
            ambient, resident_scalar(self, "depth", 0, next_depth), next_depth
        )

        # A pop on an empty stack, and (when the raise above is disabled) a
        # push past the maximum depth, are both stack inconsistencies and
        # are counted together for Diagnostics.medium_stack_underflows.
        inconsistent = underflow | overflow
        rays.medium_stack_underflows = be.where(
            inconsistent,
            rays.medium_stack_underflows + 1,
            rays.medium_stack_underflows,
        )

        # Update bounce count
        rays.bounce = be.where(hit_mask, rays.bounce + 1, rays.bounce)

        # Apply BSDF scatter if present (compute for all rays, use where to select)
        if bsdf_ir.kind != "none":
            # Compute BSDF for all N rays; where-select only hit rays
            lobe_in_dirs = be.stack([rays.L, rays.M, rays.N], axis=1)
            bsdf_dirs, bsdf_weights, bsdf_transmitted = self.bsdf.sample(
                rays.num_rays,
                lobe_in_dirs,
                normals,
                rays.wavelength,
                rng,
                ray_id_key,
                bounce_key,
                frame=_resident_transform(self)[1],
            )
            # Route only a scatter_fraction of the hit rays through the BSDF;
            # the rest keep the refracted direction computed above.
            scatters, sf_gate = scatter_branch(
                self.scatter_fraction, hit_mask, rng, ray_id_key, bounce_key,
                owner=self,
            )
            # A detached decision with a compensating weight: unbiased, but
            # not weight-preserving on this realisation.
            self.book_residual(rays.flux, 1.0 - sf_gate, hit_mask)
            rays.flux = rays.flux * be.where(hit_mask, sf_gate, be.ones_like(sf_gate))

            scatter_col = scatters[:, None]
            cur_dirs = be.stack([rays.L, rays.M, rays.N], axis=1)
            new_dirs = be.where(scatter_col, bsdf_dirs, cur_dirs)
            rays.L = new_dirs[:, 0]
            rays.M = new_dirs[:, 1]
            rays.N = new_dirs[:, 2]
            bsdf_gate = be.where(scatters, bsdf_weights, be.ones_like(bsdf_weights))
            # What the lobe did not return is a surface loss when the weight
            # is a physical fraction of the incident flux, and a surface loss
            # plus a zero-mean event residual when it is a sampling weight --
            # BaseBSDF.weight_is_albedo says which.
            albedo_gate = bsdf_gate
            if not self.bsdf.weight_is_albedo:
                albedo = self.bsdf.reflectance(
                    lobe_in_dirs, normals, rays.wavelength
                )
                albedo_gate = be.where(scatters, albedo, be.ones_like(albedo))
            self.book_lobe(rays.flux, bsdf_gate, albedo_gate, hit_mask)
            rays.flux = rays.flux * bsdf_gate

            # D-4: a scattered ray's medium is decided by its own lobe's
            # reflect/transmit side, not by the Fresnel branch draw above --
            # that draw only describes what happens to a ray that does NOT
            # enter the BSDF lobe. Re-resolves n_current/k_current for
            # exactly the scattered rays.
            bsdf_in_medium2 = bsdf_transmitted
            rays.n_current = be.where(
                scatters, be.where(bsdf_in_medium2, n2, n1), rays.n_current
            )
            rays.k_current = be.where(
                scatters, be.where(bsdf_in_medium2, k2, k1), rays.k_current
            )
            # Note: the medium stack above is not re-synced for scattered
            # rays -- a transmissive BSDF lobe can cross the boundary
            # opposite to the deterministic Fresnel branch's push/pop, so
            # scatter_fraction > 0 on a volume boundary can drift the stack
            # out of sync with n_current (diagnostic-only; n1/n2 stay
            # correct either way).

        # R-07-6: the outgoing ray starts a conservative distance clear of
        # this surface, on the side it leaves into. Last, so the direction
        # it is signed by is the final one -- a BSDF lobe above may have
        # replaced the specular/refracted direction, and a transmissive
        # lobe leaves on the other side from a reflective one.
        self.offset_from_surface(rays, n_geom, hit_mask)
