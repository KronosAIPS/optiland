"""ArrayBackend -- the one array-based trace loop.

One Monte Carlo bounce loop serves every array library the engine runs on.
It is written in ``optiland.backend`` (``be.*``) operations and the ray
bundle's own methods; nothing in it names NumPy or torch.  A backend
subclass supplies only four things:

1. **array creation and device placement** -- :meth:`ArrayBackend._prepare_bundle`
   promotes a freshly generated bundle onto the library and device the
   backend works in;
2. **compaction** -- :meth:`ArrayBackend._maybe_compact` drops dead rays, or
   keeps the bundle at a fixed width;
3. **loop control** -- :attr:`ArrayBackend.host_reads_free` says whether a
   reduction to a Python bool is free (a host array) or a device
   synchronisation (a device array), which decides whether the loop exits
   early on "no ray alive" or runs a fixed trip count;
4. **capability** -- :attr:`ArrayBackend.supports_splitting`, since bounded
   splitting grows the bundle and so needs a variable width.

Everything else -- intersection, dispatch, detector recording, the medium
stack, the kill checks, roulette and the flux ledger -- is one body of code.

Kramer Harrison, 2026
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential._utils import (
    DEFAULT_BATCH_SIZE,
    distribute_ray_budget,
    estimate_bounding_scale,
    get_detector_names,
)
from optiland.nonsequential.backends.base import TracerBackend
from optiland.nonsequential._tally import Tally
from optiland.nonsequential.detectors.dispatch import intersect_detectors
from optiland.nonsequential.diagnostics import build_diagnostics
from optiland.nonsequential.ir.interpreter import apply_primitive_interactions
from optiland.nonsequential.ir.lower import lower
from optiland.nonsequential.path_recording import (  # noqa: F401
    _EVENT_DTYPE,
    PathRecorder,
)
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.rng import EventSlot
from optiland.nonsequential.sampling import russian_roulette

if TYPE_CHECKING:
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult


# Floor on the split-budget culling survival probability, mirroring
# sampling._RR_SURVIVE_FLOOR: bounds the worst-case flux boost so a nearly
# -saturated budget cannot produce an arbitrarily large boosted flux.
_BUDGET_CULL_SURVIVE_FLOOR = 0.02


def _cull_to_budget(
    spawned: NSQRayBundle, headroom: int, rng
) -> tuple[NSQRayBundle, np.ndarray, np.ndarray]:
    """Russian-roulette a spawned batch down to ``headroom`` rays.

    Bounded splitting (``ir.sampling.split_depth > 0``) caps live rays at
    ``split_budget * batch_size``; a batch of spawned transmit children that
    would exceed the remaining headroom is culled by roulette rather than
    truncated, so the excess is an unbiased kill + boost (like
    :func:`optiland.nonsequential.sampling.russian_roulette`) instead of a
    silent flux-truncation bias.

    Args:
        spawned: The spawned-ray batch to cull (already smaller-than
            -headroom batches must not be passed in -- callers check
            ``spawned.num_rays > headroom`` first).
        headroom: Number of additional live rays the budget can still admit.
            May be 0 (budget already saturated).
        rng: Keyed PCG32 RNG.

    Returns:
        ``(kept, culled_flux, culled_mask)``: the surviving (boosted-flux)
        subset as a new bundle, the pre-cull flux of the rays that were
        killed (for ``total_flux_lost`` bookkeeping), and the NumPy bool
        mask of which input rows were culled.
    """
    n = spawned.num_rays
    keep_prob = max(headroom / n, _BUDGET_CULL_SURVIVE_FLOOR) if n > 0 else 1.0
    ray_id_np = to_numpy(spawned.ray_id)
    bounce_np = to_numpy(spawned.bounce)
    u = to_numpy(rng.uniform(ray_id_np, bounce_np, EventSlot.RR, offset=1))
    keep_np = u < keep_prob
    culled_np = ~keep_np

    flux_np = to_numpy(spawned.flux)
    culled_flux = flux_np[culled_np].copy()

    idx = np.where(keep_np)[0]
    kept = spawned.select(idx)
    kept.flux = kept.flux / keep_prob  # unbiased boost
    return kept, culled_flux, culled_np


class ArrayBackend(TracerBackend):
    """The array-based trace loop, shared by every array backend.

    Attributes:
        host_reads_free: True when the ray state lives in host memory, so
            reducing a per-ray mask to a Python bool costs nothing and the
            loop may use it to skip an empty block or to exit early. False
            when the ray state is device-resident: every such reduction is
            a device-to-host copy and a stream synchronisation, so the loop
            runs the block unconditionally (it is a no-op on an empty mask)
            and takes a fixed trip count.
        supports_splitting: True when the backend can carry a bundle whose
            width changes during a bounce, which bounded splitting requires.
        alive_check_every: On a device backend only -- how often the loop
            may ask "is any ray still alive". 0 is a strict fixed trip
            count (zero synchronisations, ``max_depth`` bounces always);
            k > 0 costs one synchronisation every k bounces and wastes at
            most k - 1 bounces of all-dead work.
    """

    host_reads_free: bool = True
    supports_splitting: bool = True
    alive_check_every: int = 0

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _prepare_bundle(self, rays: NSQRayBundle) -> NSQRayBundle:
        """Place a freshly generated bundle on this backend's library/device.

        Args:
            rays: Bundle as ``source.generate()`` built it.

        Returns:
            The same bundle, on this backend's array library and device.
        """
        return rays

    def _maybe_compact(self, rays: NSQRayBundle) -> NSQRayBundle:
        """Post-bounce hook: optionally compact dead rays from the bundle.

        Default is a no-op. ``NumpyBackend`` overrides it to call
        ``rays.compact()``; ``TorchBackend`` keeps the default so tensor
        shapes stay fixed.

        Args:
            rays: Current ray bundle.

        Returns:
            Possibly compacted ray bundle.
        """
        return rays

    def _check_sampling_support(self, ir) -> None:
        """Warn about a sampling policy this backend cannot honour.

        Args:
            ir: The scene's lowered IR.
        """
        return

    # ------------------------------------------------------------------
    # Shared per-bounce pieces
    # ------------------------------------------------------------------

    def _empty(self, mask) -> bool:
        """True when ``mask`` selects no ray *and* asking is free.

        Consulted only to skip a block that is a no-op on an empty mask, so
        a backend whose state is device-resident answers False without
        looking: the reduction would be a synchronisation and the block
        costs nothing to run.

        Args:
            mask: Per-ray boolean mask.

        Returns:
            True only on a host-resident backend with no ray selected.
        """
        if not self.host_reads_free:
            return False
        return not bool(be.any(mask))

    def intersect_scene(
        self,
        rays: NSQRayBundle,
        components: list[BaseComponent],
    ) -> tuple[object, object, object, object]:
        """Find the nearest component intersection for every ray.

        A running minimum over the component list -- no ``argmin``, the
        winning component index carried alongside in an integer array of
        the same library and device as the ray state.

        ``t_min``/``hit_normals``/``hit_n_geom`` stay attached to the active
        backend's autograd graph (a no-op detail under NumPy); the component
        index is an integer array, so no gradient can flow through the
        choice of surface.

        Args:
            rays: Current ray bundle.
            components: List of scene components.

        Returns:
            ``(t_min, hit_normals, component_indices, hit_n_geom)``.
        """
        from optiland.nonsequential.ray_bundle import (  # noqa: PLC0415
            backend_int_full,
        )

        n = rays.num_rays
        t_min = be.ones(n) * be.inf
        hit_normals = be.zeros((n, 3))
        hit_n_geom = be.zeros((n, 3))
        comp_indices = backend_int_full((n,), -1, like=rays.x, bits=32)

        for i, comp in enumerate(components):
            t_c, normals_c, hit_c, n_geom_c = comp.intersect(rays)
            better = hit_c & (t_c < t_min)
            t_min = be.where(better, t_c, t_min)
            hit_normals = be.where(better[:, None], normals_c, hit_normals)
            hit_n_geom = be.where(better[:, None], n_geom_c, hit_n_geom)
            comp_indices = be.where(
                better, backend_int_full((n,), i, like=rays.x, bits=32), comp_indices
            )

        return t_min, hit_normals, comp_indices, hit_n_geom

    # ------------------------------------------------------------------
    # The trace
    # ------------------------------------------------------------------

    def trace(
        self,
        scene: NSQScene,
        num_rays: int,
        max_depth: int = 16,
        min_flux_fraction: float = 1e-6,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int | None = None,
        record_paths: bool | int = False,
    ) -> SimulationResult:
        """Run the full Monte Carlo simulation.

        Args:
            scene: The NSQScene to simulate.
            num_rays: Total rays to launch.
            max_depth: Maximum surface hits per ray.
            min_flux_fraction: Russian-roulette threshold, relative to
                per-ray initial flux -- combined with the scene's
                ``sampling_policy.rr_start_flux`` (the larger of the two
                wins). Below threshold, rays are killed with an unbiased
                probability and survivors' flux is boosted accordingly,
                rather than truncated outright.
            batch_size: Rays per processing batch. Does not change the result,
                only the speed; see ``DEFAULT_BATCH_SIZE``.
            seed: RNG seed for reproducibility.
            record_paths: ``False`` (default) records nothing. ``True``
                records every ray's full path -- fine for small traces, but
                O(rays x bounces) memory for large ones. A positive ``int``
                records an approximately that-many-ray subset, selected by a
                PCG32 hash of ``ray_id`` so the trace stays
                full-size and cheap while a bounded, deterministic sample is
                available for visualization/diagnosis -- e.g.
                ``scene.trace(num_rays=10_000_000, record_paths=1_000)``.

        Returns:
            SimulationResult.
        """
        from optiland.nonsequential.components.absorbing import (
            AbsorbingComponent,  # noqa: PLC0415
        )
        from optiland.nonsequential.rng import NSQRng  # noqa: PLC0415
        from optiland.nonsequential.tracer import (
            SimulationResult,  # noqa: PLC0415, I001
        )

        if seed is not None:
            self.rng = NSQRng(seed)

        # Reset detectors and absorber stats
        for det in scene.detectors:
            det.reset()
        for comp in scene.surfaces:
            if isinstance(comp, AbsorbingComponent):
                comp.reset_stats()
            if hasattr(comp, "reset_ledger"):
                comp.reset_ledger()

        # The per-bounce interaction loop below is driven by this IR, not by
        # iterating scene.surfaces and branching on Python class identity.
        ir = lower(scene, strict=False)
        for component in scene.surfaces:
            component.refresh_backend_transform()
        self._check_sampling_support(ir)

        t_start = time.perf_counter()

        sources = scene.sources
        # Float-cast for stats / kill-threshold; source.generate() uses the
        # raw total_flux (may be a torch Tensor for autograd).
        total_flux_in = sum(float(s.total_flux) for s in sources)
        num_rays_total = int(num_rays)

        flux_per_ray = total_flux_in / num_rays_total if num_rays_total > 0 else 1.0

        # Per-trace tallies. Each lives where the ray state lives -- a
        # Python scalar on a host backend, a 0-dim device value on a device
        # backend -- and is read back exactly once, after the loop.
        num_rays_escaped = Tally(is_int=True)
        num_rays_flux_killed = Tally(is_int=True)
        num_rays_depth_killed = Tally(is_int=True)
        total_flux_escaped = Tally()
        total_flux_bulk_absorbed = Tally()
        # Tracked separately for Diagnostics: depth truncation
        # is an inherent, reported bias, while RR/split-budget culling is
        # unbiased in expectation -- conflating them into one total_flux_lost
        # would hide which mechanism a large loss actually came from.
        total_flux_depth_killed = Tally()
        total_flux_rr_killed = Tally()
        # Ch. 10 sec 10.1: the sampling residual that makes (10.1) close on
        # every realisation rather than only in expectation. The surfaces
        # book their own share of it; this tally holds the loop's, which is
        # roulette.
        total_flux_sampling_residual = Tally()
        total_medium_stack_underflows = Tally(is_int=True)
        # Per-primitive nearest-hit counts, accumulated on the device and
        # read once at the end (the unreached-geometry diagnostic, and the
        # per-surface hit counter of ch. 10 R-10-6).
        hit_counts = Tally.vector(len(scene.surfaces))
        split_budget_saturated = False

        # Hoisted out of the bounce loop: the scene's bounding box does not
        # change during a trace, and rebuilding it every bounce was O(S) of
        # Python plus a host read per differentiable bounding-box edge.
        bounding_scale = estimate_bounding_scale(scene)

        # Distribute ray budget across sources proportional to flux
        rays_per_source = distribute_ray_budget(
            num_rays_total, [float(s.total_flux) for s in sources]
        )

        # Vectorised columnar path recording: PathRecorder
        # replaces the old per-event Python dict + repeated to_numpy()
        # closures with preallocated array writes, and implements the
        # record_paths: int subset contract.
        path_recorder = PathRecorder(record_paths, num_rays_total, self.rng.seed)

        _next_ray_id: list[int] = [0]

        def _alloc_ray_ids(n: int) -> np.ndarray:
            """Allocate ``n`` fresh ray ids for bounded splitting.

            Shares the same monotonic counter as source-birth ray ids, so
            a spawned ray's id never collides with any other ray's -- its
            PCG32 stream (keyed by ray_id) is therefore independent.
            """
            start = _next_ray_id[0]
            _next_ray_id[0] += n
            return np.arange(start, start + n, dtype=np.int64)

        allocator = _alloc_ray_ids if self.supports_splitting else None

        # Main trace loop
        for source_idx, (source, source_num_rays) in enumerate(
            zip(sources, rays_per_source, strict=False)
        ):
            source_name = getattr(source, "name", f"source_{source_idx}")
            source_remaining = source_num_rays

            while source_remaining > 0:
                batch = min(batch_size, source_remaining)
                ray_id = np.arange(
                    _next_ray_id[0], _next_ray_id[0] + batch, dtype=np.int64
                )
                _next_ray_id[0] += batch
                rays = source.generate(ray_id, self.rng)
                # source.generate() spreads the source's whole total_flux over
                # the rays it is asked for, so a batched source would re-emit
                # the full flux once per batch. Rescale to this batch's share
                # of the source's ray budget. A no-op when batch == the budget.
                if batch != source_num_rays:
                    rays.flux = rays.flux * (batch / source_num_rays)

                rays = self._prepare_bundle(rays)
                path_recorder.log_birth(rays, source_name)

                depth = 0
                while True:
                    if not self._continue_bounce(rays, depth, max_depth):
                        break

                    # --- traversal -------------------------------------
                    t_min, hit_normals, comp_idx, hit_n_geom = self.intersect_scene(
                        rays, scene.surfaces
                    )
                    det_t_min, _det_normals, det_idx, det_absorb = intersect_detectors(
                        rays, scene.detectors
                    )

                    # Nearest hit: component vs detector
                    comp_closer = t_min <= det_t_min
                    any_comp_hit = comp_idx >= 0
                    any_det_hit = det_idx >= 0

                    det_first = any_det_hit & (~comp_closer | ~any_comp_hit)
                    comp_first = any_comp_hit & (~det_first)

                    # Rays that reach no detector carry t = inf. Zero those
                    # before multiplying by a direction: inf * 0 is NaN, which
                    # the be.where below discards but not before NumPy warns.
                    det_t_safe = be.where(
                        det_first, det_t_min, be.zeros_like(det_t_min)
                    )

                    # --- Beer-Lambert bulk absorption -------------------
                    # Attenuate flux over the segment each ray just
                    # travelled through its *current* medium (rays.k_current,
                    # set at its last crossing or its source's ambient
                    # medium) before this bounce's nearest hit -- component
                    # or detector, whichever is closer. Applied before
                    # interact()/detector recording touch flux or k_current
                    # so both see the already-attenuated value; k_current
                    # itself is only updated afterwards, by
                    # RefractiveComponent.interact(), for the medium the ray
                    # is now entering.
                    hit_first = comp_first | det_first
                    if not self._empty(hit_first):
                        comp_t_safe = be.where(
                            comp_first, t_min, be.zeros_like(t_min)
                        )
                        hit_t = be.where(comp_first, comp_t_safe, det_t_safe)
                        alpha = 4.0 * be.pi * rays.k_current / rays.wavelength
                        # hit_t is in mm; alpha is in 1/um -> convert to um.
                        transmittance = be.exp(-alpha * hit_t * 1e3)
                        flux_before = rays.flux
                        rays.flux = flux_before * be.where(
                            hit_first, transmittance, be.ones_like(rays.flux)
                        )
                        total_flux_bulk_absorbed.add(
                            be.sum(flux_before - rays.flux)
                        )

                    # --- detector recording -----------------------------
                    for di, det in enumerate(scene.detectors):
                        mask_di = det_first & (det_idx == di)
                        if self._empty(mask_di):
                            continue
                        det_name = getattr(det, "name", f"detector_{di}")
                        path_recorder.log_hits(
                            rays, mask_di, det_name, t_offset=det_t_safe
                        )
                        det.record(rays, det_t_safe, mask_di)

                    # Advance detector-hit rays. Absorbing detectors
                    # terminate the ray; absorb=False detectors are
                    # transmissive: the hit is recorded (above) and the ray
                    # continues on its unchanged direction.
                    if not self._empty(det_first):
                        dx = det_t_safe * rays.L
                        dy = det_t_safe * rays.M
                        dz = det_t_safe * rays.N
                        rays.x = be.where(det_first, rays.x + dx, rays.x)
                        rays.y = be.where(det_first, rays.y + dy, rays.y)
                        rays.z = be.where(det_first, rays.z + dz, rays.z)
                        rays.bounce = be.where(det_first, rays.bounce + 1, rays.bounce)
                        rays.alive = rays.alive & ~(det_first & det_absorb)

                    # --- component interactions -------------------------
                    # Dispatched from the IR (ir.primitives[i].component_kind
                    # / .bsdf.kind) rather than by iterating scene.surfaces
                    # and checking isinstance. ray_id_allocator enables
                    # bounded splitting (NumPy forward engine only): a hit
                    # ray below ir.sampling.split_depth spawns both Fresnel
                    # children instead of drawing one, and the transmit child
                    # comes back as spawned (merged into `rays` below, after
                    # this bounce's own kill checks -- see the merge
                    # comment).
                    spawned = apply_primitive_interactions(
                        rays,
                        ir,
                        scene.surfaces,
                        t_min,
                        hit_normals,
                        hit_n_geom,
                        comp_idx,
                        comp_first,
                        self.rng,
                        log_hit_fn=path_recorder.log_hits,
                        ray_id_allocator=allocator,
                        skip_unhit=self.host_reads_free,
                        hit_counts=hit_counts,
                    )

                    # --- escape -----------------------------------------
                    no_hit = ~any_comp_hit & ~any_det_hit
                    escaped_now = no_hit & rays.alive
                    if not self._empty(escaped_now):
                        num_rays_escaped.add_count(escaped_now)
                        total_flux_escaped.add_masked_sum(rays.flux, escaped_now)
                        path_recorder.log_deaths(rays, escaped_now, "escaped")
                        ex = bounding_scale * rays.L
                        ey = bounding_scale * rays.M
                        ez = bounding_scale * rays.N
                        rays.x = be.where(escaped_now, rays.x + ex, rays.x)
                        rays.y = be.where(escaped_now, rays.y + ey, rays.y)
                        rays.z = be.where(escaped_now, rays.z + ez, rays.z)
                    rays.alive = rays.alive & ~no_hit

                    # --- depth truncation -------------------------------
                    # Hard kill. Inherent, reported bias (unlike roulette
                    # below, there is no unbiased way to "continue" a ray
                    # past a hard bounce-count cap).
                    alive_depth = rays.bounce < max_depth
                    newly_depth_killed = rays.alive & ~alive_depth
                    if not self._empty(newly_depth_killed):
                        num_rays_depth_killed.add_count(newly_depth_killed)
                        total_flux_depth_killed.add_masked_sum(
                            rays.flux, newly_depth_killed
                        )
                        path_recorder.log_deaths(
                            rays, newly_depth_killed, "depth_killed"
                        )
                    rays.alive = rays.alive & alive_depth

                    # --- Russian roulette -------------------------------
                    # Unbiased stochastic termination of low-flux rays (kill
                    # with probability p, boost survivors by 1/(1-p)), so
                    # total_flux_lost reports a genuine diagnostic -- ~0 for
                    # a well-configured scene -- rather than an expected
                    # bookkeeping entry.
                    rr_threshold_fraction = max(
                        min_flux_fraction, ir.sampling.rr_start_flux
                    )
                    flux_before_rr = rays.flux
                    rays.flux, rays.alive, rr_killed = russian_roulette(
                        rays.flux,
                        rays.alive,
                        rr_threshold_fraction,
                        flux_per_ray,
                        self.rng,
                        rays.ray_id,
                        rays.bounce,
                        fast_path=self.host_reads_free,
                    )
                    # Ch. 10 (10.2): roulette does not preserve weight on a
                    # realisation. A killed ray takes its whole weight out
                    # of the trace and a survivor is handed -w(1-q)/q that
                    # came from nowhere; both are the event residual, and
                    # booking only the first is the 3.13% error of sec 10.2.
                    # flux is left untouched on a killed ray, so the first
                    # term below is zero there and the second picks it up.
                    total_flux_sampling_residual.add(
                        be.sum(flux_before_rr - rays.flux)
                    )
                    total_flux_sampling_residual.add_masked_sum(
                        flux_before_rr, rr_killed
                    )
                    if not self._empty(rr_killed):
                        num_rays_flux_killed.add_count(rr_killed)
                        total_flux_rr_killed.add_masked_sum(
                            flux_before_rr, rr_killed
                        )
                        path_recorder.log_deaths(rays, rr_killed, "flux_killed")

                    # --- bounded-splitting merge ------------------------
                    # Now that this bounce's own escape/depth/RR kill checks
                    # (all sized to the pre-spawn ray count) are done.
                    # Spawned rays start fresh at the next iteration's
                    # intersect_scene call, same as any other live ray.
                    if spawned is not None and spawned.num_rays > 0:
                        budget = int(ir.sampling.split_budget * batch_size)
                        headroom = max(0, budget - rays.num_rays_alive)
                        if spawned.num_rays > headroom:
                            split_budget_saturated = True
                            spawned, culled_flux_np, culled_np = _cull_to_budget(
                                spawned, headroom, self.rng
                            )
                            if culled_np.any():
                                num_rays_flux_killed.add(int(culled_np.sum()))
                                total_flux_rr_killed.add(float(culled_flux_np.sum()))
                                total_flux_sampling_residual.add(
                                    float(culled_flux_np.sum())
                                )
                        if spawned.num_rays > 0:
                            rays = NSQRayBundle.concat([rays, spawned])

                    # Flush this bounce's medium-stack inconsistency counts
                    # (see RefractiveComponent.interact) into the running
                    # total, then reset so they are counted exactly once
                    # regardless of subsequent compaction/concat.
                    total_medium_stack_underflows.add(
                        be.sum(rays.medium_stack_underflows)
                    )
                    rays.medium_stack_underflows = be.where(
                        rays.medium_stack_underflows > 0,
                        be.zeros_like(rays.medium_stack_underflows),
                        rays.medium_stack_underflows,
                    )

                    rays = self._maybe_compact(rays)
                    depth += 1
                    if rays.num_rays == 0:
                        break

                source_remaining -= batch

        t_end = time.perf_counter()

        # Collect absorbed stats from AbsorbingComponents
        total_flux_absorbed = 0.0
        num_rays_absorbed = 0
        for comp in scene.surfaces:
            if isinstance(comp, AbsorbingComponent):
                total_flux_absorbed += float(to_numpy(comp._absorbed_flux))
                num_rays_absorbed += int(to_numpy(comp._absorbed_count))

        # Collect the mirror and coating loss, and each surface's share of
        # the sampling residual, from the surfaces that booked them.
        coating_loss = 0.0
        for comp in scene.surfaces:
            if hasattr(comp, "coating_loss"):
                coating_loss += comp.coating_loss
                total_flux_sampling_residual.add(comp.sampling_residual)

        # Collect detector results. Ch. 10 sec 10.1 books flux where it
        # *leaves* the trace. A transmissive (absorb=False) detector reads
        # the beam and lets it continue, so the same watt is still in the
        # trace and will be booked again at whatever finally absorbs it;
        # counting the tap in the identity books it twice. The reading is
        # reported as it always was, and reported separately, so the
        # identity can leave it out.
        detector_results: dict[str, object] = {}
        total_flux_detected = 0.0
        total_flux_tapped = 0.0
        det_names = get_detector_names(scene)
        for i, det in enumerate(scene.detectors):
            name = det_names[i] if i < len(det_names) else (det.name or f"detector_{i}")
            result = det.get_result()
            detector_results[name] = result
            if hasattr(result, "total_flux"):
                # IrradianceMap.total_flux may be an attached backend array;
                # SimulationResult's aggregate stays a plain float.
                flux_here = float(to_numpy(result.total_flux))
                total_flux_detected += flux_here
                if not getattr(det, "absorb", True):
                    total_flux_tapped += flux_here

        escaped = total_flux_escaped.value()
        bulk = total_flux_bulk_absorbed.value()
        depth_killed = total_flux_depth_killed.value()
        rr_killed_flux = total_flux_rr_killed.value()
        sampling_residual = total_flux_sampling_residual.value()
        total_flux_lost = depth_killed + rr_killed_flux

        # Ch. 10 (10.1). Every watt a source emitted is detected at a
        # detector that removed the ray, absorbed at a surface, lost in a
        # mirror or coating, absorbed in the bulk, escaped, truncated by the
        # depth cap, or booked into the sampling residual. Roulette-killed
        # flux is not a separate term: it is part of the residual, together
        # with the boost handed to the rays that survived, and adding it
        # again here is the 3.13% error of sec 10.2 with the sign reversed.
        flux_err = (
            abs(
                total_flux_in
                - (total_flux_detected - total_flux_tapped)
                - total_flux_absorbed
                - coating_loss
                - bulk
                - escaped
                - depth_killed
                - sampling_residual
            )
            / total_flux_in
            if total_flux_in > 0
            else 0.0
        )

        # Vectorised: single conversion of the columnar buffers
        # to the structured-array format, done once here rather than
        # incrementally per event.
        ray_paths = path_recorder.finalize()

        hit_component_ids = {
            i for i, n in enumerate(hit_counts.values()) if n > 0
        }

        diagnostics = build_diagnostics(
            scene,
            hit_component_ids,
            num_rays_total,
            total_flux_in,
            depth_killed,
            rr_killed_flux,
            flux_err,
            split_budget_saturated,
            detector_results,
            medium_stack_underflows=total_medium_stack_underflows.value(),
            coating_loss=coating_loss,
            sampling_residual=sampling_residual,
        )

        return SimulationResult(
            detectors=detector_results,
            num_rays_total=num_rays_total,
            num_rays_absorbed=num_rays_absorbed,
            num_rays_escaped=num_rays_escaped.value(),
            num_rays_flux_killed=num_rays_flux_killed.value(),
            num_rays_depth_killed=num_rays_depth_killed.value(),
            total_flux_in=total_flux_in,
            total_flux_detected=total_flux_detected,
            total_flux_tapped=total_flux_tapped,
            total_flux_absorbed=total_flux_absorbed,
            total_flux_coating=coating_loss,
            total_flux_bulk_absorbed=bulk,
            total_flux_escaped=escaped,
            total_flux_lost=total_flux_lost,
            total_flux_sampling_residual=sampling_residual,
            flux_conservation_error=flux_err,
            trace_time_sec=t_end - t_start,
            ray_paths=ray_paths,
            diagnostics=diagnostics,
        )

    def _continue_bounce(
        self, rays: NSQRayBundle, depth: int, max_depth: int
    ) -> bool:
        """Decide whether to run one more bounce.

        On a host backend the alive count is free to read, so the loop stops
        the moment nothing is alive. On a device backend the loop runs a
        fixed ``max_depth`` trips and consults the alive count only every
        :attr:`alive_check_every` bounces (0 = never), because each such
        read is a device synchronisation.

        Args:
            rays: Current ray bundle.
            depth: Bounces already run for this batch.
            max_depth: The configured depth cap.

        Returns:
            True to run another bounce.
        """
        if self.host_reads_free:
            return rays.num_rays_alive > 0
        if depth >= max_depth:
            return False
        k = self.alive_check_every
        if k and depth % k == 0 and rays.num_rays_alive == 0:
            return False
        return True

    def _to_numpy(self, arr: object) -> np.ndarray:
        """Backward-compatible alias."""
        return to_numpy(arr)
