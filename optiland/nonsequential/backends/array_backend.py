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
from optiland.backend.base import BackendCapabilityError
from optiland.backend.utils import to_numpy
from optiland.nonsequential._utils import (
    DEFAULT_BATCH_SIZE,
    distribute_ray_budget,
    estimate_bounding_scale,
    get_detector_names,
    resident_scalar,
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
from optiland.nonsequential.rng import EventSlot, NSQRng, uniform_bits
from optiland.nonsequential.sampling import russian_roulette

if TYPE_CHECKING:
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult


# Floor on the split-budget culling survival probability, mirroring
# sampling._RR_SURVIVE_FLOOR: bounds the worst-case flux boost so a nearly
# -saturated budget cannot produce an arbitrarily large boosted flux.
_BUDGET_CULL_SURVIVE_FLOOR = 0.02

# Smallest width the bucket ladder offers. Two independent reasons for the
# floor, both pointing at about the same number:
#
# - a bundle narrower than this is below one launch's worth of work on a
#   device, so rounding further down buys nothing and only adds another
#   shape to compile;
# - measured: the shared library's material property cache builds its key
#   from the *contents* of the wavelength array once that array holds 1024
#   elements or fewer (``optiland.materials.base.BaseMaterial
#   ._MAX_VALUE_KEY_ARRAY_SIZE``), which copies the whole array to the host
#   on every evaluation. A rung below that would put a per-bounce host read
#   back into the loop -- the exact cost compaction exists to remove.
BUCKET_MIN_WIDTH = 2048


def bucketed_width(n_live: int, n_now: int, min_width: int = BUCKET_MIN_WIDTH) -> int:
    """The width a bundle of ``n_live`` live rays is compacted to.

    ``docs/theory/12_gpu_mapping.md`` R-12-6: the compacted width comes from
    a fixed ladder -- here 0, ``min_width``, and the powers of two above it,
    capped at the current width -- rather than from the live count itself,
    so a whole trace sees O(log N) distinct shapes and each is compiled,
    captured or autotuned once.

    Rounding up to a power of two also *is* the alive-fraction trigger R-12-6
    asks for, with no second rule: the next rung down is only reached once
    fewer than half the rays are alive, so a bundle that is still mostly live
    returns its own width and the caller skips the gather.

    Args:
        n_live: Live rays in the bundle.
        n_now: The bundle's current width.
        min_width: Smallest non-zero rung of the ladder.

    Returns:
        The target width, in ``[0, n_now]``.
    """
    if n_live <= 0:
        return 0
    width = min_width
    while width < n_live:
        width <<= 1
    return width if width < n_now else n_now


def birth_flux_like(flux, per_ray):
    """A bundle's birth weights, every ray given ``per_ray``.

    The weight a source hands each of the ``N`` rays of its budget is
    ``total_flux / N``, computed once. A batch of ``b < N`` rays used to be
    rescaled from the source's own ``total_flux / b`` by ``b / N``, and
    ``(total_flux / b) * (b / N)`` rounds differently from ``total_flux / N``
    for some ``b``: a ray born in a remainder batch carried a weight one unit
    in the last place off its siblings (KronosNSRT issue 24; section 10.3 of
    the theory requires per-ray weights bit-identical across batch sizes).

    The array is built the way the sources build theirs (``ones * w`` for a
    tensor, a filled array otherwise), so a ray of a partial batch carries
    exactly the bytes a ray of a full batch carries, and a ``per_ray`` that
    is a tensor keeps its autograd graph.

    Args:
        flux: The bundle's flux as ``source.generate`` built it, shape (b,).
        per_ray: ``total_flux / N``: a Python float or a 0-dim tensor.

    Returns:
        An array or tensor like ``flux``, every entry ``per_ray``.
    """
    if be.is_torch_tensor(flux):
        import torch  # noqa: PLC0415

        return torch.ones_like(flux) * per_ray
    if be.is_torch_tensor(per_ray):
        return be.ones(np.shape(flux)[0]) * per_ray
    return np.full(np.shape(flux), per_ray, dtype=np.asarray(flux).dtype)


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



class _BounceContext:
    """What one bounce of a trace reads and books, fixed for the whole trace.

    The scene's surfaces and detectors and its lowered IR, the path
    recorder, the ray-id allocator of bounded splitting, the per-trace
    tallies and the loop's constants. Built once per ``trace`` call and
    handed to :func:`bounce_body` with the ray bundle; the tallies and the
    recorder are the same objects the trace reads back after the loop.

    The surface and detector lists are taken from the scene once, here: the
    scene rebuilds them on every read (``NSQScene.surfaces`` walks its
    registry, whose single-surface wrappers are classes made per
    ``add_component`` call), which a compiled step would have to guard on
    and compile again for every new scene. The lists hold the same objects
    in the same order as every read would.
    """

    def __init__(
        self,
        *,
        scene,
        ir,
        path_recorder,
        allocator,
        hit_counts,
        bounding_scale,
        max_depth,
        min_flux_fraction,
        flux_per_ray,
        num_rays_escaped,
        num_rays_flux_killed,
        num_rays_depth_killed,
        total_flux_escaped,
        total_flux_bulk_absorbed,
        total_flux_depth_killed,
        total_flux_rr_killed,
        total_flux_sampling_residual,
    ) -> None:
        self.surfaces = scene.surfaces
        self.detectors = scene.detectors
        self.ir = ir
        self.path_recorder = path_recorder
        self.allocator = allocator
        self.hit_counts = hit_counts
        self.bounding_scale = bounding_scale
        self.max_depth = max_depth
        self.min_flux_fraction = min_flux_fraction
        self.flux_per_ray = flux_per_ray
        self.num_rays_escaped = num_rays_escaped
        self.num_rays_flux_killed = num_rays_flux_killed
        self.num_rays_depth_killed = num_rays_depth_killed
        self.total_flux_escaped = total_flux_escaped
        self.total_flux_bulk_absorbed = total_flux_bulk_absorbed
        self.total_flux_depth_killed = total_flux_depth_killed
        self.total_flux_rr_killed = total_flux_rr_killed
        self.total_flux_sampling_residual = total_flux_sampling_residual


def bounce_body(backend, ctx: _BounceContext, rays: NSQRayBundle):
    """One bounce of the trace loop, from traversal to Russian roulette.

    Traversal (components and detectors), Beer-Lambert attenuation over the
    segment, detector recording and the detector advance, the component
    interactions, escape, depth truncation and roulette: the part of a bounce
    that works on the bundle at one width. The bundle is updated in place and
    the tallies in ``ctx`` are booked. The splitting merge, the medium-stack
    flush and compaction stay in the loop, where the host reads they need
    are allowed.

    Args:
        backend: The :class:`ArrayBackend` running the trace.
        ctx: The trace's :class:`_BounceContext`.
        rays: The live bundle, updated in place.

    Returns:
        The transmit children spawned by bounded splitting this bounce, or
        ``None``.
    """
    # --- traversal -------------------------------------
    t_min, hit_normals, comp_idx, hit_n_geom = backend.intersect_scene(
        rays, ctx.surfaces
    )
    (
        det_t_min,
        _det_normals,
        det_idx,
        det_absorb,
        det_n_geom,
    ) = intersect_detectors(rays, ctx.detectors)

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
    if not backend._empty(hit_first):
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
        ctx.total_flux_bulk_absorbed.add(
            be.sum(flux_before - rays.flux)
        )

    # --- detector recording -----------------------------
    for di, det in enumerate(ctx.detectors):
        mask_di = det_first & (det_idx == di)
        if backend._empty(mask_di):
            continue
        det_name = getattr(det, "name", f"detector_{di}")
        ctx.path_recorder.log_hits(
            rays, mask_di, det_name, t_offset=det_t_safe
        )
        det.record(rays, det_t_safe, mask_di)
        # Arriving flux by ghost order; a no-op unless the
        # detector was built with reflection_bins.
        det.record_reflections(rays, mask_di)

    # Advance detector-hit rays. Absorbing detectors
    # terminate the ray; absorb=False detectors are
    # transmissive: the hit is recorded (above) and the ray
    # continues on its unchanged direction.
    if not backend._empty(det_first):
        # An absorbing detector is terminal, so the plain
        # global advance is all its rays' positions are ever
        # used for -- p + t*d carries u*|t| of rounding and
        # nothing reads it again.
        terminal = det_first & det_absorb
        dx = det_t_safe * rays.L
        dy = det_t_safe * rays.M
        dz = det_t_safe * rays.N
        rays.x = be.where(terminal, rays.x + dx, rays.x)
        rays.y = be.where(terminal, rays.y + dy, rays.y)
        rays.z = be.where(terminal, rays.z + dz, rays.z)
        # A transmissive detector is not terminal: the ray
        # carries on from where this puts it and the next
        # bounce tests the same plane again. It therefore
        # gets a surface's treatment -- the hit point
        # rebuilt in the detector's own frame, then pushed
        # clear of the plane along the geometric normal, or
        # a grazing crossing is recorded twice (R-07-6,
        # docs/build/X7_grazing_exit.md section 6).
        for di, det in enumerate(ctx.detectors):
            if det.absorb:
                continue
            crossed = det_first & (det_idx == di)
            det.advance_to_hit(rays, det_t_safe, crossed)
            det.offset_from_surface(rays, det_n_geom, crossed)
        rays.bounce = be.where(det_first, rays.bounce + 1, rays.bounce)
        rays.alive = rays.alive & ~terminal

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
        ctx.ir,
        ctx.surfaces,
        t_min,
        hit_normals,
        hit_n_geom,
        comp_idx,
        comp_first,
        backend.rng,
        log_hit_fn=ctx.path_recorder.log_hits,
        ray_id_allocator=ctx.allocator,
        skip_unhit=backend.host_reads_free,
        hit_counts=ctx.hit_counts,
    )

    # --- escape -----------------------------------------
    no_hit = ~any_comp_hit & ~any_det_hit
    escaped_now = no_hit & rays.alive
    if not backend._empty(escaped_now):
        ctx.num_rays_escaped.add_count(escaped_now)
        ctx.total_flux_escaped.add_masked_sum(rays.flux, escaped_now)
        ctx.path_recorder.log_deaths(rays, escaped_now, "escaped")
        ex = ctx.bounding_scale * rays.L
        ey = ctx.bounding_scale * rays.M
        ez = ctx.bounding_scale * rays.N
        rays.x = be.where(escaped_now, rays.x + ex, rays.x)
        rays.y = be.where(escaped_now, rays.y + ey, rays.y)
        rays.z = be.where(escaped_now, rays.z + ez, rays.z)
    rays.alive = rays.alive & ~no_hit

    # --- depth truncation -------------------------------
    # Hard kill. Inherent, reported bias (unlike roulette
    # below, there is no unbiased way to "continue" a ray
    # past a hard bounce-count cap).
    alive_depth = rays.bounce < ctx.max_depth
    newly_depth_killed = rays.alive & ~alive_depth
    if not backend._empty(newly_depth_killed):
        ctx.num_rays_depth_killed.add_count(newly_depth_killed)
        ctx.total_flux_depth_killed.add_masked_sum(
            rays.flux, newly_depth_killed
        )
        ctx.path_recorder.log_deaths(
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
        ctx.min_flux_fraction, ctx.ir.sampling.rr_start_flux
    )
    flux_before_rr = rays.flux
    rays.flux, rays.alive, rr_killed = russian_roulette(
        rays.flux,
        rays.alive,
        rr_threshold_fraction,
        ctx.flux_per_ray,
        backend.rng,
        rays.ray_id,
        rays.bounce,
        fast_path=backend.host_reads_free,
    )
    # Ch. 10 (10.2): roulette does not preserve weight on a
    # realisation. A killed ray takes its whole weight out
    # of the trace and a survivor is handed -w(1-q)/q that
    # came from nowhere; both are the event residual, and
    # booking only the first is the 3.13% error of sec 10.2.
    # flux is left untouched on a killed ray, so the first
    # term below is zero there and the second picks it up.
    ctx.total_flux_sampling_residual.add(
        be.sum(flux_before_rr - rays.flux)
    )
    ctx.total_flux_sampling_residual.add_masked_sum(
        flux_before_rr, rr_killed
    )
    if not backend._empty(rr_killed):
        ctx.num_rays_flux_killed.add_count(rr_killed)
        ctx.total_flux_rr_killed.add_masked_sum(
            flux_before_rr, rr_killed
        )
        ctx.path_recorder.log_deaths(rays, rr_killed, "flux_killed")

    return spawned


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
        compact_every: How often a device backend may compact to a bucketed
            width. ``None`` follows :attr:`alive_check_every`, so the live
            count the alive check already reads is the same one compaction
            selects its bucket with and the loop keeps exactly one
            synchronisation per period.
    """

    host_reads_free: bool = True
    supports_splitting: bool = True
    alive_check_every: int = 0
    compact_every: int | None = None

    # Live count for the current bounce, read at most once and shared by
    # the alive check and the compaction bucket. None means "not read yet".
    _live_count_cache: int | None = None

    @property
    def compaction_period(self) -> int:
        """Bounces between compaction attempts; 0 disables compaction."""
        k = self.compact_every
        return int(self.alive_check_every if k is None else k)

    def _live_count(self, rays: NSQRayBundle) -> int:
        """The number of live rays, read from the device at most once a bounce.

        Both the early-exit check and the compaction bucket need the same
        number, and on a device backend reading it is the loop's only
        synchronisation. Reading it once and handing it to both is what
        keeps compaction free of a synchronisation of its own.

        Args:
            rays: Current ray bundle.

        Returns:
            The live-ray count as a Python int.
        """
        if self._live_count_cache is None:
            self._live_count_cache = rays.num_rays_alive
        return self._live_count_cache

    def _gradient_mode(self, rays: NSQRayBundle) -> bool:
        """True when some field of the bundle carries a gradient.

        Args:
            rays: Current ray bundle.

        Returns:
            False on a backend with no autograd.
        """
        return False

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

    def _maybe_compact(self, rays: NSQRayBundle, depth: int) -> NSQRayBundle:
        """Post-bounce hook: optionally compact dead rays from the bundle.

        Default is a no-op. ``NumpyBackend`` overrides it to call
        ``rays.compact()`` every bounce; ``TorchBackend`` gathers to a
        bucketed width on a schedule.

        Args:
            rays: Current ray bundle.
            depth: Bounces already run for this batch, counting from 0.

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

    def _splitting_enabled(self) -> bool:
        """Whether this trace may split rays (``split_depth > 0`` honoured).

        Returns:
            :attr:`supports_splitting` by default; a backend that can split
            only in some modes (the Torch backend: forward-only, and only
            when asked) narrows it.
        """
        return bool(self.supports_splitting)

    def _bounce_step(self, ctx: _BounceContext):
        """The function the trace loop calls for each bounce.

        :func:`bounce_body` itself by default. ``TorchBackend`` returns a
        compiled version of it when built with ``compile_step=True``.

        Args:
            ctx: The trace's bounce context.

        Returns:
            A callable ``step(backend, ctx, rays)`` returning the spawned
            bundle or ``None``.
        """
        return bounce_body

    def _check_graph_replay(self, scene, ir, record_paths) -> None:
        """Refuse a replayed bounce where it cannot apply; no-op by default.

        Args:
            scene: The scene about to be traced.
            ir: The scene's lowered IR.
            record_paths: The trace's ``record_paths`` argument.
        """
        return

    def _graph_handover_depth(self, rays: NSQRayBundle) -> int | None:
        """The bounce at which this batch is handed to a replayed graph, if any.

        The default backend replays nothing, and every bounce runs through
        the loop below.

        Args:
            rays: The freshly prepared batch.

        Returns:
            ``None``.
        """
        return None

    def _replay_bounces(
        self, rays, bounce, depth, max_depth, scene, tallies
    ) -> NSQRayBundle:
        """Run bounces ``depth .. max_depth - 1`` as a replayed graph.

        Only called for a batch :meth:`_graph_handover_depth` handed over.

        Args:
            rays: The batch after the eager bounces.
            bounce: The loop's bounce body, ``bounce(rays, depth) -> rays``.
            depth: Bounces already run.
            max_depth: The depth cap.
            scene: The scene being traced.
            tallies: The trace's own tallies.

        Returns:
            The bundle after the last bounce.
        """
        raise NotImplementedError
    def _trace_rng(self, seed: int | None) -> NSQRng:
        """The keyed generator this trace draws from.

        Called once at the start of every trace. The default keeps the
        backend's generator unless the trace names a seed, in which case it
        is rebuilt with that seed. A backend with a choice of generator
        kernel (the Torch backend's ``rng_kernel``) makes the choice here,
        where the device is known.

        Args:
            seed: The trace's seed argument, or None.

        Returns:
            The generator for this trace.
        """
        return self.rng if seed is None else NSQRng(seed)

    def _environment(self) -> dict[str, object]:
        """What this trace ran on, for ``SimulationResult.environment``.

        Returns:
            The array library, the device, the working precision and the
            generator kernel that drew the trace's random numbers; at
            float32 also ``uniform_bits``, the number of the generator's 32
            output bits its uniforms are made of (24 since issue 62 of the
            research repository: the top 24 bits times 2**-24, so no draw is
            1.0). At float64 the key is absent: the uniform is the whole
            32-bit output times 2**-32, as in every record made before the
            key existed.
        """
        try:
            device = str(be.get_device())
        except (AttributeError, BackendCapabilityError):
            device = "cpu"
        env: dict[str, object] = {
            "array_backend": be.get_backend(),
            "device": device,
            "precision": f"float{be.get_precision()}",
            "rng_kernel": getattr(self.rng, "kernel", be.get_backend()),
        }
        if be.get_precision() == 32:
            env["uniform_bits"] = uniform_bits(32)
        return env

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
        from optiland.nonsequential.tracer import (
            SimulationResult,  # noqa: PLC0415, I001
        )

        self.rng = self._trace_rng(seed)

        # Reset detectors and absorber stats
        for det in scene.detectors:
            det.reset()
        for comp in scene.surfaces:
            if isinstance(comp, AbsorbingComponent):
                comp.reset_stats()
            if hasattr(comp, "reset_ledger"):
                comp.reset_ledger()

        # Clear every material's n()/k() identity memo (NSQMaterial.reset_memo)
        # so a wavelength array object reused across two separate trace()
        # calls -- the same ray bundle traced twice, with a material's own
        # parameters changed in between -- cannot read back a result computed
        # under the old parameters. One attribute write per distinct
        # material, no host read; a component's material_front/material_back
        # are frequently the same NSQMaterial instance shared with a
        # neighbour, so this is deduplicated by object identity.
        seen_material_ids: set[int] = set()
        for comp in scene.surfaces:
            for mat in (comp.material_front, comp.material_back):
                if id(mat) not in seen_material_ids:
                    seen_material_ids.add(id(mat))
                    mat.reset_memo()

        # The per-bounce interaction loop below is driven by this IR, not by
        # iterating scene.surfaces and branching on Python class identity.
        ir = lower(scene, strict=False)
        for component in scene.surfaces:
            component.refresh_backend_transform()
        self._check_sampling_support(ir)
        self._check_graph_replay(scene, ir, record_paths)

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
        tallies = (
            num_rays_escaped,
            num_rays_flux_killed,
            num_rays_depth_killed,
            total_flux_escaped,
            total_flux_bulk_absorbed,
            total_flux_depth_killed,
            total_flux_rr_killed,
            total_flux_sampling_residual,
            total_medium_stack_underflows,
            hit_counts,
        )

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

        allocator = _alloc_ray_ids if self._splitting_enabled() else None

        # Everything the bounce body reads for the whole trace, in one
        # object: the body lives outside this method (:func:`bounce_body`)
        # so a backend can hand the loop a compiled version of it
        # (``TorchBackend(compile_step=True)``). With the default step the
        # loop runs exactly the statements it always ran, in the same order.
        ctx = _BounceContext(
            scene=scene,
            ir=ir,
            path_recorder=path_recorder,
            allocator=allocator,
            hit_counts=hit_counts,
            bounding_scale=bounding_scale,
            max_depth=max_depth,
            min_flux_fraction=min_flux_fraction,
            flux_per_ray=flux_per_ray,
            num_rays_escaped=num_rays_escaped,
            num_rays_flux_killed=num_rays_flux_killed,
            num_rays_depth_killed=num_rays_depth_killed,
            total_flux_escaped=total_flux_escaped,
            total_flux_bulk_absorbed=total_flux_bulk_absorbed,
            total_flux_depth_killed=total_flux_depth_killed,
            total_flux_rr_killed=total_flux_rr_killed,
            total_flux_sampling_residual=total_flux_sampling_residual,
        )
        step = self._bounce_step(ctx)

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
                # the full flux once per batch. Every ray of the source carries
                # the one weight total_flux / source_num_rays, whatever batch it
                # is born in. A no-op when batch == the budget.
                if batch != source_num_rays:
                    rays.flux = birth_flux_like(
                        rays.flux, source.total_flux / source_num_rays
                    )

                rays = self._prepare_bundle(rays)
                path_recorder.log_birth(rays, source_name)

                depth = 0
                self._live_count_cache = None

                # One bounce, as a function of the bundle and the bounce
                # index. The loop below calls it once per bounce; a backend
                # that replays a recorded bounce (TorchBackend(graph_replay=
                # ...)) hands it to _replay_bounces, which records it once.
                def _bounce(rays, depth):
                    nonlocal split_budget_saturated

                    spawned = step(self, ctx, rays)

                    # --- bounded-splitting merge ------------------------
                    # Now that this bounce's own escape/depth/RR kill checks
                    # (all sized to the pre-spawn ray count) are done.
                    # Spawned rays start fresh at the next iteration's
                    # intersect_scene call, same as any other live ray.
                    if spawned is not None and spawned.num_rays > 0:
                        # The depth cap binds a spawned child exactly as it
                        # binds its sibling, which kept the parent's row and
                        # met the check above: a child that has just made its
                        # max_depth-th interaction is truncated here, not
                        # merged. Without this a transmit child ran one
                        # interaction past the cap on a host backend (whose
                        # loop continues while any ray is alive) and was
                        # dropped unbooked on a device backend (whose loop
                        # stops after max_depth trips), so the two
                        # backends truncated a split tree differently and
                        # the device ledger did not close.
                        # Splitting already chose its rows on the host, so
                        # reading this mask back costs nothing new.
                        over_depth = spawned.bounce >= max_depth
                        over_np = np.asarray(be.to_numpy(over_depth), dtype=bool)
                        if over_np.any():
                            num_rays_depth_killed.add_count(over_depth)
                            total_flux_depth_killed.add_masked_sum(
                                spawned.flux, over_depth
                            )
                            path_recorder.log_deaths(
                                spawned, over_depth, "depth_killed"
                            )
                            spawned = spawned.select(np.where(~over_np)[0])
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
                    # regardless of subsequent compaction/concat. The reset
                    # writes an integer zero of the field's own dtype: the
                    # working-float zero it used to write turned the counter
                    # into floats after the first bounce, and the tally with
                    # it (issue 60 of the research repository).
                    total_medium_stack_underflows.add(
                        be.sum(rays.medium_stack_underflows)
                    )
                    underflows = rays.medium_stack_underflows
                    rays.medium_stack_underflows = be.where(
                        underflows > 0,
                        resident_scalar(self, "underflow_reset", 0, underflows),
                        underflows,
                    )

                    rays = self._maybe_compact(rays, depth)
                    return rays

                handover = self._graph_handover_depth(rays)
                while True:
                    if not self._continue_bounce(rays, depth, max_depth):
                        break
                    # Whatever live count was read for this bounce's
                    # early-exit check described the state *before* the
                    # bounce body, which is about to change it. Compaction
                    # at the end of the bounce reads it again, once, and
                    # that read is the one the next bounce's check reuses.
                    self._live_count_cache = None
                    if handover is not None and depth == handover:
                        rays = self._replay_bounces(
                            rays, _bounce, depth, max_depth, scene, tallies
                        )
                        break
                    rays = _bounce(rays, depth)
                    depth += 1
                    # A shape, not a value: free on every backend. With
                    # compaction on, a bundle whose last ray has died comes
                    # back at width 0 and the loop stops here without
                    # anything being read.
                    if rays.num_rays == 0:
                        break

                source_remaining -= batch

        t_end = time.perf_counter()

        # Collect absorbed stats from AbsorbingComponents
        total_flux_absorbed = 0.0
        num_rays_absorbed = 0
        for comp in scene.surfaces:
            if isinstance(comp, AbsorbingComponent):
                total_flux_absorbed += comp.absorbed_flux()
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
        reflection_histograms: dict[str, object] = {}
        total_flux_detected = 0.0
        total_flux_tapped = 0.0
        det_names = get_detector_names(scene)
        for i, det in enumerate(scene.detectors):
            name = det_names[i] if i < len(det_names) else (det.name or f"detector_{i}")
            result = det.get_result()
            detector_results[name] = result
            histogram = det.reflection_histogram()
            if histogram is not None:
                reflection_histograms[name] = histogram
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
            reflection_histograms=reflection_histograms,
            environment=self._environment(),
        )

    def _continue_bounce(
        self, rays: NSQRayBundle, depth: int, max_depth: int
    ) -> bool:
        """Decide whether to run one more bounce.

        On a host backend the alive count is free to read, so the loop stops
        the moment nothing is alive. On a device backend the loop runs a
        fixed ``max_depth`` trips and consults the alive count only every
        :attr:`alive_check_every` bounces (0 = never), because each such
        read is a device synchronisation. The count comes through
        :meth:`_live_count`, so when compaction ran at the end of the
        previous bounce this check reuses that read instead of making its
        own -- the state cannot have changed in between.

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
        if k and depth % k == 0 and self._live_count(rays) == 0:
            return False
        return True

    def _to_numpy(self, arr: object) -> np.ndarray:
        """Backward-compatible alias."""
        return to_numpy(arr)
