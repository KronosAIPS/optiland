"""TorchBackend -- the differentiable, device-resident array backend.

The trace loop itself lives in :class:`~optiland.nonsequential.backends
.array_backend.ArrayBackend`; this class is the thin part that is specific
to running it on torch tensors on a device:

- **device placement**: :meth:`TorchBackend._prepare_bundle` promotes every
  field of a freshly generated bundle -- floats, the alive flag, the bounce
  and ray-id integers, and the medium-stack table -- onto the backend's
  device. Nothing in the trace path then refers to a host array.
- **bucketed compaction**: in forward-only mode the bundle is gathered down
  to a width from a fixed ladder once fewer than half its rays are alive, so
  dead rays stop costing a full-width bounce while the set of tensor shapes
  a trace uses stays small and known. In gradient mode the width is left
  alone, so the autograd graph is a single chain of fixed-shape operations.
- **loop control**: ``host_reads_free = False`` tells the loop that
  reducing a per-ray mask to a Python bool is a synchronisation, so the
  loop skips nothing and takes a bounded trip count, asking whether any ray
  is still alive at most once every ``alive_check_every`` bounces.
- **capability**: bounded splitting grows the bundle. It is refused by
  default, and in gradient mode always; ``TorchBackend(allow_splitting=True)``
  runs it in forward-only mode, at the cost of reading the split rows back
  to the host at every bounce that splits.
- **replayed bounces** (opt-in): ``TorchBackend(graph_replay=True)`` records
  one fixed-width bounce as a CUDA graph and replays it, on a CUDA device
  only (:mod:`~optiland.nonsequential.backends.graph_replay`).
- **generator kernel**: the keyed generator runs as int64 limb arithmetic
  in torch by default. ``TorchBackend(rng_kernel="warp")`` draws the same
  numbers from one Warp kernel per draw (:mod:`optiland.nonsequential
  .rng_warp`) when Warp is installed and the device is CUDA; anywhere else
  the limb path is kept without a warning, and the result's
  ``environment`` says which kernel drew the trace and why.

Memory scaling: O(num_rays x max_depth) activations when gradient_mode is
"autograd". The recommended envelope is ~1e5 rays at depth 16 on a single
GPU.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.backends import graph_replay as _graph
from optiland.nonsequential.backends.array_backend import (
    BUCKET_MIN_WIDTH,
    ArrayBackend,
    bucketed_width,
)
from optiland.nonsequential.backends.graph_replay import GraphReplayUnavailable
from optiland.nonsequential.ray_bundle import backend_live_permutation
from optiland.nonsequential.rng import NSQRng

if TYPE_CHECKING:
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class TorchBackend(ArrayBackend):
    """Differentiable, device-resident PyTorch backend for NSQ raytracing.

    Uses ``optiland.backend`` (configured to torch) for all computation.
    The fixed-depth wavefront loop lets PyTorch build an autograd graph
    through the entire trace so that ``result.detectors[name].data.backward()``
    propagates gradients to scene parameters.

    In **gradient mode** compaction is disabled: dead rays
    (``alive=False``) carry zero throughput and participate in all
    operations as no-ops, and the tensor shape stays fixed across bounces so
    the graph remains a clean chain. The gather itself would be
    differentiable -- its adjoint is a scatter into a zero-filled buffer of
    the original width, exact and cheap -- so this is a shape decision, not
    a correctness one (``docs/theory/12_gpu_mapping.md`` section 12.3,
    ``docs/theory/09_differentiation.md`` R-09-11).

    In **forward-only mode** the bundle is compacted to a bucketed width
    (R-12-6). Two things make that free of a synchronisation of its own:
    the width comes from a fixed ladder (:func:`~optiland.nonsequential
    .backends.array_backend.bucketed_width`) rather than from the live count
    directly, and the one read the ladder needs is the live count the
    alive check already makes -- taken once per period and handed to both
    (:meth:`~optiland.nonsequential.backends.array_backend.ArrayBackend
    ._live_count`). A plain boolean-mask gather was the formulation ruled
    out here before, and for the right reason: the width of its own output
    is a property of the data, so it costs a read per bounce whatever the
    loop does.

    Gradient strategy is "autograd" (naive attached graph) in v1. A pluggable
    ``gradient_mode`` seam is provided for future Path Replay Backpropagation.

    Attributes:
        seed: RNG seed.
        gradient_mode: Gradient strategy (currently only "autograd").
        rng: Keyed PCG32 RNG for detached sampling decisions (see
            :mod:`optiland.nonsequential.rng`).
        alive_check_every: How often the loop may ask whether any ray is
            still alive. Each such question is one device synchronisation,
            and it is the only one the loop makes. 0 is a strict fixed trip
            count -- no synchronisation at all, ``max_depth`` bounces
            always, which is what ``docs/theory/12_gpu_mapping.md`` R-12-3
            and R-12-4 ask for.

            The default is 1, not 0, and the reason is measured rather than
            assumed. A fixed trip count runs every bounce at full width
            whether or not anything is still alive, and on a scene whose
            rays die at bounce 3 of 16 that is four times the work. On the
            quick-start singlet at 1e6 rays on this CPU, in float64, before
            compaction existed:

                period   host reads/bounce   rays/s
                0                        0   68,067
                1                        1   185,174
                2                      0.5   174,604
                4                     0.25   134,620

            The same read now also selects the compaction bucket, so at the
            default the loop makes one synchronisation per bounce and gets
            both the early exit and the narrowing out of it. 0 turns off
            both, for a device run that would rather have neither.
        compact_every: How often the bundle may be compacted to a bucketed
            width. ``None`` (the default) follows ``alive_check_every``, so
            the two share their one read. Set it explicitly only to measure
            one without the other.
        supports_splitting: Whether this backend honours
            ``SamplingPolicy.split_depth``. False unless the backend was
            built with ``allow_splitting=True``, and even then only in
            forward-only mode (:meth:`_splitting_enabled`). Off by default
            for the reason the loop avoids every other host read: the rows
            to split are chosen on the host, so a bounce that splits reads
            its hit masks back from the device. The exhaustive split is the
            estimator a deterministic ghost series needs (every order to
            floating point, ``docs/theory/03_monte_carlo.md`` 3.6), so a
            device run that needs it can ask for it.
        graph_replay: Replay one recorded bounce instead of dispatching its
            kernels one by one (:mod:`~optiland.nonsequential.backends
            .graph_replay`). Off by default, and with it off nothing in the
            trace changes. With ``True``, each batch of at least
            ``BUCKET_MIN_WIDTH`` rays runs two bounces eagerly, is recorded
            as a CUDA graph for one bounce, and the graph is replayed to
            ``max_depth``; the live count is still read between replays
            every ``alive_check_every`` bounces (0: never), and compaction is
            off, so every number is the eager fixed-width trace's
            (``compact_every=0``), which differs from the default compacted
            trace only in the summation order of its totals. It pays where
            the loop is launch-bound: on the catalogue's integrating sphere
            (r1_16, 1e6 rays, float64) an A100 measured 814.1 s eager and
            189.9 s replayed, in the hybrid-engine study of 2026-09-24 (report
            H4, a patched copy of this engine at 6e175a2f; issue 2 of the
            research repository).

            It needs a CUDA device: on the CPU and on Apple's ``mps`` torch
            has no graph capture, and ``True`` is refused there with
            :class:`GraphReplayUnavailable` rather than silently run
            eagerly. It is refused the same way in gradient mode, when
            splitting would run, with ``record_paths``, with a ray-database
            detector, and when the capture's guard finds an accumulator
            rebound inside the recorded bounce. ``"emulate"`` runs the same
            bookkeeping eagerly on any device -- the check that a scene is
            capture-safe, with the eager fixed-width numbers and no speed-up.
        rng_kernel: Which implementation of the keyed generator is asked
            for: ``"torch"`` (the default, the limb path of
            :mod:`optiland.nonsequential.rng`) or ``"warp"`` (one Warp
            kernel per draw, :mod:`optiland.nonsequential.rng_warp`). The
            values are the same bit for bit at float64 and float32; only
            the number of dispatched operations changes (on the catalogue's
            sphere cavity, from about 2,470 per bounce to about 650). The
            Warp kernel is used only when ``warp`` imports and the device
            is CUDA; otherwise the trace uses the limb path, silently, and
            ``SimulationResult.environment`` records ``rng_kernel`` (the
            kernel that drew), ``rng_kernel_requested`` and, on a fallback,
            ``rng_kernel_note`` (why).
        rng_kernel_in_use: The kernel the last trace drew from; None before
            the first trace.
    """

    host_reads_free = False
    supports_splitting = False
    alive_check_every = 1

    RNG_KERNELS: tuple[str, ...] = ("torch", "warp")

    def __init__(
        self,
        seed: int | None = None,
        gradient_mode: Literal["autograd"] = "autograd",
        alive_check_every: int | None = None,
        compact_every: int | None = None,
        allow_splitting: bool = False,
        graph_replay: bool | Literal["emulate"] = False,
        rng_kernel: Literal["torch", "warp"] = "torch",
    ) -> None:
        """Initialize TorchBackend.

        Args:
            seed: Optional random seed for reproducibility.
            gradient_mode: Gradient computation strategy. Currently only
                ``"autograd"`` is supported; "prb" is the planned follow-up.
            alive_check_every: Override the class default (see the class
                docstring). 0 disables the check entirely.
            compact_every: Override the compaction period, which otherwise
                follows ``alive_check_every``. 0 disables compaction, which
                is what gradient mode does for itself.
            allow_splitting: Honour ``SamplingPolicy.split_depth`` in
                forward-only mode (default False: warn and fall back to the
                single-branch draw, as before). In gradient mode splitting
                is refused whatever this says.
            graph_replay: Replay a recorded bounce as a CUDA graph (default
                False; see the class docstring). ``True`` needs a CUDA
                device; ``"emulate"`` runs the same bookkeeping eagerly on
                any device, as a check. Either keeps each batch at its full
                width, so ``compact_every`` must be left unset or 0.
            rng_kernel: ``"torch"`` (default) or ``"warp"``; see the class
                docstring. The choice is made at the start of each trace,
                when the device is known.

        Raises:
            ValueError: If ``graph_replay`` is not ``False``, ``True`` or
                ``"emulate"``, or is combined with a non-zero
                ``compact_every``; or if ``rng_kernel`` is not one of
                :attr:`RNG_KERNELS`.
        """
        if graph_replay not in (False, True, "emulate"):
            raise ValueError(
                f"graph_replay must be False, True or 'emulate', got {graph_replay!r}"
            )
        if graph_replay and compact_every:
            raise ValueError(
                "graph_replay replays one bundle width per batch, so it cannot be "
                f"combined with compaction (compact_every={compact_every}); leave "
                "compact_every unset or 0"
            )
        if rng_kernel not in self.RNG_KERNELS:
            raise ValueError(
                f"rng_kernel must be one of {self.RNG_KERNELS}, got {rng_kernel!r}"
            )
        self.seed = seed
        self.gradient_mode = gradient_mode
        self.supports_splitting = bool(allow_splitting)
        # Detached sampling uses a keyed RNG (sampling decisions are detached)
        self.rng = NSQRng(seed)
        if alive_check_every is not None:
            self.alive_check_every = int(alive_check_every)
        if graph_replay != "emulate":
            graph_replay = bool(graph_replay)
        self.graph_replay = graph_replay
        self.compact_every = 0 if graph_replay else compact_every
        self.rng_kernel = rng_kernel
        self.rng_kernel_in_use: str | None = None
        # The current batch's replay buffers, between _graph_handover_depth and
        # _replay_bounces (graph_replay only).
        self._graph_static: dict | None = None
        self._rng_kernel_note: str | None = None

    def _warp_rng_unavailable(self) -> str | None:
        """Why the Warp generator cannot serve this trace, or None if it can.

        Returns:
            None when the Warp kernel will draw; otherwise one sentence
            naming the reason (Warp not installed, not CUDA, no kernel load).
        """
        try:
            from optiland.nonsequential import rng_warp  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 - not installed, or it failed to load
            return f"warp is not importable ({type(exc).__name__})"
        return rng_warp.availability(be.get_device())

    def _trace_rng(self, seed: int | None) -> NSQRng:
        """The generator for this trace: the limb path, or the Warp kernel.

        With the default ``rng_kernel="torch"`` this is exactly the base
        class's choice. With ``"warp"`` the kernel is checked (and loaded on
        the device) here, before the first bounce, so no kernel is compiled
        or loaded inside the bounce loop.

        Args:
            seed: The trace's seed argument, or None to keep the backend's.

        Returns:
            The generator for this trace.
        """
        self._rng_kernel_note = None
        if self.rng_kernel == "warp":
            note = self._warp_rng_unavailable()
            if note is None:
                from optiland.nonsequential.rng_warp import (  # noqa: PLC0415
                    WarpNSQRng,
                )

                self.rng_kernel_in_use = "warp"
                return WarpNSQRng(self.rng.seed if seed is None else seed)
            self._rng_kernel_note = note
        self.rng_kernel_in_use = "torch"
        rng = super()._trace_rng(seed)
        if getattr(rng, "kernel", None) == "warp":
            # A previous trace on a CUDA device drew through the kernel;
            # this one cannot, so it gets the limb path with the same seed.
            rng = NSQRng(rng.seed)
        return rng

    def _environment(self) -> dict[str, object]:
        """The base environment, plus the generator kernel asked for.

        Returns:
            The base class's entries, ``rng_kernel_requested`` and, when the
            request could not be honoured, ``rng_kernel_note``.
        """
        env = super()._environment()
        env["rng_kernel_requested"] = self.rng_kernel
        if self._rng_kernel_note is not None:
            env["rng_kernel_note"] = self._rng_kernel_note
        return env

    def _gradient_mode(self, rays: NSQRayBundle) -> bool:
        """True when any field of the bundle is on an autograd graph.

        A host-side attribute read on each of the bundle's float fields --
        ``requires_grad`` is a property of the tensor, not of its contents,
        so asking costs no synchronisation. True for a leaf that requires a
        gradient and for anything derived from one, which is what makes a
        scene whose gradients enter at a surface (rather than at the source)
        switch to fixed shapes as soon as that surface is reached.

        Args:
            rays: Current ray bundle.

        Returns:
            True when the width must stay fixed.
        """
        return any(
            getattr(field, "requires_grad", False)
            for field in (
                rays.x,
                rays.y,
                rays.z,
                rays.L,
                rays.M,
                rays.N,
                rays.flux,
                rays.wavelength,
                rays.n_current,
                rays.k_current,
            )
        )

    def _maybe_compact(self, rays: NSQRayBundle, depth: int) -> NSQRayBundle:
        """Gather the live rays down to a bucketed width.

        Runs at the end of every ``compaction_period``-th bounce, and only
        in forward-only mode. The width comes from
        :func:`~optiland.nonsequential.backends.array_backend
        .bucketed_width`, which returns the bundle's current width while
        more than half its rays are alive -- so the gather is skipped
        entirely until it pays for itself, which is the alive-fraction
        trigger R-12-6 asks for.

        Args:
            rays: Current ray bundle.
            depth: Bounces already run for this batch, counting from 0.

        Returns:
            The bundle, narrowed or unchanged.
        """
        period = self.compaction_period
        if not period or (depth + 1) % period:
            return rays
        if self._gradient_mode(rays):
            return rays

        n_live = self._live_count(rays)
        width = bucketed_width(n_live, rays.num_rays)
        if width >= rays.num_rays:
            return rays
        perm = backend_live_permutation(rays.alive, n_live)
        return rays.take(perm[:width])

    def _splitting_enabled(self) -> bool:
        """Split only when asked to, and only in forward-only mode.

        Gradient mode keeps the bundle at one fixed width for the whole
        trace, so the autograd graph is a single chain of fixed-shape
        operations -- the same reason compaction is off there.

        Returns:
            True when ``allow_splitting`` was given and the backend's
            gradient mode is off.
        """
        if not self.supports_splitting:
            return False
        try:
            grad_on = bool(be.grad_mode.requires_grad)
        except Exception:  # noqa: BLE001 - a backend with no grad mode cannot be in it
            grad_on = False
        return not grad_on

    def _check_sampling_support(self, ir) -> None:
        """Refuse bounded splitting loudly where it is not honoured.

        Bounded splitting grows the live ray bundle, which conflicts with
        the fixed tensor shapes this backend's autograd graph requires, and
        on a device costs a host read per splitting bounce. Unless the
        backend was built with ``allow_splitting=True`` and the trace is
        forward-only, a non-zero ``split_depth`` is never silently ignored:
        warn and fall back to importance-biased single-branch sampling.

        Args:
            ir: The scene's lowered IR.
        """
        if ir.sampling.split_depth > 0 and self.supports_splitting:
            if not self._splitting_enabled():
                warnings.warn(
                    f"TorchBackend does not split in gradient mode "
                    f"(sampling_policy.split_depth={ir.sampling.split_depth}); "
                    "fixed tensor shapes are required for the autograd graph. "
                    "Falling back to importance-biased single-branch sampling "
                    "(split_depth is ignored for this trace).",
                    stacklevel=2,
                )
            return
        if ir.sampling.split_depth > 0:
            warnings.warn(
                f"TorchBackend does not support bounded splitting "
                f"(sampling_policy.split_depth={ir.sampling.split_depth}); "
                "fixed tensor shapes are required for the autograd graph. "
                "Falling back to importance-biased single-branch sampling "
                "(split_depth is ignored). Use NumpyBackend for bounded "
                "splitting.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Replayed bounces (graph_replay)
    # ------------------------------------------------------------------

    def _check_graph_replay(self, scene, ir, record_paths) -> None:
        """Refuse ``graph_replay`` where a replayed bounce cannot be right.

        Each refusal is a :class:`GraphReplayUnavailable` naming its reason;
        none falls back silently.

        Args:
            scene: The scene about to be traced.
            ir: The scene's lowered IR.
            record_paths: The trace's ``record_paths`` argument.

        Raises:
            GraphReplayUnavailable: On a device without CUDA graphs (for
                ``graph_replay=True``), in gradient mode, when splitting would
                run, with path recording, or with a ray-database detector.
        """
        if not self.graph_replay:
            return
        import torch as _torch  # noqa: PLC0415

        from optiland.nonsequential.detectors.ray_database import (  # noqa: PLC0415
            RayDatabaseDetector,
        )

        if self.graph_replay is True:
            device = str(be.get_device())
            if not (device.startswith("cuda") and _torch.cuda.is_available()):
                raise GraphReplayUnavailable(
                    "graph_replay=True records a CUDA graph, and the active device "
                    f"is {device!r}: torch has no graph capture on the CPU or on "
                    "Apple's mps. Trace eagerly (graph_replay=False), or check a "
                    "scene's capture safety with graph_replay='emulate'"
                )
        try:
            grad_on = bool(be.grad_mode.requires_grad)
        except Exception:  # noqa: BLE001 - a backend with no grad mode cannot be in it
            grad_on = False
        if grad_on:
            raise GraphReplayUnavailable(
                "graph_replay: gradient mode is on; a replayed bounce records no "
                "autograd graph, so a gradient trace runs eagerly"
            )
        if ir.sampling.split_depth > 0 and self._splitting_enabled():
            raise GraphReplayUnavailable(
                "graph_replay: bounded splitting grows the bundle and chooses its "
                "rows on the host, which a replayed bounce cannot do; trace with "
                "graph_replay=False, or without allow_splitting"
            )
        if record_paths:
            raise GraphReplayUnavailable(
                "graph_replay: path recording copies ray state to the host at "
                "every bounce; trace with record_paths=False"
            )
        databases = [
            getattr(d, "name", "") or type(d).__name__
            for d in scene.detectors
            if isinstance(d, RayDatabaseDetector)
        ]
        if databases:
            raise GraphReplayUnavailable(
                "graph_replay: a ray database copies its hits to the host at every "
                f"bounce ({', '.join(databases)})"
            )

    def _graph_handover_depth(self, rays: NSQRayBundle) -> int | None:
        """Hand a batch to the replayed graph after the eager bounces.

        A batch narrower than the compaction ladder's floor
        (:data:`~optiland.nonsequential.backends.array_backend
        .BUCKET_MIN_WIDTH`) runs eagerly at its fixed width instead: below it
        the shared material cache keys on the wavelength array's contents,
        a host read at every bounce that no graph can hold, and a graph buys
        nothing at that width anyway. The values are the same either way.

        A batch that will be replayed has its fields moved into the replay's
        static buffers here, before its eager bounces
        (:func:`~optiland.nonsequential.backends.graph_replay.static_buffers`),
        so the glass memo the eager bounces fill is keyed on the wavelength
        tensor the recorded bounce reads.

        Args:
            rays: The freshly prepared batch.

        Returns:
            :data:`~optiland.nonsequential.backends.graph_replay.EAGER_BOUNCES`,
            or ``None`` for an eager batch.
        """
        self._graph_static = None
        if not self.graph_replay or rays.num_rays < BUCKET_MIN_WIDTH:
            return None
        self._graph_static = _graph.static_buffers(rays)
        return _graph.EAGER_BOUNCES

    def _replay_bounces(
        self, rays, bounce, depth, max_depth, scene, tallies
    ) -> NSQRayBundle:
        """Record the bounce once and replay it to the depth cap.

        See :func:`~optiland.nonsequential.backends.graph_replay
        .replay_bounces`.

        Args:
            rays: The batch after the eager bounces.
            bounce: The loop's bounce body.
            depth: Bounces already run.
            max_depth: The depth cap.
            scene: The scene being traced.
            tallies: The trace's own tallies.

        Returns:
            The bundle after the last bounce.
        """
        mode = "emulate" if self.graph_replay == "emulate" else "cuda"
        static, self._graph_static = self._graph_static, None
        return _graph.replay_bounces(
            self,
            rays,
            bounce,
            depth,
            max_depth,
            lambda: _graph.accumulator_identities(scene, tallies),
            static=static,
            mode=mode,
        )

    def _prepare_bundle(self, rays: NSQRayBundle) -> NSQRayBundle:
        """Promote a generated bundle onto this backend's device.

        Args:
            rays: Ray bundle from ``source.generate()``.

        Returns:
            The same bundle with every field a torch tensor on one device.
        """
        return self._ensure_torch_bundle(rays)

    def _ensure_torch_bundle(self, rays: NSQRayBundle) -> NSQRayBundle:
        """Convert every NSQRayBundle field to a tensor on one device.

        Sources produce NumPy arrays.  NumPy 2.0 disallows mixed
        numpy/torch arithmetic, so every field is promoted at batch start.
        Gradient-carrying fields (flux) are left untouched if already a
        Tensor.

        Every field is placed on the *same* device -- including the ones
        that used to be built with ``torch.from_numpy`` and so always
        landed on the CPU (``alive``, ``bounce``) and the one that used to
        stay NumPy outright (``ray_id``, uploaded again on every keyed
        draw). Those were the first thing a non-CPU device hit.

        Args:
            rays: Ray bundle from source.generate().

        Returns:
            Same ray bundle with all arrays as torch Tensors.
        """
        import torch as _torch  # noqa: PLC0415

        def _to_float(x: object) -> _torch.Tensor:
            if isinstance(x, _torch.Tensor):
                return x
            return be.array(x)

        rays.x = _to_float(rays.x)
        rays.y = _to_float(rays.y)
        rays.z = _to_float(rays.z)
        rays.L = _to_float(rays.L)
        rays.M = _to_float(rays.M)
        rays.N = _to_float(rays.N)
        rays.flux = _to_float(rays.flux)
        rays.wavelength = _to_float(rays.wavelength)
        rays.n_current = _to_float(rays.n_current)
        rays.k_current = _to_float(rays.k_current)

        device = rays.x.device

        def _to_bool(x: object) -> _torch.Tensor:
            if isinstance(x, _torch.Tensor):
                return x.to(device=device, dtype=_torch.bool)
            return _torch.as_tensor(
                np.asarray(x, dtype=bool).copy(), dtype=_torch.bool, device=device
            )

        def _to_int(x: object, dtype: _torch.dtype) -> _torch.Tensor:
            if isinstance(x, _torch.Tensor):
                return x.to(device=device, dtype=dtype)
            np_dtype = np.int64 if dtype == _torch.int64 else np.int32
            return _torch.as_tensor(
                np.asarray(x, dtype=np_dtype).copy(), dtype=dtype, device=device
            )

        rays.alive = _to_bool(rays.alive)
        rays.bounce = _to_int(rays.bounce, _torch.int32)
        if rays.ray_id is not None:
            # The generator is keyed by (seed, ray_id, bounce, slot) and
            # evaluates on the device; leaving ray_id on the host meant
            # uploading it again on every draw.
            rays.ray_id = _to_int(rays.ray_id, _torch.int64)

        # The medium stack is ray state like any other: it lives on the same
        # device as the rest of the bundle, as an integer table, so
        # RefractiveComponent.interact can push and pop it without moving
        # anything to the host.
        rays.medium_stack = _to_int(rays.medium_stack, _torch.int64)
        rays.medium_depth = _to_int(rays.medium_depth, _torch.int32)
        rays.medium_stack_underflows = _to_int(
            rays.medium_stack_underflows, _torch.int32
        )
        # The reflection count is ray state too, beside the bounce count.
        rays.reflections = _to_int(rays.reflections, _torch.int32)
        return rays

    def _to_numpy(self, arr: object) -> np.ndarray:
        """Backward-compatible alias."""
        return to_numpy(arr)
