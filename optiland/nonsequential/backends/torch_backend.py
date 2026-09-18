"""TorchBackend -- the differentiable, device-resident array backend.

The trace loop itself lives in :class:`~optiland.nonsequential.backends
.array_backend.ArrayBackend`; this class is the thin part that is specific
to running it on torch tensors on a device:

- **device placement**: :meth:`TorchBackend._prepare_bundle` promotes every
  field of a freshly generated bundle -- floats, the alive flag, the bounce
  and ray-id integers, and the medium-stack table -- onto the backend's
  device. Nothing in the trace path then refers to a host array.
- **fixed shapes**: the bundle is never compacted, so the autograd graph is
  a single clean chain of fixed-shape operations and no bounce needs a
  boolean-mask gather (whose output shape is a device read).
- **loop control**: ``host_reads_free = False`` tells the loop that
  reducing a per-ray mask to a Python bool is a synchronisation, so the
  loop skips nothing and takes a bounded trip count, asking whether any ray
  is still alive at most once every ``alive_check_every`` bounces.
- **capability**: bounded splitting grows the bundle and is refused.

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
from optiland.nonsequential.backends.array_backend import ArrayBackend
from optiland.nonsequential.rng import NSQRng

if TYPE_CHECKING:
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class TorchBackend(ArrayBackend):
    """Differentiable, device-resident PyTorch backend for NSQ raytracing.

    Uses ``optiland.backend`` (configured to torch) for all computation.
    The fixed-depth wavefront loop lets PyTorch build an autograd graph
    through the entire trace so that ``result.detectors[name].data.backward()``
    propagates gradients to scene parameters.

    Compaction is disabled: dead rays (``alive=False``) carry zero throughput
    and participate in all operations as no-ops; the tensor shape stays fixed
    across bounces so the graph remains clean.

    It stays disabled in forward-only mode too, which is a decision and not
    an oversight. Compaction by boolean mask costs one device-to-host
    synchronisation per bounce, because the width of its own output is a
    property of the data -- which is the cost this backend exists to avoid,
    and it would put back per bounce exactly what the rest of the loop gives
    up. The saving is real and worth having: ``docs/theory/12_gpu_mapping.md``
    R-12-6 gets it without the read by rounding the live count up to a fixed
    ladder of bucketed widths, so the shape is chosen from a small known set
    rather than measured. That is the next item, not this one.

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
            quick-start singlet at 1e6 rays on this CPU, in float64:

                period   host reads/bounce   rays/s
                0                        0   68,067
                1                        1   185,174
                2                      0.5   174,604
                4                     0.25   134,620

            So the last synchronisation is worth keeping until the dead
            rays stop costing anything, and what makes them stop costing is
            compaction to bucketed widths (R-12-6), not the trip count.
            Until that lands, 1 removes 96% of this engine's per-bounce
            synchronisations and keeps the early exit that pays for the
            other 4%; 0 is one argument away for a device run that would
            rather have neither.
    """

    host_reads_free = False
    supports_splitting = False
    alive_check_every = 1

    def __init__(
        self,
        seed: int | None = None,
        gradient_mode: Literal["autograd"] = "autograd",
        alive_check_every: int | None = None,
    ) -> None:
        """Initialize TorchBackend.

        Args:
            seed: Optional random seed for reproducibility.
            gradient_mode: Gradient computation strategy. Currently only
                ``"autograd"`` is supported; "prb" is the planned follow-up.
            alive_check_every: Override the class default (see the class
                docstring). 0 disables the check entirely.
        """
        self.seed = seed
        self.gradient_mode = gradient_mode
        # Detached sampling uses a keyed RNG (sampling decisions are detached)
        self.rng = NSQRng(seed)
        if alive_check_every is not None:
            self.alive_check_every = int(alive_check_every)

    def _check_sampling_support(self, ir) -> None:
        """Refuse bounded splitting, loudly.

        Bounded splitting grows the live ray bundle, which conflicts with
        the fixed tensor shapes this backend's autograd graph requires.
        Never silently ignored -- warn and fall back to importance-biased
        single-branch sampling, which this backend always uses regardless of
        ``split_depth``.

        Args:
            ir: The scene's lowered IR.
        """
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
        return rays

    def _to_numpy(self, arr: object) -> np.ndarray:
        """Backward-compatible alias."""
        return to_numpy(arr)
