"""CUDA-graph replay of the fixed-width bounce, for the torch backend.

On the catalogue's integrating-sphere cases the torch loop is host-bound: at
16,384-ray batches each bounce dispatches about 2,470 small kernels, and the
host time to launch them, not the arithmetic, sets the wall time. A CUDA graph
records one bounce's kernels once and relaunches the whole sequence with one
call. Measured on an A100 in the hybrid-engine study of 2026-09-24 (report H4,
issue 2 of the research repository): r1_16 at 1e6 rays, 814.1 s eager to
189.9 s with the bounce replayed, the multiplier identical to the last bit on
24 of 25 traces and 2 ulp apart on the 25th (a summation over the compacted
width the eager default uses).

What a replay needs from the bounce, and how each need is met:

- **One width.** A graph replays fixed shapes, so the batch is never
  compacted: it keeps its full width and dead rays ride along as masked lanes
  (``compact_every=0``). Values are those of the eager fixed-width loop.
- **Nothing crosses to the host.** No host read (the device loop reads
  nothing per bounce by construction, ``test_nsq_host_reads.py``) and no
  upload (``test_nsq_host_uploads.py``). A scene that needs a host transfer
  per bounce -- path recording, a ray database, splitting -- is refused.
- **Accumulators add in place.** A graph writes to the memory it recorded; a
  tally rebound to a new tensor inside the recorded bounce would silently
  stop accumulating after the first replay. Every tally and detector counter
  adds in place after its first term, and the identity of every accumulator
  is compared before and after the capture (:func:`accumulator_identities`);
  a change refuses the replay rather than returning a wrong ledger.
- **Every dtype settled.** The first bounces run eagerly
  (:data:`EAGER_BOUNCES`): they build every lazy table and constant and give
  every tally its first term, so the recorded bounce is the steady one.

The ray state lives in static buffers: each tensor field of the bundle is
cloned once before the capture, the recorded bounce reads them, and at its end
copies every field it rebound back into them, so one replay advances the
bundle one bounce.

``mode="emulate"`` runs the same bookkeeping without CUDA: the bounce runs
eagerly through the static buffers and the same guard, trip for trip. It is a
check of the data flow and of a scene's capture safety on a machine without a
CUDA device, not a graph -- torch has no graph capture on the CPU or on
Apple's ``mps`` -- and its numbers are the eager fixed-width numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.scene import NSQScene

#: Bounces run eagerly before the capture. The first builds every lazily
#: created table and constant and gives every tally its first term; the
#: second is the measured recipe's margin (H4 captured at bounce 2).
EAGER_BOUNCES = 2


class GraphReplayUnavailable(RuntimeError):
    """``graph_replay`` asked for where a replayed bounce would be wrong or impossible.

    Raised, never worked around: a device without CUDA graphs, gradient mode,
    splitting, per-bounce host transfers (path recording, a ray database), or a
    capture the guard found unsafe. The message names the reason.
    """


def accumulator_identities(scene: NSQScene, tallies: Iterable[Any]) -> dict[str, Any]:
    """The storage identity of every per-trace accumulator a bounce writes.

    The trace's own tallies, and every tensor attribute and tally of every
    scene surface and detector (the ledger tallies, the absorbed counts, the
    detector buffers and hit counters). The identity is the tensor object, its
    storage address, dtype, shape and whether it carries a gradient, or
    ``None`` before the first term.

    Args:
        scene: The scene being traced.
        tallies: The trace's own ``Tally`` and ``_TallyVector`` objects.

    Returns:
        ``{label: identity}``.
    """
    import torch  # noqa: PLC0415

    from optiland.nonsequential._tally import Tally, _TallyVector  # noqa: PLC0415

    def ident(value):
        if torch.is_tensor(value):
            return (
                id(value),
                value.data_ptr(),
                str(value.dtype),
                tuple(value.shape),
                bool(value.requires_grad),
            )
        return None

    out: dict[str, Any] = {}
    for i, tally in enumerate(tallies):
        out[f"trace tally {i}"] = ident(tally._dev)
    for kind, owners in (("surface", scene.surfaces), ("detector", scene.detectors)):
        for j, owner in enumerate(owners):
            label = getattr(owner, "name", "") or f"{kind} {j}"
            for attr, value in getattr(owner, "__dict__", {}).items():
                if isinstance(value, (Tally, _TallyVector)):
                    out[f"{label}.{attr}"] = ident(value._dev)
                elif torch.is_tensor(value):
                    out[f"{label}.{attr}"] = ident(value)
    return out


def _rebound(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    keys = before.keys() | after.keys()
    return sorted(k for k in keys if before.get(k) != after.get(k))


def replay_bounces(
    backend: Any,
    rays: NSQRayBundle,
    bounce: Callable[[NSQRayBundle, int], NSQRayBundle],
    depth: int,
    max_depth: int,
    accumulators: Callable[[], dict[str, Any]],
    mode: str = "cuda",
) -> NSQRayBundle:
    """Run bounces ``depth .. max_depth - 1`` of one batch as a replayed graph.

    Args:
        backend: The torch backend; its ``alive_check_every`` sets how often
            the live count is read between replays (0: never), as the eager
            loop reads it between bounces.
        rays: The batch after the eager bounces, at its full width.
        bounce: The loop's bounce body, ``bounce(rays, depth) -> rays``.
        depth: Bounces already run.
        max_depth: The trace's depth cap.
        accumulators: Returns :func:`accumulator_identities` for this trace.
        mode: ``"cuda"`` records and replays a CUDA graph; ``"emulate"`` runs
            the same bookkeeping eagerly (see the module docstring).

    Returns:
        The bundle after the last bounce, its fields in the static buffers.

    Raises:
        GraphReplayUnavailable: If the bundle carries a gradient, if the
            bounce replaces the bundle or rebinds an accumulator, or if the
            CUDA capture itself fails.
    """
    import torch  # noqa: PLC0415

    fields = sorted(k for k, v in vars(rays).items() if torch.is_tensor(v))
    before = accumulators()
    attached = [f for f in fields if getattr(rays, f).requires_grad]
    attached += [label for label, ident in before.items() if ident and ident[4]]
    if attached:
        raise GraphReplayUnavailable(
            "graph_replay: the trace carries a gradient "
            f"({', '.join(attached)}); a replayed bounce records no autograd "
            "graph, so gradient mode runs eagerly (graph_replay=False)"
        )
    static = {f: getattr(rays, f).clone() for f in fields}
    for f in fields:
        setattr(rays, f, static[f])

    def write_back() -> None:
        # Copy every field the bounce rebound into its static buffer, and point
        # the bundle back at the buffers the next bounce (or replay) reads.
        for f in fields:
            current = getattr(rays, f)
            if current is not static[f]:
                static[f].copy_(current)

    def check_bundle(out: NSQRayBundle) -> None:
        if out is not rays:
            raise GraphReplayUnavailable(
                "graph_replay: the bounce replaced the ray bundle (compaction or "
                "splitting); a replay needs one bundle of one width"
            )

    k = int(getattr(backend, "alive_check_every", 0) or 0)

    if mode == "emulate":
        # The recorded bounce, run instead of recorded: it executes, so the
        # replays that follow are one fewer than on CUDA.
        check_bundle(bounce(rays, depth))
        write_back()
        for f in fields:
            setattr(rays, f, static[f])
        rebound = _rebound(before, accumulators())
        if rebound:
            raise GraphReplayUnavailable(
                "graph_replay: the bounce rebinds accumulators instead of adding in "
                f"place, which a replay would silently reset: {rebound}"
            )
        d = depth + 1
        while d < max_depth:
            if k and d % k == 0 and rays.num_rays_alive == 0:
                break
            check_bundle(bounce(rays, d))
            write_back()
            for f in fields:
                setattr(rays, f, static[f])
            d += 1
        return rays

    if mode != "cuda":
        raise ValueError(f"graph_replay mode must be 'cuda' or 'emulate', got {mode!r}")

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            check_bundle(bounce(rays, depth))
            write_back()
    except GraphReplayUnavailable:
        raise
    except RuntimeError as exc:
        raise GraphReplayUnavailable(
            f"graph_replay: the CUDA capture of the bounce failed ({exc}); a "
            "per-bounce host transfer or a data-dependent shape is the usual cause"
        ) from exc
    for f in fields:
        setattr(rays, f, static[f])
    rebound = _rebound(before, accumulators())
    if rebound:
        raise GraphReplayUnavailable(
            "graph_replay: the captured bounce rebinds accumulators instead of "
            f"adding in place, which a replay would silently reset: {rebound}"
        )
    # The capture recorded the bounce without running it: bounce `depth` is
    # the first replay.
    d = depth
    while d < max_depth:
        if d > depth and k and d % k == 0 and rays.num_rays_alive == 0:
            break
        graph.replay()
        d += 1
    torch.cuda.synchronize()
    del graph
    return rays
