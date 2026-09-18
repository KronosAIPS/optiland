"""No device-to-host read inside the bounce loop, on the device backend.

``docs/theory/12_gpu_mapping.md`` R-12-4 asks for zero host synchronisations
per bounce, with all accounting device-side and flushed once per trace, and
T-12-1 asks for it to be asserted by an instrumented backend rather than
inspected. This module is that instrument.

Every route a torch tensor's value can take to the host is wrapped:
``Tensor.cpu``, ``.item``, ``.tolist``, ``.__array__`` and the scalar
conversions ``__bool__``/``__float__``/``__int__``/``__index__``.
``optiland.backend.utils.to_numpy`` is ``detach().cpu().numpy()``, so it is
caught by ``cpu``; ``numpy`` is deliberately not wrapped, because counting
it as well would count every ``to_numpy`` twice.

The loop is bracketed by the backend's own ``intersect_scene``, which runs
exactly once at the top of every bounce, so no hook is needed in the traced
code and the same instrument measures any engine that has that method. The
count between two consecutive entries is one bounce's reads; the interval
after the last entry is excluded, because it holds the once-per-trace flush
as well as the last bounce.

Kramer Harrison, 2026
"""

from __future__ import annotations

import collections
import traceback

import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.array_backend import ArrayBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend

# One entry per host transfer. See the module docstring for why ``numpy``
# is not in this list.
_TRANSFERS = (
    "cpu",
    "item",
    "tolist",
    "__array__",
    "__bool__",
    "__float__",
    "__int__",
    "__index__",
)

# The one per-bounce read this branch does not own. ``BaseComponent
# .intersect`` resolves its component's placement by calling
# ``components/base.py:_get_transform`` on every bounce, which round-trips
# the coordinate system through the host and uploads the result again --
# ``docs/theory/12_gpu_mapping.md`` R-12-8 wants scene data resolved once
# per scene change instead. The detectors in this package already do that
# (``BaseDetector.frame``); the components do not, and that file belongs to
# the geometry work. Counted separately so it cannot grow unnoticed.
_UNOWNED = "components/base.py"


class _ReadCounter:
    """Count and attribute host transfers during a trace."""

    def __init__(self) -> None:
        self.total = 0
        self.sites: collections.Counter[str] = collections.Counter()
        self.marks: list[tuple[int, collections.Counter[str], int, int]] = []
        self._orig: dict[str, object] = {}
        self._orig_intersect = None
        # The bundle the previous bounce ran on, held by reference rather
        # than by id(): CPython reuses the id of a collected object, and a
        # compacted bundle is collected the moment the loop rebinds past
        # it. One strong reference is enough to make "is it still the same
        # bundle?" answerable for adjacent bounces, and the sequence number
        # below carries that answer into the mark list.
        self._last_bundle = None
        self._bundle_seq = 0

    # -- attribution ---------------------------------------------------

    def _site(self) -> str:
        stack = traceback.extract_stack(limit=25)[:-2]
        for frame in reversed(stack):
            if "/tests/" in frame.filename:
                continue
            if "/nonsequential/" in frame.filename:
                short = frame.filename.split("/nonsequential/")[-1]
                return f"{short}:{frame.lineno} {frame.name}"
        return "outside the engine"

    def _record(self) -> None:
        self.total += 1
        self.sites[self._site()] += 1

    # -- installation --------------------------------------------------

    def __enter__(self) -> _ReadCounter:
        for name in _TRANSFERS:
            orig = getattr(torch.Tensor, name)
            self._orig[name] = orig

            def wrapper(inner, *a, _orig=orig, **k):
                self._record()
                return _orig(inner, *a, **k)

            setattr(torch.Tensor, name, wrapper)

        self._orig_intersect = ArrayBackend.intersect_scene

        def bracketed(backend, rays, *a, **k):
            if rays is not self._last_bundle:
                self._bundle_seq += 1
                self._last_bundle = rays
            self.mark(rays.num_rays, self._bundle_seq)
            return self._orig_intersect(backend, rays, *a, **k)

        ArrayBackend.intersect_scene = bracketed
        return self

    def __exit__(self, *exc) -> None:
        for name, orig in self._orig.items():
            setattr(torch.Tensor, name, orig)
        ArrayBackend.intersect_scene = self._orig_intersect

    def mark(self, width: int = 0, bundle: int = 0) -> None:
        self.marks.append((self.total, self.sites.copy(), width, bundle))

    # -- results -------------------------------------------------------

    @property
    def bounces(self) -> int:
        """Bounce-to-bounce intervals fully inside the loop."""
        return max(len(self.marks) - 1, 0)

    def per_bounce_sites(self) -> collections.Counter[str]:
        """Reads per site over the complete bounce-to-bounce intervals."""
        if len(self.marks) < 2:
            return collections.Counter()
        return self.marks[-1][1] - self.marks[0][1]

    def widths(self) -> list[int]:
        """The bundle width the loop ran each bounce at, in order."""
        return [width for _, _, width, _ in self.marks]

    def _steady(self, i: int) -> bool:
        """True when interval ``i`` is pure bounce work on one ray-state buffer.

        The interval runs from bounce ``i``'s traversal to bounce
        ``i + 1``'s, so it holds bounce ``i``'s body and whatever came after
        it. It is pure bounce work only when the bundle it started on was
        already the previous bounce's (nothing about the ray state's shape
        or storage is new) and is still the next bounce's (the interval did
        not build a replacement -- a compaction, or the next batch's freshly
        generated rays).
        """
        return (
            0 < i < len(self.marks) - 1
            and self.marks[i - 1][3] == self.marks[i][3] == self.marks[i + 1][3]
        )

    def steady_bounces(self) -> int:
        """Bounces whose bundle is the object the previous bounce ran on."""
        return sum(1 for i in range(len(self.marks) - 1) if self._steady(i))

    def recurring_sites(self) -> collections.Counter[str]:
        """How many *steady* bounces each site read in, after its first read.

        Two kinds of read are not per-bounce costs and must not be counted
        as one.

        A read that fires **once per trace** -- a glass catalogue entry
        resolved on its first use -- shows up in one interval whichever
        bounce it lands in. It is excluded by counting a site only from its
        second read.

        A read that fires **once per ray-state buffer** -- the source
        building a batch, or a cache in the shared library keyed on an
        array's shape and identity, which every fresh batch and every
        compaction event presents a new one of -- is the per-shape cost that
        bucketed compaction trades a per-bounce cost for
        (``docs/theory/12_gpu_mapping.md`` section 12.3: O(log N) shapes,
        each paid for once). It is excluded by counting only *steady*
        intervals: those that start and end on the bundle object the loop
        was already running on, so nothing about the ray state's shape or
        storage is new in them.

        A genuinely per-bounce read survives both rules, because it reads in
        every steady bounce and a trace has many.
        """
        seen: set[str] = set()
        recurring: collections.Counter[str] = collections.Counter()
        for i in range(len(self.marks) - 1):
            steady = self._steady(i)
            for site in self.marks[i + 1][1] - self.marks[i][1]:
                if steady and site in seen:
                    recurring[site] += 1
                seen.add(site)
        return recurring


def _singlet() -> NSQScene:
    """The package quick-start scene: 1 W collimated beam, singlet, detector."""
    scene = NSQScene()
    spec = Spectrum.monochromatic(0.55)
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=5.0),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50),
        LensConfig(
            r1=100.0,
            r2=-100.0,
            thickness=5.0,
            material="N-BK7",
            front_aperture_radius=12.5,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=150),
        IrradianceDetectorConfig(
            width=20, height=20, num_pixels_x=64, num_pixels_y=64, splat="bilinear"
        ),
    )
    return scene


def _trace_and_count(
    num_rays: int,
    max_depth: int,
    alive_check_every,
    batch_size: int | None = None,
    scene_fn=None,
):
    be.set_backend("torch")
    be.set_precision("float64")
    try:
        scene = (scene_fn or _singlet)()
        backend = TorchBackend(seed=42, alive_check_every=alive_check_every)
        with _ReadCounter() as counter:
            scene.trace(
                num_rays=num_rays,
                seed=42,
                max_depth=max_depth,
                batch_size=batch_size or num_rays,
                backend=backend,
            )
        return counter
    finally:
        be.set_backend("numpy")


def _assert_steady_bounces(counter: _ReadCounter) -> None:
    """The control for :meth:`_ReadCounter.recurring_sites`.

    That method counts a read only in a bounce whose bundle is the one the
    previous bounce already ran on, so a trace in which the bundle is
    replaced at every bounce would report nothing at all and every assertion
    built on it would pass vacuously. Assert that a useful share of the
    bounces are steady ones.
    """
    steady = counter.steady_bounces()
    assert steady >= 3, (
        f"only {steady} of {counter.bounces} bounces ran on an unchanged "
        "bundle -- too few to detect a per-bounce read"
    )


class TestNoHostReadPerBounce:
    """R-12-4: the bounce loop reads nothing back to the host."""

    def test_fixed_trip_count_reads_nothing(self):
        """With the alive check off, the loop is free of host reads.

        Everything the loop used to read -- the alive count, the nearest-hit
        comparison, the branch probabilities, the roulette decision, the
        detector bin indices, the flux and ray-count totals -- is now either
        a device mask or a device accumulator read back once after the
        trace.
        """
        counter = _trace_and_count(
            100_000, max_depth=16, alive_check_every=0, batch_size=25_000
        )
        assert counter.bounces >= 8
        _assert_steady_bounces(counter)

        # A read that fires once per trace (a glass catalogue entry) or once
        # per bundle width (a shape-keyed cache) is not a per-bounce cost --
        # see _ReadCounter.recurring_sites for how the three are separated.
        recurring = {
            site: n
            for site, n in counter.recurring_sites().items()
            if _UNOWNED not in site and n > 0
        }
        assert recurring == {}, f"per-bounce host reads remain: {recurring}"

    def test_the_default_reads_only_the_alive_count(self):
        """The shipped default keeps one read and drops every other.

        That one read is the live count, and compaction takes its bucket
        from the same read rather than making a second one.
        """
        counter = _trace_and_count(
            100_000, max_depth=16, alive_check_every=None, batch_size=25_000
        )
        _assert_steady_bounces(counter)
        recurring = {
            site: n
            for site, n in counter.recurring_sites().items()
            if _UNOWNED not in site and "num_rays_alive" not in site and n > 0
        }
        assert recurring == {}, f"per-bounce host reads remain: {recurring}"

    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_alive_check_costs_one_read_every_k_bounces(self, k):
        """An every-k check costs one read per k bounces and nothing else."""
        counter = _trace_and_count(100_000, max_depth=16, alive_check_every=k)
        alive_reads = sum(
            n
            for site, n in counter.per_bounce_sites().items()
            if "num_rays_alive" in site
        )
        # One read per k bounces, and never more.
        assert alive_reads <= -(-counter.bounces // k)
        # And it is the only thing in the loop that reads at all.
        recurring = {
            site: n
            for site, n in counter.recurring_sites().items()
            if _UNOWNED not in site and "num_rays_alive" not in site and n > 0
        }
        assert recurring == {}, f"per-bounce host reads remain: {recurring}"

    def test_the_remaining_site_is_the_scene_transform(self):
        """The reads that are left are the per-bounce transform re-upload.

        Recorded rather than fixed: ``components/base.py`` is the geometry
        work's file. The bound keeps it from growing, and names what to
        change if it does.
        """
        counter = _trace_and_count(20_000, max_depth=8, alive_check_every=0)
        unowned = sum(
            n for site, n in counter.per_bounce_sites().items() if _UNOWNED in site
        )
        per_bounce = unowned / max(counter.bounces, 1)
        # Measured at 66 per bounce on this three-surface scene; the bound
        # allows for a little variation, not for a new site.
        assert per_bounce < 80
