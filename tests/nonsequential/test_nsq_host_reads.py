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
        self.marks: list[tuple[int, collections.Counter[str]]] = []
        self._orig: dict[str, object] = {}
        self._orig_intersect = None

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

        def bracketed(backend, *a, **k):
            self.mark()
            return self._orig_intersect(backend, *a, **k)

        ArrayBackend.intersect_scene = bracketed
        return self

    def __exit__(self, *exc) -> None:
        for name, orig in self._orig.items():
            setattr(torch.Tensor, name, orig)
        ArrayBackend.intersect_scene = self._orig_intersect

    def mark(self) -> None:
        self.marks.append((self.total, self.sites.copy()))

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

    def intervals_touched(self) -> collections.Counter[str]:
        """How many separate bounces each site read in.

        A read that fires once per trace shows up in one interval whichever
        bounce it lands in; a read that is genuinely per-bounce shows up in
        every one. Counting intervals rather than reads separates the two
        without having to know which bounce a one-off belongs to.
        """
        touched: collections.Counter[str] = collections.Counter()
        for (_, before), (_, after) in zip(
            self.marks, self.marks[1:], strict=False
        ):
            for site in (after - before):
                touched[site] += 1
        return touched


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


def _trace_and_count(num_rays: int, max_depth: int, alive_check_every: int):
    be.set_backend("torch")
    be.set_precision("float64")
    try:
        scene = _singlet()
        backend = TorchBackend(seed=42, alive_check_every=alive_check_every)
        with _ReadCounter() as counter:
            scene.trace(
                num_rays=num_rays,
                seed=42,
                max_depth=max_depth,
                batch_size=num_rays,
                backend=backend,
            )
        return counter
    finally:
        be.set_backend("numpy")


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
        counter = _trace_and_count(100_000, max_depth=16, alive_check_every=0)
        assert counter.bounces >= 8

        # A material's first evaluation caches a catalogue lookup: one
        # read, in whichever bounce it happens to fall in. A per-bounce read
        # shows up in every interval instead, so count intervals.
        recurring = {
            site: n
            for site, n in counter.intervals_touched().items()
            if _UNOWNED not in site and n > 1
        }
        assert recurring == {}, f"per-bounce host reads remain: {recurring}"

    def test_alive_check_costs_one_read_every_k_bounces(self):
        """The default is an every-k check, and k is what it costs."""
        k = 4
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
            for site, n in counter.intervals_touched().items()
            if _UNOWNED not in site and "num_rays_alive" not in site and n > 1
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
