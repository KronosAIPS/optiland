"""The per-trace accumulators keep their dtype and their storage across bounces.

A running total that changes dtype mid-trace, or that is rebound to a new
tensor at every addition, gives the right number in an eager trace -- which
is why neither was noticed -- but it is exactly what a replayed bounce cannot
tolerate: a CUDA graph writes to the memory it recorded, so an accumulator
rebound inside the recorded bounce stops accumulating after the first replay
(issue 2 of the research repository, where the capture's guard found both).

Kramer Harrison, 2026
"""

from __future__ import annotations

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
from optiland.nonsequential import _tally
from optiland.nonsequential.backends.array_backend import ArrayBackend
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend


def _singlet() -> NSQScene:
    """A lens, so every bounce pushes and pops the medium stack."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55),
            total_flux=1.0,
            aperture_radius=5.0,
        ),
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
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=16, num_pixels_y=16),
    )
    return scene


_BACKENDS = {
    "numpy-float64": ("numpy", "float64", lambda: NumpyBackend(seed=5)),
    "torch-float64-fixed": (
        "torch", "float64", lambda: TorchBackend(seed=5, alive_check_every=0)
    ),
    "torch-float32-fixed": (
        "torch", "float32", lambda: TorchBackend(seed=5, alive_check_every=0)
    ),
    "torch-float64-default": ("torch", "float64", lambda: TorchBackend(seed=5)),
}


@pytest.fixture
def configured(request):
    backend, precision, make = _BACKENDS[request.param]
    be.set_backend(backend)
    be.set_precision(precision)
    try:
        yield make
    finally:
        be.set_backend("numpy")
        be.set_precision("float64")


class TestTheUnderflowCounterKeepsItsDtype:
    """Issue 60: the medium stack's underflow counter stays an integer after the reset."""

    @pytest.mark.parametrize("configured", sorted(_BACKENDS), indirect=True)
    def test_every_tally_and_the_per_ray_counter_keep_one_dtype(
        self, configured, monkeypatch
    ):
        tally_dtypes: dict[int, tuple[object, set[str]]] = {}
        field_dtypes: list[str] = []

        original_add = _tally.Tally.add

        def recording_add(self, term):
            original_add(self, term)
            if self._dev is not None:
                # The tally itself is kept alive here, so its id is never
                # reused by another tally during the test.
                entry = tally_dtypes.setdefault(id(self), (self, set()))
                entry[1].add(str(self._dev.dtype))

        original_intersect = ArrayBackend.intersect_scene

        def recording_intersect(backend, rays, *a, **k):
            field_dtypes.append(str(rays.medium_stack_underflows.dtype))
            return original_intersect(backend, rays, *a, **k)

        monkeypatch.setattr(_tally.Tally, "add", recording_add)
        monkeypatch.setattr(ArrayBackend, "intersect_scene", recording_intersect)
        result = _singlet().trace(
            num_rays=8_192, seed=5, max_depth=10, batch_size=4_096, backend=configured()
        )

        assert len(field_dtypes) >= 10
        # Every device tally: one dtype from its first term to the last (the
        # underflow tally used to go from int64 to the working float).
        drifted = {i: d for i, (_, d) in tally_dtypes.items() if len(d) > 1}
        assert drifted == {}, drifted
        # The per-ray counter: an integer at birth, and the same integer after
        # every reset (it used to turn into the working float at bounce 1).
        assert len(set(field_dtypes)) == 1, field_dtypes
        assert "int" in field_dtypes[0]
        # And the count it feeds is still an int.
        assert isinstance(result.diagnostics.medium_stack_underflows, int)

    def test_the_device_tallies_were_observed(self, monkeypatch):
        """The control for the test above: on torch, the tallies are device tallies."""
        be.set_backend("torch")
        be.set_precision("float64")
        seen = []
        original_add = _tally.Tally.add

        def recording_add(self, term):
            original_add(self, term)
            seen.append(self._dev is not None)

        monkeypatch.setattr(_tally.Tally, "add", recording_add)
        try:
            _singlet().trace(
                num_rays=4_096, seed=5, max_depth=4,
                backend=TorchBackend(seed=5, alive_check_every=0),
            )
        finally:
            be.set_backend("numpy")
        assert seen and all(seen)
