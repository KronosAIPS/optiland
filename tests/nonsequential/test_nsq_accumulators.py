"""The per-trace accumulators keep their dtype and their storage across bounces.

A running total that changes dtype mid-trace, or that is rebound to a new
tensor at every addition, gives the right number in an eager trace -- which
is why neither was noticed -- but it is exactly what a replayed bounce cannot
tolerate: a CUDA graph writes to the memory it recorded, so an accumulator
rebound inside the recorded bounce stops accumulating after the first replay
(issue 2 of the research repository, where the capture's guard found both).
"""

from __future__ import annotations

import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.nonsequential import (
    AnnularPlaneGeometry,
    CollimatedSourceConfig,
    FarFieldDetectorConfig,
    HarveyShackBSDF,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQScene,
    SpectralDetectorConfig,
    Spectrum,
    SurfaceConfig,
)
from optiland.nonsequential import _tally
from optiland.nonsequential.backends.array_backend import ArrayBackend
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.components.absorbing import AbsorbingComponent


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
    """Issue 60: the medium stack's underflow counter stays an integer."""

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


def _stopped_singlet() -> NSQScene:
    """The singlet behind an annular absorber, and a detector with a ghost histogram."""
    scene = _singlet()
    scene.add_component(
        "stop",
        AbsorbingComponent(
            cs=CoordinateSystem(z=20.0),
            geometry=AnnularPlaneGeometry(inner_radius=4.0, outer_radius=20.0),
            name="stop",
        ),
    )
    scene.add_detector(
        "D2",
        CoordinateSystem(z=160),
        IrradianceDetectorConfig(width=30, height=30, reflection_bins=3),
    )
    return scene


def _scattering() -> NSQScene:
    """A lossy scattering mirror between a spectral tap and a far-field detector."""
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
    scene.add_detector(
        "TAP",
        CoordinateSystem(z=40),
        SpectralDetectorConfig(
            width=200, height=200, num_pixels_x=8, num_pixels_y=8,
            wl_min=0.4, wl_max=0.7, num_bins=4, splat="bilinear", absorb=False,
        ),
    )
    scene.add_mirror(
        "M1",
        CoordinateSystem(z=100),
        MirrorConfig(
            radius=0.0, reflectance=0.9, aperture_radius=50.0,
            surface=SurfaceConfig(
                bsdf=HarveyShackBSDF(b0=1e-3, l0=0.05, s=2.0), scatter_fraction=0.7
            ),
        ),
    )
    scene.add_detector(
        "FF", CoordinateSystem(z=-60), FarFieldDetectorConfig(num_theta=16, num_phi=32)
    )
    return scene


def _storage(scene: NSQScene, tallies: list) -> dict[str, tuple[int, int]]:
    """Object and storage identity of every tensor accumulator a trace writes.

    Found independently of the engine's own inventory: the tallies the trace
    added to (collected by the caller), and every tensor or tally held as an
    attribute of a scene surface or detector.
    """
    out = {}

    def put(label, value):
        if torch.is_tensor(value):
            out[label] = (id(value), value.data_ptr())

    for i, tally in enumerate(tallies):
        put(f"tally {i}", tally._dev)
    for owner in [*scene.surfaces, *scene.detectors]:
        for attr, value in vars(owner).items():
            label = f"{type(owner).__name__} {id(owner)}.{attr}"
            if isinstance(value, (_tally.Tally, _tally._TallyVector)):
                put(label, value._dev)
            else:
                put(label, value)
    return out


class TestAccumulatorsAddInPlace:
    """A tally or counter keeps its storage from its first term to the trace's end."""

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    @pytest.mark.parametrize("scene_fn", [_stopped_singlet, _scattering])
    def test_storage_survives_every_bounce(self, scene_fn, precision, monkeypatch):
        be.set_backend("torch")
        be.set_precision(precision)
        scene = scene_fn()
        tallies: list = []
        snapshots: list[dict] = []

        original_add = _tally.Tally.add

        def registering_add(self, term):
            if not any(t is self for t in tallies):
                tallies.append(self)
            original_add(self, term)

        original_intersect = ArrayBackend.intersect_scene

        def snapshotting_intersect(backend, rays, *a, **k):
            snapshots.append(_storage(scene, tallies))
            return original_intersect(backend, rays, *a, **k)

        monkeypatch.setattr(_tally.Tally, "add", registering_add)
        monkeypatch.setattr(ArrayBackend, "intersect_scene", snapshotting_intersect)
        try:
            scene.trace(
                num_rays=8_192, seed=5, max_depth=8, batch_size=4_096,
                backend=TorchBackend(seed=5, alive_check_every=0),
            )
        finally:
            be.set_backend("numpy")
            be.set_precision("float64")

        # From the start of bounce 1 (every accumulator has had its first term
        # in bounce 0) to the last bounce of the last batch: same objects,
        # same storage.
        assert len(snapshots) >= 16
        reference = snapshots[1]
        assert len(reference) >= 10, sorted(reference)
        moved = {
            label: n
            for n, snap in enumerate(snapshots[1:], start=1)
            for label, ident in reference.items()
            if snap.get(label) != ident
        }
        assert moved == {}, moved
