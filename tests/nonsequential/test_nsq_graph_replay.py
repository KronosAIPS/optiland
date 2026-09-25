"""The torch backend's ``graph_replay`` option: its refusals, and its bookkeeping.

``TorchBackend(graph_replay=True)`` records one fixed-width bounce as a CUDA
graph after two eager bounces and replays it to the depth cap (issue 2 of the
research repository; measured 4.3 times on r1_16 at 1e6 rays on an A100 in
the hybrid-engine study of 2026-09-24, and 8.3 times with the fused
generator). Torch has no graph capture on the CPU or on Apple's ``mps``, so
nothing here records a graph. What runs here:

- every refusal: a device without CUDA graphs, gradient mode, splitting, path
  recording, a ray database, conflicting options, and a bounce that rebinds
  an accumulator (the guard the capture relies on);
- ``graph_replay="emulate"``: the same static-buffer bookkeeping, eager
  bounces, guard and trip count, with the bounce run instead of recorded. Its
  every number must equal the eager fixed-width trace's, bit for bit; that is
  the data-flow half of the proof. The CUDA half (the graph itself, on a GPU)
  is a separate run and is not in this suite; ``TestOnCuda`` runs it where a
  CUDA device is present.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.nonsequential import (
    CollimatedSourceConfig,
    FarFieldDetectorConfig,
    HarveyShackBSDF,
    IrradianceDetectorConfig,
    LambertianBSDF,
    LensConfig,
    MirrorConfig,
    NSQScene,
    RayDatabaseConfig,
    ReflectiveComponent,
    SpectralDetectorConfig,
    Spectrum,
    SphericalCavityGeometry,
    SphericalPort,
    SurfaceConfig,
)
from optiland.nonsequential.backends import graph_replay as gr
from optiland.nonsequential.backends.torch_backend import (
    GraphReplayUnavailable,
    TorchBackend,
)
from optiland.nonsequential.ir.scene_ir import SamplingPolicy


@pytest.fixture(autouse=True)
def _torch_float64():
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _source(scene, z=0.0, radius=5.0, total_flux=1.0):
    scene.add_source(
        "S1",
        CoordinateSystem(z=z),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55),
            total_flux=total_flux,
            aperture_radius=radius,
        ),
    )


def _singlet(total_flux=1.0) -> NSQScene:
    scene = NSQScene()
    _source(scene, total_flux=total_flux)
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
            width=20, height=20, num_pixels_x=32, num_pixels_y=32, splat="bilinear",
            reflection_bins=3,
        ),
    )
    return scene


def _scattering() -> NSQScene:
    scene = NSQScene()
    _source(scene)
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


def _sphere() -> NSQScene:
    scene = NSQScene()
    scene.add_component(
        "wall",
        ReflectiveComponent(
            CoordinateSystem(),
            SphericalCavityGeometry(
                50.0, [SphericalPort.from_area_fraction((0.0, 0.0, -1.0), 0.02)]
            ),
            reflectance=0.95,
            bsdf=LambertianBSDF(reflectance_value=1.0),
            name="wall",
        ),
    )
    _source(scene, z=-40.0, radius=2.5)
    scene.add_detector(
        "patch",
        CoordinateSystem(y=49.95, rx=np.deg2rad(-90.0)),
        IrradianceDetectorConfig(
            width=5.0, height=5.0, num_pixels_x=1, num_pixels_y=1,
            splat="hard", absorb=False, side="front",
        ),
    )
    return scene


_SCENES = {
    "singlet": (_singlet, 16),
    "scattering": (_scattering, 10),
    "sphere": (_sphere, 40),
}

_SKIP = (
    "trace_time_sec",
    "ray_paths",
    "detectors",
    "diagnostics",
    "reflection_histograms",
    # The test compares numbers; the environment records how the trace ran.
    "environment",
)


def _host(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return value


def _bits(value):
    value = _host(value)
    if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0:
        arr = np.ascontiguousarray(np.asarray(value))
        return (str(arr.dtype), arr.tobytes().hex())
    if isinstance(value, (float, np.floating)) or (
        hasattr(value, "dtype") and np.issubdtype(value.dtype, np.floating)
    ):
        return float(value).hex()
    return value


def _ledger(result) -> dict:
    """Every number a trace returns, as exact bit patterns."""
    out = {key: _bits(v) for key, v in vars(result).items() if key not in _SKIP}
    out["underflows"] = result.diagnostics.medium_stack_underflows
    for name, det in result.detectors.items():
        for key, value in vars(det).items():
            out[f"{name}.{key}"] = _bits(value)
    for name, hist in (result.reflection_histograms or {}).items():
        for key, value in vars(hist).items():
            out[f"{name}.hist.{key}"] = _bits(value)
    return out


def _trace(scene_fn, depth, backend, num_rays=6_000, batch_size=2_048):
    return scene_fn().trace(
        num_rays=num_rays,
        seed=11,
        max_depth=depth,
        batch_size=batch_size,
        backend=backend,
    )


class TestEmulatedReplayIsTheEagerFixedWidthTrace:
    """The replay's bookkeeping reproduces the eager fixed-width loop, bit for bit."""

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    @pytest.mark.parametrize("alive_check_every", [0, 1, 3])
    @pytest.mark.parametrize("scene_name", sorted(_SCENES))
    def test_every_number_is_identical(self, scene_name, alive_check_every, precision):
        be.set_precision(precision)
        scene_fn, depth = _SCENES[scene_name]
        eager = TorchBackend(
            seed=11, alive_check_every=alive_check_every, compact_every=0
        )
        emulated = TorchBackend(
            seed=11, alive_check_every=alive_check_every, graph_replay="emulate"
        )
        a = _ledger(_trace(scene_fn, depth, eager))
        b = _ledger(_trace(scene_fn, depth, emulated))
        assert a == b, sorted(k for k in a if a[k] != b.get(k))

    def test_the_replay_path_was_taken(self, monkeypatch):
        """The control for the test above: the emulated replay really ran."""
        calls = []
        original = gr.replay_bounces

        def spy(*args, **kwargs):
            calls.append((args[3], args[4], kwargs.get("mode")))
            return original(*args, **kwargs)

        monkeypatch.setattr(gr, "replay_bounces", spy)
        backend = TorchBackend(seed=11, alive_check_every=0, graph_replay="emulate")
        _trace(_singlet, 16, backend)
        # 6,000 rays in batches of 2,048: two full batches handed over at
        # bounce 2 and run to the cap of 16, and a third of 1,904 rays, below
        # the 2,048-ray floor, that runs eagerly.
        assert calls == [(gr.EAGER_BOUNCES, 16, "emulate")] * 2

    def test_a_narrow_batch_runs_eagerly(self, monkeypatch):
        """A batch below the ladder's floor is not replayed; its values are the same."""
        calls = []
        original = gr.replay_bounces
        monkeypatch.setattr(
            gr, "replay_bounces", lambda *a, **k: calls.append(1) or original(*a, **k)
        )
        eager = TorchBackend(seed=11, compact_every=0)
        emulated = TorchBackend(seed=11, graph_replay="emulate")
        a = _ledger(_trace(_singlet, 16, eager, num_rays=5_000, batch_size=4_096))
        b = _ledger(_trace(_singlet, 16, emulated, num_rays=5_000, batch_size=4_096))
        # 4,096 + 904: the second batch is below the 2,048-ray floor.
        assert len(calls) == 1
        assert a == b


class TestTheGuard:
    """An accumulator rebound inside the recorded bounce refuses the replay."""

    def test_a_rebinding_tally_is_refused(self, monkeypatch):
        from optiland.nonsequential import _tally

        def rebinding_add(self, term):
            if be.is_torch_tensor(term):
                self._dev = term if self._dev is None else self._dev + term
            elif self.is_int:
                self._host += int(term)
            else:
                self._host += float(term)

        monkeypatch.setattr(_tally.Tally, "add", rebinding_add)
        with pytest.raises(GraphReplayUnavailable, match="rebinds accumulators"):
            _trace(_singlet, 16, TorchBackend(seed=11, graph_replay="emulate"))

    def test_the_inventory_covers_every_kind_of_accumulator(self):
        """The guard sees the detector buffers, hit counters, histograms and ledgers."""
        scene = _singlet()
        scene.trace(num_rays=2_048, seed=1, max_depth=4, backend=TorchBackend(seed=1))
        labels = set(gr.accumulator_identities(scene, []))
        for suffix in ("._data", "._num_rays_hit", "._refl_flux", "._coating_loss"):
            assert any(label.endswith(suffix) for label in labels), (suffix, labels)


class TestRefusals:
    """Where a replay cannot be right, it is refused by name, never worked around."""

    def test_no_cuda_graphs_on_this_device(self):
        if torch.cuda.is_available():
            pytest.skip("this machine has CUDA graphs")
        with pytest.raises(GraphReplayUnavailable, match="CUDA"):
            _trace(_singlet, 8, TorchBackend(seed=1, graph_replay=True))

    def test_a_parameter_that_carries_a_gradient(self):
        flux = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        scene = _singlet(total_flux=flux)
        with pytest.raises(GraphReplayUnavailable, match="gradient"):
            scene.trace(
                num_rays=4_096, seed=1, max_depth=6,
                backend=TorchBackend(seed=1, graph_replay="emulate"),
            )

    def test_gradient_mode(self):
        be.grad_mode.enable()
        try:
            with pytest.raises(GraphReplayUnavailable, match="gradient"):
                _singlet().trace(
                    num_rays=4_096, seed=1, max_depth=6,
                    backend=TorchBackend(seed=1, graph_replay="emulate"),
                )
        finally:
            be.grad_mode.disable()

    def test_splitting(self):
        scene = _singlet()
        scene.sampling_policy = SamplingPolicy(split_depth=2)
        backend = TorchBackend(seed=1, allow_splitting=True, graph_replay="emulate")
        with pytest.raises(GraphReplayUnavailable, match="splitting"):
            scene.trace(num_rays=4_096, seed=1, max_depth=6, backend=backend)

    def test_path_recording(self):
        with pytest.raises(GraphReplayUnavailable, match="path recording"):
            _singlet().trace(
                num_rays=4_096, seed=1, max_depth=6, record_paths=True,
                backend=TorchBackend(seed=1, graph_replay="emulate"),
            )

    def test_a_ray_database(self):
        scene = _singlet()
        scene.add_detector(
            "RDB", CoordinateSystem(z=120), RayDatabaseConfig(width=20.0, height=20.0)
        )
        with pytest.raises(GraphReplayUnavailable, match="ray database"):
            scene.trace(
                num_rays=4_096, seed=1, max_depth=6,
                backend=TorchBackend(seed=1, graph_replay="emulate"),
            )

    def test_compaction_conflicts_with_a_replay(self):
        with pytest.raises(ValueError, match="compact_every"):
            TorchBackend(graph_replay=True, compact_every=1)

    def test_an_unknown_mode(self):
        with pytest.raises(ValueError, match="graph_replay"):
            TorchBackend(graph_replay="sometimes")

    def test_off_by_default(self):
        backend = TorchBackend()
        assert backend.graph_replay is False
        assert backend.compact_every is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestOnCuda:
    """The recorded graph itself: every number equals the eager fixed-width trace's."""

    @pytest.mark.parametrize("scene_name", sorted(_SCENES))
    def test_the_replayed_trace_is_the_eager_fixed_width_trace(self, scene_name):
        be.set_device("cuda")
        try:
            scene_fn, depth = _SCENES[scene_name]
            eager = TorchBackend(seed=11, alive_check_every=0, compact_every=0)
            replayed = TorchBackend(seed=11, alive_check_every=0, graph_replay=True)
            a = _ledger(_trace(scene_fn, depth, eager))
            b = _ledger(_trace(scene_fn, depth, replayed))
        finally:
            be.set_device("cpu")
        assert a == b, sorted(k for k in a if a[k] != b.get(k))
