"""The compiled bounce step of the torch backend (``TorchBackend(compile_step=True)``).

The option runs each bounce through ``torch.compile`` of
``array_backend.bounce_body``, the bounce lifted out of the trace loop. What is
asserted here:

* **Off, it is not there.** The default backend's step *is* ``bounce_body``,
  so the loop runs the statements it always ran; the catalogue's numbers with
  the option off are the eager loop's to the bit (checked on the research
  repository's fast test, both backends, float64 and float32).
* **The switch.** Explicit argument first, then the environment variable
  ``OPTILAND_NSQ_COMPILE_STEP``, off when neither says otherwise.
* **Forward only.** Gradient mode is refused with ``CompiledStepError``, for
  the global switch and for a bundle that arrives carrying a gradient.
* **The traced step books what the eager step books.** With torch's
  ``"eager"`` compile backend (the step traced by dynamo, every operation run
  as it is, no kernel generated) the trace is bit-identical to the eager loop:
  every detector pixel, every ledger entry and every ray count. That checks the
  plumbing -- the context object, the tallies, the in-place detector
  accumulation and the side effects dynamo replays -- on any host. The
  generated kernels themselves (the inductor, on the Apple GPU) fuse and
  reorder arithmetic and are compared statistically, on the catalogue.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.array_backend import bounce_body
from optiland.nonsequential.backends.torch_backend import (
    COMPILE_STEP_ENV,
    CompiledStepError,
    TorchBackend,
)

_TRACED = {"backend": "eager"}


@pytest.fixture
def torch_state():
    """Torch backend on the CPU for the test; numpy, float64, no grad after."""
    be.set_backend("torch")
    be.set_device("cpu")
    be.grad_mode.disable()
    yield
    be.set_backend("torch")
    be.grad_mode.disable()
    be.set_precision("float64")
    be.set_backend("numpy")
    be.set_precision("float64")


def _singlet() -> NSQScene:
    """The package quick-start scene: 1 W collimated beam, singlet, detector."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0
        ),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50),
        LensConfig(r1=100.0, r2=-100.0, thickness=5.0, material="N-BK7",
                   front_aperture_radius=12.5),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=150),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=16, num_pixels_y=16,
                                 splat="bilinear"),
    )
    return scene


def _trace(backend, num_rays=600, max_depth=6):
    return _singlet().trace(num_rays=num_rays, seed=7, max_depth=max_depth,
                            batch_size=num_rays, backend=backend)


_LEDGER = (
    "num_rays_absorbed",
    "num_rays_escaped",
    "num_rays_flux_killed",
    "num_rays_depth_killed",
    "total_flux_detected",
    "total_flux_absorbed",
    "total_flux_coating",
    "total_flux_bulk_absorbed",
    "total_flux_escaped",
    "total_flux_lost",
    "total_flux_sampling_residual",
    "flux_conservation_error",
)


class TestTheOption:
    def test_off_by_default_and_the_step_is_the_eager_body(self, monkeypatch):
        monkeypatch.delenv(COMPILE_STEP_ENV, raising=False)
        backend = TorchBackend()
        assert backend.compile_step is False
        assert backend._bounce_step(None) is bounce_body

    @pytest.mark.parametrize("value, on", [("1", True), ("true", True), ("ON", True),
                                           ("mps", "mps"), ("CUDA", "cuda"),
                                           ("0", False), ("", False), ("no", False)])
    def test_the_environment_switch(self, monkeypatch, value, on):
        monkeypatch.setenv(COMPILE_STEP_ENV, value)
        assert TorchBackend().compile_step == on
        assert type(TorchBackend().compile_step) is type(on)

    def test_the_argument_overrides_the_environment(self, monkeypatch):
        monkeypatch.setenv(COMPILE_STEP_ENV, "1")
        assert TorchBackend(compile_step=False).compile_step is False
        monkeypatch.setenv(COMPILE_STEP_ENV, "0")
        assert TorchBackend(compile_step=True).compile_step is True
        with pytest.raises(ValueError, match="compile_step"):
            TorchBackend(compile_step="metal")

    def test_a_device_type_compiles_only_on_that_device(self, torch_state):
        be.set_precision("float64")
        backend = TorchBackend(compile_step="mps")
        assert backend._bounce_step(None) is bounce_body  # this trace is on the cpu
        assert TorchBackend(compile_step="cpu", compile_options=_TRACED)._bounce_step(None) is not bounce_body


class TestForwardOnly:
    def test_refused_in_gradient_mode(self, torch_state):
        be.set_precision("float64")
        be.grad_mode.enable()
        with pytest.raises(CompiledStepError, match="forward traces only"):
            _trace(TorchBackend(seed=7, compile_step=True, compile_options=_TRACED))

    def test_refused_for_a_bundle_carrying_a_gradient(self, torch_state):
        be.set_precision("float64")
        backend = TorchBackend(seed=7, compile_step=True, compile_options=_TRACED)
        step = backend._bounce_step(None)
        fields = ("x", "y", "z", "L", "M", "N", "flux", "wavelength", "n_current", "k_current")
        rays = types.SimpleNamespace(**{f: torch.zeros(4) for f in fields})
        rays.flux = torch.ones(4, requires_grad=True)
        with pytest.raises(CompiledStepError, match="carries a gradient"):
            step(backend, None, rays)


class TestTracedStepBooksWhatTheEagerStepBooks:
    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_bit_identical_to_the_eager_loop(self, torch_state, precision):
        be.set_precision(precision)
        eager = _trace(TorchBackend(seed=7))
        traced = _trace(TorchBackend(seed=7, compile_step=True, compile_options=_TRACED))
        for name in _LEDGER:
            assert getattr(traced, name) == getattr(eager, name), name
        a = np.asarray(be.to_numpy(eager.detectors["D1"].data))
        b = np.asarray(be.to_numpy(traced.detectors["D1"].data))
        assert np.array_equal(a, b)
        assert eager.total_flux_detected > 0.5  # the beam reached the detector
