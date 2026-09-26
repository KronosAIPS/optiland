"""The polarization state on the ray bundle and its run switch (the research repository's issue 5, item 3).

The switch is the backend's ``polarization`` argument (``"off"``, the default,
or ``"stokes"``), or :data:`~optiland.nonsequential.polarization.POLARIZATION_ENV`
for a harness that builds its own backends. What is pinned:

* **Off is the engine as it was.** No state field exists, the environment block
  has no ``polarization`` key, and nothing else changes (the fork's suite and the
  catalogue's fast test, run on both backends and both precisions, are the proof
  of "bit-identical"; this file holds the unit checks).
* **On carries six fields** -- ``pol_q``, ``pol_u``, ``pol_v`` and the reference
  axis ``pol_ex``, ``pol_ey``, ``pol_ez`` -- through every row operation the loop
  uses (``compact``, ``take``, ``select``, ``concat``), torch compaction, bounded
  splitting and the replay's static buffers, and the result records the mode.
* **Stokes I is the scalar flux.** With an unpolarized source, every number a
  trace returns is the same, bit for bit, with the switch on and off, on the two
  catalogue geometries where the scalar answer is exact (chapter 06 section 6.9,
  conditions 1 and 2): a single tilted interface (r1_07's rig) and the window at
  normal incidence with every ghost order (r1_05's rig), with roulette and with
  exhaustive splitting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    PlaneGeometry,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential import polarization as P
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.nonsequential.materials import VACUUM, NSQMaterial
from optiland.nonsequential.ray_bundle import NSQRayBundle

N_BK7 = 1.5168


@pytest.fixture(autouse=True)
def _restore_backend(monkeypatch):
    monkeypatch.delenv(P.POLARIZATION_ENV, raising=False)
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _glass():
    return NSQMaterial(optiland_material=IdealMaterial(n=N_BK7, k=0.0))


def _window_scene(split: bool = False) -> NSQScene:
    """r1_05's rig: a 1e9 mm-radius N-BK7 window, 5 mm thick, collimated beam."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-50.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=2.5
        ),
    )
    scene.add_lens(
        "W",
        CoordinateSystem(z=0.0),
        LensConfig(r1=1.0e9, r2=1.0e9, thickness=5.0, material=_glass(), front_aperture_radius=10.0),
    )
    scene.add_detector(
        "T",
        CoordinateSystem(z=55.0),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=4, num_pixels_y=4, splat="hard"),
    )
    if split:
        scene.sampling_policy = SamplingPolicy(split_depth=8, split_budget=64.0)
    return scene


def _interface_scene(theta_deg: float = 45.0) -> NSQScene:
    """r1_07's rig: one tilted vacuum/N-BK7 plane, a detector 1 um behind it along its normal."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=0.02
        ),
    )
    rx = math.radians(theta_deg)
    scene.add_component(
        "IF",
        RefractiveComponent(
            CoordinateSystem(z=10.0, rx=rx), PlaneGeometry(), material_front=VACUUM, material_back=_glass()
        ),
    )
    gap = 0.001
    scene.add_detector(
        "T",
        CoordinateSystem(y=-gap * math.sin(rx), z=10.0 + gap * math.cos(rx), rx=rx),
        IrradianceDetectorConfig(width=500, height=500, num_pixels_x=1, num_pixels_y=1, splat="hard"),
    )
    return scene


def _host(value):
    if be.is_torch_tensor(value):
        value = value.detach().cpu().numpy()
    return value


def _bits(value):
    value = _host(value)
    if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0:
        arr = np.ascontiguousarray(np.asarray(value))
        return (str(arr.dtype), arr.tobytes().hex())
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    return value


def _ledger(result) -> dict:
    """Every number a trace returns, as exact bit patterns (the environment and the time aside)."""
    skip = {"trace_time_sec", "diagnostics", "detectors", "ray_paths", "reflection_histograms", "environment"}
    out = {k: _bits(v) for k, v in vars(result).items() if k not in skip}
    for name, det in result.detectors.items():
        for key, value in vars(det).items():
            out[f"{name}.{key}"] = _bits(value)
    return out


def _backend(kind: str, polarization, **kw):
    if kind == "numpy":
        return NumpyBackend(seed=7, polarization=polarization)
    from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: PLC0415

    return TorchBackend(seed=7, polarization=polarization, **kw)


def _configure(kind: str, precision: str) -> None:
    be.set_backend(kind)
    if kind == "torch":
        be.set_device("cpu")
        be.grad_mode.disable()
    be.set_precision(precision)


LEGS = [("numpy", "float64"), ("torch", "float64"), ("torch", "float32")]
LEG_IDS = [f"{b}-{p}" for b, p in LEGS]


# ---------------------------------------------------------------------------
# The bundle's row operations
# ---------------------------------------------------------------------------


def _bundle(n: int = 6, polarized: bool = True) -> NSQRayBundle:
    z = np.zeros(n)
    b = NSQRayBundle(
        x=np.arange(n, dtype=float), y=z.copy(), z=z.copy(), L=z.copy(), M=z.copy(), N=np.ones(n),
        flux=np.ones(n), wavelength=np.full(n, 0.55), n_current=np.ones(n),
        bounce=np.zeros(n, dtype=np.int32), alive=np.array([True, False] * (n // 2)),
        ray_id=np.arange(n, dtype=np.int64),
    )
    if polarized:
        b.pol_q = np.linspace(-0.5, 0.5, n)
        b.pol_u = np.linspace(0.1, 0.2, n)
        b.pol_v = np.linspace(-0.3, 0.0, n)
        b.pol_ex, b.pol_ey, b.pol_ez = np.ones(n), np.zeros(n), np.zeros(n)
    return b


class TestBundle:
    def test_off_has_no_fields(self):
        b = _bundle(polarized=False)
        assert not b.polarized
        for op in (b.compact(), b.take(np.array([0, 2])), b.select(np.array([1, 3]))):
            assert all(getattr(op, f) is None for f in P.POL_FIELDS)
        assert NSQRayBundle.concat([b, b]).pol_q is None

    def test_every_row_operation_carries_the_state(self):
        b = _bundle()
        cases = {
            "compact": (b.compact(), b.alive),
            "take": (b.take(np.array([5, 0, 3])), np.array([5, 0, 3])),
            "select": (b.select(np.array([4, 1])), np.array([4, 1])),
        }
        for name, (out, rows) in cases.items():
            for f in P.POL_FIELDS:
                assert np.array_equal(getattr(out, f), getattr(b, f)[rows]), (name, f)
        cat = NSQRayBundle.concat([b, b.select(np.array([2]))])
        for f in P.POL_FIELDS:
            assert np.array_equal(getattr(cat, f), np.concatenate([getattr(b, f), getattr(b, f)[[2]]]))

    def test_select_copies(self):
        """A snapshot for splitting owns its state: writing the parent does not reach it."""
        b = _bundle()
        child = b.select(np.array([0, 1]))
        b.pol_q[0] = 99.0
        assert child.pol_q[0] != 99.0

    def test_concat_refuses_a_mixed_list(self):
        with pytest.raises(ValueError, match="polarization"):
            NSQRayBundle.concat([_bundle(), _bundle(polarized=False)])


# ---------------------------------------------------------------------------
# The switch and the record
# ---------------------------------------------------------------------------


class TestSwitch:
    def test_default_is_off_and_the_record_says_nothing(self):
        backend = NumpyBackend(seed=1)
        assert backend.polarization == "off"
        result = _window_scene().trace(num_rays=200, seed=1, backend=backend, max_depth=8)
        assert "polarization" not in result.environment

    @pytest.mark.parametrize("kind", ["numpy", "torch"])
    def test_on_is_recorded(self, kind):
        _configure(kind, "float64")
        result = _window_scene().trace(
            num_rays=200, seed=1, backend=_backend(kind, "stokes"), max_depth=8
        )
        assert result.environment["polarization"] == "stokes"

    def test_the_environment_variable(self, monkeypatch):
        monkeypatch.setenv(P.POLARIZATION_ENV, "stokes")
        assert NumpyBackend(seed=1).polarization == "stokes"
        assert NumpyBackend(seed=1, polarization="off").polarization == "off"
        monkeypatch.setenv(P.POLARIZATION_ENV, "jones")
        with pytest.raises(ValueError):
            NumpyBackend(seed=1)

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_the_state_is_born_and_carried(self, leg, monkeypatch):
        """Every bounce of a Stokes trace sees six fields as wide as the bundle, on its library."""
        kind, precision = leg
        _configure(kind, precision)
        backend = _backend(kind, "stokes")
        seen = []
        original = type(backend)._maybe_compact

        def spy(self, rays, depth):
            seen.append(
                (
                    rays.num_rays,
                    [getattr(rays, f).shape[0] for f in P.POL_FIELDS],
                    {type(getattr(rays, f)).__name__ for f in P.POL_FIELDS},
                    {str(getattr(rays, f).dtype) for f in P.POL_FIELDS},
                )
            )
            return original(self, rays, depth)

        monkeypatch.setattr(type(backend), "_maybe_compact", spy)
        _window_scene().trace(num_rays=3000, seed=3, backend=backend, max_depth=12, batch_size=2048)
        assert seen
        for n, widths, kinds, dtypes in seen:
            assert widths == [n] * 6
            assert kinds == {"Tensor" if kind == "torch" else "ndarray"}
            assert len(dtypes) == 1 and precision in next(iter(dtypes))

    def test_birth_state_is_unpolarized_with_e_on_x(self, monkeypatch):
        captured = {}
        original = P.prepare_bundle

        def spy(rays):
            original(rays)
            captured.update({f: np.array(getattr(rays, f)) for f in P.POL_FIELDS})
            captured["k"] = np.stack([rays.L, rays.M, rays.N], axis=1)

        monkeypatch.setattr(P, "prepare_bundle", spy)
        _window_scene().trace(num_rays=100, seed=1, backend=NumpyBackend(seed=1, polarization="stokes"), max_depth=4)
        assert np.all(captured["pol_q"] == 0) and np.all(captured["pol_u"] == 0) and np.all(captured["pol_v"] == 0)
        e = np.stack([captured["pol_ex"], captured["pol_ey"], captured["pol_ez"]], axis=1)
        assert np.max(np.abs(np.sum(e * captured["k"], axis=1))) < 1e-15
        assert np.max(np.abs(np.sum(e * e, axis=1) - 1.0)) < 1e-15


# ---------------------------------------------------------------------------
# Stokes I equals the scalar flux, bit for bit, where the scalar answer is exact
# ---------------------------------------------------------------------------


class TestScalarEquivalence:
    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    @pytest.mark.parametrize("theta", [0.0, 30.0, 45.0, 60.0, 89.0])
    def test_single_interface(self, leg, theta):
        """r1_07's rig, condition 1 of section 6.9: one interaction from an unpolarized source."""
        _configure(*leg)
        off = _interface_scene(theta).trace(num_rays=4000, seed=5, max_depth=2, backend=_backend(leg[0], "off"))
        on = _interface_scene(theta).trace(num_rays=4000, seed=5, max_depth=2, backend=_backend(leg[0], "stokes"))
        a, b = _ledger(off), _ledger(on)
        assert a == b, sorted(k for k in a if a[k] != b.get(k))

    @pytest.mark.parametrize("leg", LEGS, ids=LEG_IDS)
    def test_window_every_ghost_order(self, leg):
        """r1_05's rig, condition 2 of section 6.9: normal incidence throughout, 40 interactions deep."""
        _configure(*leg)
        off = _window_scene().trace(num_rays=6000, seed=5, max_depth=40, batch_size=2048, backend=_backend(leg[0], "off"))
        on = _window_scene().trace(num_rays=6000, seed=5, max_depth=40, batch_size=2048, backend=_backend(leg[0], "stokes"))
        a, b = _ledger(off), _ledger(on)
        assert a == b, sorted(k for k in a if a[k] != b.get(k))

    def test_window_exhaustive_split(self):
        """The same with bounded splitting (NumPy): select and concat carry the state."""
        off = _window_scene(split=True).trace(num_rays=500, seed=5, max_depth=12, backend=NumpyBackend(5, "off"))
        on = _window_scene(split=True).trace(num_rays=500, seed=5, max_depth=12, backend=NumpyBackend(5, "stokes"))
        a, b = _ledger(off), _ledger(on)
        assert a == b, sorted(k for k in a if a[k] != b.get(k))


class TestReplay:
    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_emulated_replay_equals_eager_with_the_state_on(self, precision):
        """The six fields are static buffers like the others: the emulated replay equals eager."""
        pytest.importorskip("torch")
        from optiland.nonsequential.backends import graph_replay as gr  # noqa: PLC0415

        _configure("torch", precision)
        calls = []
        original = gr.static_buffers

        def spy(rays):
            out = original(rays)
            calls.append(sorted(k for k in out if k.startswith("pol_")))
            return out

        gr.static_buffers = spy
        try:
            eager = _window_scene().trace(
                num_rays=4096, seed=9, max_depth=16, batch_size=2048,
                backend=_backend("torch", "stokes", compact_every=0),
            )
            replay = _window_scene().trace(
                num_rays=4096, seed=9, max_depth=16, batch_size=2048,
                backend=_backend("torch", "stokes", graph_replay="emulate"),
            )
        finally:
            gr.static_buffers = original
        assert calls == [sorted(P.POL_FIELDS)] * 2
        a, b = _ledger(eager), _ledger(replay)
        assert a == b, sorted(k for k in a if a[k] != b.get(k))
