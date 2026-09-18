"""Bucketed compaction on the device path.

``docs/theory/12_gpu_mapping.md`` R-12-6: dead rays are gathered out of the
bundle so they stop costing a full-width bounce, and the compacted width
comes from a fixed ladder rather than from the live count, so a whole trace
sees O(log N) distinct shapes. T-12-4 asks that compaction not change the
answer and T-12-5 that the ladder be short; chapter 09's R-09-11 keeps the
permutation outside the autodiff graph, and this backend goes further and
leaves the width alone whenever a gradient is in flight.

Kramer Harrison, 2026
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
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.array_backend import (
    BUCKET_MIN_WIDTH,
    ArrayBackend,
    bucketed_width,
)
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.ray_bundle import backend_live_permutation


class _WidthLog:
    """Record the bundle width every bounce ran at."""

    def __init__(self) -> None:
        self.widths: list[int] = []
        self._orig = None

    def __enter__(self) -> _WidthLog:
        self._orig = ArrayBackend.intersect_scene
        log = self

        def bracketed(backend, rays, *a, **k):
            log.widths.append(rays.num_rays)
            return log._orig(backend, rays, *a, **k)

        ArrayBackend.intersect_scene = bracketed
        return self

    def __exit__(self, *exc) -> None:
        ArrayBackend.intersect_scene = self._orig


def _singlet(scatter: float = 0.0) -> NSQScene:
    """The quick-start scene: 1 W collimated beam, singlet, detector."""
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
        IrradianceDetectorConfig(
            width=20, height=20, num_pixels_x=32, num_pixels_y=32, splat="bilinear"
        ),
    )
    return scene


def _trace(num_rays: int, batch: int, compact_every, max_depth: int = 16):
    """Trace the singlet on torch CPU and return (result, widths)."""
    be.set_backend("torch")
    be.set_precision("float64")
    try:
        scene = _singlet()
        backend = TorchBackend(seed=42, compact_every=compact_every)
        with _WidthLog() as log:
            result = scene.trace(
                num_rays=num_rays,
                seed=42,
                max_depth=max_depth,
                batch_size=batch,
                backend=backend,
            )
        return result, log.widths
    finally:
        be.set_backend("numpy")


class TestTheLadder:
    """The widths compaction is allowed to choose from."""

    def test_nothing_alive_gives_width_zero(self):
        assert bucketed_width(0, 16_384) == 0

    def test_a_mostly_live_bundle_keeps_its_width(self):
        """Above half alive the next rung up is the current width itself.

        Which is the alive-fraction trigger R-12-6 asks for, with no second
        rule: the caller compares the answer against the current width and
        skips the gather when they are equal.
        """
        for n_now in (4096, 8192, 16_384):
            assert bucketed_width(n_now // 2 + 1, n_now) == n_now

    def test_the_rungs_are_powers_of_two_above_the_floor(self):
        for n_live in (1, 5, 2047, 2049, 4095, 9000):
            width = bucketed_width(n_live, 1 << 20)
            assert width >= n_live
            assert width >= BUCKET_MIN_WIDTH
            assert width & (width - 1) == 0, f"{width} is not a power of two"

    def test_the_floor_clears_the_shared_content_keyed_cache(self):
        """A rung of 1024 or below would put a per-bounce host read back.

        ``optiland.materials.base.BaseMaterial`` keys its refractive-index
        cache on the *contents* of the wavelength array once that array
        holds ``_MAX_VALUE_KEY_ARRAY_SIZE`` elements or fewer, which copies
        the whole array to the host on every evaluation. The ladder's floor
        sits above that on purpose.
        """
        from optiland.materials.base import BaseMaterial

        assert BUCKET_MIN_WIDTH > BaseMaterial._MAX_VALUE_KEY_ARRAY_SIZE


class TestThePermutation:
    """The prefix-sum partition the gather reads through."""

    @pytest.mark.parametrize("n", [1, 7, 64, 1000])
    def test_it_is_a_permutation_with_the_live_rows_first(self, n):
        rng = np.random.default_rng(7)
        alive_np = rng.random(n) < 0.4
        n_live = int(alive_np.sum())
        for alive in (alive_np, torch.from_numpy(alive_np)):
            perm = backend_live_permutation(alive, n_live)
            perm_np = np.asarray(perm)
            assert sorted(perm_np.tolist()) == list(range(n))
            assert alive_np[perm_np[:n_live]].all()
            assert not alive_np[perm_np[n_live:]].any()

    def test_it_preserves_order_within_each_group(self):
        alive = np.array([True, False, True, True, False, False, True])
        perm = backend_live_permutation(alive, 4)
        assert perm.tolist() == [0, 2, 3, 6, 1, 4, 5]

    def test_all_dead_and_all_alive(self):
        alive = np.ones(16, dtype=bool)
        assert backend_live_permutation(alive, 16).tolist() == list(range(16))
        dead = np.zeros(16, dtype=bool)
        assert backend_live_permutation(dead, 0).tolist() == list(range(16))


class TestCompactionDoesNotChangeTheAnswer:
    """T-12-4, at the precision the reordering allows."""

    def test_flux_and_image_agree_with_the_uncompacted_trace(self):
        """Not bit-identical, and it cannot be: compaction reorders the rows.

        The generator is keyed by ``ray_id`` (chapter 12 section 12.4), so
        every ray draws exactly what it drew before and no decision changes.
        What changes is the order the detector's scatter-add sums a pixel's
        contributions in, and the order a masked flux total is reduced in.
        Both are float64 sums over at most ``num_rays`` terms, so the
        tolerance is the operation count times the float64 step.
        """
        num_rays = 100_000
        off, widths_off = _trace(num_rays, 25_000, compact_every=0)
        on, widths_on = _trace(num_rays, 25_000, compact_every=1)

        assert min(widths_on) < min(widths_off), "compaction never narrowed"

        tol = num_rays * np.finfo(np.float64).eps

        assert on.num_rays_escaped == off.num_rays_escaped
        assert on.num_rays_depth_killed == off.num_rays_depth_killed
        assert on.num_rays_flux_killed == off.num_rays_flux_killed
        assert on.detectors["D1"].num_rays_hit == off.detectors["D1"].num_rays_hit

        for name in (
            "total_flux_detected",
            "total_flux_escaped",
            "total_flux_bulk_absorbed",
            "total_flux_lost",
        ):
            a = float(getattr(on, name))
            b = float(getattr(off, name))
            assert abs(a - b) <= tol * max(abs(b), 1e-12), name

        image_on = np.asarray(on.detectors["D1"].irradiance)
        image_off = np.asarray(off.detectors["D1"].irradiance)
        assert image_on.shape == image_off.shape
        # Same pixels lit, to the same value within the reordering.
        assert ((image_on > 0) == (image_off > 0)).all()
        np.testing.assert_allclose(image_on, image_off, rtol=tol, atol=0.0)

    def test_the_period_does_not_change_the_answer(self):
        """Compacting every bounce and every fourth agree the same way."""
        num_rays = 40_000
        every_1, _ = _trace(num_rays, 20_000, compact_every=1)
        every_4, _ = _trace(num_rays, 20_000, compact_every=4)
        tol = num_rays * np.finfo(np.float64).eps
        assert abs(
            float(every_1.total_flux_detected) - float(every_4.total_flux_detected)
        ) <= tol * float(every_4.total_flux_detected)


class TestBucketedShapes:
    """T-12-5: the trace uses O(log N) distinct widths."""

    def test_the_widths_are_a_short_ladder(self):
        num_rays = 100_000
        _result, widths = _trace(num_rays, 25_000, compact_every=1)
        distinct = sorted(set(widths))
        # Every rung is the batch width or a power of two at or above the
        # floor, and there are at most log2(N) of them.
        for width in distinct:
            assert width == 25_000 or (
                width >= BUCKET_MIN_WIDTH and width & (width - 1) == 0
            )
        assert len(distinct) <= int(np.log2(num_rays)) + 1
        assert len(distinct) > 1, "the bundle never narrowed"

    def test_without_compaction_the_width_never_changes(self):
        """The control: the same trace with compaction off is one shape."""
        _result, widths = _trace(100_000, 25_000, compact_every=0)
        assert set(widths) == {25_000}


class TestGradientModeKeepsFixedShapes:
    """The width is left alone whenever a gradient is in flight."""

    def _scene_with_a_trainable_source(self):
        scene = NSQScene()
        flux = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        scene.add_source(
            "S1",
            CoordinateSystem(),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=flux,
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
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=16, num_pixels_y=16
            ),
        )
        return scene, flux

    def test_a_grad_carrying_bundle_is_never_narrowed(self):
        be.set_backend("torch")
        be.set_precision("float64")
        try:
            scene, flux = self._scene_with_a_trainable_source()
            backend = TorchBackend(seed=42, compact_every=1)
            with _WidthLog() as log:
                result = scene.trace(
                    num_rays=20_000,
                    seed=42,
                    max_depth=8,
                    batch_size=20_000,
                    backend=backend,
                )
            assert set(log.widths) == {20_000}
            # And the trace is still differentiable end to end.
            total = result.detectors["D1"].total_flux
            total.backward()
            assert flux.grad is not None
            assert float(flux.grad) > 0.0
        finally:
            be.set_backend("numpy")

    def test_the_forward_only_control_does_narrow(self):
        """The same scene without the gradient compacts, so the test above
        is measuring gradient mode and not an inert configuration."""
        _result, widths = _trace(20_000, 20_000, compact_every=1, max_depth=8)
        assert min(widths) < 20_000
