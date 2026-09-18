"""W4 — dtype-aware tolerances, the origin advance, and the mps device.

Covers optiland.nonsequential._tol (the ulp/accept_t_min/tiny_for/
radicand_floor primitives), the origin advance in BaseComponent.intersect
(the long-path collimated-source case), the float32 singlet flux and
gradient recovery, and the torch backend's mps device validation.

Scene parameters throughout match NS2_precision_and_torch_measurements.md
section 3 (E1) and section 6 (E3): a 1 W collimated source, 5 mm aperture
radius, biconvex N-BK7 lens r1=100/r2=-100/t=5, semi-diameter 12.5 at
z=50, and a 20x20 mm detector at z=150 ("the quickstart singlet").

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available — skip precision tests")

# Imports below intentionally follow importorskip: they must not run when
# torch is unavailable.
# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    RayDatabaseConfig,
    Spectrum,
)
from optiland.nonsequential import _tol as tol


def _reset_backend() -> None:
    be.set_backend("numpy")


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    _reset_backend()


def _build_singlet(source_z: float, detector_cls: str = "irradiance") -> NSQScene:
    """The quickstart singlet (see module docstring), source at ``source_z``."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(z=source_z),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0
        ),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50),
        LensConfig(
            r1=100, r2=-100, thickness=5, material="N-BK7", front_aperture_radius=12.5
        ),
    )
    if detector_cls == "irradiance":
        scene.add_detector(
            "D1",
            CoordinateSystem(z=150),
            IrradianceDetectorConfig(width=20, height=20, num_pixels_x=64, num_pixels_y=64),
        )
    else:
        scene.add_detector("D1", CoordinateSystem(z=150), RayDatabaseConfig(width=20, height=20))
    return scene


def _direct_cluster_rms(db, n_clip: int = 4, k_sigma: float = 3.0) -> tuple[float, int, int]:
    """Flux-weighted rms radius [mm] after iterative sigma-clipping.

    Isolates the tight, well-focused core from ghosts/aberrated-halo rays.
    Not NS2's own "direct cluster" algorithm (unpublished) -- a documented
    proxy with the same intent, verified in docs/build/W4_tolerances.md to
    reproduce NS2's reported figures (12.4358 / ~568 / ~640 um) closely.
    """
    x = np.asarray(db.x, dtype=np.float64)
    y = np.asarray(db.y, dtype=np.float64)
    w = np.asarray(db.flux, dtype=np.float64)
    mask = np.ones(x.shape, dtype=bool)
    for _ in range(n_clip):
        xw, yw, ww = x[mask], y[mask], w[mask]
        total = ww.sum()
        cx = (xw * ww).sum() / total
        cy = (yw * ww).sum() / total
        r = np.sqrt((xw - cx) ** 2 + (yw - cy) ** 2)
        rms = np.sqrt((ww * r**2).sum() / total)
        new_mask = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < k_sigma * rms
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
    xw, yw, ww = x[mask], y[mask], w[mask]
    total = ww.sum()
    cx = (xw * ww).sum() / total
    cy = (yw * ww).sum() / total
    r2 = (xw - cx) ** 2 + (yw - cy) ** 2
    return float(np.sqrt((ww * r2).sum() / total)), int(mask.sum()), int(x.size)


# ---------------------------------------------------------------------------
# _tol primitives
# ---------------------------------------------------------------------------


class TestUlp:
    """ulp() is the exact IEEE-754 spacing, not the cruder approximation a
    pure-Python reference script uses when it cannot ask NumPy/Torch
    directly (see _tol.py's module docstring and docs/build/W4_tolerances.md
    for the ~2x discrepancy this implies against that script's own figure).
    """

    def test_ulp_50mm_numpy_float64(self):
        assert tol.ulp(np.float64(50.0)) == pytest.approx(7.105427357601002e-15, rel=1e-9)

    def test_ulp_50mm_numpy_float32(self):
        assert tol.ulp(np.float32(50.0)) == pytest.approx(3.8146973e-06, rel=1e-5)

    def test_25_ulp_50mm_matches_dtype_table(self):
        """docs/theory tables state 25 ulp of 50 mm as ~9.5e-5 mm (f32) and
        ~1.78e-13 mm (f64), computed via the exact IEEE spacing -- not the
        bench script's "about 25 ulp" approximation (1.9e-4 mm), which is
        exactly 2x larger because it rounds the coordinate up to the next
        power of two using the dtype's *epsilon* (2u) rather than its unit
        roundoff (u). Both are safe thresholds -- see W4_tolerances.md.
        """
        f32 = 25 * tol.ulp(np.float32(50.0))
        f64 = 25 * tol.ulp(np.float64(50.0))
        assert f32 == pytest.approx(9.5367e-05, rel=1e-3)
        assert f64 == pytest.approx(1.7764e-13, rel=1e-3)

    def test_ulp_matches_torch_nextafter(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor(50.0, dtype=dtype)
            expected = torch.nextafter(x, torch.full_like(x, float("inf"))) - x
            assert float(tol.ulp(x)) == pytest.approx(float(expected), rel=1e-9)

    def test_ulp_torch_and_numpy_agree(self):
        assert float(tol.ulp(torch.tensor(50.0, dtype=torch.float64))) == pytest.approx(
            float(tol.ulp(np.float64(50.0))), rel=1e-9
        )


class TestAcceptTMin:
    def test_scales_with_k(self):
        base = tol.accept_t_min(50.0, k=1)
        assert tol.accept_t_min(50.0, k=25) == pytest.approx(25 * base, rel=1e-9)

    def test_floor_at_small_magnitude(self):
        """Below the 1 mm floor, the threshold is pinned at ulp(1 mm)."""
        assert tol.accept_t_min(0.0) == pytest.approx(tol.accept_t_min(1.0), rel=1e-9)
        assert tol.accept_t_min(0.5) == pytest.approx(tol.accept_t_min(1.0), rel=1e-9)

    def test_float32_threshold_larger_than_float64(self):
        f32 = tol.accept_t_min(torch.tensor(50.0, dtype=torch.float32))
        f64 = tol.accept_t_min(torch.tensor(50.0, dtype=torch.float64))
        assert float(f32) > float(f64)


class TestTinyFor:
    def test_float32_sqrt_smallest_normal(self):
        assert tol.tiny_for(np.float32) == pytest.approx(1.0842022e-19, rel=1e-4)

    def test_float64_sqrt_smallest_normal(self):
        assert tol.tiny_for(np.float64) == pytest.approx(1.4916681e-154, rel=1e-4)

    def test_square_does_not_underflow_float32(self):
        """The whole point of tiny_for: squaring it in the backward pass of
        a bare reciprocal must not underflow to zero in the given dtype.
        """
        eps = tol.tiny_for(np.float32)
        assert np.float32(eps) ** 2 > 0.0

    def test_accepts_array_tensor_and_dtype(self):
        arr = np.ones(3, dtype=np.float32)
        t = torch.ones(3, dtype=torch.float32)
        assert tol.tiny_for(arr) == pytest.approx(tol.tiny_for(t), rel=1e-6)
        assert tol.tiny_for(np.float32) == pytest.approx(tol.tiny_for(t), rel=1e-6)


class TestRadicandFloor:
    def test_float64_reduces_to_calibrated_constant(self):
        floor = tol.radicand_floor(np.ones(3, dtype=np.float64), floor_at_f64=1e-12)
        assert floor == pytest.approx(1e-12, rel=1e-12)

    def test_float32_scaled_up(self):
        f32 = tol.radicand_floor(np.ones(3, dtype=np.float32), floor_at_f64=1e-12)
        assert f32[0] > 1e-12


# ---------------------------------------------------------------------------
# Item 2: the origin advance / long-path case
# ---------------------------------------------------------------------------


class TestLongPathOriginAdvance:
    """NS2 section 6 (E3): a collimated source far behind the singlet.

    Without the origin advance, the rms spot radius at the detector grows
    from 12.4 um (source at z=0) to 569/642 um at z=-1e5/-1e6 mm, because
    the ray-conic discriminant loses accuracy as (L/R)^2 for the long throw
    L. With the advance, the result must equal the z=0 value at every
    distance.
    """

    @pytest.mark.parametrize("source_z", [0.0, -1e5, -1e6])
    def test_rms_radius_independent_of_source_distance(self, source_z):
        be.set_backend("numpy")
        scene = _build_singlet(source_z, detector_cls="raydb")
        result = scene.trace(num_rays=200_000, seed=42, max_depth=8)
        rms_mm, n_direct, n_total = _direct_cluster_rms(result.detectors["D1"])
        rms_um = rms_mm * 1000.0
        assert n_direct > 0.95 * n_total
        assert rms_um == pytest.approx(12.4358, abs=0.01), (
            f"source_z={source_z}: rms {rms_um:.4f} um does not match the "
            "z=0 value -- the origin advance should make this distance-"
            "independent to a fraction of a micron"
        )

    def test_flux_and_ray_count_independent_of_source_distance(self):
        be.set_backend("numpy")
        base = _build_singlet(0.0, detector_cls="raydb")
        base_result = base.trace(num_rays=200_000, seed=42, max_depth=8)
        far = _build_singlet(-1e6, detector_cls="raydb")
        far_result = far.trace(num_rays=200_000, seed=42, max_depth=8)
        assert far_result.total_flux_detected == pytest.approx(
            base_result.total_flux_detected, rel=1e-4
        )
        assert len(far_result.detectors["D1"].x) == len(base_result.detectors["D1"].x)


# ---------------------------------------------------------------------------
# Item 3: the float32 singlet and its gradient
# ---------------------------------------------------------------------------


class TestFloat32Singlet:
    """NS1/NS2: at 1e6 rays the float32 singlet used to detect 0.305 W
    (vs. 0.917 W at float64) with 314,599 of 1e6 rays killed by the depth
    cap -- a self-intersection epsilon (1e-9 mm) below the float32 step at
    this scene's coordinates. Both symptoms must be gone.
    """

    def test_zero_rays_depth_killed(self):
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float32")
        scene = _build_singlet(0.0)
        result = scene.trace(num_rays=200_000, seed=42, max_depth=16)
        assert result.num_rays_depth_killed == 0

    def test_per_ray_flux_matches_float64_tightly(self):
        """The ray-tracing math itself (this task's scope): summing each
        ray's own flux (via a RayDatabase, before any detector splatting)
        agrees with float64 to a few parts in 1e9 -- see
        docs/build/W4_tolerances.md for why the *detector's* own float32
        accumulation (out of this task's scope) does not.
        """
        fluxes = {}
        for precision in ("float64", "float32"):
            be.set_backend("torch")
            be.set_device("cpu")
            be.set_precision(precision)
            scene = _build_singlet(0.0, detector_cls="raydb")
            result = scene.trace(num_rays=200_000, seed=42, max_depth=16)
            db = result.detectors["D1"]
            flux_np = np.asarray(db.flux, dtype=np.float64)
            fluxes[precision] = float(flux_np.sum())
        assert fluxes["float32"] == pytest.approx(fluxes["float64"], abs=1e-6)

    def test_detector_flux_within_documented_bound(self):
        """The IrradianceDetector's own total (not just the per-ray sum)
        is now close to float64 -- a 66% error shrunk to well under 1% --
        though not within the 1e-6 W the ray-tracing math alone achieves
        (see test_per_ray_flux_matches_float64_tightly and
        docs/build/W4_tolerances.md): its bilinear-splat accumulation runs
        in float32 too and is flagged there for the detector work.
        """
        fluxes = {}
        for precision in ("float64", "float32"):
            be.set_backend("torch")
            be.set_device("cpu")
            be.set_precision(precision)
            scene = _build_singlet(0.0)
            result = scene.trace(num_rays=200_000, seed=42, max_depth=16)
            fluxes[precision] = float(result.total_flux_detected)
        rel_err = abs(fluxes["float32"] - fluxes["float64"]) / fluxes["float64"]
        assert rel_err < 0.01, f"float32 detector flux off by {rel_err:.4%}"


class TestFloat32Gradient:
    """NS2 section 9 (E6, "probe F"): the frustum's guarded reciprocal
    produced NaN above 40,000 rays in float32 autograd. Fixed by masking
    the input rather than enlarging the additive epsilon.
    """

    @staticmethod
    def _probe_f(r1, num_rays: int = 50_000, seed: int = 42):
        """Detector short of focus, 6x6 mm: mean-square spot radius vs r1."""
        scene = NSQScene()
        scene.add_source(
            "S1",
            CoordinateSystem(z=0.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0
            ),
        )
        scene.add_lens(
            "L1",
            CoordinateSystem(z=50),
            LensConfig(r1=r1, r2=-100, thickness=5, material="N-BK7", front_aperture_radius=12.5),
        )
        scene.add_detector(
            "D1",
            CoordinateSystem(z=110),
            IrradianceDetectorConfig(width=6, height=6, num_pixels_x=64, num_pixels_y=64),
        )
        result = scene.trace(num_rays=num_rays, seed=seed, max_depth=8)
        det = result.detectors["D1"]
        xs, ys = np.meshgrid(det.x_coords, det.y_coords)
        weights = be.array((xs**2 + ys**2).ravel())
        data = det.data
        return be.sum(data * weights) / be.sum(data)

    def test_gradient_is_finite_in_float32(self):
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float32")
        r1 = torch.tensor(100.0, dtype=torch.float32, requires_grad=True)
        loss = self._probe_f(r1)
        loss.backward()
        assert np.isfinite(r1.grad.item()), "gradient is NaN or inf in float32"

    def test_gradient_within_one_percent_of_float64(self):
        grads = {}
        for precision in ("float64", "float32"):
            be.set_backend("torch")
            be.set_device("cpu")
            be.set_precision(precision)
            dtype = torch.float64 if precision == "float64" else torch.float32
            r1 = torch.tensor(100.0, dtype=dtype, requires_grad=True)
            loss = self._probe_f(r1)
            loss.backward()
            grads[precision] = r1.grad.item()
        rel_err = abs(grads["float32"] - grads["float64"]) / abs(grads["float64"])
        assert rel_err < 0.01, (
            f"float32 grad {grads['float32']!r} vs float64 {grads['float64']!r}, "
            f"relative error {rel_err:.4%}"
        )


# ---------------------------------------------------------------------------
# Item 4: torch backend mps device validation
# ---------------------------------------------------------------------------


class TestMpsDevice:
    def test_invalid_device_string_rejected(self):
        be.set_backend("torch")
        with pytest.raises(ValueError, match="cpu.*cuda.*mps"):
            be.set_device("tpu")

    def test_mps_unavailable_raises_clear_error(self):
        """On this machine mps is not reachable (see the docstring rules on
        sandboxed hosts); the rejection must name mps, not fail generically.
        """
        be.set_backend("torch")
        if torch.backends.mps.is_available():
            pytest.skip("mps is available on this host; covered by the float64 tests below")
        with pytest.raises(ValueError, match="MPS"):
            be.set_device("mps")

    def test_float64_on_mps_refused_device_first(self, monkeypatch):
        """set_device('mps') then set_precision('float64') must both be
        blocked, without needing real mps hardware.
        """
        from optiland.backend.torch_backend.config import _Config

        cfg = _Config()
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        cfg.set_device("mps")
        assert cfg.get_device() == "mps"
        with pytest.raises(ValueError, match="float64"):
            cfg.set_precision("float64")

    def test_float64_on_mps_refused_precision_first(self, monkeypatch):
        from optiland.backend.torch_backend.config import _Config

        cfg = _Config()
        cfg.set_precision("float64")
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        with pytest.raises(ValueError, match="float64"):
            cfg.set_device("mps")

    def test_mps_float32_accepted_when_available(self, monkeypatch):
        from optiland.backend.torch_backend.config import _Config

        cfg = _Config()
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        cfg.set_device("mps")
        assert cfg.get_device() == "mps"
