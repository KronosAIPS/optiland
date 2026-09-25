"""The Fresnel split's branch-probability clamp, per working dtype (issue 59).

The detached branch probability ``p`` is clamped away from 0 and 1 so the
compensating weights ``R / p`` and ``T / (1 - p)`` stay finite
(``docs/theory/09_differentiation.md`` R-09-1, ``08_precision.md`` section
8.7). The clamp was the float64 literal pair ``[1e-12, 1 - 1e-12]``; in
float32 the upper literal is exactly 1.0. A probability at the clamp then
never leaves the reflect branch -- except on the draw that itself rounds to
1.0 in float32 (a 32-bit draw within 128 of 2**32, about one in 3e7), which
takes the transmit branch with a weight of ``T / tiny_for`` ~ ``9e18 T``.

The upper bound is now ``min(1 - 1e-12, 1 - 4 u)``: the literal at float64,
where every number stays what it was, and ``1 - 2**-22`` at float32.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.coatings import SimpleCoating
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    FinitePlaneGeometry,
    IrradianceDetectorConfig,
    NSQMaterial,
    NSQRng,
    NSQScene,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential import _tol
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.ir.bsdf_ir import BsdfIR
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.rng import EventSlot

U32 = 2.0**-24
U64 = 2.0**-53

#: A coating whose reflectance rounds to exactly 1.0 in float32 and not in
#: float64: 1 - 2**-26 lies within half a float32 step of 1.
_DELTA = 2.0**-26

#: Seed 0, bounce 0, Fresnel-branch slot: the 32-bit draw of this ray id is
#: 2**32 - 14, which rounds to 2**32 in float32, so its uniform is exactly 1.0
#: there (found by a search over the ids; it is the sixth below 2**28). In
#: float64 the same draw is 1 - 14 / 2**32, above 1 - _DELTA, so both
#: precisions take the transmit branch for it.
_RAY_ID_DRAWING_ONE = 193_628_428


def _coated_window() -> RefractiveComponent:
    return RefractiveComponent(
        CoordinateSystem(z=10.0),
        FinitePlaneGeometry(40.0, 40.0),
        VACUUM,
        NSQMaterial.from_glass("N-BK7"),
        coating=SimpleCoating(transmittance=_DELTA, reflectance=1.0 - _DELTA),
    )


@pytest.fixture
def torch_backend():
    be.set_backend("torch")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


class TestTheBounds:
    """What the clamp is in each dtype."""

    @pytest.mark.parametrize("lib", ["numpy", "torch"])
    def test_float64_keeps_the_literal_pair(self, lib):
        like = np.zeros(2) if lib == "numpy" else torch.zeros(2, dtype=torch.float64)
        assert _tol.branch_probability_bounds(like) == (1e-12, 1.0 - 1e-12)

    @pytest.mark.parametrize("lib", ["numpy", "torch"])
    def test_float32_upper_bound_is_below_one(self, lib):
        if lib == "numpy":
            like = np.zeros(2, np.float32)
        else:
            like = torch.zeros(2, dtype=torch.float32)
        lo, hi = _tol.branch_probability_bounds(like)
        assert lo == 1e-12
        assert hi == 1.0 - 4 * U32
        # Exact in float32, and four unit roundoffs short of one there.
        hi32 = torch.tensor(hi, dtype=torch.float32)
        assert hi32.item() == hi
        assert (torch.tensor(1.0, dtype=torch.float32) - hi32).item() == 4 * U32

    def test_the_old_upper_literal_is_one_in_float32(self):
        """The defect itself: 1 - 1e-12 cannot be told from 1 in float32."""
        assert torch.tensor(1.0 - 1e-12, dtype=torch.float32).item() == 1.0


class TestTheDrawThatRoundsToOne:
    """At the clamp, the float32 transmit weight is bounded by T / (4 u)."""

    def _interact(self, precision):
        be.set_precision(precision)
        dtype = torch.float32 if precision == "float32" else torch.float64
        rays = NSQRayBundle(
            x=np.zeros(1),
            y=np.zeros(1),
            z=np.zeros(1),
            L=np.zeros(1),
            M=np.zeros(1),
            N=np.ones(1),
            flux=np.ones(1),
            wavelength=np.full(1, 0.55),
            n_current=np.ones(1),
            bounce=np.zeros(1, dtype=np.int32),
            alive=np.ones(1, dtype=bool),
            ray_id=np.array([_RAY_ID_DRAWING_ONE], dtype=np.int64),
        )
        rays = TorchBackend(seed=0)._prepare_bundle(rays)
        flux_in = torch.ones(1, dtype=dtype, requires_grad=True)
        rays.flux = flux_in * 1.0
        rng = NSQRng(0)
        u = rng.uniform(rays.ray_id, rays.bounce, EventSlot.FRESNEL_BRANCH)
        comp = _coated_window()
        comp.interact(
            rays,
            torch.full((1,), 10.0, dtype=dtype),
            torch.tensor([[0.0, 0.0, -1.0]], dtype=dtype),
            torch.ones(1, dtype=torch.bool),
            rng,
            BsdfIR(kind="none"),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype),
        )
        rays.flux.sum().backward()
        transmitted = bool(rays.N.item() > 0)
        return u.item(), transmitted, rays.flux.item(), flux_in.grad.item()

    def test_float32_transmit_weight_is_bounded(self, torch_backend):
        u, transmitted, weight, grad = self._interact("float32")
        assert u == 1.0  # the draw this test is about
        assert transmitted
        # T / (1 - p) with 1 - p = 4 u32: 2**-26 / 2**-22. Before the bound
        # moved, p was 1.0 and this read T / tiny_for, about 1.4e11.
        assert weight == pytest.approx(_DELTA / (4 * U32), rel=4 * U32)
        assert weight <= 1.0
        assert math.isfinite(grad) and grad == weight

    def test_float64_takes_the_same_branch_unclamped(self, torch_backend):
        """The float64 reference: p = R (no clamp binds), weight T / (1 - R) = 1."""
        u, transmitted, weight, grad = self._interact("float64")
        assert u == 1.0 - 14 / 2.0**32
        assert transmitted
        assert weight == pytest.approx(1.0, rel=16 * U64)
        assert math.isfinite(grad) and grad == weight


def _reflecting_scene(total_flux) -> NSQScene:
    """A beam onto the coated window; the reflection lands behind the source."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55),
            total_flux=total_flux,
            aperture_radius=5.0,
        ),
    )
    scene.add_component("window", _coated_window())
    scene.add_detector(
        "back",
        CoordinateSystem(z=-10.0),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=8, num_pixels_y=8),
    )
    return scene


class TestTheReversePassThroughTheClamp:
    """The float32 gradient through a split at the clamp: finite, and float64's."""

    N = 4_096

    def _gradient(self, precision):
        be.set_precision(precision)
        dtype = torch.float32 if precision == "float32" else torch.float64
        phi = torch.tensor(1.0, dtype=dtype, requires_grad=True)
        result = _reflecting_scene(phi).trace(
            num_rays=self.N, seed=3, max_depth=4, backend=TorchBackend(seed=3)
        )
        detected = result.detectors["back"].total_flux
        detected.backward()
        return float(detected), phi.grad.item(), result.total_flux_escaped

    def test_finite_and_within_the_scaled_window(self, torch_backend):
        d64, g64, esc64 = self._gradient("float64")
        d32, g32, esc32 = self._gradient("float32")
        # The control: every ray reflected in both precisions (no draw of this
        # seed reaches the float32 clamp), so the two runs take the same
        # branches and differ only by rounding and by the clamp itself.
        assert esc64 == 0.0 and esc32 == 0.0
        assert math.isfinite(g32)
        # The window, derived: the clamp's own bias R / p = 1 / (1 - 4 u32),
        # 4 u32; one rounding each for the weight's division and the flux
        # product; and the float32 sum of the N per-ray terms into the
        # gradient, at most ceil(log2 N) roundings on its pairwise tree.
        window = (4 + 2 + math.ceil(math.log2(self.N))) * U32
        assert abs(g32 / g64 - 1.0) <= window, (g32, g64, window)
        assert abs(d32 / d64 - 1.0) <= window, (d32, d64, window)
