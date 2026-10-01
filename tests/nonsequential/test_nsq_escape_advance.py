"""The escape advance in a scene with an unbounded surface (KronosNSRT issue 79).

An escaped ray is moved past the scene by the bounding scale. A scene with an
infinite plane had an infinite scale: escaped positions became infinite or NaN
(inf * 0), and in a gradient trace the zero cotangent of the discarded lane
times the advance's infinite derivative, or a later bounce's arithmetic on the
infinite position (the detector plane's division), made every gradient NaN.
The scale is now the diagonal of the finite bounds only. Pinned here, on torch
float64 with autograd:

* the scale ignores infinite bounds and is unchanged for a finite scene;
* d(transmitted flux)/dn through one infinite vacuum/n interface at normal
  incidence against the closed form -4 (n - 1) / (n + 1)**3 / (1 - p), with p the
  Fresnel branch probability clipped to 1e-12, within K u of it, K the
  operation count of the recorded graph (the maintainer's ruling 3 of
  2026-09-27, the form T-09-6 uses);
* the same derivative against a central difference;
* a scene with escaping rays and oblique incidence has a finite gradient, equal
  bit for bit to the same scene with the plane replaced by a finite plane wider
  than the beam (the escape distance never reaches a detector).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import optiland.backend as be  # noqa: E402
from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.materials import IdealMaterial  # noqa: E402
from optiland.nonsequential import (  # noqa: E402
    CollimatedSourceConfig,
    FinitePlaneGeometry,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    PlaneGeometry,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential._utils import estimate_bounding_scale  # noqa: E402
from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: E402
from optiland.nonsequential.components.geometry.base import AABB  # noqa: E402
from optiland.nonsequential.ir.scene_ir import SamplingPolicy  # noqa: E402
from optiland.nonsequential.materials import VACUUM, NSQMaterial  # noqa: E402

U64 = 2.0**-53
P_CLIP = 1e-12


@pytest.fixture(autouse=True)
def _torch_float64_with_grad():
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.enable()
    yield
    be.grad_mode.disable()
    be.set_backend("numpy")
    be.set_precision("float64")


def _t(x):
    return torch.tensor(float(x), dtype=torch.float64, requires_grad=True)


def _operation_count(output, leaf) -> int:
    """Operations of ``output``'s autograd graph from which ``leaf`` is reachable, each once."""
    reach: dict[int, bool] = {}
    stack = [(output.grad_fn, False)]
    while stack:
        fn, expanded = stack.pop()
        if fn is None or (id(fn) in reach and not expanded):
            continue
        children = [nxt for nxt, _ in fn.next_functions if nxt is not None]
        if not expanded:
            reach[id(fn)] = False
            stack.append((fn, True))
            stack.extend((c, False) for c in children if id(c) not in reach)
            continue
        hit = getattr(fn, "variable", None) is leaf
        reach[id(fn)] = hit or any(reach.get(id(c), False) for c in children)
    return sum(reach.values())


class _Box:
    def __init__(self, lo, hi):
        self.bounding_box = AABB(np.array(lo, dtype=float), np.array(hi, dtype=float))


class _Scene:
    def __init__(self, surfaces):
        self.surfaces = surfaces


class TestBoundingScale:
    def test_infinite_bounds_are_left_out(self):
        inf = math.inf
        finite = _Box([0.0, 0.0, 0.0], [30.0, 40.0, 0.0])
        plane = _Box([-inf, -inf, -inf], [inf, inf, inf])
        assert estimate_bounding_scale(_Scene([finite, plane])) == 50.0
        assert estimate_bounding_scale(_Scene([plane, finite])) == 50.0

    def test_only_infinite_bounds_fall_back(self):
        inf = math.inf
        plane = _Box([-inf, -inf, -inf], [inf, inf, inf])
        assert estimate_bounding_scale(_Scene([plane])) == 100.0

    def test_finite_scene_is_unchanged(self):
        boxes = [_Box([-1.5, 2.0, -3.25], [4.0, 5.5, 6.0]), _Box([0.1, -7.0, 0.0], [0.2, 0.3, 9.75])]
        dx, dy, dz = 4.0 - (-1.5), 5.5 - (-7.0), 9.75 - (-3.25)
        assert estimate_bounding_scale(_Scene(boxes)) == (dx**2 + dy**2 + dz**2) ** 0.5


def _interface_flux(n, geometry, theta_deg=0.0, aperture=0.01, num_rays=64):
    """Transmitted flux through one vacuum/n interface (the branch probability pinned)."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.5876), total_flux=1.0, aperture_radius=aperture
        ),
    )
    glass = NSQMaterial(optiland_material=IdealMaterial(n=n, k=0.0))
    scene.add_component(
        "IF",
        RefractiveComponent(
            CoordinateSystem(z=10.0, ry=math.radians(theta_deg)), geometry, VACUUM, glass, name="IF"
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=30.0),
        IrradianceDetectorConfig(width=40, height=40, num_pixels_x=1, num_pixels_y=1, splat="hard"),
    )
    scene.sampling_policy = SamplingPolicy(reflect_prob=0.0, rr_start_flux=1e-30)
    res = scene.trace(num_rays=num_rays, seed=1, max_depth=2, backend=TorchBackend(seed=1))
    return res.detectors["D"].total_flux


class TestInfinitePlaneGradient:
    def test_against_the_closed_form(self):
        n0 = 1.5
        n = _t(n0)
        flux = _interface_flux(n, PlaneGeometry())
        (g,) = torch.autograd.grad(flux, n, retain_graph=True)
        assert torch.isfinite(g)
        closed = -4.0 * (n0 - 1.0) / (n0 + 1.0) ** 3 / (1.0 - P_CLIP)
        k = _operation_count(flux, n)
        assert abs(float(g) - closed) <= k * U64 * abs(closed), (float(g), closed, k)

    def test_against_a_central_difference(self):
        """Central difference, h = 1e-6: truncation h**2 |f'''| / 6 below 1e-12 (|f'''| < 1
        for the transmittance near n = 1.5) and rounding about K u |f| / h with
        K the graph's operation count."""
        n0, h = 1.5, 1e-6
        n = _t(n0)
        flux = _interface_flux(n, PlaneGeometry())
        (g,) = torch.autograd.grad(flux, n, retain_graph=True)
        k = _operation_count(flux, n)

        def f(x):
            return float(_interface_flux(x, PlaneGeometry()))

        fd = (f(n0 + h) - f(n0 - h)) / (2.0 * h)
        assert abs(float(g) - fd) <= 1e-12 + k * U64 * 1.0 / h, (float(g), fd, k)

    @pytest.mark.parametrize("theta", [0.0, 3.0])
    def test_infinite_plane_equals_a_wide_finite_plane(self, theta):
        """Rays escape past a small detector; the infinite plane's gradient is finite and
        equal to the finite plane's (both escape distances are finite and no escaped ray
        is tallied)."""

        def run(geometry):
            n = _t(1.5)
            scene = NSQScene()
            scene.add_source(
                "S",
                CoordinateSystem(),
                CollimatedSourceConfig(
                    spectrum=Spectrum.monochromatic(0.5876), total_flux=1.0, aperture_radius=8.0
                ),
            )
            glass = NSQMaterial(optiland_material=IdealMaterial(n=n, k=0.0))
            scene.add_component(
                "IF",
                RefractiveComponent(
                    CoordinateSystem(z=10.0, ry=math.radians(theta)), geometry, VACUUM, glass, name="IF"
                ),
            )
            scene.add_lens(
                "L1",
                CoordinateSystem(x=0.3, z=40.0),
                LensConfig(r1=60.0, r2=float("inf"), thickness=5.0, material="N-BK7", front_aperture_radius=6.0),
            )
            scene.add_detector(
                "D",
                CoordinateSystem(z=120.0),
                IrradianceDetectorConfig(width=6, height=6, num_pixels_x=8, num_pixels_y=8),
            )
            scene.sampling_policy = SamplingPolicy(reflect_prob=1e-6, rr_start_flux=1e-30)
            res = scene.trace(num_rays=400, seed=3, max_depth=8, backend=TorchBackend(seed=3))
            data = res.detectors["D"].data
            (g,) = torch.autograd.grad(data.sum(), n)
            return data.detach(), g, res.num_rays_escaped

        d_inf, g_inf, esc_inf = run(PlaneGeometry())
        d_fin, g_fin, esc_fin = run(FinitePlaneGeometry(aperture_radius=50.0))
        assert esc_inf > 0
        assert esc_inf == esc_fin
        assert torch.isfinite(g_inf)
        assert torch.equal(d_inf, d_fin)
        assert float(g_inf) == float(g_fin)
