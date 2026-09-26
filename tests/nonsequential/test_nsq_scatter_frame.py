"""A scatter lobe's tangent frame belongs to the surface, not to the world axes.

Issue 26 of the research repository: the samplers that draw about a normal
(Lambertian, Harvey-Shack, tabulated) built their tangent frame from the
world components of the normal, so a rigidly rotated scene sent the same
random numbers to a different physical direction and the traced paths parted
(case r2_03 on the diffuser and the sphere; chapter 10 section 10.3,
invariant 3). The frame is now built in the surface's own axes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    HarveyShackBSDF,
    IrradianceDetectorConfig,
    LambertianBSDF,
    NSQScene,
    Spectrum,
    TabulatedBSDF,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.bsdf.lambertian import _orthonormal_basis
from optiland.nonsequential.components.geometry import FinitePlaneGeometry
from optiland.nonsequential.components.reflective import ReflectiveComponent


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _unit(n, seed):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestTheFrameBuilder:
    def setup_method(self):
        be.set_backend("numpy")

    def test_the_identity_frame_is_the_world_frame_bit_for_bit(self):
        n = _unit(5000, 1)
        t0, b0 = _orthonormal_basis(n)
        t1, b1 = _orthonormal_basis(n, np.eye(3))
        assert np.array_equal(t0, t1) and np.array_equal(b0, b1)

    def test_the_frame_turns_with_the_surface(self):
        """Rotate the surface and its normals by Q: the frame turns by Q."""
        n = _unit(5000, 2)
        r = _rotation([0.2, -0.7, 0.4], 0.9)
        q = _rotation([0.36, 0.48, 0.8], math.radians(37.0))
        t, b = _orthonormal_basis(n, r)
        tq, bq = _orthonormal_basis(n @ q.T, q @ r)
        np.testing.assert_allclose(tq, t @ q.T, atol=1e-14)
        np.testing.assert_allclose(bq, b @ q.T, atol=1e-14)
        # Still an orthonormal frame beside n.
        np.testing.assert_allclose(np.sum(t * n, axis=1), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.sum(b * n, axis=1), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.sum(t * b, axis=1), 0.0, atol=1e-14)

    def test_the_world_frame_does_not(self):
        """The control: without the surface's frame the world-axis frame does
        not turn with a rotation about the normal itself."""
        n = np.tile([0.0, 0.0, -1.0], (4, 1))
        q = _rotation([0.0, 0.0, 1.0], 0.6)
        t, _ = _orthonormal_basis(n)
        tq, _ = _orthonormal_basis(n @ q.T)
        assert not np.allclose(tq, t @ q.T, atol=1e-3)


def _tabulated(tmp_path):
    path = tmp_path / "scatter.csv"
    rows = []
    for ti in (0.0, 45.0, 90.0):
        for ts, value in ((0.0, 0.30), (45.0, 0.12), (90.0, 0.01)):
            rows.append(f"{ti},{ts},{value}")
    path.write_text("\n".join(rows) + "\n")
    return TabulatedBSDF(path)


def _scene(bsdf, gamma: float, shift) -> NSQScene:
    """A beam onto a tilted scattering plate and a detector facing it.

    Every placement is rotated about z by ``gamma`` and moved by ``shift``:
    each original placement has rotation Rx(alpha), and Rz(gamma) Rx(alpha)
    is the engine's own Euler form (R = Rz Ry Rx) with rx = alpha, rz = gamma,
    so the transformed scene is written without any angle conversion.
    """
    q = _rotation([0.0, 0.0, 1.0], gamma)
    d = np.asarray(shift, dtype=float)

    def cs(pos, rx=0.0):
        p = q @ np.asarray(pos, dtype=float) + d
        return CoordinateSystem(x=p[0], y=p[1], z=p[2], rx=rx, rz=gamma)

    scene = NSQScene()
    scene.add_source(
        "S",
        cs([0.0, 0.0, 0.0]),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=2.0
        ),
    )
    scene.add_component(
        "plate",
        ReflectiveComponent(
            cs([0.0, 0.0, 30.0], rx=0.35),
            FinitePlaneGeometry(width=40.0, height=40.0),
            reflectance=1.0,
            bsdf=bsdf,
            name="plate",
            scatter_fraction=1.0,
        ),
    )
    scene.add_detector(
        "D",
        cs([0.0, 0.0, -20.0], rx=math.pi),
        IrradianceDetectorConfig(width=400, height=400, num_pixels_x=8, num_pixels_y=8),
    )
    return scene, q, d


def _hits(result, name):
    ev = result.ray_paths["events"]
    ev = ev[(ev["event_type"] == "hit") & (ev["component_name"] == name)]
    order = np.argsort(ev["ray_id"], kind="stable")
    return ev[order]


@pytest.mark.parametrize("lobe", ["lambertian", "harvey_shack", "tabulated"])
def test_a_rotated_scene_traces_the_rotated_rays(lobe, tmp_path):
    def bsdf():
        if lobe == "lambertian":
            return LambertianBSDF(reflectance_value=1.0)
        if lobe == "harvey_shack":
            return HarveyShackBSDF(b0=0.5, l0=0.3, s=2.0)
        return _tabulated(tmp_path)

    be.set_backend("numpy")
    runs = []
    for gamma, shift in ((0.0, (0.0, 0.0, 0.0)), (0.6, (11.0, -7.0, 23.0))):
        scene, q, d = _scene(bsdf(), gamma, shift)
        result = scene.trace(
            num_rays=2000, seed=3, max_depth=4, record_paths=True,
            backend=NumpyBackend(seed=3),
        )
        runs.append((result, q, d))
    (r0, _, _), (r1, q, d) = runs
    a, b = _hits(r0, "D"), _hits(r1, "D")
    assert a.size > 100
    assert np.array_equal(a["ray_id"], b["ray_id"])
    p1 = np.stack([b["x"], b["y"], b["z"]], axis=1)
    back = (p1 - d) @ q  # q^T (p - d), row vectors
    p0 = np.stack([a["x"], a["y"], a["z"]], axis=1)
    np.testing.assert_allclose(back, p0, atol=1e-9)
    np.testing.assert_allclose(b["flux"], a["flux"], rtol=1e-12)
