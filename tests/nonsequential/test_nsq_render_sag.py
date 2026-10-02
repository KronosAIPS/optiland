"""The lens, doublet and mirror renderers draw the sag the face kind evaluates
(KronosNSRT issue 81).

The renderers drew every face from its config's radius and conic constant, so
an asphere face (issue 30) was drawn as its base conic. The drawn points are
compared here with the face geometry's own sag: the same function on the same
abscissae, placed by an identity transform (a product with 1 and 0 and a sum
with 0, all exact), so the comparison is bit for bit. Each scene's polynomial
moves the rim sag by more than 0.1 mm from the base conic, so a renderer that
drops the polynomial fails by orders of magnitude more than any rounding.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import optiland.backend as be  # noqa: E402
from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.nonsequential.components.configs import (  # noqa: E402
    DoubletConfig,
    LensConfig,
    MirrorConfig,
)
from optiland.nonsequential.components.doublet import Doublet  # noqa: E402
from optiland.nonsequential.components.lens import Lens  # noqa: E402
from optiland.nonsequential.components.mirror import Mirror  # noqa: E402
from optiland.nonsequential.visualization.renderers import lens as lens_renderers  # noqa: E402
from optiland.nonsequential.visualization.renderers import mirror as mirror_renderers  # noqa: E402

N_PTS = 64


@pytest.fixture(autouse=True)
def _numpy_backend():
    be.set_backend("numpy")
    yield
    be.set_backend("numpy")
    plt.close("all")


def _kind_sag(face, y):
    g = face.geometry
    return g.sag(np.zeros_like(y), y) if hasattr(g, "sag") else g._sag(np.zeros_like(y), y)


def _asphere_lens():
    return Lens(
        "L",
        CoordinateSystem(),
        LensConfig(
            r1=40.0,
            r2=-60.0,
            thickness=6.0,
            material="N-BK7",
            front_aperture_radius=10.0,
            coefficients1=[0.0, -2e-5, 1e-8],
            coefficients2=[1e-2, 1e-3],
            odd2=True,
        ),
    )


def _drawn_lines(render, component):
    fig, ax = plt.subplots()
    render(component, ax)
    return [np.asarray(line.get_xydata()) for line in ax.lines]


def _base_conic(radius, conic, y):
    return lens_renderers._sag_array(radius, conic, y)


class TestLens2D:
    def test_faces_follow_the_kind(self):
        lens = _asphere_lens()
        (contour,) = _drawn_lines(lens_renderers.LensRenderer2D().render, lens)
        front, back = lens.surfaces[0], lens.surfaces[1]
        # YZ projection: horizontal z, vertical y; contour = front (64), 2 rim
        # points, back reversed (64), 2 rim points.
        fz, fy = contour[:N_PTS, 0], contour[:N_PTS, 1]
        bz = contour[N_PTS + 2 : 2 * N_PTS + 2, 0][::-1]
        by = contour[N_PTS + 2 : 2 * N_PTS + 2, 1][::-1]
        np.testing.assert_array_equal(fz, _kind_sag(front, fy))
        np.testing.assert_array_equal(bz, _kind_sag(back, by) + 6.0)
        assert np.max(np.abs(fz - _base_conic(40.0, 0.0, fy))) > 0.1
        assert np.max(np.abs(bz - 6.0 - _base_conic(-60.0, 0.0, by))) > 0.1

    def test_a_conic_lens_follows_the_conic_kind(self):
        lens = Lens(
            "C",
            CoordinateSystem(),
            LensConfig(r1=50.0, r2=-50.0, thickness=5.0, material="N-BK7", front_aperture_radius=12.5),
        )
        (contour,) = _drawn_lines(lens_renderers.LensRenderer2D().render, lens)
        fz, fy = contour[:N_PTS, 0], contour[:N_PTS, 1]
        np.testing.assert_array_equal(fz, _kind_sag(lens.surfaces[0], fy))


class TestDoublet2D:
    def test_three_faces_follow_the_kind(self):
        d = Doublet(
            "D",
            CoordinateSystem(),
            DoubletConfig(
                r1=60.0, r2=-45.0, r3=-150.0, thickness1=6.0, thickness2=3.0,
                material1="N-BK7", material2="N-SF5", aperture_radius=10.0,
                coefficients1=[0.0, -3e-5], coefficients2=[0.0, 4e-5], coefficients3=[2e-2], odd3=True,
            ),
        )
        c1, c2 = _drawn_lines(lens_renderers.DoubletRenderer2D().render, d)
        f = d.surfaces
        z1, y1 = c1[:N_PTS, 0], c1[:N_PTS, 1]
        z2, y2 = c2[:N_PTS, 0], c2[:N_PTS, 1]
        z3 = c2[N_PTS + 1 : 2 * N_PTS + 1, 0][::-1]
        y3 = c2[N_PTS + 1 : 2 * N_PTS + 1, 1][::-1]
        np.testing.assert_array_equal(z1, _kind_sag(f[0], y1))
        np.testing.assert_array_equal(z2, _kind_sag(f[1], y2) + 6.0)
        np.testing.assert_array_equal(z3, _kind_sag(f[2], y3) + 6.0 + 3.0)
        assert np.max(np.abs(z3 - 9.0 - _base_conic(-150.0, 0.0, y3))) > 0.1


class TestMirror2D:
    def test_an_asphere_mirror_follows_the_kind(self):
        m = Mirror(
            "M",
            CoordinateSystem(),
            MirrorConfig(radius=-200.0, reflectance=1.0, aperture_radius=20.0, coefficients=[0.0, 1e-6]),
        )
        (line,) = _drawn_lines(mirror_renderers.MirrorRenderer2D().render, m)
        z, y = line[:, 0], line[:, 1]
        np.testing.assert_array_equal(z, _kind_sag(m.surfaces[0], y))
        assert np.max(np.abs(z - _base_conic(-200.0, 0.0, y))) > 0.1


class TestRenderers3D:
    """The revolved contours, captured at ``revolve_contour``."""

    @staticmethod
    def _captured(render, component):
        pytest.importorskip("vtk")
        calls = []

        def fake_revolve(x, y, z):
            calls.append((np.asarray(x), np.asarray(y), np.asarray(z)))
            return mock.MagicMock()

        with mock.patch("optiland.visualization.system.utils.revolve_contour", fake_revolve):
            render(component, mock.MagicMock())
        return calls

    def test_lens(self):
        lens = _asphere_lens()
        ((_, y, z),) = self._captured(lens_renderers.LensRenderer3D().render, lens)
        np.testing.assert_array_equal(z[:N_PTS], _kind_sag(lens.surfaces[0], y[:N_PTS]))

    def test_mirror(self):
        m = Mirror(
            "M",
            CoordinateSystem(),
            MirrorConfig(radius=-200.0, reflectance=1.0, aperture_radius=20.0, coefficients=[0.0, 1e-6]),
        )
        ((_, y, z),) = self._captured(mirror_renderers.MirrorRenderer3D().render, m)
        np.testing.assert_array_equal(z, _kind_sag(m.surfaces[0], y))


class TestNurbsMirror:
    """A NURBS mirror is drawn from its surface, sampled along the section by the
    kind's own intersection (KronosNSRT issue 88), not as the config's base conic.

    The face is the revolved parabola of the NURBS tests, z = r^2 / (4 f) exactly
    (a polynomial quadratic Bezier arc, so the patch is the paraboloid with no
    representation error): every drawn point lies on it to the kind's root
    accuracy, over the patches' extent (10 mm), where the config's radius 0
    would draw a flat line."""

    F = 50.0

    def _mirror(self):
        from tests.nonsequential.test_nsq_nurbs import contract, revolved  # noqa: PLC0415

        h = 10.0
        arrays = contract([revolved([0, 0, 0, 1, 1, 1], [[0, 0], [h / 2, 0], [h, h * h / (4 * self.F)]], [1, 1, 1], 2)
                           + (False,)])
        return Mirror("M", CoordinateSystem(), MirrorConfig(radius=0.0, reflectance=1.0, nurbs=arrays))

    def test_2d_points_lie_on_the_surface(self):
        m = self._mirror()
        (line,) = _drawn_lines(mirror_renderers.MirrorRenderer2D().render, m)
        z, y = line[:, 0], line[:, 1]
        ok = np.isfinite(z)
        # drawn over the part of the section on the patches: from the probe's first hit to its last,
        # within one probe step (2 x 10 sqrt(2) mm / 512) of the rim at 10 mm
        assert ok.all()
        assert -y.min() > 10.0 - 0.06 and y.max() > 10.0 - 0.06 and np.abs(y).max() <= 10.0
        # the kind's own surface, z = y^2 / (4 f): within the kind's root tolerance
        # (32 u at the coordinate scale, about 2 * 10 mm with the ray's start below)
        np.testing.assert_allclose(z[ok], y[ok] ** 2 / (4 * self.F), rtol=0, atol=64 * 2.0**-53 * 40.0)
        # and not the config's base conic (radius 0: flat), which the drawing used before
        assert np.max(np.abs(z[ok] - _base_conic(0.0, 0.0, y[ok]))) > 0.4

    def test_3d_contour_lies_on_the_surface(self):
        m = self._mirror()
        ((_, y, z),) = TestRenderers3D._captured(mirror_renderers.MirrorRenderer3D().render, m)
        assert np.all(np.isfinite(z)) and 10.0 - 0.06 < y.max() <= 10.0
        np.testing.assert_allclose(z, y**2 / (4 * self.F), rtol=0, atol=64 * 2.0**-53 * 40.0)
