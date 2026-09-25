"""Tests for LensletArrayGeometry (KronosNSRT issue 40): the analytic,
periodic conic-cap array.

An independent route against a brute-force per-cell ``ConicGeometry``
(cell membership imposed by hand, not by the grid march under test) is the
main correctness check; the rest cover the vertex case, multi-cell
crossing, per-cell offsets, a traced focus case and the IR lowering.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensletArrayGeometry,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
    kinds,
)
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.ir import lower as lower_scene
from optiland.nonsequential.ir.lower import _lower_geometry

torch = pytest.importorskip(
    "torch", reason="Torch not available -- float32 runs on its CPU backend"
)


@pytest.fixture(autouse=True)
def _numpy_float64():
    be.set_backend("numpy")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


# ---------------------------------------------------------------------------
# An independent route: one ConicGeometry per cell, cell membership imposed
# by hand on the caller's side (not by LensletArrayGeometry's own grid
# march), the nearest valid hit kept over every cell.
# ---------------------------------------------------------------------------


def _brute_force(pitch_x, pitch_y, radius, conic, num_x, num_y, origins, directions):
    """Nearest valid hit over every cell, an independent route against
    LensletArrayGeometry's own grid march: each cell's conic quadratic is
    solved directly (both roots, not just whichever root a bare
    ``ConicGeometry.ray_intersect`` call happens to prefer under its own
    circular-aperture pick -- the nearer of the two roots is sometimes
    outside the cell while the farther one is inside it, which a black-box
    call on the whole conic would silently lose), and cell membership is
    imposed by hand on the caller's side.
    """
    N = origins.shape[0]
    ox0, oy0, oz0 = origins[:, 0], origins[:, 1], origins[:, 2]
    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

    c = 0.0 if (radius == 0.0 or not np.isfinite(radius)) else 1.0 / radius
    kp = 1.0 + conic

    best_t = np.full(N, np.inf)
    best_mask = np.zeros(N, dtype=bool)
    best_px_rel = np.zeros(N)
    best_py_rel = np.zeros(N)

    for i in range(num_x):
        xc = (i - (num_x - 1) / 2.0) * pitch_x
        for j in range(num_y):
            yc = (j - (num_y - 1) / 2.0) * pitch_y
            ox = ox0 - xc
            oy = oy0 - yc

            a = c * (dx**2 + dy**2 + kp * dz**2)
            b = 2.0 * (c * (ox * dx + oy * dy + kp * oz0 * dz) - dz)
            c0 = c * (ox**2 + oy**2 + kp * oz0**2) - 2.0 * oz0
            disc = b**2 - 4.0 * a * c0
            disc_ok = disc >= 0.0
            sq = np.sqrt(np.maximum(disc, 0.0))

            a_ok = np.abs(a) > 1e-12
            a_safe = np.where(a_ok, a, 1.0)
            t_quad_1 = (-b + sq) / (2.0 * a_safe)
            t_quad_2 = (-b - sq) / (2.0 * a_safe)
            # a ~ 0 (flat limit, or a ray whose combination of components
            # cancels it exactly): the equation degenerates to linear.
            b_ok = np.abs(b) > 1e-12
            t_lin = -c0 / np.where(b_ok, b, 1.0)

            roots = (
                (np.where(a_ok, t_quad_1, t_lin), disc_ok & a_ok | (~a_ok & b_ok)),
                (np.where(a_ok, t_quad_2, np.inf), disc_ok & a_ok),
            )
            for t_candidate, solvable in roots:
                px = ox0 + t_candidate * dx
                py = oy0 + t_candidate * dy
                pz = oz0 + t_candidate * dz
                on_sheet = (1.0 - kp * c * pz) >= 0.0
                in_cell = (np.abs(px - xc) <= pitch_x / 2.0) & (
                    np.abs(py - yc) <= pitch_y / 2.0
                )
                valid = (
                    solvable
                    & np.isfinite(t_candidate)
                    & (t_candidate > 1e-9)
                    & on_sheet
                    & in_cell
                )
                better = valid & (t_candidate < best_t)
                best_t = np.where(better, t_candidate, best_t)
                best_px_rel = np.where(better, px - xc, best_px_rel)
                best_py_rel = np.where(better, py - yc, best_py_rel)
                best_mask = best_mask | valid

    # The normal formula itself is not what this route re-derives (it is a
    # plain gradient of the sag function); reuse ConicGeometry's for it,
    # evaluated at the (independently found) winning cell's relative
    # coordinates.
    cap = ConicGeometry(radius=radius, conic=conic, aperture_radius=1e12)
    n_raw = cap._normal_local(best_px_rel, best_py_rel)
    n_len = np.sqrt((n_raw**2).sum(axis=1, keepdims=True))
    n_geom = n_raw / n_len
    dot = (directions * n_geom).sum(axis=1, keepdims=True)
    best_normals = np.where(dot > 0, -n_geom, n_geom)

    return best_t, best_mask, best_normals


def _cell_centre(i, j, num_x, num_y, pitch_x, pitch_y):
    xc = (i - (num_x - 1) / 2.0) * pitch_x
    yc = (j - (num_y - 1) / 2.0) * pitch_y
    return xc, yc


def _random_rays(
    n, seed, half_x, half_y, z_lo, z_hi, dz_log_lo=-3.0, spread=1.1
):
    """Origins above and below the array, directions from steep to grazing.

    Args:
        half_x, half_y: The array footprint's own half-extents; origins are
            spread a little past them (``spread``) so some rays start
            outside the footprint too.
        z_lo, z_hi: Range of |z| for the origin, on a random side of the
            array (``+z`` and ``-z`` both exercised).
        dz_log_lo: log10 of the smallest |direction z| sampled -- 0.0 is
            head-on; more negative reaches further into grazing incidence.
    """
    rng = np.random.default_rng(seed)
    origins = np.empty((n, 3))
    origins[:, 0] = rng.uniform(-spread * half_x, spread * half_x, n)
    origins[:, 1] = rng.uniform(-spread * half_y, spread * half_y, n)
    side = rng.choice([-1.0, 1.0], size=n)
    origins[:, 2] = side * rng.uniform(z_lo, z_hi, n)

    directions = rng.normal(size=(n, 3))
    # Log-uniform magnitude: mostly steep, with a share of near-grazing rays.
    dz_mag = 10.0 ** rng.uniform(dz_log_lo, 0.0, n)
    directions[:, 2] = -side * dz_mag
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origins, directions


class TestBruteForceAgreement:
    """Random rays, above and below, steep to grazing: same mask, t, normals."""

    def test_matches_brute_force(self):
        pitch_x, pitch_y, radius, conic = 1.7, 1.3, 6.0, -0.3
        num_x, num_y = 4, 3
        half_x = num_x * pitch_x / 2.0
        half_y = num_y * pitch_y / 2.0

        g = LensletArrayGeometry(
            pitch_x=pitch_x, pitch_y=pitch_y, radius=radius, conic=conic,
            num_x=num_x, num_y=num_y,
        )
        origins, directions = _random_rays(
            6000, seed=0, half_x=half_x, half_y=half_y, z_lo=1.5, z_hi=5.0,
            dz_log_lo=-0.7, spread=1.05,
        )

        t, normals, hit, n_geom = g.ray_intersect(origins, directions)
        bt, bhit, bn = _brute_force(
            pitch_x, pitch_y, radius, conic, num_x, num_y, origins, directions
        )

        assert np.array_equal(hit, bhit)
        both = hit & bhit
        assert both.sum() > 100  # the comparison is only meaningful with hits
        np.testing.assert_allclose(t[both], bt[both], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(normals[both], bn[both], atol=1e-9)


class TestVertexHit:
    """A ray straight down each cell centre hits at the cap vertex."""

    def _check(self, backend, precision):
        be.set_backend(backend)
        be.set_precision(precision)
        try:
            pitch_x, pitch_y, radius, conic = 1.5, 1.5, 8.0, 0.0
            num_x, num_y = 3, 3
            g = LensletArrayGeometry(
                pitch_x=pitch_x, pitch_y=pitch_y, radius=radius, conic=conic,
                num_x=num_x, num_y=num_y,
            )
            D = 12.0
            cells = [(i, j) for i in range(num_x) for j in range(num_y)]
            origins = np.zeros((len(cells), 3))
            for k, (i, j) in enumerate(cells):
                xc, yc = _cell_centre(i, j, num_x, num_y, pitch_x, pitch_y)
                origins[k] = [xc, yc, -D]
            directions = np.tile([0.0, 0.0, 1.0], (len(cells), 1))

            if backend == "torch":
                dtype = torch.float32 if precision == "float32" else torch.float64
                origins_b = torch.tensor(origins, dtype=dtype)
                directions_b = torch.tensor(directions, dtype=dtype)
            else:
                origins_b, directions_b = origins, directions

            t, normals, hit, n_geom = g.ray_intersect(origins_b, directions_b)
            t_np = be.to_numpy(t) if backend == "torch" else np.asarray(t)
            hit_np = be.to_numpy(hit) if backend == "torch" else np.asarray(hit)

            assert np.all(hit_np)
            tol = 1e-5 if precision == "float32" else 1e-9
            np.testing.assert_allclose(t_np, D, rtol=tol, atol=tol)
        finally:
            be.set_backend("numpy")
            be.set_precision("float64")

    def test_numpy_float64(self):
        self._check("numpy", "float64")

    def test_torch_float64(self):
        self._check("torch", "float64")

    def test_torch_float32(self):
        self._check("torch", "float32")


class TestGrazingMultiCellCrossing:
    """Some grazing rays are found only after crossing several cells."""

    def test_some_hits_cross_more_than_one_cell(self):
        pitch_x, pitch_y, radius, conic = 1.0, 1.0, 4.0, 0.0
        num_x, num_y = 5, 5
        half_x = num_x * pitch_x / 2.0
        half_y = num_y * pitch_y / 2.0

        g = LensletArrayGeometry(
            pitch_x=pitch_x, pitch_y=pitch_y, radius=radius, conic=conic,
            num_x=num_x, num_y=num_y,
        )
        origins, directions = _random_rays(
            6000, seed=1, half_x=half_x, half_y=half_y, z_lo=2.0, z_hi=8.0,
            dz_log_lo=-3.0, spread=1.2,
        )
        t, normals, hit, n_geom = g.ray_intersect(origins, directions)

        x_min = -(num_x * pitch_x) / 2.0
        y_min = -(num_y * pitch_y) / 2.0
        # The entry cell, from where each ray's footprint first reaches the
        # array's own coordinate origin plane (z=0) -- a coarse but valid
        # proxy for "which cell the footprint starts in", independent of
        # the geometry's own internal box-clip math.
        t0 = -origins[:, 2] / directions[:, 2]
        ex = origins[:, 0] + t0 * directions[:, 0]
        ey = origins[:, 1] + t0 * directions[:, 1]
        i0 = np.clip(np.floor((ex - x_min) / pitch_x), 0, num_x - 1)
        j0 = np.clip(np.floor((ey - y_min) / pitch_y), 0, num_y - 1)

        px = origins[:, 0] + t * directions[:, 0]
        py = origins[:, 1] + t * directions[:, 1]
        ih = np.clip(np.floor((px - x_min) / pitch_x), 0, num_x - 1)
        jh = np.clip(np.floor((py - y_min) / pitch_y), 0, num_y - 1)

        crossed = hit & ((i0 != ih) | (j0 != jh))
        assert hit.sum() > 0
        assert crossed.sum() > 0


class TestSagOffsets:
    """A per-cell sag offset moves a vertical ray's hit by exactly that offset."""

    def test_offset_shifts_vertical_hit(self):
        pitch_x, pitch_y, radius, conic = 1.2, 1.2, 6.0, 0.0
        num_x, num_y = 2, 2
        rng = np.random.default_rng(2)
        offsets = rng.uniform(-0.3, 0.3, size=(num_y, num_x))

        g_plain = LensletArrayGeometry(
            pitch_x=pitch_x, pitch_y=pitch_y, radius=radius, conic=conic,
            num_x=num_x, num_y=num_y,
        )
        g_offset = LensletArrayGeometry(
            pitch_x=pitch_x, pitch_y=pitch_y, radius=radius, conic=conic,
            num_x=num_x, num_y=num_y, sag_offsets=offsets,
        )

        cells = [(i, j) for i in range(num_x) for j in range(num_y)]
        origins = np.zeros((len(cells), 3))
        for k, (i, j) in enumerate(cells):
            xc, yc = _cell_centre(i, j, num_x, num_y, pitch_x, pitch_y)
            origins[k] = [xc, yc, -10.0]
        directions = np.tile([0.0, 0.0, 1.0], (len(cells), 1))

        t_plain, _, hit_plain, _ = g_plain.ray_intersect(origins, directions)
        t_off, _, hit_off, _ = g_offset.ray_intersect(origins, directions)

        assert np.all(hit_plain) and np.all(hit_off)
        for k, (i, j) in enumerate(cells):
            expected_shift = offsets[j, i]
            np.testing.assert_allclose(
                t_off[k] - t_plain[k], expected_shift, atol=1e-10
            )


class TestTracedFocus:
    """A collimated beam through a plano-convex lenslet array focuses per cell."""

    def test_flux_closes_and_spots_are_centred(self):
        pitch = 1.0
        num = 3
        radius = 5.0
        conic = 0.0
        n_glass = 1.5
        focal_length = radius / (n_glass - 1.0)
        thickness = 0.1  # thin compared to focal_length; > the cap's own sag
        footprint = num * pitch

        glass = NSQMaterial(optiland_material=IdealMaterial(n=n_glass, k=0.0))
        geom = LensletArrayGeometry(
            pitch_x=pitch, pitch_y=pitch, radius=radius, conic=conic,
            num_x=num, num_y=num,
        )

        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-10.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                # A circumscribing circle: every one of the 3x3 cells,
                # corners included, is fully illuminated ("square-ish").
                aperture_radius=(footprint / 2.0) * 2**0.5,
            ),
        )
        scene.add_component(
            "front",
            RefractiveComponent(
                cs=CoordinateSystem(z=0.0),
                geometry=geom,
                material_front=VACUUM,
                material_back=glass,
                name="front",
            ),
        )
        # The flat back face shares the front array's own rectangular
        # footprint, so a ray that misses the array (outside the 3x3
        # cells) also misses this plane rather than being treated as
        # arriving from inside glass it never entered.
        scene.add_component(
            "back",
            RefractiveComponent(
                cs=CoordinateSystem(z=thickness),
                geometry=FinitePlaneGeometry(width=footprint, height=footprint),
                material_front=glass,
                material_back=VACUUM,
                name="back",
            ),
        )
        det_width = footprint + 1.0
        scene.add_detector(
            "D",
            CoordinateSystem(z=focal_length),
            IrradianceDetectorConfig(
                width=det_width, height=det_width,
                num_pixels_x=240, num_pixels_y=240,
            ),
        )

        result = scene.trace(num_rays=20_000, seed=7, max_depth=8)
        assert result.flux_conservation_error < 1e-12

        det = result.detectors["D"]
        irr = np.asarray(det.irradiance)
        X, Y = np.meshgrid(det.x_coords, det.y_coords)
        half = pitch / 2.0
        for i in range(num):
            for j in range(num):
                xc = (i - (num - 1) / 2.0) * pitch
                yc = (j - (num - 1) / 2.0) * pitch
                mask = (np.abs(X - xc) <= half) & (np.abs(Y - yc) <= half)
                w = irr[mask]
                assert w.sum() > 0
                cx = (X[mask] * w).sum() / w.sum()
                cy = (Y[mask] * w).sum() / w.sum()
                assert abs(cx - xc) < 0.05 * pitch
                assert abs(cy - yc) < 0.05 * pitch


class TestIRLowering:
    """The kind lowers through optiland.nonsequential.ir.lower."""

    def test_kind_is_registered(self):
        reg = kinds.registered_kinds()
        assert "lenslet_array" in reg["geometry"]

    def test_lower_geometry_params(self):
        offsets = [[0.0, 0.1], [0.2, 0.0], [0.0, -0.05]]
        g = LensletArrayGeometry(
            pitch_x=1.0, pitch_y=1.5, radius=5.0, conic=-0.5,
            num_x=2, num_y=3, sag_offsets=offsets,
        )
        kind, params = _lower_geometry(g)
        assert kind == "lenslet_array"
        assert params == {
            "pitch_x": 1.0, "pitch_y": 1.5, "radius": 5.0, "conic": -0.5,
            "num_x": 2, "num_y": 3, "sag_offsets": offsets,
        }

        g_default = LensletArrayGeometry(
            pitch_x=1.0, pitch_y=1.5, radius=5.0, conic=-0.5, num_x=2, num_y=3,
        )
        _, params_default = _lower_geometry(g_default)
        assert params_default["sag_offsets"] is None

    def test_scene_lowers_with_lenslet_array_primitive(self):
        geom = LensletArrayGeometry(
            pitch_x=1.0, pitch_y=1.0, radius=5.0, conic=0.0, num_x=3, num_y=3
        )
        glass = NSQMaterial.from_glass("N-BK7")
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-10.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55), total_flux=1.0,
                aperture_radius=2.0,
            ),
        )
        scene.add_component(
            "front",
            RefractiveComponent(
                cs=CoordinateSystem(z=0.0), geometry=geom,
                material_front=VACUUM, material_back=glass, name="front",
            ),
        )
        scene.add_detector(
            "D",
            CoordinateSystem(z=10.0),
            IrradianceDetectorConfig(
                width=5, height=5, num_pixels_x=10, num_pixels_y=10
            ),
        )
        ir = lower_scene(scene)
        assert [p.kind for p in ir.primitives] == ["lenslet_array"]
