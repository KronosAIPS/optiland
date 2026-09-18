"""The Harvey-Shack and tabulated lobes sample where the ray state is.

Both used to drop to host NumPy for the whole of ``sample()``: the normal
and the direction copied down, the inverse CDF read with ``numpy.interp``,
the measured grid read with a SciPy interpolator, and the result uploaded
again -- four device-to-host round trips per scattering surface per bounce
(``docs/theory/12_gpu_mapping.md`` R-12-1, R-12-4). They now run in
``optiland.backend`` operations against tables uploaded once.

Two things have to hold for that to be an improvement rather than a
rewrite: the lookups must agree with the reference implementations they
replace, to the precision of the arithmetic; and the returned arrays must
stay on the backend they were given.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

import optiland.backend as be
from optiland.nonsequential import HarveyShackBSDF, TabulatedBSDF
from optiland.nonsequential.rng import NSQRng

torch = pytest.importorskip("torch", reason="Torch not available")

EPS64 = np.finfo(np.float64).eps


@pytest.fixture(autouse=True)
def _numpy_float64():
    be.set_backend("numpy")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


@pytest.fixture
def scatter_table(tmp_path):
    """A small measured BSDF table, non-uniform in both angles."""
    path = tmp_path / "scatter.csv"
    rows = []
    theta_i = [0.0, 10.0, 45.0, 80.0]
    theta_s = [0.0, 5.0, 45.0, 70.0, 90.0]
    for i, ti in enumerate(theta_i):
        for j, ts in enumerate(theta_s):
            rows.append(f"{ti},{ts},{0.30 - 0.02 * i - 0.03 * j:.6f}")
    path.write_text("\n".join(rows) + "\n")
    return path, np.array(theta_i), np.array(theta_s)


class TestTheInverseCdfMatchesNumpyInterp:
    """The Harvey-Shack radial lookup, against the routine it replaces."""

    def test_agrees_to_a_few_ulps(self):
        bsdf = HarveyShackBSDF(b0=1e-3, l0=0.01, s=2.0)
        _ = bsdf.total_integrated_scatter  # builds the tables
        rng = np.random.default_rng(11)
        u = rng.random(50_000)

        got = np.asarray(bsdf._inverse_cdf(u))
        want = np.interp(u, bsdf._cdf_grid, bsdf._beta_grid)

        # Both are one linear interpolation inside one cell: a handful of
        # rounding steps, so a few ulps of the result.
        np.testing.assert_allclose(got, want, rtol=8 * EPS64, atol=8 * EPS64)

    def test_the_end_points_are_exact(self):
        bsdf = HarveyShackBSDF(b0=1e-3, l0=0.01, s=2.0)
        _ = bsdf.total_integrated_scatter
        ends = np.array([0.0, 1.0])
        got = np.asarray(bsdf._inverse_cdf(ends))
        assert got[0] == bsdf._beta_grid[0]
        assert got[1] == pytest.approx(bsdf._beta_grid[-1], rel=8 * EPS64)

    def test_the_lookup_runs_on_torch(self):
        bsdf = HarveyShackBSDF(b0=1e-3, l0=0.01, s=2.0)
        _ = bsdf.total_integrated_scatter
        u_np = np.random.default_rng(3).random(1000)
        want = np.asarray(bsdf._inverse_cdf(u_np))

        be.set_backend("torch")
        be.set_precision("float64")
        got = bsdf._inverse_cdf(torch.as_tensor(u_np, dtype=torch.float64))
        assert isinstance(got, torch.Tensor)
        np.testing.assert_allclose(
            got.numpy(), want, rtol=8 * EPS64, atol=8 * EPS64
        )


class TestTheTabulatedLookupMatchesTheInterpolator:
    """The measured-grid lookup, against SciPy's regular-grid interpolator."""

    def test_agrees_inside_the_table(self, scatter_table):
        path, theta_i, theta_s = scatter_table
        bsdf = TabulatedBSDF(path)
        reference = RegularGridInterpolator(
            (theta_i, theta_s),
            bsdf._grid_flat.reshape(theta_i.size, theta_s.size),
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        rng = np.random.default_rng(5)
        qi = rng.uniform(theta_i[0], theta_i[-1], 20_000)
        qs = rng.uniform(theta_s[0], theta_s[-1], 20_000)

        got = np.asarray(bsdf._evaluate(qi, qs))
        want = reference(np.column_stack([qi, qs]))
        np.testing.assert_allclose(got, want, rtol=32 * EPS64, atol=32 * EPS64)

    def test_outside_the_table_is_zero(self, scatter_table):
        path, theta_i, theta_s = scatter_table
        bsdf = TabulatedBSDF(path)
        qi = np.array([-1.0, 0.0, 40.0, 80.0, 81.0, 40.0])
        qs = np.array([40.0, 40.0, -0.5, 40.0, 40.0, 91.0])
        got = np.asarray(bsdf._evaluate(qi, qs))
        assert got[0] == 0.0
        assert got[2] == 0.0
        assert got[4] == 0.0
        assert got[5] == 0.0
        assert got[1] > 0.0
        assert got[3] > 0.0

    def test_the_grid_nodes_come_back_exactly(self, scatter_table):
        path, theta_i, theta_s = scatter_table
        bsdf = TabulatedBSDF(path)
        grid = bsdf._grid_flat.reshape(theta_i.size, theta_s.size)
        qi, qs = np.meshgrid(theta_i, theta_s, indexing="ij")
        got = np.asarray(bsdf._evaluate(qi.ravel(), qs.ravel()))
        np.testing.assert_allclose(got, grid.ravel(), rtol=8 * EPS64, atol=0.0)

    def test_the_lookup_runs_on_torch(self, scatter_table):
        path, theta_i, theta_s = scatter_table
        bsdf = TabulatedBSDF(path)
        qi = np.linspace(0.0, 80.0, 500)
        qs = np.linspace(0.0, 90.0, 500)
        want = np.asarray(bsdf._evaluate(qi, qs))

        be.set_backend("torch")
        be.set_precision("float64")
        got = bsdf._evaluate(
            torch.as_tensor(qi, dtype=torch.float64),
            torch.as_tensor(qs, dtype=torch.float64),
        )
        assert isinstance(got, torch.Tensor)
        np.testing.assert_allclose(
            got.numpy(), want, rtol=32 * EPS64, atol=32 * EPS64
        )


def _sample_on_torch(bsdf, n_rays: int = 4096):
    """Sample the lobe with every input already a tensor."""
    be.set_backend("torch")
    be.set_precision("float64")
    normals = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).repeat(n_rays, 1)
    dirs = torch.tensor([0.0, 0.3, -0.9539392], dtype=torch.float64).repeat(
        n_rays, 1
    )
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    wavelengths = torch.full((n_rays,), 0.55, dtype=torch.float64)
    ray_id = torch.arange(n_rays, dtype=torch.int64)
    bounce = torch.zeros(n_rays, dtype=torch.int32)
    return bsdf.sample(
        n_rays, dirs, normals, wavelengths, NSQRng(9), ray_id, bounce
    )


class TestTheLobesStayOnTheBackend:
    """R-12-1: nothing a sampler returns comes back as a host array."""

    def test_harvey_shack(self):
        bsdf = HarveyShackBSDF(b0=1e-3, l0=0.05, s=2.0, transmissive_fraction=0.3)
        dirs, weights, transmitted = _sample_on_torch(bsdf)
        assert isinstance(dirs, torch.Tensor)
        assert isinstance(weights, torch.Tensor)
        assert isinstance(transmitted, torch.Tensor)
        assert transmitted.dtype == torch.bool
        norms = dirs.norm(dim=1)
        # Every reachable sample is a unit vector; an unreachable one keeps
        # the reference direction, which is also a unit vector.
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=0.0, atol=64 * EPS64
        )
        assert float(transmitted.double().mean()) == pytest.approx(0.3, abs=0.03)

    def test_tabulated(self, scatter_table):
        path, _ti, _ts = scatter_table
        bsdf = TabulatedBSDF(path, transmissive_fraction=0.4)
        dirs, weights, transmitted = _sample_on_torch(bsdf)
        assert isinstance(dirs, torch.Tensor)
        assert isinstance(weights, torch.Tensor)
        assert isinstance(transmitted, torch.Tensor)
        assert transmitted.dtype == torch.bool
        norms = dirs.norm(dim=1)
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=0.0, atol=64 * EPS64
        )
        assert float(weights.min()) >= 0.0
        assert float(weights.max()) <= 1.0
        assert float(transmitted.double().mean()) == pytest.approx(0.4, abs=0.03)

    def test_tabulated_reflectance(self, scatter_table):
        path, _ti, _ts = scatter_table
        bsdf = TabulatedBSDF(path)
        be.set_backend("torch")
        be.set_precision("float64")
        n = 128
        dirs = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64).repeat(n, 1)
        normals = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).repeat(n, 1)
        wl = torch.full((n,), 0.55, dtype=torch.float64)
        refl = bsdf.reflectance(dirs, normals, wl)
        assert isinstance(refl, torch.Tensor)
        assert refl.shape == (n,)
        assert float(refl.min()) >= 0.0
        assert float(refl.max()) <= 1.0

    def test_the_numpy_path_still_returns_numpy(self, scatter_table):
        """The control: the same samplers on the host backend."""
        path, _ti, _ts = scatter_table
        n = 256
        normals = np.tile([0.0, 0.0, 1.0], (n, 1))
        dirs = np.tile([0.0, 0.3, -0.9539392], (n, 1))
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        wl = np.full(n, 0.55)
        for bsdf in (
            HarveyShackBSDF(b0=1e-3, l0=0.05, s=2.0),
            TabulatedBSDF(path),
        ):
            out, weights, transmitted = bsdf.sample(
                n,
                dirs,
                normals,
                wl,
                NSQRng(1),
                np.arange(n),
                np.zeros(n, dtype=np.int32),
            )
            assert isinstance(out, np.ndarray)
            assert isinstance(weights, np.ndarray)
            assert isinstance(transmitted, np.ndarray)
