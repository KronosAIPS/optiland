"""The sequential NurbsGeometry's derivative of t on the torch backend.

The research repository's issue 72: the basis is evaluated in NumPy, so the
converged (u, v) carried no gradient and dt/dtheta was the derivative at a hit
point held at fixed (u, v). ``distance`` now adds one zero-valued Newton step on
``S(u, v) - P0 - t d`` with the Jacobian detached (the non-sequential NURBS
kind's adjoint), so the value is unchanged and the derivative is the implicit
one. Checked against a central difference of the geometry's own root.

The window: with h = 1e-5 the central difference carries a truncation error of
order h^2 |t'''| / 6 (about 1e-10 here) and the root's noise, the Newton
tolerance 1e-14 over |cos| >= 0.9 divided by 2h (about 6e-10); 1e-6 of the
largest derivative is three decades above both, and the detached derivative
misses it by 6 percent (control point) and 43 percent (weight) of that scale.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be

torch = pytest.importorskip("torch")

from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.geometries.nurbs.nurbs_geometry import NurbsGeometry  # noqa: E402
from optiland.rays import RealRays  # noqa: E402

_G = np.linspace(-8.0, 8.0, 4)
_X, _Y = np.meshgrid(_G, _G, indexing="ij")
_Z = np.array(
    [
        [0.0, 0.4, 0.3, 0.0],
        [0.5, 2.0, 1.6, 0.2],
        [0.2, 1.7, 2.4, 0.6],
        [0.0, 0.3, 0.5, 0.1],
    ]
)
P = np.stack([_X, _Y, _Z])  # a bicubic bump of a few mm over [-8, 8]^2
W = np.array(
    [
        [1.0, 0.8, 1.2, 1.0],
        [0.9, 1.6, 0.7, 1.1],
        [1.0, 1.3, 1.5, 0.9],
        [1.0, 1.0, 0.8, 1.0],
    ]
)
KNOTS = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
N_RAYS = 60
H = 1e-5


def _rays():
    rng = np.random.default_rng(5)
    x = rng.uniform(-5, 5, N_RAYS)
    y = rng.uniform(-5, 5, N_RAYS)
    th = np.radians(rng.uniform(0, 25, N_RAYS))
    ph = rng.uniform(0, 2 * np.pi, N_RAYS)
    L, M, N = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
    return x, y, L, M, N


@pytest.fixture
def torch64():
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


def _geometry(dz=0.0, dw=0.0, grad=False):
    zs = torch.tensor(float(dz), dtype=torch.float64, requires_grad=grad)
    ws = torch.tensor(float(dw), dtype=torch.float64, requires_grad=grad)
    e = torch.zeros(3, 4, 4, dtype=torch.float64)
    e[2, 1, 2] = 1.0
    f = torch.zeros(4, 4, dtype=torch.float64)
    f[2, 1] = 1.0
    geo = NurbsGeometry(
        CoordinateSystem(),
        control_points=torch.tensor(P) + zs * e,
        weights=torch.tensor(W) + ws * f,
        u_degree=3,
        v_degree=3,
        u_knots=torch.tensor(KNOTS),
        v_knots=torch.tensor(KNOTS),
        tol=1e-14,
        max_iter=100,
    )
    return geo, zs, ws


def _distance(geo):
    x, y, L, M, N = _rays()
    rays = RealRays(x, y, np.full(N_RAYS, -30.0), L, M, N, np.ones(N_RAYS), 0.55)
    return geo.distance(rays)


def _hits(t):
    # the converged point lies on the surface (a ray whose root is off the
    # patch is pinned at its rim and is not a hit)
    x, y, L, M, N = _rays()
    t = be.to_numpy(t)
    px, py, pz = x + t * L, y + t * M, -30.0 + t * N
    geo, _, _ = _geometry()
    return np.abs(be.to_numpy(geo.sag(px, py)) - pz) < 1e-9


@pytest.mark.parametrize("which", ["control_point_z", "weight"])
def test_derivative_of_t_is_the_implicit_one(torch64, which):
    geo, zs, ws = _geometry(grad=True)
    t = _distance(geo)
    param = zs if which == "control_point_z" else ws
    grad = np.array(
        [
            torch.autograd.grad(t[i], param, retain_graph=True)[0].item()
            for i in range(N_RAYS)
        ]
    )
    kw = "dz" if which == "control_point_z" else "dw"
    plus = be.to_numpy(_distance(_geometry(**{kw: H})[0]))
    minus = be.to_numpy(_distance(_geometry(**{kw: -H})[0]))
    fd = (plus - minus) / (2 * H)
    hit = _hits(t.detach())
    assert hit.sum() >= N_RAYS // 2
    scale = np.abs(fd[hit]).max()
    assert scale > 1e-3
    assert np.abs(grad - fd)[hit].max() <= 1e-6 * scale


def test_value_is_unchanged_by_the_attached_step(torch64):
    geo, _, _ = _geometry(grad=True)
    attached = be.to_numpy(_distance(geo).detach())
    plain = be.to_numpy(_distance(_geometry(grad=False)[0]))
    assert np.array_equal(attached.view(np.int64), plain.view(np.int64))


def test_numpy_backend_is_untouched():
    be.set_backend("numpy")
    geo = NurbsGeometry(
        CoordinateSystem(),
        control_points=P,
        weights=W,
        u_degree=3,
        v_degree=3,
        u_knots=KNOTS,
        v_knots=KNOTS,
        tol=1e-14,
        max_iter=100,
    )
    t = _distance(geo)
    assert np.all(np.isfinite(t))
