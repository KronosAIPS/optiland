"""The component-intersection stage as one Warp kernel per component (prototype).

The eager torch bounce intersects every component through
:meth:`~optiland.nonsequential.components.base.BaseComponent.intersect`: a
frame transform, the origin advance, the per-ray accept threshold, the
geometry's own root solve and root selection, and the transform of the two
normals back to the global frame -- about 160 dispatched torch operations
for the integrating sphere's ported cavity and about 185 for a conic face, each one a
kernel launch on a device. This module computes the same stage in one Warp
kernel launch per component, for the two analytic kinds it covers
(:class:`~optiland.nonsequential.components.geometry.analytic.spherical_cavity
.SphericalCavityGeometry` and :class:`~optiland.nonsequential.components
.geometry.analytic.conic.ConicGeometry`), and hands every other component to
its own ``intersect``.

**The interface with the torch loop.** What crosses into a kernel is the ray
state the torch stage reads (``x, y, z, L, M, N`` and ``alive``), the
per-ray accept threshold (computed once per bounce in torch, where the stage
used to compute it once per component), the component's placement as twelve
numbers (translation and rotation) and the geometry's scalars, each formed on
the host by exactly the expression the torch stage evaluates. What comes back
is exactly what ``BaseComponent.intersect`` returns (``t, normals, hit_mask,
n_geom``) and the two halves of the hit distance it stores for
``advance_to_hit`` (``_local_root``). The nearest-hit select, the
interaction, the detectors and compaction stay in torch and see ordinary
tensors, so compaction and the CUDA-graph replay treat the stage like any
other operation: the launch runs on torch's current CUDA stream and the
kernels are loaded before the first bounce (:func:`prepare`), so a recorded
bounce records the launch.

**Same numbers.** The kernels are compiled with floating-point contraction
off (Warp's ``fuse_fp`` module option), so no multiply-add is fused that the
torch stage does not fuse, and every expression is evaluated in the order the
torch stage evaluates it; the scalars a torch expression takes from Python
floats are cast to the working dtype on the host as torch casts them. Three
orders are torch's own and are reproduced from measurement rather than from
the source text: the matrix product of the frame transform is a chain of
fused multiply-adds (written with an explicit ``fma``); a sum over the last
axis of an (N, 3) array is ordered by the device's vector reduction (probed
at load, :func:`sum_order`); and a division by a Python number may be a
product with the rounded reciprocal (probed at load, :func:`scalar_division`).
On the Apple silicon CPU the stage is bit-identical to ``BaseComponent
.intersect`` at float64 and float32, rotated placements included. On CUDA
the first A100 run found one-ulp differences in the cavity's normals and
larger ones for a rotated placement (cuBLAS orders the transform otherwise);
see the tests.

**Gradients.** In gradient mode the launch is recorded on a Warp tape inside
a ``torch.autograd.Function``, and the backward pass replays the tape's
adjoint kernels: the gradient reaches the ray state, the placement and the
geometry's parameters (through the host-formed scalars, whose own graph is
torch's). The hit mask is a decision and carries none.

Use it through the torch backend, ``TorchBackend(intersect_kernel="warp")``;
like the Warp generator it is used only when Warp imports and the device is
CUDA, and the result's ``environment`` says what ran.
"""

import math
from typing import Any

import torch
import warp as wp

import optiland.backend as be
from optiland.nonsequential import _tol

#: The device types on which the torch backend uses the kernels. Warp has no
#: Metal backend, and on the CPU Warp runs one serial thread.
SUPPORTED_DEVICE_TYPES: tuple[str, ...] = ("cuda",)

# No fused multiply-add unless written: the torch stage rounds every product
# of its elementwise arithmetic before the sum, on the CPU and on CUDA alike.
# The one place torch does fuse is its matrix product (the frame transform),
# which is written with an explicit fused multiply-add below.
wp.set_module_options({"fuse_fp": False, "enable_backward": True})

_FMA_SNIPPET = "return fma(a, b, c);"
_FMA_ADJ = "adj_a += b * adj_ret; adj_b += a * adj_ret; adj_c += adj_ret;"


@wp.func_native(_FMA_SNIPPET, _FMA_ADJ)
def _fma64(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64: ...


@wp.func_native(_FMA_SNIPPET, _FMA_ADJ)
def _fma32(a: wp.float32, b: wp.float32, c: wp.float32) -> wp.float32: ...

#: Layout of the placement vector: translation (3), then the rotation matrix
#: row-major (9), local = (global - T) @ R, global normal = local normal @ R^T.
_XF = 12


def _make_kernels(FT, fma):
    """The two intersection kernels for one float type.

    Two orders are the torch stage's own and are reproduced here, both
    measured on the Apple silicon CPU with torch 2.14:

    - the (N, 3) @ (3, 3) matrix product is ``fma(a2, R2j, fma(a1, R1j,
      a0 * R0j))`` for every entry, at both precisions;
    - a sum over the last axis of an (N, 3) array (``.sum(axis=1)``) is
      ``(x + z) + y`` at float64 and ``(x + y) + z`` at float32, the lane
      order of the CPU's vector reduction. The order is a kernel argument
      (``sum_order``), probed from torch on the device at load time
      (:func:`sum_order`), so a device whose reduction orders the terms
      otherwise gets its own.
    """

    @wp.func
    def _sum3(x: FT, y: FT, z: FT, order: int):
        if order == 1:
            return (x + z) + y
        return (x + y) + z

    @wp.func
    def _div_scalar(x: FT, s: FT, inv_s: FT, recip: int):
        if recip == 1:
            return x * inv_s
        return x / s

    @wp.func
    def _mm(a0: FT, a1: FT, a2: FT, r0: FT, r1: FT, r2: FT):
        return fma(a2, r2, fma(a1, r1, a0 * r0))

    @wp.func
    def _to_local(
        x: FT, y: FT, z: FT, xf: wp.array(dtype=FT)
    ):
        px = x - xf[0]
        py = y - xf[1]
        pz = z - xf[2]
        lx = _mm(px, py, pz, xf[3], xf[6], xf[9])
        ly = _mm(px, py, pz, xf[4], xf[7], xf[10])
        lz = _mm(px, py, pz, xf[5], xf[8], xf[11])
        return lx, ly, lz

    @wp.func
    def _dir_local(
        x: FT, y: FT, z: FT, xf: wp.array(dtype=FT)
    ):
        lx = _mm(x, y, z, xf[3], xf[6], xf[9])
        ly = _mm(x, y, z, xf[4], xf[7], xf[10])
        lz = _mm(x, y, z, xf[5], xf[8], xf[11])
        return lx, ly, lz

    @wp.func
    def _to_global_normal(
        x: FT, y: FT, z: FT, xf: wp.array(dtype=FT)
    ):
        gx = _mm(x, y, z, xf[3], xf[4], xf[5])
        gy = _mm(x, y, z, xf[6], xf[7], xf[8])
        gz = _mm(x, y, z, xf[9], xf[10], xf[11])
        return gx, gy, gz

    @wp.func
    def _cavity_root_valid(
        t: FT,
        disc_ok: bool,
        eps: FT,
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT,
        radius: FT,
        inv_radius: FT,
        recip: int,
        ports: wp.array(dtype=FT),
        nports: int,
    ):
        forward = disc_ok and (t > eps)
        safe_t = FT(0.0)
        if forward:
            safe_t = t
        hx = ox + safe_t * dx
        hy = oy + safe_t * dy
        hz = oz + safe_t * dz
        in_port = int(0)
        for p in range(nports):
            cos_angle = _div_scalar(
                (hx * ports[4 * p] + hy * ports[4 * p + 1]) + hz * ports[4 * p + 2],
                radius, inv_radius, recip,
            )
            if cos_angle >= ports[4 * p + 3]:
                in_port = 1
        return forward and (in_port == 0)

    @wp.kernel
    def k_cavity(
        x: wp.array(dtype=FT), y: wp.array(dtype=FT), z: wp.array(dtype=FT),
        L: wp.array(dtype=FT), M: wp.array(dtype=FT), N: wp.array(dtype=FT),
        alive: wp.array(dtype=wp.bool),
        t_min: wp.array(dtype=FT),
        xf: wp.array(dtype=FT),
        gp: wp.array(dtype=FT),
        ports: wp.array(dtype=FT),
        nports: int,
        sum_order: int,
        recip: int,
        t_hit: wp.array(dtype=FT),
        normals: wp.array2d(dtype=FT),
        hit: wp.array(dtype=wp.bool),
        n_geom: wp.array2d(dtype=FT),
        t_adv_out: wp.array(dtype=FT),
        t_local_out: wp.array(dtype=FT),
    ):
        # gp: radius, radius**2, radicand floor, inf, 1 / radius
        i = wp.tid()
        radius = gp[0]
        inv_radius = gp[4]
        r2 = gp[1]
        floor = gp[2]
        inf = gp[3]
        plx, ply, plz = _to_local(x[i], y[i], z[i], xf)
        dx, dy, dz = _dir_local(L[i], M[i], N[i], xf)
        t_adv = -_sum3(plx * dx, ply * dy, plz * dz, sum_order)
        ox = plx + t_adv * dx
        oy = ply + t_adv * dy
        oz = plz + t_adv * dz
        eps = t_min[i] - t_adv

        b = FT(2.0) * ((ox * dx + oy * dy) + oz * dz)
        c = ((ox * ox + oy * oy) + oz * oz) - r2
        disc = b * b - FT(4.0) * c
        disc_ok = disc >= FT(0.0)
        sqrt_disc = FT(0.0)
        if disc_ok:
            sqrt_disc = wp.sqrt(wp.max(disc, floor))
        t_near = (-b - sqrt_disc) / FT(2.0)
        t_far = (-b + sqrt_disc) / FT(2.0)
        use_near = _cavity_root_valid(t_near, disc_ok, eps, ox, oy, oz, dx, dy, dz, radius, inv_radius, recip, ports, nports)
        use_far = _cavity_root_valid(t_far, disc_ok, eps, ox, oy, oz, dx, dy, dz, radius, inv_radius, recip, ports, nports)
        use_far = use_far and (not use_near)
        hit_l = use_near or use_far
        t = inf
        if use_far:
            t = t_far
        if use_near:
            t = t_near
        safe_t = FT(0.0)
        if hit_l:
            safe_t = t
        hx = ox + safe_t * dx
        hy = oy + safe_t * dy
        hz = oz + safe_t * dz
        nx = FT(0.0)
        ny = FT(0.0)
        nz = FT(0.0)
        if hit_l:
            nx = _div_scalar(hx, radius, inv_radius, recip)
            ny = _div_scalar(hy, radius, inv_radius, recip)
            nz = _div_scalar(hz, radius, inv_radius, recip)
        dot = (dx * nx + dy * ny) + dz * nz
        flip = FT(1.0)
        if dot > FT(0.0):
            flip = FT(-1.0)

        th = t + t_adv
        accepted = th > t_min[i]
        if not accepted:
            th = inf
        if not alive[i]:
            th = inf
        t_hit[i] = th
        hit[i] = hit_l and accepted and alive[i]
        gx, gy, gz = _to_global_normal(nx * flip, ny * flip, nz * flip, xf)
        normals[i, 0] = gx
        normals[i, 1] = gy
        normals[i, 2] = gz
        qx, qy, qz = _to_global_normal(-nx, -ny, -nz, xf)
        n_geom[i, 0] = qx
        n_geom[i, 1] = qy
        n_geom[i, 2] = qz
        t_adv_out[i] = t_adv
        t_local_out[i] = t

    @wp.func
    def _conic_root_valid(
        t: FT,
        solvable: bool,
        eps: FT,
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT,
        ap2: FT, kc: FT,
    ):
        px = ox + t * dx
        py = oy + t * dy
        pz = oz + t * dz
        in_aperture = (px * px + py * py) <= ap2
        on_sheet = (FT(1.0) - kc * pz) >= FT(0.0)
        valid = solvable and wp.isfinite(t) and (t > eps) and in_aperture and on_sheet
        return valid, px, py

    @wp.kernel
    def k_conic(
        x: wp.array(dtype=FT), y: wp.array(dtype=FT), z: wp.array(dtype=FT),
        L: wp.array(dtype=FT), M: wp.array(dtype=FT), N: wp.array(dtype=FT),
        alive: wp.array(dtype=wp.bool),
        t_min: wp.array(dtype=FT),
        xf: wp.array(dtype=FT),
        gp: wp.array(dtype=FT),
        sum_order: int,
        t_hit: wp.array(dtype=FT),
        normals: wp.array2d(dtype=FT),
        hit: wp.array(dtype=wp.bool),
        n_geom: wp.array2d(dtype=FT),
        t_adv_out: wp.array(dtype=FT),
        t_local_out: wp.array(dtype=FT),
    ):
        # gp: c, 1 + K, aperture**2, (1 + K) c, (1 + K) c**2, -c,
        #     radicand floor, tiny, inf
        i = wp.tid()
        c = gp[0]
        kp = gp[1]
        ap2 = gp[2]
        kc = gp[3]
        kc2 = gp[4]
        negc = gp[5]
        floor = gp[6]
        tiny = gp[7]
        inf = gp[8]
        plx, ply, plz = _to_local(x[i], y[i], z[i], xf)
        dx, dy, dz = _dir_local(L[i], M[i], N[i], xf)
        t_adv = -_sum3(plx * dx, ply * dy, plz * dz, sum_order)
        ox = plx + t_adv * dx
        oy = ply + t_adv * dy
        oz = plz + t_adv * dz
        eps = t_min[i] - t_adv

        a = c * ((dx * dx + dy * dy) + kp * (dz * dz))
        b = FT(2.0) * (c * ((ox * dx + oy * dy) + (kp * oz) * dz) - dz)
        c0 = c * ((ox * ox + oy * oy) + kp * (oz * oz)) - FT(2.0) * oz
        disc = b * b - (FT(4.0) * a) * c0
        disc_ok = disc >= FT(0.0)
        sqrt_disc = FT(0.0)
        if disc_ok:
            sqrt_disc = wp.sqrt(wp.max(disc, floor))
        sign_b = FT(-1.0)
        if b >= FT(0.0):
            sign_b = FT(1.0)
        q = FT(-0.5) * (b + sign_b * sqrt_disc)
        a_ok = wp.abs(a) > tiny
        q_ok = wp.abs(q) > tiny
        a_den = FT(1.0)
        if a_ok:
            a_den = a
        q_den = FT(1.0)
        if q_ok:
            q_den = q
        t1 = q / a_den
        t2 = c0 / q_den
        valid1, px1, py1 = _conic_root_valid(t1, disc_ok and a_ok, eps, ox, oy, oz, dx, dy, dz, ap2, kc)
        valid2, px2, py2 = _conic_root_valid(t2, disc_ok and q_ok, eps, ox, oy, oz, dx, dy, dz, ap2, kc)
        pick1 = valid1 and ((not valid2) or (t1 <= t2))
        pick2 = valid2 and (not pick1)
        hit_l = pick1 or pick2
        t = FT(0.0)
        px = FT(0.0)
        py = FT(0.0)
        if pick2:
            t = t2
            px = px2
            py = py2
        if pick1:
            t = t1
            px = px1
            py = py1
        t_out = inf
        if hit_l:
            t_out = t

        r2 = px * px + py * py
        s = wp.sqrt(wp.max(FT(1.0) - kc2 * r2, floor))
        gx = (negc * px) / s
        gy = (negc * py) / s
        gz = FT(1.0)
        n_len = wp.sqrt(_sum3(gx * gx, gy * gy, gz * gz, sum_order))
        den = n_len + tiny
        ngx = gx / den
        ngy = gy / den
        ngz = gz / den
        dot = _sum3(dx * ngx, dy * ngy, dz * ngz, sum_order)
        nlx = ngx
        nly = ngy
        nlz = ngz
        if dot > FT(0.0):
            nlx = -ngx
            nly = -ngy
            nlz = -ngz

        th = t_out + t_adv
        accepted = th > t_min[i]
        if not accepted:
            th = inf
        if not alive[i]:
            th = inf
        t_hit[i] = th
        hit[i] = hit_l and accepted and alive[i]
        ax, ay, az = _to_global_normal(nlx, nly, nlz, xf)
        normals[i, 0] = ax
        normals[i, 1] = ay
        normals[i, 2] = az
        bx, by, bz = _to_global_normal(ngx, ngy, ngz, xf)
        n_geom[i, 0] = bx
        n_geom[i, 1] = by
        n_geom[i, 2] = bz
        t_adv_out[i] = t_adv
        t_local_out[i] = t_out

    return {"cavity": k_cavity, "conic": k_conic}


_KERNELS = {
    torch.float64: _make_kernels(wp.float64, _fma64),
    torch.float32: _make_kernels(wp.float32, _fma32),
}
_SUM_ORDER: dict = {}


def sum_order(dtype: torch.dtype, device: Any) -> int:
    """How torch orders a three-term sum over the last axis on this device and dtype.

    0 for ``(x + y) + z``, 1 for ``(x + z) + y``. Probed once with the row
    ``(s, 1, s)``, ``s`` half an ulp of 1: summed as ``(s + 1) + s`` it rounds
    to 1 twice (ties to even), summed as ``(s + s) + 1`` it is ``1 + 2 s``
    exactly. Cached per dtype and device.

    Args:
        dtype: The working float dtype.
        device: The torch device.

    Returns:
        The order flag the kernels take.
    """
    key = (dtype, str(device))
    order = _SUM_ORDER.get(key)
    if order is None:
        half_ulp = 2.0**-53 if dtype == torch.float64 else 2.0**-24
        probe = torch.tensor([[half_ulp, 1.0, half_ulp]], dtype=dtype, device=device)
        order = 0 if float(probe.sum(dim=1)[0]) == 1.0 else 1
        _SUM_ORDER[key] = order
    return order


_SCALAR_DIVISION: dict = {}


def scalar_division(dtype: torch.dtype, device: Any) -> int:
    """How torch divides a tensor by a Python number on this device and dtype.

    0 for a true division ``x / s``, 1 for a product with the reciprocal
    ``x * (1 / s)`` (the reciprocal rounded once in the working dtype). The
    cavity's two divisions by its radius are written against a Python float,
    so the kernel follows whichever the device does. Probed once on 4,096
    values divided by 3, many of which round differently under the two, and
    cached; a device that matches neither keeps the true division.

    Args:
        dtype: The working float dtype.
        device: The torch device.

    Returns:
        The flag the cavity kernel takes.
    """
    import numpy as np  # noqa: PLC0415

    key = (dtype, str(device))
    mode = _SCALAR_DIVISION.get(key)
    if mode is None:
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        x = (np.arange(1, 4097, dtype=np.float64) * 0.7310585786300049).astype(np_dtype)
        divisor = 3.0
        got = (torch.as_tensor(x, device=device) / divisor).cpu().numpy()
        true = x / np_dtype(divisor)
        recip = x * (np_dtype(1.0) / np_dtype(divisor))
        mode = 1 if (np.array_equal(got, recip) and not np.array_equal(got, true)) else 0
        _SCALAR_DIVISION[key] = mode
    return mode
_WP_FLOAT = {torch.float64: wp.float64, torch.float32: wp.float32}

_prepared: set = set()
_initialised = [False]


def _init() -> None:
    if not _initialised[0]:
        wp.init()
        _initialised[0] = True


def prepare(device: Any) -> None:
    """Load the kernels on ``device`` before any bounce (and any graph capture).

    Args:
        device: The torch device the ray arrays live on.
    """
    _init()
    tdev = torch.device(device)
    if tdev.type == "cuda" and tdev.index is None:
        tdev = torch.device("cuda", torch.cuda.current_device())
    name = str(tdev)
    if name in _prepared:
        return
    wp.load_module(module=__name__, device=wp.device_from_torch(tdev))
    # The sum-order probe reads a value to the host: done here, before any
    # bounce, never inside a recorded one.
    for dtype in (torch.float64, torch.float32):
        sum_order(dtype, tdev)
        scalar_division(dtype, tdev)
    _prepared.add(name)


def availability(device: Any) -> str | None:
    """Why the kernels cannot serve ``device``, or None when they can."""
    tdev = torch.device(device)
    if tdev.type not in SUPPORTED_DEVICE_TYPES:
        return (
            f"the Warp intersection kernels run on {', '.join(SUPPORTED_DEVICE_TYPES)} "
            f"only; the device is {tdev.type}"
        )
    try:
        _init()
        if tdev.type == "cuda" and not wp.is_cuda_available():
            return "Warp sees no CUDA device"
        prepare(tdev)
    except Exception as exc:  # noqa: BLE001 - any failure means "use the torch stage"
        return f"Warp could not load its kernels ({type(exc).__name__})"
    return None


# ---------------------------------------------------------------------------
# Which components the kernels take, and their scalars
# ---------------------------------------------------------------------------


def kind_of(component) -> str | None:
    """``"cavity"``, ``"conic"``, or None when the component keeps its own intersect."""
    from optiland.nonsequential.components.base import BaseComponent  # noqa: PLC0415
    from optiland.nonsequential.components.geometry.analytic.conic import (  # noqa: PLC0415
        ConicGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.spherical_cavity import (  # noqa: PLC0415
        SphericalCavityGeometry,
    )

    if type(component).intersect is not BaseComponent.intersect:
        return None
    geometry = getattr(component, "geometry", None)
    gtype = type(geometry)
    if (
        isinstance(geometry, SphericalCavityGeometry)
        and gtype.ray_intersect is SphericalCavityGeometry.ray_intersect
        and gtype._on_wall is SphericalCavityGeometry._on_wall
        and gtype._root_valid is SphericalCavityGeometry._root_valid
    ):
        return "cavity"
    if (
        isinstance(geometry, ConicGeometry)
        and gtype.ray_intersect is ConicGeometry.ray_intersect
        and gtype._root_valid is ConicGeometry._root_valid
        and gtype._normal_local is ConicGeometry._normal_local
        and gtype._curvature is ConicGeometry._curvature
    ):
        return "conic"
    return None


def _is_tensor(v) -> bool:
    return isinstance(v, torch.Tensor)


def _scalar_vector(values, like: torch.Tensor) -> torch.Tensor:
    """The geometry's scalars in the working dtype, each cast as torch casts a Python float.

    A value that is a tensor (a parameter carrying a gradient) is kept on
    its graph; the others are uploaded once per trace and cached by the
    caller.
    """
    if not any(_is_tensor(v) for v in values):
        return torch.tensor([float(v) for v in values], dtype=like.dtype, device=like.device)
    parts = [
        v.to(dtype=like.dtype, device=like.device).reshape(1)
        if _is_tensor(v)
        else torch.tensor([float(v)], dtype=like.dtype, device=like.device)
        for v in values
    ]
    return torch.cat(parts)


_FLOOR_TINY: dict = {}


def _floor_and_tiny(like: torch.Tensor) -> tuple[float, float]:
    """The radicand floor and the division guard of the working dtype, once per dtype."""
    cached = _FLOOR_TINY.get(like.dtype)
    if cached is None:
        ones = torch.ones((), dtype=like.dtype)
        cached = (float(_tol.radicand_floor(ones)), float(_tol.tiny_for(like)))
        _FLOOR_TINY[like.dtype] = cached
    return cached


def _cavity_scalars(geometry, like):
    floor, _ = _floor_and_tiny(like)
    radius = geometry.radius
    inv = 1.0 if _is_tensor(radius) else float(
        torch.tensor(1.0, dtype=like.dtype) / torch.tensor(float(radius), dtype=like.dtype)
    )
    return [radius, radius**2, floor, math.inf, inv]


def _cavity_ports(geometry, like):
    values = []
    for port in geometry.ports:
        ax, ay, az = port.unit_axis
        values += [ax, ay, az, port.cos_half_angle]
    if not values:
        values = [0.0, 0.0, 0.0, 2.0]
    return torch.tensor(values, dtype=like.dtype, device=like.device), len(geometry.ports)


def _conic_scalars(geometry, like):
    floor, tiny = _floor_and_tiny(like)
    c = geometry._curvature()
    K = geometry.conic
    kp = 1.0 + geometry.conic
    ap2 = geometry.aperture_radius**2
    kc = (1.0 + geometry.conic) * geometry._curvature()
    kc2 = (1.0 + K) * c**2
    negc = -c
    return [c, kp, ap2, kc, kc2, negc, floor, tiny, math.inf]


def _key(like: torch.Tensor) -> tuple:
    return (like.dtype, str(like.device))


def _component_inputs(component, kind: str, like: torch.Tensor):
    """Placement vector, geometry scalars and (cavity) port table for one component.

    Constant inputs are built once per trace (``reset``) and dtype/device,
    and cached on the component; a parameter carrying a gradient is rebuilt
    at every call so it stays on the graph.
    """
    from optiland.nonsequential.components.base import _resident_transform  # noqa: PLC0415

    t_be, r_be = _resident_transform(component)
    cache = getattr(component, "_warp_stage_cache", None)
    key = _key(like)
    if cache is None or cache.get("key") != key or cache.get("t_be") is not t_be or cache.get("r_be") is not r_be:
        cache = {"key": key, "t_be": t_be, "r_be": r_be}
        component._warp_stage_cache = cache
    if "xf" not in cache:
        xf = torch.cat([t_be.reshape(3), r_be.reshape(9)]).to(dtype=like.dtype, device=like.device)
        xf = xf.contiguous()
        if not xf.requires_grad:
            cache["xf"] = xf
    else:
        xf = cache["xf"]
    geometry = component.geometry
    if kind == "cavity":
        values = _cavity_scalars(geometry, like)
    else:
        values = _conic_scalars(geometry, like)
    if any(_is_tensor(v) for v in values):
        gp = _scalar_vector(values, like)
    else:
        gp = cache.get("gp")
        if gp is None or cache.get("gp_values") != values:
            gp = _scalar_vector(values, like)
            cache["gp"] = gp
            cache["gp_values"] = values
    ports, nports = None, 0
    if kind == "cavity":
        if "ports" not in cache:
            cache["ports"] = _cavity_ports(geometry, like)
        ports, nports = cache["ports"]
    return xf, gp, ports, nports


# ---------------------------------------------------------------------------
# Launch, the custom operator and the autograd function
# ---------------------------------------------------------------------------


def _outputs(n: int, like: torch.Tensor):
    return (
        torch.empty(n, dtype=like.dtype, device=like.device),
        torch.empty((n, 3), dtype=like.dtype, device=like.device),
        torch.empty(n, dtype=torch.bool, device=like.device),
        torch.empty((n, 3), dtype=like.dtype, device=like.device),
        torch.empty(n, dtype=like.dtype, device=like.device),
        torch.empty(n, dtype=like.dtype, device=like.device),
    )


def _launch(kind, inputs_t, outputs_t, nports, *, recip=0, requires_grad=False, tape=None):
    """Wrap the torch tensors as Warp arrays and launch on torch's stream.

    Returns the Warp arrays (inputs, outputs) so a tape can find their
    gradients.
    """
    _init()
    like = inputs_t[0]
    kernel = _KERNELS[like.dtype][kind]
    tdev = like.device
    wp_in = []
    for i, t in enumerate(inputs_t):
        if t is None:
            continue
        rg = requires_grad and t.dtype.is_floating_point and i != 7  # t_min is detached
        wp_in.append(wp.from_torch(t, requires_grad=rg))
    wp_out = [
        wp.from_torch(t, requires_grad=requires_grad and t.dtype.is_floating_point)
        for t in outputs_t
    ]
    args = list(wp_in)
    if kind == "cavity":
        args.append(int(nports))
    args.append(sum_order(like.dtype, like.device))
    if kind == "cavity":
        args.append(int(recip))
    n = int(like.shape[0])
    kwargs = {"dim": n, "inputs": args, "outputs": wp_out}
    if tdev.type == "cuda":
        kwargs["stream"] = wp.stream_from_torch(torch.cuda.current_stream(tdev))
    else:
        kwargs["device"] = wp.device_from_torch(tdev)
    if n > 0:
        if tape is not None:
            with tape:
                wp.launch(kernel, **kwargs)
        else:
            wp.launch(kernel, **kwargs)
    return wp_in, wp_out


@torch.library.custom_op("optiland_nsq::intersect_component", mutates_args=())
def _op_intersect(
    kind: str,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    L: torch.Tensor,
    M: torch.Tensor,
    N: torch.Tensor,
    alive: torch.Tensor,
    t_min: torch.Tensor,
    xf: torch.Tensor,
    gp: torch.Tensor,
    ports: torch.Tensor,
    nports: int,
    recip: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outs = _outputs(x.shape[0], x)
    ins = [x, y, z, L, M, N, alive, t_min, xf, gp] + ([ports] if kind == "cavity" else [])
    _launch(kind, ins, outs, nports, recip=recip)
    return outs


@_op_intersect.register_fake
def _(kind, x, y, z, L, M, N, alive, t_min, xf, gp, ports, nports, recip):
    return _outputs(x.shape[0], x)


class _IntersectFunction(torch.autograd.Function):
    """The kernel with its adjoint: the forward launch recorded on a Warp tape."""

    @staticmethod
    def forward(ctx, kind, nports, recip, x, y, z, L, M, N, alive, t_min, xf, gp, ports):
        ins = [t.detach().contiguous() for t in (x, y, z, L, M, N)]
        ins += [alive, t_min.detach(), xf.detach().contiguous(), gp.detach().contiguous()]
        if kind == "cavity":
            ins.append(ports)
        outs = _outputs(x.shape[0], x)
        tape = wp.Tape()
        wp_in, wp_out = _launch(kind, ins, outs, nports, recip=recip, requires_grad=True, tape=tape)
        ctx.tape = tape
        ctx.wp_in = wp_in
        ctx.wp_out = wp_out
        ctx.mark_non_differentiable(outs[2])
        return outs

    @staticmethod
    def backward(ctx, g_t, g_n, g_hit, g_ng, g_adv, g_loc):
        grads = {}
        for arr, g in zip(ctx.wp_out, (g_t, g_n, None, g_ng, g_adv, g_loc), strict=True):
            if g is None or not arr.requires_grad:
                continue
            grads[arr] = wp.from_torch(g.contiguous())
        if grads:
            ctx.tape.backward(grads=grads)
        wp_in = ctx.wp_in
        # wp_in: x y z L M N alive t_min xf gp [ports]
        out = [None, None, None]
        for k in range(6):
            out.append(wp.to_torch(wp_in[k].grad).clone() if wp_in[k].requires_grad else None)
        out += [None, None]
        out.append(wp.to_torch(wp_in[8].grad).clone())
        out.append(wp.to_torch(wp_in[9].grad).clone())
        out.append(None)
        ctx.tape.zero()
        return tuple(out)


def intersect_component(component, kind: str, rays, t_min):
    """One component's intersection through its kernel: ``BaseComponent.intersect``'s result.

    Also sets the component's ``_local_root`` exactly as its own
    ``intersect`` does, for ``advance_to_hit``.

    Args:
        component: A component :func:`kind_of` accepted.
        kind: Its kind.
        rays: The ray bundle.
        t_min: The per-ray accept threshold of this bounce.

    Returns:
        ``(t, normals, hit_mask, n_geom)``.
    """
    like = rays.x
    xf, gp, ports, nports = _component_inputs(component, kind, like)
    if ports is None:
        ports = xf  # unused placeholder for the operator's signature
    fields = [rays.x, rays.y, rays.z, rays.L, rays.M, rays.N]
    needs_grad = torch.is_grad_enabled() and (
        any(f.requires_grad for f in fields) or xf.requires_grad or gp.requires_grad
    )
    recip = 0
    if kind == "cavity" and not _is_tensor(component.geometry.radius):
        recip = scalar_division(like.dtype, like.device)
    if needs_grad:
        t, normals, hit, n_geom, t_adv, t_local = _IntersectFunction.apply(
            kind, nports, recip, *fields, rays.alive, t_min, xf, gp, ports
        )
    else:
        # Strided fields are passed as they are (a Warp array carries its
        # strides): the ray state is often a column of an (N, 3) product, and
        # a copy per field would be three more launches per component.
        t, normals, hit, n_geom, t_adv, t_local = torch.ops.optiland_nsq.intersect_component(
            kind, *fields, rays.alive, t_min, xf, gp, ports, nports, recip,
        )
    component._local_root = (t_adv, t_local)
    return t, normals, hit, n_geom


def accept_threshold(rays):
    """The per-ray accept threshold every component of this bounce uses.

    ``BaseComponent.intersect`` computes it from the ray's global position
    alone, once per component; it is the same tensor for every component,
    so the stage computes it once per bounce.
    """
    from optiland.nonsequential.components.base import coordinate_magnitude  # noqa: PLC0415

    return _tol.accept_t_min(coordinate_magnitude(rays))


def intersect_scene(rays, components):
    """``ArrayBackend.intersect_scene`` with the covered components through their kernels.

    The running minimum over the components is the torch stage's, statement
    for statement; only the per-component intersection changes.
    """
    from optiland.nonsequential.ray_bundle import backend_int_full  # noqa: PLC0415

    n = rays.num_rays
    t_min = be.ones(n) * be.inf
    hit_normals = be.zeros((n, 3))
    hit_n_geom = be.zeros((n, 3))
    comp_indices = backend_int_full((n,), -1, like=rays.x, bits=32)
    threshold = None
    for i, comp in enumerate(components):
        kind = kind_of(comp)
        if kind is None:
            t_c, normals_c, hit_c, n_geom_c = comp.intersect(rays)
        else:
            if threshold is None:
                threshold = accept_threshold(rays)
            t_c, normals_c, hit_c, n_geom_c = intersect_component(comp, kind, rays, threshold)
        better = hit_c & (t_c < t_min)
        t_min = be.where(better, t_c, t_min)
        hit_normals = be.where(better[:, None], normals_c, hit_normals)
        hit_n_geom = be.where(better[:, None], n_geom_c, hit_n_geom)
        comp_indices = be.where(
            better, backend_int_full((n,), i, like=rays.x, bits=32), comp_indices
        )
    return t_min, hit_normals, comp_indices, hit_n_geom


__all__ = [
    "SUPPORTED_DEVICE_TYPES",
    "accept_threshold",
    "availability",
    "intersect_component",
    "intersect_scene",
    "kind_of",
    "prepare",
]
