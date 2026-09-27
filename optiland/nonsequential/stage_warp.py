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
the source text, each probed on the device at load: the placement product
``(N, 3) @ (3, 3)``, whose rounding order the matrix library chooses by the
width and, on the CPU, row by row (:func:`matmul_orders`; mostly a chain of
fused multiply-adds, written with an explicit ``fma``); a sum over the last
axis of an (N, 3) array, ordered by the device's reduction
(:func:`sum_order`); and a division by a Python number, which torch's CUDA
kernel takes as a product with the rounded reciprocal
(:func:`scalar_division`). Measured on the A100 (2026-09-27, research
repository issue 77): the first CUDA run's differences (one ulp in the
cavity's normals, up to 3,209 ulp in a rotated one's) were the division
alone, and cuBLAS rounds the product as the CPU does from 17 rows up but
otherwise below; with all three probed the stage is bit-identical to
``BaseComponent.intersect`` (see the tests, which run on CUDA where present).

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

import numpy as np

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

#: The orders a three-term dot product ``a0 r0 + a1 r1 + a2 r2`` can be
#: rounded in, as the matrix product of a library forms it: the products taken
#: in one of six orders, the second and the third term added either by a
#: fused multiply-add or as a separately rounded product. Code
#: ``4 * permutation + 2 * fused_second + fused_third``.
MM_ORDERS: tuple[tuple[tuple[int, int, int], int, int], ...] = tuple(
    (perm, f1, f2)
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    for f1 in (0, 1)
    for f2 in (0, 1)
)
#: The fused multiply-add chain ``fma(a2, r2, fma(a1, r1, a0 r0))``: torch's
#: product on the Apple silicon CPU and cuBLAS's on the A100 for most widths.
MM_FMA_CHAIN = wp.constant(3)
#: Widths below this are probed row by row at load time; from it on the
#: product is modelled as one main order with the last ``n mod block`` rows in
#: their own orders, probed at widths 4,096 to 4,111 and 65,536 to 65,539.
MM_SMALL = 65


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
    def _term(k: int, a0: FT, a1: FT, a2: FT, r0: FT, r1: FT, r2: FT):
        if k == 0:
            return a0, r0
        if k == 1:
            return a1, r1
        return a2, r2

    @wp.func
    def _mm(a0: FT, a1: FT, a2: FT, r0: FT, r1: FT, r2: FT, code: int):
        # One entry of an (N, 3) @ (3, 3) product in the rounding order
        # ``code`` names (see MM_ORDERS): the three products taken in the
        # order of the permutation, the second and the third added either
        # with a fused multiply-add or as a rounded product.
        if code == MM_FMA_CHAIN:
            return fma(a2, r2, fma(a1, r1, a0 * r0))
        p = code // 4
        i0 = int(0)
        i1 = int(1)
        i2 = int(2)
        if p == 1:
            i1 = 2
            i2 = 1
        elif p == 2:
            i0 = 1
            i1 = 0
        elif p == 3:
            i0 = 1
            i1 = 2
            i2 = 0
        elif p == 4:
            i0 = 2
            i1 = 0
            i2 = 1
        elif p == 5:
            i0 = 2
            i2 = 0
        x0, y0 = _term(i0, a0, a1, a2, r0, r1, r2)
        x1, y1 = _term(i1, a0, a1, a2, r0, r1, r2)
        x2, y2 = _term(i2, a0, a1, a2, r0, r1, r2)
        acc = x0 * y0
        if (code // 2) % 2 == 1:
            acc = fma(x1, y1, acc)
        else:
            acc = acc + x1 * y1
        if code % 2 == 1:
            acc = fma(x2, y2, acc)
        else:
            acc = acc + x2 * y2
        return acc

    @wp.func
    def _row_code(
        i: int, n: int, form: int,
        small: wp.array3d(dtype=int), meta: wp.array(dtype=int), tail: wp.array3d(dtype=int),
    ):
        # The rounding order torch's product gave row ``i`` of an ``n``-row
        # product at load time (form 0: ``a @ R``, form 1: ``a @ R.T``):
        # measured row by row below MM_SMALL rows, above it a main order with
        # the last ``n mod block`` rows in their own orders, in one of two
        # regimes (MatmulOrders.code is the same function on the host).
        if n < small.shape[1]:
            return small[form, n, i]
        reg = int(1)
        if n < meta[5 * form + 4]:
            reg = 0
        block = meta[5 * form + 2 * reg + 1]
        base = n - n % block
        if i < base:
            return meta[5 * form + 2 * reg]
        return tail[2 * form + reg, n % block, i - base]

    @wp.func
    def _to_local(
        x: FT, y: FT, z: FT, xf: wp.array(dtype=FT), code: int
    ):
        px = x - xf[0]
        py = y - xf[1]
        pz = z - xf[2]
        lx = _mm(px, py, pz, xf[3], xf[6], xf[9], code)
        ly = _mm(px, py, pz, xf[4], xf[7], xf[10], code)
        lz = _mm(px, py, pz, xf[5], xf[8], xf[11], code)
        return lx, ly, lz

    @wp.func
    def _dir_local(
        x: FT, y: FT, z: FT, xf: wp.array(dtype=FT), code: int
    ):
        lx = _mm(x, y, z, xf[3], xf[6], xf[9], code)
        ly = _mm(x, y, z, xf[4], xf[7], xf[10], code)
        lz = _mm(x, y, z, xf[5], xf[8], xf[11], code)
        return lx, ly, lz

    @wp.func
    def _to_global_normal(
        x: FT, y: FT, z: FT, xf: wp.array(dtype=FT), code: int
    ):
        gx = _mm(x, y, z, xf[3], xf[4], xf[5], code)
        gy = _mm(x, y, z, xf[6], xf[7], xf[8], code)
        gz = _mm(x, y, z, xf[9], xf[10], xf[11], code)
        return gx, gy, gz

    @wp.kernel
    def k_mm_probe(
        a: wp.array2d(dtype=FT), r: wp.array2d(dtype=FT), out: wp.array3d(dtype=FT)
    ):
        # Every candidate order of ``a @ r`` (r already transposed for the
        # second form), for the load-time probe.
        i, j, code = wp.tid()
        out[i, j, code] = _mm(a[i, 0], a[i, 1], a[i, 2], r[0, j], r[1, j], r[2, j], code)

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
        mm_small: wp.array3d(dtype=int),
        mm_meta: wp.array(dtype=int),
        mm_tail: wp.array3d(dtype=int),
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
        n = x.shape[0]
        code_f = _row_code(i, n, 0, mm_small, mm_meta, mm_tail)
        code_b = _row_code(i, n, 1, mm_small, mm_meta, mm_tail)
        plx, ply, plz = _to_local(x[i], y[i], z[i], xf, code_f)
        dx, dy, dz = _dir_local(L[i], M[i], N[i], xf, code_f)
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
        gx, gy, gz = _to_global_normal(nx * flip, ny * flip, nz * flip, xf, code_b)
        normals[i, 0] = gx
        normals[i, 1] = gy
        normals[i, 2] = gz
        qx, qy, qz = _to_global_normal(-nx, -ny, -nz, xf, code_b)
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
        mm_small: wp.array3d(dtype=int),
        mm_meta: wp.array(dtype=int),
        mm_tail: wp.array3d(dtype=int),
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
        n = x.shape[0]
        code_f = _row_code(i, n, 0, mm_small, mm_meta, mm_tail)
        code_b = _row_code(i, n, 1, mm_small, mm_meta, mm_tail)
        plx, ply, plz = _to_local(x[i], y[i], z[i], xf, code_f)
        dx, dy, dz = _dir_local(L[i], M[i], N[i], xf, code_f)
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
        ax, ay, az = _to_global_normal(nlx, nly, nlz, xf, code_b)
        normals[i, 0] = ax
        normals[i, 1] = ay
        normals[i, 2] = az
        bx, by, bz = _to_global_normal(ngx, ngy, ngz, xf, code_b)
        n_geom[i, 0] = bx
        n_geom[i, 1] = by
        n_geom[i, 2] = bz
        t_adv_out[i] = t_adv
        t_local_out[i] = t_out

    return {"cavity": k_cavity, "conic": k_conic, "mm_probe": k_mm_probe}


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


# ---------------------------------------------------------------------------
# The order of torch's (N, 3) @ (3, 3) product, per width and row
# ---------------------------------------------------------------------------

#: Tie-break among orders that match every probed value: the FMA chain, then
#: the plain rounded sum, then the chain with the last term unfused, then the
#: rest in code order.
_MM_PREFERENCE = (3, 0, 2) + tuple(c for c in range(24) if c not in (3, 0, 2))
_MM_BLOCKS = (1, 2, 4, 8, 16)
#: Widths the two regimes above ``MM_SMALL`` are learnt at: every residue
#: mod 16 just above the row-by-row table, and every residue at 4,096 plus a
#: check at 65,536.
_MM_MID = tuple(range(MM_SMALL, MM_SMALL + 16))
_MM_LARGE = tuple(range(4096, 4112)) + (65536, 65537, 65538, 65539)
_MM_TABLES: dict = {}


class MatmulOrders:
    """torch's rounding order for ``a @ R`` (form 0) and ``a @ R.T`` (form 1), measured on one device.

    Attributes:
        small: ``(2, MM_SMALL, MM_SMALL - 1)`` codes, row ``i`` of an ``n``-row
            product at ``[form, n, i]``; -1 where no order of
            :data:`MM_ORDERS` matched (that width then runs the torch stage).
        meta: ``(2, 5)``: per form the main order and block of the mid
            regime, the same of the large regime, and the width from which
            the large regime holds.
        tail: ``(4, 16, 16)``: per form and regime (``2 form + regime``) the
            codes of the last ``n mod block`` rows.
        refusal: None, or why the kernels cannot follow the product above
            ``MM_SMALL`` on this device.
        tensors: The three tables on the device, as the kernels take them.
    """

    def __init__(self, small, meta, tail, refusal, device):
        self.small = small
        self.meta = meta
        self.tail = tail
        self.refusal = refusal
        self.small_ok = [bool((small[:, n, :n] >= 0).all()) for n in range(MM_SMALL)]
        self.tensors = (
            torch.as_tensor(small, dtype=torch.int32, device=device).contiguous(),
            torch.as_tensor(meta.reshape(-1), dtype=torch.int32, device=device).contiguous(),
            torch.as_tensor(tail, dtype=torch.int32, device=device).contiguous(),
        )

    def code(self, form: int, n: int, i: int) -> int:
        """The order of row ``i`` of an ``n``-row product (the kernels' ``_row_code``)."""
        if n < MM_SMALL:
            return int(self.small[form, n, i])
        reg = 0 if n < self.meta[form, 4] else 1
        main, block = self.meta[form, 2 * reg], self.meta[form, 2 * reg + 1]
        base = n - n % block
        return int(main) if i < base else int(self.tail[2 * form + reg, n % block, i - base])

    def covers(self, n: int) -> bool:
        """Whether a launch of ``n`` rays reproduces torch's product on every row."""
        if self.refusal is not None:
            return False
        if n < MM_SMALL:
            return self.small_ok[n]
        return True


def _mm_choose(match_row) -> int:
    for code in _MM_PREFERENCE:
        if match_row[code]:
            return code
    return -1


def _mm_probe_rows(dtype, device, form: int, n: int, trials: int, rng) -> torch.Tensor:
    """For each of ``n`` rows, which of the 24 orders reproduce torch's ``a @ R`` (or ``@ R.T``)
    on every one of ``trials`` random products of width ``n``; an (n, 24) boolean tensor."""
    kernel = _KERNELS[dtype]["mm_probe"]
    ivt = torch.int64 if dtype == torch.float64 else torch.int32
    acc = torch.ones((n, 24), dtype=torch.bool, device=device)
    for _ in range(trials):
        # Unit normal entries: products of one size, so the orders' roundings
        # differ on 15 to 45 percent of the entries (a spread of magnitudes
        # hides the smaller terms' rounding and discriminates less).
        a = torch.tensor(rng.standard_normal((n, 3)), dtype=dtype, device=device)
        r = torch.tensor(rng.standard_normal((3, 3)), dtype=dtype, device=device)
        got = a @ (r.T if form else r)
        rk = (r.T if form else r).contiguous()
        cand = torch.empty((n, 3, 24), dtype=dtype, device=device)
        kwargs = {"dim": (n, 3, 24), "inputs": [wp.from_torch(a), wp.from_torch(rk)],
                  "outputs": [wp.from_torch(cand)]}
        if device.type == "cuda":
            kwargs["stream"] = wp.stream_from_torch(torch.cuda.current_stream(device))
        else:
            kwargs["device"] = wp.device_from_torch(device)
        wp.launch(kernel, **kwargs)
        acc &= (cand.view(ivt) == got.view(ivt)[:, :, None]).all(dim=1)
    return acc


def _mm_regime(widths, matches):
    """A main order, a block and the tail rows' orders that every probed width shares, or None."""
    common = np.ones(24, dtype=bool)
    for n, m in zip(widths, matches, strict=True):
        common &= m[: n - n % 16].all(axis=0)
    main = _mm_choose(common)
    if main < 0:
        return None
    for block in _MM_BLOCKS:
        if all(m[: n - n % block, main].all() for n, m in zip(widths, matches, strict=True)):
            break
    else:
        return None
    tail = np.full((16, 16), -1, dtype=np.int32)
    for r in range(1, block):
        share = np.ones((r, 24), dtype=bool)
        for n, m in zip(widths, matches, strict=True):
            if n % block == r:
                share &= m[n - r:]
        codes = [_mm_choose(row) for row in share]
        if min(codes) < 0:
            return None
        tail[r, :r] = codes
    return main, block, tail


def _mm_follows(regime, dtype, device, form, n0, rng) -> bool:
    """Whether widths ``n0`` to ``n0 + 15`` round every row as ``regime`` says."""
    main, block, tail = regime
    for n in range(n0, n0 + 16):
        m = _mm_probe_rows(dtype, device, form, n, 6, rng).cpu().numpy()
        base = n - n % block
        codes = [main] * base + [int(tail[n % block, j]) for j in range(n - base)]
        if not all(m[i, c] for i, c in enumerate(codes)):
            return False
    return True


def matmul_orders(dtype: torch.dtype, device: Any) -> MatmulOrders:
    """How torch rounds the placement products of ``BaseComponent.intersect`` on this device.

    The stage transforms the ray state with ``(p - T) @ R`` and ``d @ R`` and
    the normals back with ``n @ R.T``: (N, 3) @ (3, 3) products that torch
    hands to its matrix library, which picks a kernel by the shapes, and the
    kernel fixes the order in which each entry's three products are rounded
    and summed. Measured (2026-09-27, research repository issue 77): on the
    A100 cuBLAS gives the fused chain :data:`MM_FMA_CHAIN` from 17 rows up and
    another order at 1 to 16 rows; on the Apple silicon CPU the chain holds
    for every row except the last ``n mod 2`` or ``n mod 4`` rows, which a
    tail loop rounds otherwise, up to a width that depends on the dtype and
    the form (about 700 rows for float64 ``@ R.T``, above 1,200 for float32),
    and for every row above it.

    So the order is probed here once per dtype and device: row by row for
    every width below :data:`MM_SMALL`; above it as two regimes of a main
    order plus the orders of a tail block, one learnt at the 16 widths from
    ``MM_SMALL`` and one at 4,096 to 4,111 (checked at 65,536 to 65,539), and
    where the two differ, the width at which the second takes over, found by
    bisection on blocks of 16 consecutive widths. The kernels read the order
    per row. Widths between the probed ones are assumed to follow the model;
    the stage's width sweep and whole-trace tests check it.

    Args:
        dtype: The working float dtype.
        device: The torch device.

    Returns:
        The tables, cached per dtype and device.
    """
    tdev = torch.device(device)
    if tdev.type == "cuda" and tdev.index is None:
        tdev = torch.device("cuda", torch.cuda.current_device())
    key = (dtype, str(tdev))
    cached = _MM_TABLES.get(key)
    if cached is not None:
        return cached
    _init()
    rng = np.random.default_rng(20260927)
    small = np.full((2, MM_SMALL, MM_SMALL - 1), -1, dtype=np.int32)
    meta = np.zeros((2, 5), dtype=np.int32)
    tail = np.full((4, 16, 16), -1, dtype=np.int32)
    refusal = None
    with torch.no_grad():
        for form in (0, 1):
            rows = [_mm_probe_rows(dtype, tdev, form, n, 48, rng) for n in range(1, MM_SMALL)]
            mid = [_mm_probe_rows(dtype, tdev, form, n, 48, rng) for n in _MM_MID]
            large = [_mm_probe_rows(dtype, tdev, form, n, 8 if n < 65536 else 2, rng) for n in _MM_LARGE]
            rows = [r.cpu().numpy() for r in rows]
            mid = [m.cpu().numpy() for m in mid]
            large = [m.cpu().numpy() for m in large]
            for n, m in zip(range(1, MM_SMALL), rows, strict=True):
                small[form, n, :n] = [_mm_choose(row) for row in m]
            regime_mid = _mm_regime(_MM_MID, mid)
            regime_large = _mm_regime(_MM_LARGE, large)
            if regime_mid is None or regime_large is None:
                refusal = f"no main order and tail block match torch's product from {MM_SMALL} rows (form {form})"
                continue
            switch = MM_SMALL
            same = (
                regime_mid[0] == regime_large[0]
                and regime_mid[1] == regime_large[1]
                and np.array_equal(regime_mid[2], regime_large[2])
            )
            if not same:
                # The large regime holds from some width on: the smallest
                # block of 16 widths it holds on, by bisection.
                lo, hi = _MM_MID[-1], _MM_LARGE[0]
                while hi - lo > 1:
                    mid_n = (lo + hi) // 2
                    if _mm_follows(regime_large, dtype, tdev, form, mid_n, rng):
                        hi = mid_n
                    else:
                        lo = mid_n
                switch = hi
                if not _mm_follows(regime_mid, dtype, tdev, form, max(MM_SMALL, switch - 16), rng):
                    refusal = (
                        f"torch's product changes order near width {switch} in a way the "
                        f"two-regime model does not hold (form {form})"
                    )
            meta[form] = (regime_mid[0], regime_mid[1], regime_large[0], regime_large[1], switch)
            tail[2 * form] = regime_mid[2]
            tail[2 * form + 1] = regime_large[2]
    tables = MatmulOrders(small, meta, tail, refusal, tdev)
    _MM_TABLES[key] = tables
    return tables


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
        matmul_orders(dtype, tdev)
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
    for dtype in (torch.float64, torch.float32):
        refusal = matmul_orders(dtype, tdev).refusal
        if refusal is not None:
            return f"the kernels cannot follow torch's placement product here: {refusal}"
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
    # 1 / radius rounded once in the working dtype, as torch forms the
    # reciprocal of a Python number; numpy's IEEE division, no torch
    # operation, so nothing here touches a device inside a recorded bounce.
    np_dtype = np.float64 if like.dtype == torch.float64 else np.float32
    inv = 1.0 if _is_tensor(radius) else float(np_dtype(1.0) / np_dtype(radius))
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
    for table in matmul_orders(like.dtype, like.device).tensors:
        args.append(wp.from_torch(table, dtype=wp.int32))
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
    if not matmul_orders(like.dtype, like.device).covers(int(like.shape[0])):
        # A width at which torch's placement product rounds a row in an order
        # the kernels do not reproduce: the component's own intersect runs.
        return component.intersect(rays)
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
