"""The component-intersection stage as Warp kernels, one launch per component (prototype).

The eager torch bounce intersects every component through
:meth:`~optiland.nonsequential.components.base.BaseComponent.intersect`: a
frame transform, the origin advance, the per-ray accept threshold, the
geometry's own root solve and root selection, and the transform of the two
normals back to the global frame, followed by the running nearest-hit select
of :meth:`~optiland.nonsequential.backends.array_backend.ArrayBackend
.intersect_scene` -- 150 to 600 dispatched torch operations per component,
each one a kernel launch on a device. This module computes the whole stage
with one Warp launch per component, for every analytic kind of the engine:
the plane and the finite plane, the annulus, the sphere, the conic (and the
paraboloid), the frustum (a lens edge; one more launch for the bundle-wide
maximum its axial window uses), the ported spherical cavity, the lenslet
array, and the even and odd aspheres (their Newton refinement inside the
kernel). A component of another kind (a mesh, a subclass that overrides the
geometry) keeps its own ``intersect`` and is merged into the running select
by one small launch.

**The interface with the torch loop.** What crosses into a kernel is the ray
state the torch stage reads (``x, y, z, L, M, N`` and ``alive``), the
component's placement as twelve numbers (translation and rotation), the
geometry's scalars, integers and table, each formed on the host by exactly
the expression the torch stage evaluates, and the orders torch rounds in on
the device (below). The accept threshold is computed per ray inside the
kernel, and the nearest-hit select is the kernel's epilogue: each launch
updates the running nearest hit (distance, both normals, component index) in
place, so what comes back is exactly what ``intersect_scene`` returns, and
every component's two halves of the hit distance (``_local_root``) for
``advance_to_hit``. Compaction, the interaction, the detectors and the
CUDA-graph replay see ordinary tensors: the launches run on torch's current
CUDA stream and the kernels are loaded, and the orders probed, before the
first bounce (:func:`prepare`), so a recorded bounce records the launches.

**Same numbers.** The kernels are compiled with floating-point contraction
off (Warp's ``fuse_fp`` module option), so no multiply-add is fused that the
torch stage does not fuse, and every expression is evaluated in the order the
torch stage evaluates it, including torch's own operator forms: a Python
number divided by a tensor is ``reciprocal(x) * s`` (``Tensor.__rtruediv__``),
a Python-number expression is evaluated in float64 on the host before it
meets a tensor, and a scalar a torch expression takes from Python is cast to
the working dtype on the host as torch casts it. Three orders are torch's own
and are reproduced from measurement rather than from the source text, each
probed on the device at load: the placement product ``(N, 3) @ (3, 3)``,
whose rounding order the matrix library chooses by the width and, on the
CPU, row by row (:func:`matmul_orders`; mostly a chain of fused multiply-adds,
written with an explicit ``fma``); a sum over the last axis of an (N, 3)
array, ordered by the device's reduction (:func:`sum_order`); and a division
by a Python number, which torch's CUDA kernel takes as a product with the
rounded reciprocal (:func:`scalar_division`). Measured on CUDA
(2026-09-27, research repository issue 77): the first CUDA run's differences
(one ulp in the cavity's normals, up to 3,209 ulp in a rotated one's) were the
division alone, and cuBLAS rounds the product as the CPU does from 17 rows up
but otherwise below; with all three probed the stage is bit-identical to
``BaseComponent.intersect`` (see the tests, which run on CUDA where present).

**Gradients.** The cavity and the conic carry the Warp tape's adjoint: in
gradient mode the launch (without the fused select, which then runs in torch)
is recorded on a tape inside a ``torch.autograd.Function``, and the backward
pass replays the tape's adjoint kernels, so the gradient reaches the ray
state and the geometry's parameters. Every other case of a gradient -- another
kind with a parameter or ray carrying one, a placement that carries one
(research repository issue 31's attached placements; the kernels take the
placement as twelve detached numbers), a forward-mode dual tensor -- is
routed to the component's own ``intersect`` with the reason counted
(:func:`routed_counts`), so a differentiable trace keeps torch's derivative.

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
# which is written with an explicit fused multiply-add below. The kernels
# live in modules of their own (:func:`_kernel`), which carry the same
# option; this module holds the shared functions.
wp.set_module_options({"fuse_fp": False, "enable_backward": True})

_FMA_SNIPPET = "return fma(a, b, c);"
_FMA_ADJ = "adj_a += b * adj_ret; adj_b += a * adj_ret; adj_c += adj_ret;"


@wp.func_native(_FMA_SNIPPET, _FMA_ADJ)
def _fma64(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64: ...


@wp.func_native(_FMA_SNIPPET, _FMA_ADJ)
def _fma32(a: wp.float32, b: wp.float32, c: wp.float32) -> wp.float32: ...


# The spacing above |a|, as _tol.ulp forms it with torch.nextafter(|a|, inf)
# - |a|: the bit pattern of |a| plus one, minus |a|. It is a tolerance, never
# differentiated (the adjoint is zero).
_ULP64_SNIPPET = (
    "union { double d; long long i; } u; u.d = a; u.i &= 0x7fffffffffffffffLL; "
    "double m = u.d; u.i += 1; return u.d - m;"
)
_ULP32_SNIPPET = (
    "union { float f; int i; } u; u.f = a; u.i &= 0x7fffffff; "
    "float m = u.f; u.i += 1; return u.f - m;"
)


# Negation that flips the sign bit, as torch's does: Warp's own unary minus is
# ``0 - x``, which turns -(+0) into +0 (measured), so a zero's sign would differ.
@wp.func_native("return -a;", "adj_a -= adj_ret;")
def _neg64(a: wp.float64) -> wp.float64: ...


@wp.func_native("return -a;", "adj_a -= adj_ret;")
def _neg32(a: wp.float32) -> wp.float32: ...


@wp.func_native(_ULP64_SNIPPET, "")
def _ulp64(a: wp.float64) -> wp.float64: ...


@wp.func_native(_ULP32_SNIPPET, "")
def _ulp32(a: wp.float32) -> wp.float32: ...


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
#: product on the Apple silicon CPU and cuBLAS's on CUDA for most widths.
MM_FMA_CHAIN = wp.constant(3)
#: The same chain from a +0 accumulator, ``fma(a2, r2, fma(a1, r1, fma(a0,
#: r0, +0)))``, which differs from it only in the sign of a zero result (a
#: row of negative zeros gives +0): torch's product on the Apple silicon CPU.
MM_FMA_CHAIN_ZERO = wp.constant(27)
#: Number of order codes: the 24 of MM_ORDERS, and each from a +0 accumulator.
MM_CODES = 48
#: Widths below this are probed row by row at load time; from it on the
#: product is modelled as one main order with the last ``n mod block`` rows in
#: their own orders, probed at widths 4,096 to 4,111 and 65,536 to 65,539.
MM_SMALL = 65

#: The accept threshold's multiple and floor (_tol.accept_t_min).
_ACCEPT_K = wp.constant(float(_tol.DEFAULT_ACCEPT_K))
_ACCEPT_FLOOR = wp.constant(float(_tol._MAGNITUDE_FLOOR))

# Status codes of the asphere kind (asphere.MISS_REASONS).
_ST_HIT = wp.constant(0.0)
_ST_NO_SEED = wp.constant(1.0)
_ST_BEHIND = wp.constant(2.0)
_ST_APERTURE = wp.constant(3.0)
_ST_DOMAIN = wp.constant(4.0)
_ST_GRAZING = wp.constant(5.0)
_ST_NOT_CONVERGED = wp.constant(6.0)

#: The launch mode: write the component's own outputs (the tape's route and
#: the tests), or fold them into the running nearest hit (the fused select).
MODE_OWN = 0
MODE_SELECT = 1


def _kernel_module_name(name: str, bits: int) -> str:
    """The Warp module one kernel of the stage lives in (one module per kernel and float type)."""
    return f"{__name__}.{name}_{bits}"


def _kernel(fn, name: str, bits: int, backward: bool = False):
    """``fn`` as a Warp kernel in a module of its own.

    One module per kernel and float type (research repository issue 85): a
    module is Warp's unit of compilation, caching and loading, so a scene
    compiles and loads only the kernels of the kinds it holds, the modules
    a scene needs compile in parallel (:func:`load_kernels`), and only the
    kinds that carry the tape's adjoint (:data:`TAPE_KINDS`) generate a
    backward pass. The expressions, and so the numbers, are the same as in
    one module: the module options are the stage's (no floating-point
    contraction).
    """
    module = _kernel_module_name(name, bits)
    wp.set_module_options({"fuse_fp": False, "enable_backward": bool(backward)}, module=module)
    return wp.kernel(fn, module=wp.get_module(module))


#: The kinds whose kernel carries the Warp tape's adjoint.
TAPE_KINDS = ("cavity", "conic")


def _make_kernels(FT, fma, ulp, flipsign):
    """The stage's kernels for one float type.

    Every expression mirrors the torch source statement for statement; the
    comments name the torch expression where its operator form matters.
    """
    bits = 64 if FT is wp.float64 else 32

    # -- the orders torch rounds in ------------------------------------------

    @wp.func
    def _sum3(x: FT, y: FT, z: FT, order: int):
        # A sum over the last axis of an (N, 3) array (``.sum(axis=1)``);
        # ``order >= 2``: from a +0 accumulator (only a zero's sign changes).
        r = (x + y) + z
        if order % 2 == 1:
            r = (x + z) + y
        if order >= 2:
            r = r + FT(0.0)
        return r

    @wp.func
    def _div_scalar(x: FT, s: FT, inv_s: FT, recip: int):
        # ``tensor / python_number``.
        if recip == 1:
            return x * inv_s
        return x / s

    @wp.func
    def _rdiv(s: FT, x: FT):
        # ``python_number / tensor`` is ``x.reciprocal() * s``.
        return (FT(1.0) / x) * s

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
        # with a fused multiply-add or as a rounded product; from 24 on, the
        # sum starts from a +0 accumulator (a zero's sign only).
        if code == MM_FMA_CHAIN_ZERO:
            return fma(a2, r2, fma(a1, r1, a0 * r0 + FT(0.0)))
        if code == MM_FMA_CHAIN:
            return fma(a2, r2, fma(a1, r1, a0 * r0))
        zinit = code >= 24
        c24 = code % 24
        p = c24 // 4
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
        if zinit:
            acc = acc + FT(0.0)
        if (c24 // 2) % 2 == 1:
            acc = fma(x1, y1, acc)
        else:
            acc = acc + x1 * y1
        if c24 % 2 == 1:
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
    def _to_local(x: FT, y: FT, z: FT, xf: wp.array(dtype=FT), code: int):
        px = x - xf[0]
        py = y - xf[1]
        pz = z - xf[2]
        lx = _mm(px, py, pz, xf[3], xf[6], xf[9], code)
        ly = _mm(px, py, pz, xf[4], xf[7], xf[10], code)
        lz = _mm(px, py, pz, xf[5], xf[8], xf[11], code)
        return lx, ly, lz

    @wp.func
    def _dir_local(x: FT, y: FT, z: FT, xf: wp.array(dtype=FT), code: int):
        lx = _mm(x, y, z, xf[3], xf[6], xf[9], code)
        ly = _mm(x, y, z, xf[4], xf[7], xf[10], code)
        lz = _mm(x, y, z, xf[5], xf[8], xf[11], code)
        return lx, ly, lz

    @wp.func
    def _to_global_normal(x: FT, y: FT, z: FT, xf: wp.array(dtype=FT), code: int):
        gx = _mm(x, y, z, xf[3], xf[4], xf[5], code)
        gy = _mm(x, y, z, xf[6], xf[7], xf[8], code)
        gz = _mm(x, y, z, xf[9], xf[10], xf[11], code)
        return gx, gy, gz

    @wp.func
    def _max_nan(a: FT, b: FT):
        # torch.maximum: NaN if either is NaN.
        if wp.isnan(a) or wp.isnan(b):
            return a + b
        if a > b:
            return a
        return b

    @wp.func
    def _min_nan(a: FT, b: FT):
        # torch.minimum: NaN if either is NaN.
        if wp.isnan(a) or wp.isnan(b):
            return a + b
        if a < b:
            return a
        return b

    @wp.func
    def _accept_t_min(mag: FT):
        # _tol.accept_t_min: k ulps of max(|mag|, 1) (torch.clamp keeps NaN).
        m = wp.abs(mag)
        if not wp.isnan(m):
            if m < FT(_ACCEPT_FLOOR):
                m = FT(_ACCEPT_FLOOR)
        return FT(_ACCEPT_K) * ulp(m)

    def k_mm_probe(a: wp.array2d(dtype=FT), r: wp.array2d(dtype=FT), out: wp.array3d(dtype=FT)):
        # Every candidate order of ``a @ r`` (r already transposed for the
        # second form), for the load-time probe.
        i, j, code = wp.tid()
        out[i, j, code] = _mm(a[i, 0], a[i, 1], a[i, 2], r[0, j], r[1, j], r[2, j], code)

    k_mm_probe = _kernel(k_mm_probe, "mm_probe", bits)

    # -- the geometries ---------------------------------------------------------
    #
    # Each takes the advanced local origin, the local direction, the shifted
    # accept threshold ``eps`` and the kind's parameters, and returns the
    # geometry's ``ray_intersect`` for one ray: (t, normal, hit, n_geom), with
    # the asphere's status and step count (0 elsewhere).

    @wp.func
    def _cavity_root_valid(
        t: FT, disc_ok: bool, eps: FT, ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT,
        radius: FT, inv_radius: FT, recip: int, tab: wp.array(dtype=FT), nports: int,
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
                (hx * tab[4 * p] + hy * tab[4 * p + 1]) + hz * tab[4 * p + 2], radius, inv_radius, recip
            )
            if cos_angle >= tab[4 * p + 3]:
                in_port = 1
        return forward and (in_port == 0)

    @wp.func
    def g_cavity(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT), aux: wp.array(dtype=FT),
        sum_order: int, recip: int,
    ):
        # gp: radius, radius**2, radicand floor, 1 / radius; gi: ports
        radius = gp[0]
        r2 = gp[1]
        floor = gp[2]
        inv_radius = gp[3]
        inf = FT(wp.inf)
        b = FT(2.0) * ((ox * dx + oy * dy) + oz * dz)
        c = ((ox * ox + oy * oy) + oz * oz) - r2
        disc = b * b - FT(4.0) * c
        disc_ok = disc >= FT(0.0)
        sqrt_disc = FT(0.0)
        if disc_ok:
            sqrt_disc = wp.sqrt(_max_nan(disc, floor))
        t_near = (flipsign(b) - sqrt_disc) * FT(0.5)
        t_far = (flipsign(b) + sqrt_disc) * FT(0.5)
        use_near = _cavity_root_valid(t_near, disc_ok, eps, ox, oy, oz, dx, dy, dz, radius, inv_radius, recip, tab, gi[0])
        use_far = _cavity_root_valid(t_far, disc_ok, eps, ox, oy, oz, dx, dy, dz, radius, inv_radius, recip, tab, gi[0])
        use_far = use_far and (not use_near)
        hit = use_near or use_far
        t = inf
        if use_far:
            t = t_far
        if use_near:
            t = t_near
        safe_t = FT(0.0)
        if hit:
            safe_t = t
        hx = ox + safe_t * dx
        hy = oy + safe_t * dy
        hz = oz + safe_t * dz
        nx = FT(0.0)
        ny = FT(0.0)
        nz = FT(0.0)
        if hit:
            nx = _div_scalar(hx, radius, inv_radius, recip)
            ny = _div_scalar(hy, radius, inv_radius, recip)
            nz = _div_scalar(hz, radius, inv_radius, recip)
        dot = (dx * nx + dy * ny) + dz * nz
        flip = FT(1.0)
        if dot > FT(0.0):
            flip = FT(-1.0)
        return t, nx * flip, ny * flip, nz * flip, hit, flipsign(nx), flipsign(ny), flipsign(nz), FT(0.0), FT(0.0)

    @wp.func
    def _conic_root_valid(
        t: FT, solvable: bool, eps: FT, ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, ap2: FT, kc: FT,
    ):
        px = ox + t * dx
        py = oy + t * dy
        pz = oz + t * dz
        in_aperture = (px * px + py * py) <= ap2
        on_sheet = (FT(1.0) - kc * pz) >= FT(0.0)
        valid = solvable and wp.isfinite(t) and (t > eps) and in_aperture and on_sheet
        return valid, px, py

    @wp.func
    def _conic_roots(ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, c: FT, kp: FT, floor: FT, tiny: FT):
        # ConicGeometry._quadratic_roots.
        a = c * ((dx * dx + dy * dy) + kp * (dz * dz))
        b = FT(2.0) * (c * ((ox * dx + oy * dy) + (kp * oz) * dz) - dz)
        c0 = c * ((ox * ox + oy * oy) + kp * (oz * oz)) - FT(2.0) * oz
        disc = b * b - (FT(4.0) * a) * c0
        disc_ok = disc >= FT(0.0)
        sqrt_disc = FT(0.0)
        if disc_ok:
            sqrt_disc = wp.sqrt(_max_nan(disc, floor))
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
        return q / a_den, c0 / q_den, disc_ok and a_ok, disc_ok and q_ok

    @wp.func
    def _conic_normal(px: FT, py: FT, kc2: FT, negc: FT, floor: FT):
        # ConicGeometry._normal_local, unnormalised.
        r2 = px * px + py * py
        s = wp.sqrt(_max_nan(FT(1.0) - kc2 * r2, floor))
        return (negc * px) / s, (negc * py) / s, FT(1.0)

    @wp.func
    def _unit_and_face(gx: FT, gy: FT, gz: FT, dx: FT, dy: FT, dz: FT, tiny: FT, sum_order: int):
        # n_len = (n * n).sum(axis=1, keepdims=True) ** 0.5; n / (n_len + tiny);
        # dot = (d * n).sum(axis=1); where(dot > 0, -n, n).
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
            nlx = flipsign(ngx)
            nly = flipsign(ngy)
            nlz = flipsign(ngz)
        return nlx, nly, nlz, ngx, ngy, ngz

    @wp.func
    def g_conic(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT), aux: wp.array(dtype=FT),
        sum_order: int, recip: int,
    ):
        # gp: c, 1 + K, aperture**2, (1 + K) c, (1 + K) c**2, -c, radicand floor, tiny
        c = gp[0]
        kp = gp[1]
        ap2 = gp[2]
        kc = gp[3]
        kc2 = gp[4]
        negc = gp[5]
        floor = gp[6]
        tiny = gp[7]
        t1, t2, solv1, solv2 = _conic_roots(ox, oy, oz, dx, dy, dz, c, kp, floor, tiny)
        valid1, px1, py1 = _conic_root_valid(t1, solv1, eps, ox, oy, oz, dx, dy, dz, ap2, kc)
        valid2, px2, py2 = _conic_root_valid(t2, solv2, eps, ox, oy, oz, dx, dy, dz, ap2, kc)
        pick1 = valid1 and ((not valid2) or (t1 <= t2))
        pick2 = valid2 and (not pick1)
        hit = pick1 or pick2
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
        t_out = FT(wp.inf)
        if hit:
            t_out = t
        gx, gy, gz = _conic_normal(px, py, kc2, negc, floor)
        nlx, nly, nlz, ngx, ngy, ngz = _unit_and_face(gx, gy, gz, dx, dy, dz, tiny, sum_order)
        return t_out, nlx, nly, nlz, hit, ngx, ngy, ngz, FT(0.0), FT(0.0)

    @wp.func
    def g_plane(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT), aux: wp.array(dtype=FT),
        sum_order: int, recip: int,
    ):
        # PlaneGeometry; gp: the parallel-ray floor 8 ulp(1)
        valid = wp.abs(dz) > gp[0]
        safe_dz = FT(1.0)
        if valid:
            safe_dz = dz
        t = FT(wp.inf)
        if valid:
            t = flipsign(oz) / safe_dz
        hit = valid and (t > eps)
        if not hit:
            t = FT(wp.inf)
        nz = FT(-1.0)
        if dz < FT(0.0):
            nz = FT(1.0)
        return t, FT(0.0), FT(0.0), nz, hit, FT(0.0), FT(0.0), FT(1.0), FT(0.0), FT(0.0)

    @wp.func
    def g_finite_plane(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT), aux: wp.array(dtype=FT),
        sum_order: int, recip: int,
    ):
        # FinitePlaneGeometry; gp: 8 ulp(1), aperture**2, width / 2, height / 2;
        # gi: 1 for a circular aperture
        inf = FT(wp.inf)
        plane_valid = wp.abs(dz) > gp[0]
        den = FT(1.0)
        if plane_valid:
            den = dz
        t = inf
        if plane_valid:
            t = flipsign(oz) / den
        if not (plane_valid and (t > eps)):
            t = inf
        safe_t = FT(0.0)
        if wp.isfinite(t):
            safe_t = t
        hx = ox + safe_t * dx
        hy = oy + safe_t * dy
        in_aperture = bool(False)
        if gi[0] == 1:
            in_aperture = (hx * hx + hy * hy) <= gp[1]
        else:
            in_aperture = (wp.abs(hx) <= gp[2]) and (wp.abs(hy) <= gp[3])
        hit = plane_valid and (t < inf) and in_aperture
        if not hit:
            t = inf
        nz = FT(-1.0)
        if dz < FT(0.0):
            nz = FT(1.0)
        return t, FT(0.0), FT(0.0), nz, hit, FT(0.0), FT(0.0), FT(1.0), FT(0.0), FT(0.0)

    @wp.func
    def g_annulus(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT), aux: wp.array(dtype=FT),
        sum_order: int, recip: int,
    ):
        # AnnularPlaneGeometry; gp: 8 ulp(1), z_offset, tiny, inner**2, outer**2
        inf = FT(wp.inf)
        t = inf
        if wp.abs(dz) > gp[0]:
            t = (gp[1] - oz) / (dz + gp[2])
        hx = ox + t * dx
        hy = oy + t * dy
        r2 = hx * hx + hy * hy
        hit = (t > eps) and (r2 >= gp[3]) and (r2 <= gp[4])
        t_out = inf
        if hit:
            t_out = t
        nz = FT(1.0)
        if dz > FT(0.0):
            nz = FT(-1.0)
        return t_out, FT(0.0), FT(0.0), nz, hit, FT(0.0), FT(0.0), FT(1.0), FT(0.0), FT(0.0)

    @wp.func
    def g_sphere(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT), aux: wp.array(dtype=FT),
        sum_order: int, recip: int,
    ):
        # SphereGeometry; gp: radius, radius**2, radicand floor, 1 / radius,
        # aperture radius; gi: 1 with an aperture
        radius = gp[0]
        inf = FT(wp.inf)
        b = FT(2.0) * ((ox * dx + oy * dy) + oz * dz)
        c = ((ox * ox + oy * oy) + oz * oz) - gp[1]
        disc = b * b - FT(4.0) * c
        disc_ok = disc >= FT(0.0)
        sqrt_disc = FT(0.0)
        if disc_ok:
            sqrt_disc = wp.sqrt(_max_nan(disc, gp[2]))
        t1 = inf
        t2 = inf
        if disc_ok:
            t1 = (flipsign(b) - sqrt_disc) * FT(0.5)
            t2 = (flipsign(b) + sqrt_disc) * FT(0.5)
        use1 = disc_ok and (t1 > eps)
        use2 = disc_ok and (not use1) and (t2 > eps)
        t = inf
        if use2:
            t = t2
        if use1:
            t = t1
        hx = ox + t * dx
        hy = oy + t * dy
        hz = oz + t * dz
        finite = t < inf
        nx = FT(0.0)
        ny = FT(0.0)
        nz = FT(0.0)
        if finite:
            nx = _div_scalar(hx, radius, gp[3], recip)
            ny = _div_scalar(hy, radius, gp[3], recip)
            nz = _div_scalar(hz, radius, gp[3], recip)
        dot = (dx * nx + dy * ny) + dz * nz
        flip = FT(1.0)
        if dot > FT(0.0):
            flip = FT(-1.0)
        hit = finite
        if gi[0] == 1:
            r_t = wp.sqrt(hx * hx + hy * hy)
            hit = hit and (r_t <= gp[4])
            if not hit:
                t = inf
        return t, nx * flip, ny * flip, nz * flip, hit, flipsign(nx), flipsign(ny), flipsign(nz), FT(0.0), FT(0.0)

    @wp.func
    def _frustum_valid(t: FT, disc: FT, eps: FT, oz: FT, dz: FT, z_lo: FT, z_hi: FT):
        z_hit = oz + t * dz
        in_z = (z_hit >= z_lo) and (z_hit <= z_hi)
        return (disc >= FT(0.0)) and (t > eps) and in_z

    @wp.func
    def g_frustum(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT), aux: wp.array(dtype=FT),
        sum_order: int, recip: int,
    ):
        # CylindricalFrustumGeometry; gp: r_front, slope, z_front, z_back,
        # tiny, the degeneracy multiple; aux[0]: max |origin| over the bundle
        r_front = gp[0]
        slope = gp[1]
        z_front = gp[2]
        z_back = gp[3]
        inf = FT(wp.inf)
        rz = r_front + slope * (oz - z_front)
        rv = slope * dz
        a = (dx * dx + dy * dy) - rv * rv
        b = FT(2.0) * ((ox * dx + oy * dy) - rz * rv)
        c = (ox * ox + oy * oy) - rz * rz
        disc = b * b - (FT(4.0) * a) * c
        disc_pos = disc > FT(0.0)
        sqrt_disc = FT(0.0)
        if disc_pos:
            sqrt_disc = wp.sqrt(disc)
        axial_slack = _accept_t_min(aux[0])
        coeff_scale = _max_nan(wp.abs(b), wp.abs(c))
        degeneracy_floor = gp[5] * ulp(coeff_scale)
        a_small = wp.abs(a) < degeneracy_floor
        b_small = wp.abs(b) < degeneracy_floor
        b_safe = b
        if b_small:
            b_safe = FT(1.0)
        t_lin = inf
        if not b_small:
            t_lin = flipsign(c) / b_safe
        a_safe = a
        if a_small:
            a_safe = FT(1.0)
        inv2a = FT(0.0)
        if not a_small:
            inv2a = _rdiv(FT(1.0), FT(2.0) * a_safe)
        t1 = (flipsign(b) - sqrt_disc) * inv2a
        t2 = (flipsign(b) + sqrt_disc) * inv2a
        z_lo = z_front - axial_slack
        z_hi = z_back + axial_slack
        valid1 = _frustum_valid(t1, disc, eps, oz, dz, z_lo, z_hi) and (not a_small)
        valid2 = _frustum_valid(t2, disc, eps, oz, dz, z_lo, z_hi) and (not a_small)
        valid_lin = _frustum_valid(t_lin, disc, eps, oz, dz, z_lo, z_hi)
        t_best = inf
        if valid2:
            t_best = t2
        if valid1:
            t_best = t1
        if a_small and valid_lin:
            t_best = t_lin
        hit = wp.isfinite(t_best)
        t_nrm = FT(0.0)
        if hit:
            t_nrm = t_best
        hx = ox + t_nrm * dx
        hy = oy + t_nrm * dy
        hz = oz + t_nrm * dz
        rz_hit = r_front + slope * (hz - z_front)
        nx = hx
        ny = hy
        nz = (flipsign(rz_hit) * slope) * FT(1.0)
        n_len = wp.sqrt(((nx * nx + ny * ny) + nz * nz) + gp[4])
        ngx = nx / n_len
        ngy = ny / n_len
        ngz = nz / n_len
        dot = _sum3(dx * ngx, dy * ngy, dz * ngz, sum_order)
        nlx = ngx
        nly = ngy
        nlz = ngz
        if dot > FT(0.0):
            nlx = flipsign(ngx)
            nly = flipsign(ngy)
            nlz = flipsign(ngz)
        t_out = inf
        if hit:
            t_out = t_best
        return t_out, nlx, nly, nlz, hit, ngx, ngy, ngz, FT(0.0), FT(0.0)

    @wp.func
    def _axis_interval(o: FT, d: FT, lo: FT, hi: FT):
        # lenslet_array._axis_interval: 1.0 / d is d.reciprocal().
        inv_d = FT(1.0) / d
        t0 = (lo - o) * inv_d
        t1 = (hi - o) * inv_d
        return _min_nan(t0, t1), _max_nan(t0, t1)

    @wp.func
    def g_lenslet(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT), aux: wp.array(dtype=FT),
        sum_order: int, recip: int,
    ):
        # LensletArrayGeometry; gp: pitch_x, pitch_y, pitch_x / 2, pitch_y / 2,
        # x_min, x_max, y_min, y_max, z_lo, z_hi, c, 1 + K, (1 + K) c,
        # (1 + K) c**2, -c, radicand floor, tiny, 1 / pitch_x, 1 / pitch_y;
        # gi: num_x, num_y; tab: the sag offsets, row-major
        pitch_x = gp[0]
        pitch_y = gp[1]
        half_px = gp[2]
        half_py = gp[3]
        x_min = gp[4]
        y_min = gp[6]
        c = gp[10]
        kp = gp[11]
        kpc = gp[12]
        floor = gp[15]
        tiny = gp[16]
        num_x = gi[0]
        num_y = gi[1]
        inf = FT(wp.inf)
        tnx, tfx = _axis_interval(ox, dx, x_min, gp[5])
        tny, tfy = _axis_interval(oy, dy, y_min, gp[7])
        tnz, tfz = _axis_interval(oz, dz, gp[8], gp[9])
        t_enter = _max_nan(_max_nan(tnx, tny), tnz)
        t_exit = _min_nan(_min_nan(tfx, tfy), tfz)
        t_start = _max_nan(t_enter, eps)
        box_valid = (t_exit >= t_enter) and (t_exit > eps) and (t_start <= t_exit)
        if not box_valid:
            t_start = FT(0.0)
        ex = ox + t_start * dx
        ey = oy + t_start * dy
        i = wp.clamp(int(wp.floor(_div_scalar(ex - x_min, pitch_x, gp[17], recip))), 0, num_x - 1)
        j = wp.clamp(int(wp.floor(_div_scalar(ey - y_min, pitch_y, gp[18], recip))), 0, num_y - 1)
        # floor_to_int(torch.sign(d)): 0 for a zero component.
        step_x = int(0)
        if dx > FT(0.0):
            step_x = 1
        if dx < FT(0.0):
            step_x = -1
        step_y = int(0)
        if dy > FT(0.0):
            step_y = 1
        if dy < FT(0.0):
            step_y = -1
        t_delta_x = _rdiv(pitch_x, wp.abs(dx))
        t_delta_y = _rdiv(pitch_y, wp.abs(dy))
        ib = i
        if step_x > 0:
            ib = i + 1
        jb = j
        if step_y > 0:
            jb = j + 1
        x_boundary = x_min + FT(ib) * pitch_x
        y_boundary = y_min + FT(jb) * pitch_y
        t_max_x = (x_boundary - ox) / dx
        t_max_y = (y_boundary - oy) / dy
        if step_x == 0:
            t_max_x = inf
        if step_y == 0:
            t_max_y = inf
        found = bool(False)
        active = box_valid
        t_result = inf
        px_rel = FT(0.0)
        py_rel = FT(0.0)
        for _step in range(num_x + num_y + 1):
            if not (active and (not found)):
                break
            i_c = wp.clamp(i, 0, num_x - 1)
            j_c = wp.clamp(j, 0, num_y - 1)
            xc = x_min + (FT(i_c) + FT(0.5)) * pitch_x
            yc = y_min + (FT(j_c) + FT(0.5)) * pitch_y
            off = tab[j_c * num_x + i_c]
            oxp = ox - xc
            oyp = oy - yc
            ozp = oz - off
            a = c * ((dx * dx + dy * dy) + (kp * dz) * dz)
            b = FT(2.0) * (c * ((oxp * dx + oyp * dy) + (kp * ozp) * dz) - dz)
            c0 = c * ((oxp * oxp + oyp * oyp) + (kp * ozp) * ozp) - FT(2.0) * ozp
            disc = b * b - (FT(4.0) * a) * c0
            disc_ok = disc >= FT(0.0)
            sqrt_disc = FT(0.0)
            if disc_ok:
                sqrt_disc = wp.sqrt(_max_nan(disc, floor))
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
            px1 = ox + t1 * dx
            py1 = oy + t1 * dy
            pz1 = oz + t1 * dz
            valid1 = (disc_ok and a_ok) and wp.isfinite(t1) and (t1 > eps)
            valid1 = valid1 and (wp.abs(px1 - xc) <= half_px) and (wp.abs(py1 - yc) <= half_py)
            valid1 = valid1 and ((FT(1.0) - kpc * (pz1 - off)) >= FT(0.0))
            px2 = ox + t2 * dx
            py2 = oy + t2 * dy
            pz2 = oz + t2 * dz
            valid2 = (disc_ok and q_ok) and wp.isfinite(t2) and (t2 > eps)
            valid2 = valid2 and (wp.abs(px2 - xc) <= half_px) and (wp.abs(py2 - yc) <= half_py)
            valid2 = valid2 and ((FT(1.0) - kpc * (pz2 - off)) >= FT(0.0))
            pick1 = valid1 and ((not valid2) or (t1 <= t2))
            pick2 = valid2 and (not pick1)
            if pick1:
                t_result = t1
                px_rel = px1 - xc
                py_rel = py1 - yc
                found = True
            elif pick2:
                t_result = t2
                px_rel = px2 - xc
                py_rel = py2 - yc
                found = True
            if not found:
                if t_max_x <= t_max_y:
                    i = i + step_x
                    t_max_x = t_max_x + t_delta_x
                else:
                    j = j + step_y
                    t_max_y = t_max_y + t_delta_y
            active = active and (i >= 0) and (i < num_x) and (j >= 0) and (j < num_y)
        gx, gy, gz = _conic_normal(px_rel, py_rel, gp[13], gp[14], floor)
        nlx, nly, nlz, ngx, ngy, ngz = _unit_and_face(gx, gy, gz, dx, dy, dz, tiny, sum_order)
        t_out = inf
        if found:
            t_out = t_result
        return t_out, nlx, nly, nlz, found, ngx, ngy, ngz, FT(0.0), FT(0.0)

    # -- the asphere ------------------------------------------------------------
    #
    # gp: c, (1 + K) c**2, (1 + K) c, 1 + K, -c, aperture**2, radicand floor,
    # radicand minimum (k u), tiny, guard eta, residual k, scan a**2, z_lo,
    # z_hi; gi: coefficients n, odd flag, max iterations, scan samples;
    # tab: the coefficients a_i, then (i + 1) a_i (formed in float64 on the
    # host, as the Python expressions are), then j / m for j = 0..m.

    @wp.func
    def _poly(r2: FT, n: int, odd: int, tab: wp.array(dtype=FT)):
        # _AsphereGeometry._poly: (P, sigma_P).
        if n == 0:
            return FT(0.0), FT(0.0)
        if odd == 0:
            p = tab[n - 1] * FT(1.0)
            d = tab[n + n - 1] * FT(1.0)
            for k in range(n - 1):
                ii = n - 2 - k
                p = tab[ii] + r2 * p
                d = tab[n + ii] + r2 * d
            return r2 * p, FT(2.0) * d
        pos = r2 > FT(0.0)
        r = FT(0.0)
        if pos:
            r = wp.sqrt(r2)
        p = tab[n - 1] * FT(1.0)
        d = tab[n + n - 1] * FT(1.0)
        for k in range(n - 1):
            ii = n - 2 - k
            p = tab[ii] + r * p
            d = tab[n + ii] + r * d
        rden = FT(1.0)
        if pos:
            rden = r
        return r * p, d / rden

    @wp.func
    def _asph_eval(
        x: FT, y: FT, z: FT, dx: FT, dy: FT, dz: FT, c: FT, kc2: FT, rmin: FT,
        n: int, odd: int, tab: wp.array(dtype=FT),
    ):
        # _AsphereGeometry._evaluate: (f, f', |grad G|, P, domain).
        r2 = x * x + y * y
        under = FT(1.0) - kc2 * r2
        dom = under > rmin
        wsq = FT(1.0)
        if dom:
            wsq = under
        w = wp.sqrt(wsq)
        poly, sigma_p = _poly(r2, n, odd, tab)
        sag = (c * r2) / (FT(1.0) + w) + poly
        sigma = _rdiv(c, w) + sigma_p
        f = z - sag
        fp = dz - sigma * (x * dx + y * dy)
        gnorm = wp.sqrt((sigma * sigma) * r2 + FT(1.0))
        return f, fp, gnorm, poly, dom

    @wp.func
    def _asph_refine(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT,
        t0: FT, seed_ok: bool, conic_seed: bool, t_neg0: FT, t_pos0: FT, bracket0: bool,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT),
    ):
        # _AsphereGeometry._refine for one lane.
        c = gp[0]
        kc2 = gp[1]
        rmin = gp[7]
        eta = gp[9]
        kres = gp[10]
        n = gi[0]
        odd = gi[1]
        max_it = gi[2]
        t = FT(0.0)
        if seed_ok:
            t = t0
        active = seed_ok
        status = FT(_ST_NO_SEED)
        if seed_ok:
            status = FT(_ST_NOT_CONVERGED)
        steps = FT(0.0)
        t_neg = t_neg0
        t_pos = t_pos0
        has_neg = bracket0
        has_pos = bracket0
        t_prev = t
        has_prev = bool(False)
        fp_last = FT(1.0)
        gn_last = FT(1.0)
        tol_last = FT(1.0)
        for it in range(max_it + 1):
            x = ox + t * dx
            y = oy + t * dy
            z = oz + t * dz
            f, fp, gn, poly, dom = _asph_eval(x, y, z, dx, dy, dz, c, kc2, rmin, n, odd, tab)
            if it == 0 and conic_seed:
                f = flipsign(poly)
            scale = _max_nan(_max_nan(wp.abs(x), wp.abs(y)), _max_nan(wp.abs(z), FT(1.0)))
            tol = (kres * ulp(scale)) * gn
            conv = wp.abs(f) <= tol
            if it == 0:
                conv = conv and (dom or conic_seed)
            else:
                conv = conv and dom
            if active:
                fp_last = fp
                gn_last = gn
                tol_last = tol
            guard = wp.abs(fp) >= eta * gn
            safe_fp = FT(1.0)
            if guard:
                safe_fp = fp
            t_newton = t - f / safe_fp
            done = active and conv
            if done and guard and wp.isfinite(t_newton):
                t = t_newton
            if done:
                status = FT(_ST_HIT)
            active = active and (not conv)
            if it == max_it:
                break
            neg = active and dom and (f < FT(0.0))
            pos = active and dom and (f > FT(0.0))
            if neg:
                t_neg = t
            if pos:
                t_pos = t
            has_neg = has_neg or neg
            has_pos = has_pos or pos
            bracket = has_neg and has_pos
            lo = _min_nan(t_neg, t_pos)
            hi = _max_nan(t_neg, t_pos)
            inside = (t_newton > lo) and (t_newton < hi)
            newton_ok = dom and wp.isfinite(t_newton) and ((bracket and inside) or ((not bracket) and guard))
            t_mid = FT(0.5) * (t_neg + t_pos)
            t_back = FT(0.5) * (t_prev + t)
            stuck = active and (not newton_ok) and (not bracket) and (dom or (not has_prev))
            if stuck and dom:
                status = FT(_ST_GRAZING)
            if stuck and (not dom):
                status = FT(_ST_DOMAIN)
            active = active and (not stuck)
            t_new = t_back
            if bracket:
                t_new = t_mid
            if newton_ok:
                t_new = t_newton
            if active and dom:
                t_prev = t
            has_prev = has_prev or (active and dom)
            if active:
                t = t_new
                steps = steps + FT(1.0)
            else:
                steps = steps + FT(0.0)
        return t, status, steps, fp_last, gn_last, tol_last

    @wp.func
    def _asph_scan(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT),
    ):
        # _AsphereGeometry._scan for one ray.
        c = gp[0]
        kc2 = gp[1]
        rmin = gp[7]
        tiny = gp[8]
        kres = gp[10]
        aa = gp[11]
        z_lo = gp[12]
        z_hi = gp[13]
        n = gi[0]
        odd = gi[1]
        m = gi[3]
        inf = FT(wp.inf)
        dz_ok = wp.abs(dz) > tiny
        dz_den = FT(1.0)
        if dz_ok:
            dz_den = dz
        inv_dz = FT(1.0) / dz_den
        ta = (z_lo - oz) * inv_dz
        tb = (z_hi - oz) * inv_dz
        in_slab = (oz >= z_lo) and (oz <= z_hi)
        tz0 = inf
        tz1 = -inf
        if in_slab:
            tz0 = -inf
            tz1 = inf
        if dz_ok:
            tz0 = _min_nan(ta, tb)
            tz1 = _max_nan(ta, tb)
        qa = dx * dx + dy * dy
        qb = ox * dx + oy * dy
        qc = (ox * ox + oy * oy) - aa
        disc = qb * qb - qa * qc
        qa_ok = qa > tiny
        dpos = FT(0.0)
        if disc > FT(0.0):
            dpos = disc
        root = wp.sqrt(dpos)
        qa_den = FT(1.0)
        if qa_ok:
            qa_den = qa
        inv_qa = FT(1.0) / qa_den
        in_cyl = qc <= FT(0.0)
        tc0 = inf
        tc1 = -inf
        if in_cyl:
            tc0 = -inf
            tc1 = inf
        if qa_ok:
            tc0 = (flipsign(qb) - root) * inv_qa
            tc1 = (flipsign(qb) + root) * inv_qa
        if qa_ok and (disc < FT(0.0)):
            tc0 = inf
        t_start = _max_nan(_max_nan(tz0, tc0), eps + FT(0.0))
        t_end = _min_nan(tz1, tc1)
        seg_ok = (t_end > t_start) and wp.isfinite(t_start) and wp.isfinite(t_end)
        span = FT(0.0)
        if seg_ok:
            span = t_end - t_start
        else:
            t_start = FT(0.0)
        found = bool(False)
        t_neg = FT(0.0)
        t_pos = FT(0.0)
        f_neg = FT(0.0)
        f_pos = FT(0.0)
        # The samples from the last down, so the earliest sign change remains.
        ts1 = t_start + span * tab[n + n + m]
        f1, fp1, gn1, poly1, dom1 = _asph_eval(ox + ts1 * dx, oy + ts1 * dy, oz + ts1 * dz, dx, dy, dz, c, kc2, rmin, n, odd, tab)
        for k in range(m):
            jj = m - 1 - k
            ts0 = t_start + span * tab[n + n + jj]
            x0 = ox + ts0 * dx
            y0 = oy + ts0 * dy
            z0 = oz + ts0 * dz
            f0, fp0, gn0, poly0, dom0 = _asph_eval(x0, y0, z0, dx, dy, dz, c, kc2, rmin, n, odd, tab)
            change = seg_ok and dom0 and dom1 and ((f0 > FT(0.0)) != (f1 > FT(0.0)))
            if jj == 0:
                scale = _max_nan(_max_nan(wp.abs(x0), wp.abs(y0)), _max_nan(wp.abs(z0), FT(1.0)))
                on_surface = wp.abs(f0) <= (kres * ulp(scale)) * gn0
                change = change and (not on_surface)
            if change:
                if not (f0 > FT(0.0)):
                    t_neg = ts0
                    t_pos = ts1
                    f_neg = f0
                    f_pos = f1
                else:
                    t_neg = ts1
                    t_pos = ts0
                    f_neg = f1
                    f_pos = f0
            found = found or change
            ts1 = ts0
            f1 = f0
            dom1 = dom0
        den = FT(1.0)
        if found:
            den = f_pos - f_neg
        t_seed = t_neg + ((t_pos - t_neg) * flipsign(f_neg)) / den
        if not found:
            t_seed = FT(0.0)
        return t_seed, t_neg, t_pos, found

    @wp.func
    def _asph_candidate(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        t_c: FT, st_c: FT, steps_c: FT, fp: FT, gn: FT, tol: FT, ap2: FT, eta: FT,
    ):
        px = ox + t_c * dx
        py = oy + t_c * dy
        in_aperture = (px * px + py * py) <= ap2
        converged = st_c == FT(0.0)
        took_steps = steps_c > FT(0.0)
        guard_root = wp.abs(fp) >= eta * gn
        grazing = converged and took_steps and (not guard_root)
        ahead = wp.isfinite(t_c) and (t_c > eps)
        valid = converged and (not grazing) and ahead and in_aperture
        st = st_c
        if grazing:
            st = FT(_ST_GRAZING)
        if converged and (not grazing) and (not ahead):
            st = FT(_ST_BEHIND)
        if converged and (not grazing) and ahead and (not in_aperture):
            st = FT(_ST_APERTURE)
        if (st_c == FT(_ST_NOT_CONVERGED)) and (not in_aperture):
            st = FT(_ST_APERTURE)
        afp = wp.abs(fp)
        uden = FT(1.0)
        if afp > FT(0.0):
            uden = afp
        return valid, st, tol / uden

    @wp.func
    def g_asphere(
        ox: FT, oy: FT, oz: FT, dx: FT, dy: FT, dz: FT, eps: FT,
        gp: wp.array(dtype=FT), gi: wp.array(dtype=int), tab: wp.array(dtype=FT), aux: wp.array(dtype=FT),
        sum_order: int, recip: int,
    ):
        c = gp[0]
        kc = gp[2]
        kp = gp[3]
        negc = gp[4]
        ap2 = gp[5]
        floor = gp[6]
        tiny = gp[8]
        eta = gp[9]
        n = gi[0]
        odd = gi[1]
        t1, t2, solv1, solv2 = _conic_roots(ox, oy, oz, dx, dy, dz, c, kp, floor, tiny)
        seed1 = solv1 and wp.isfinite(t1) and ((FT(1.0) - kc * (oz + t1 * dz)) >= FT(0.0))
        seed2 = solv2 and wp.isfinite(t2) and ((FT(1.0) - kc * (oz + t2 * dz)) >= FT(0.0))
        t3, tn3, tp3, found3 = _asph_scan(ox, oy, oz, dx, dy, dz, eps, gp, gi, tab)
        ta, sa0, na, fa, ga, ola = _asph_refine(ox, oy, oz, dx, dy, dz, t1, seed1, True, FT(0.0), FT(0.0), False, gp, gi, tab)
        tb, sb0, nb, fb, gb, olb = _asph_refine(ox, oy, oz, dx, dy, dz, t2, seed2, True, FT(0.0), FT(0.0), False, gp, gi, tab)
        ts, ss0, ns, fs, gs, ols = _asph_refine(ox, oy, oz, dx, dy, dz, t3, found3, False, tn3, tp3, found3, gp, gi, tab)
        va, sa, ua = _asph_candidate(ox, oy, oz, dx, dy, dz, eps, ta, sa0, na, fa, ga, ola, ap2, eta)
        vb, sb, ub = _asph_candidate(ox, oy, oz, dx, dy, dz, eps, tb, sb0, nb, fb, gb, olb, ap2, eta)
        vs, ss, us = _asph_candidate(ox, oy, oz, dx, dy, dz, eps, ts, ss0, ns, fs, gs, ols, ap2, eta)
        pick1 = va and ((not vb) or (ta <= tb))
        pick2 = vb and (not pick1)
        hit_c = pick1 or pick2
        t_cc = FT(0.0)
        u_c = FT(0.0)
        if pick2:
            t_cc = tb
            u_c = ub
        if pick1:
            t_cc = ta
            u_c = ua
        pick3 = vs and ((not hit_c) or (ts < t_cc - (u_c + us)))
        pick1 = pick1 and (not pick3)
        pick2 = pick2 and (not pick3)
        hit = pick1 or pick2 or pick3
        t_star = FT(0.0)
        if pick2:
            t_star = tb
        if pick1:
            t_star = ta
        if pick3:
            t_star = ts
        status = FT(_ST_HIT)
        if not hit:
            status = _max_nan(_max_nan(sa, sb), ss)
        steps = na
        if pick2:
            steps = nb
        if pick3:
            steps = ns
        t_out = FT(wp.inf)
        if hit:
            t_out = t_star
        px = ox + t_star * dx
        py = oy + t_star * dy
        if not hit:
            px = FT(0.0)
            py = FT(0.0)
        gx, gy, gz = _conic_normal(px, py, gp[1], negc, floor)
        pr, sigma_p = _poly(px * px + py * py, n, odd, tab)
        gx = gx - (px * sigma_p + FT(0.0))
        gy = gy - (py * sigma_p + FT(0.0))
        gz = gz - FT(0.0)
        nlx, nly, nlz, ngx, ngy, ngz = _unit_and_face(gx, gy, gz, dx, dy, dz, tiny, sum_order)
        return t_out, nlx, nly, nlz, hit, ngx, ngy, ngz, status, steps

    # -- the kernels ------------------------------------------------------------

    def _stage_kernel(geom, kind):
        def k(
            x: wp.array(dtype=FT), y: wp.array(dtype=FT), z: wp.array(dtype=FT),
            L: wp.array(dtype=FT), M: wp.array(dtype=FT), N: wp.array(dtype=FT),
            alive: wp.array(dtype=wp.bool),
            xf: wp.array(dtype=FT),
            gp: wp.array(dtype=FT),
            gi: wp.array(dtype=int),
            tab: wp.array(dtype=FT),
            aux: wp.array(dtype=FT),
            sum_order: int,
            recip: int,
            mm_small: wp.array3d(dtype=int),
            mm_meta: wp.array(dtype=int),
            mm_tail: wp.array3d(dtype=int),
            mode: int,
            comp_index: int,
            first: int,
            t_hit: wp.array(dtype=FT),
            normals: wp.array2d(dtype=FT),
            hit: wp.array(dtype=wp.bool),
            n_geom: wp.array2d(dtype=FT),
            t_adv_out: wp.array(dtype=FT),
            t_local_out: wp.array(dtype=FT),
            st_out: wp.array(dtype=FT),
            steps_out: wp.array(dtype=FT),
            t_run: wp.array(dtype=FT),
            n_run: wp.array2d(dtype=FT),
            g_run: wp.array2d(dtype=FT),
            idx_run: wp.array(dtype=wp.int32),
        ):
            i = wp.tid()
            n = x.shape[0]
            code_f = _row_code(i, n, 0, mm_small, mm_meta, mm_tail)
            code_b = _row_code(i, n, 1, mm_small, mm_meta, mm_tail)
            xi = x[i]
            yi = y[i]
            zi = z[i]
            plx, ply, plz = _to_local(xi, yi, zi, xf, code_f)
            dx, dy, dz = _dir_local(L[i], M[i], N[i], xf, code_f)
            t_adv = flipsign(_sum3(plx * dx, ply * dy, plz * dz, sum_order))
            ox = plx + t_adv * dx
            oy = ply + t_adv * dy
            oz = plz + t_adv * dz
            # BaseComponent.intersect: the threshold from the global position.
            t_min = _accept_t_min(_max_nan(_max_nan(wp.abs(xi), wp.abs(yi)), wp.abs(zi)))
            eps = t_min - t_adv
            t_loc, nlx, nly, nlz, hit_l, ngx, ngy, ngz, st, stp = geom(
                ox, oy, oz, dx, dy, dz, eps, gp, gi, tab, aux, sum_order, recip
            )
            th = t_loc + t_adv
            accepted = th > t_min
            if not accepted:
                th = FT(wp.inf)
            if not alive[i]:
                th = FT(wp.inf)
            h = hit_l and accepted and alive[i]
            gx, gy, gz = _to_global_normal(nlx, nly, nlz, xf, code_b)
            qx, qy, qz = _to_global_normal(ngx, ngy, ngz, xf, code_b)
            t_adv_out[i] = t_adv
            t_local_out[i] = t_loc
            if st_out.shape[0] == n:
                st_out[i] = st
                steps_out[i] = stp
            if mode == 0:
                t_hit[i] = th
                hit[i] = h
                normals[i, 0] = gx
                normals[i, 1] = gy
                normals[i, 2] = gz
                n_geom[i, 0] = qx
                n_geom[i, 1] = qy
                n_geom[i, 2] = qz
            else:
                if first == 1:
                    t_run[i] = FT(wp.inf)
                    n_run[i, 0] = FT(0.0)
                    n_run[i, 1] = FT(0.0)
                    n_run[i, 2] = FT(0.0)
                    g_run[i, 0] = FT(0.0)
                    g_run[i, 1] = FT(0.0)
                    g_run[i, 2] = FT(0.0)
                    idx_run[i] = wp.int32(-1)
                if h and (th < t_run[i]):
                    t_run[i] = th
                    n_run[i, 0] = gx
                    n_run[i, 1] = gy
                    n_run[i, 2] = gz
                    g_run[i, 0] = qx
                    g_run[i, 1] = qy
                    g_run[i, 2] = qz
                    idx_run[i] = wp.int32(comp_index)

        return _kernel(k, kind, bits, backward=kind in TAPE_KINDS)

    def k_absmax(
        x: wp.array(dtype=FT), y: wp.array(dtype=FT), z: wp.array(dtype=FT),
        L: wp.array(dtype=FT), M: wp.array(dtype=FT), N: wp.array(dtype=FT),
        xf: wp.array(dtype=FT),
        sum_order: int,
        mm_small: wp.array3d(dtype=int),
        mm_meta: wp.array(dtype=int),
        mm_tail: wp.array3d(dtype=int),
        out: wp.array(dtype=FT),
    ):
        # ``be.abs(origins).max()`` over the advanced local origins, the
        # frustum's axial window: the same preamble as the stage kernel.
        i = wp.tid()
        n = x.shape[0]
        code_f = _row_code(i, n, 0, mm_small, mm_meta, mm_tail)
        plx, ply, plz = _to_local(x[i], y[i], z[i], xf, code_f)
        dx, dy, dz = _dir_local(L[i], M[i], N[i], xf, code_f)
        t_adv = flipsign(_sum3(plx * dx, ply * dy, plz * dz, sum_order))
        ox = plx + t_adv * dx
        oy = ply + t_adv * dy
        oz = plz + t_adv * dz
        wp.atomic_max(out, 0, wp.max(wp.max(wp.abs(ox), wp.abs(oy)), wp.abs(oz)))

    k_absmax = _kernel(k_absmax, "absmax", bits)

    def k_merge(
        t_c: wp.array(dtype=FT),
        normals_c: wp.array2d(dtype=FT),
        hit_c: wp.array(dtype=wp.bool),
        n_geom_c: wp.array2d(dtype=FT),
        comp_index: int,
        first: int,
        t_run: wp.array(dtype=FT),
        n_run: wp.array2d(dtype=FT),
        g_run: wp.array2d(dtype=FT),
        idx_run: wp.array(dtype=wp.int32),
    ):
        # A component's own intersect folded into the running nearest hit.
        i = wp.tid()
        if first == 1:
            t_run[i] = FT(wp.inf)
            n_run[i, 0] = FT(0.0)
            n_run[i, 1] = FT(0.0)
            n_run[i, 2] = FT(0.0)
            g_run[i, 0] = FT(0.0)
            g_run[i, 1] = FT(0.0)
            g_run[i, 2] = FT(0.0)
            idx_run[i] = wp.int32(-1)
        if hit_c[i] and (t_c[i] < t_run[i]):
            t_run[i] = t_c[i]
            n_run[i, 0] = normals_c[i, 0]
            n_run[i, 1] = normals_c[i, 1]
            n_run[i, 2] = normals_c[i, 2]
            g_run[i, 0] = n_geom_c[i, 0]
            g_run[i, 1] = n_geom_c[i, 1]
            g_run[i, 2] = n_geom_c[i, 2]
            idx_run[i] = wp.int32(comp_index)

    k_merge = _kernel(k_merge, "merge", bits)

    kernels = {
        "cavity": _stage_kernel(g_cavity, "cavity"),
        "conic": _stage_kernel(g_conic, "conic"),
        "plane": _stage_kernel(g_plane, "plane"),
        "finite_plane": _stage_kernel(g_finite_plane, "finite_plane"),
        "annulus": _stage_kernel(g_annulus, "annulus"),
        "sphere": _stage_kernel(g_sphere, "sphere"),
        "frustum": _stage_kernel(g_frustum, "frustum"),
        "lenslet": _stage_kernel(g_lenslet, "lenslet"),
        "asphere": _stage_kernel(g_asphere, "asphere"),
    }
    kernels["mm_probe"] = k_mm_probe
    kernels["absmax"] = k_absmax
    kernels["merge"] = k_merge
    return kernels


_KERNELS = {
    torch.float64: _make_kernels(wp.float64, _fma64, _ulp64, _neg64),
    torch.float32: _make_kernels(wp.float32, _fma32, _ulp32, _neg32),
}


_SUM_ORDER: dict = {}


def sum_order(dtype: torch.dtype, device: Any) -> int:
    """How torch orders a three-term sum over the last axis on this device and dtype.

    0 for ``(x + y) + z``, 1 for ``(x + z) + y``, plus 2 when the reduction
    starts from a +0 accumulator (a sum of negative zeros is then +0, as it is
    on the Apple silicon CPU). Probed once with the row ``(s, 1, s)``, ``s``
    half an ulp of 1: summed as ``(s + 1) + s`` it rounds to 1 twice (ties to
    even), summed as ``(s + s) + 1`` it is ``1 + 2 s`` exactly; and with a row
    of negative zeros. Cached per dtype and device.

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
        # A row of negative zeros: a reduction that starts from +0 returns
        # +0, one that starts from the first term returns -0.
        zeros = torch.full((1, 3), -0.0, dtype=dtype, device=device)
        if not bool(torch.signbit(zeros.sum(dim=1))[0]):
            order += 2
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
_MM_PREFERENCE = (27, 3, 24, 0, 26, 2) + tuple(c for c in range(48) if c not in (27, 3, 24, 0, 26, 2))
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
    """For each of ``n`` rows, which of the 48 order codes reproduce torch's ``a @ R`` (or ``@ R.T``)
    on every one of ``trials`` random products of width ``n`` and on one product of negative
    zeros (which tells a +0 accumulator from none); an (n, 48) boolean tensor."""
    kernel = _KERNELS[dtype]["mm_probe"]
    ivt = torch.int64 if dtype == torch.float64 else torch.int32
    acc = torch.ones((n, MM_CODES), dtype=torch.bool, device=device)
    for trial in range(trials + 1):
        # Unit normal entries: products of one size, so the orders' roundings
        # differ on 15 to 45 percent of the entries (a spread of magnitudes
        # hides the smaller terms' rounding and discriminates less).
        # Drawn on the device into fresh (aligned) tensors, as the stage's
        # own operands are: no host copy per trial.
        a = torch.randn((n, 3), generator=rng, dtype=dtype, device=device)
        r = torch.randn((3, 3), generator=rng, dtype=dtype, device=device)
        if trial == trials:
            a = torch.full((n, 3), -0.0, dtype=dtype, device=device)
            r = torch.abs(r)
        got = a @ (r.T if form else r)
        rk = (r.T if form else r).contiguous()
        cand = torch.empty((n, 3, MM_CODES), dtype=dtype, device=device)
        kwargs = {"dim": (n, 3, MM_CODES), "inputs": [wp.from_torch(a), wp.from_torch(rk)],
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
    common = np.ones(MM_CODES, dtype=bool)
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
        share = np.ones((r, MM_CODES), dtype=bool)
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
    widths = list(range(n0, n0 + 16))
    probes = [_mm_probe_rows(dtype, device, form, n, 6, rng) for n in widths]
    parts = np.split(torch.cat(probes).cpu().numpy(), np.cumsum(widths)[:-1])
    for n, m in zip(widths, parts, strict=True):
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
    and summed. Measured (2026-09-27, research repository issue 77): on CUDA
    cuBLAS gives the fused chain :data:`MM_FMA_CHAIN` from 17 rows up and
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
    rng = torch.Generator(device=tdev)
    rng.manual_seed(20260927)
    small = np.full((2, MM_SMALL, MM_SMALL - 1), -1, dtype=np.int32)
    meta = np.zeros((2, 5), dtype=np.int32)
    tail = np.full((4, 16, 16), -1, dtype=np.int32)
    refusal = None
    with torch.no_grad():
        for form in (0, 1):
            rows = [_mm_probe_rows(dtype, tdev, form, n, 24, rng) for n in range(1, MM_SMALL)]
            mid = [_mm_probe_rows(dtype, tdev, form, n, 24, rng) for n in _MM_MID]
            large = [_mm_probe_rows(dtype, tdev, form, n, 8 if n < 65536 else 2, rng) for n in _MM_LARGE]
            sizes = [m.shape[0] for m in rows + mid + large]
            flat = torch.cat(rows + mid + large).cpu().numpy()
            parts = np.split(flat, np.cumsum(sizes)[:-1])
            rows, mid, large = parts[: len(rows)], parts[len(rows): len(rows) + len(mid)], parts[len(rows) + len(mid):]
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
_NP_FLOAT = {torch.float64: np.float64, torch.float32: np.float32}

_prepared: set = set()
_initialised = [False]
#: ``(device, dtype, kernel name)`` of the kernels loaded on a device.
_LOADED: set = set()
#: How many kernel modules compile at once when a scene's kernels load
#: (:func:`load_kernels`); None lets Warp choose (up to four threads).
LOAD_WORKERS: int | None = None


def _init() -> None:
    if not _initialised[0]:
        wp.init()
        _initialised[0] = True


def prepare(device: Any) -> None:
    """Load the kernels on ``device`` and probe torch's orders there, before any bounce (and any graph capture).

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
    # Only the probe's kernels here; a scene's kernels load when it first
    # needs them (load_kernels), before the bounce a replay records.
    load_kernels(tdev, kinds=(), extras=("mm_probe",))
    # The probes read values to the host: done here, before any bounce,
    # never inside a recorded one.
    for dtype in (torch.float64, torch.float32):
        sum_order(dtype, tdev)
        scalar_division(dtype, tdev)
        matmul_orders(dtype, tdev)
        # The one-element arguments a launch does not use, allocated here
        # rather than first inside a recorded bounce.
        _placeholders(torch.empty(0, dtype=dtype, device=tdev))
    _prepared.add(name)


def _torch_device(device: Any) -> torch.device:
    tdev = torch.device(device)
    if tdev.type == "cuda" and tdev.index is None:
        tdev = torch.device("cuda", torch.cuda.current_device())
    return tdev


def load_kernels(
    device: Any,
    kinds=None,
    dtypes=(torch.float64, torch.float32),
    extras=("absmax", "merge"),
    max_workers: int | None = None,
) -> list:
    """Compile (or read from Warp's kernel cache) and load the kernels of ``kinds`` on ``device``.

    Each kernel is a module of its own (:func:`_kernel`), so only what is
    asked for compiles, and the modules not yet loaded compile in parallel
    (``max_workers`` threads; :data:`LOAD_WORKERS` when None). A kernel
    already loaded on the device is skipped. Called by :func:`prepare` for
    the probe's kernel, and by :func:`intersect_scene` for the kinds of the
    scene in hand, on the first bounce that meets them (an eager bounce, so
    a CUDA-graph replay records launches only).

    Args:
        device: The torch device.
        kinds: The geometry kinds to load (None: every kind of :data:`KINDS`).
        dtypes: The float types to load them for.
        extras: Other kernels of the stage to load with them (``"absmax"``,
            ``"merge"``, ``"mm_probe"``).
        max_workers: Parallel compilations; None for :data:`LOAD_WORKERS`.

    Returns:
        The names ``(name, dtype)`` loaded by this call.
    """
    _init()
    tdev = _torch_device(device)
    names = list(KINDS if kinds is None else kinds) + list(extras)
    todo, modules = [], []
    for dtype in dtypes:
        for name in names:
            key = (str(tdev), dtype, name)
            if key in _LOADED:
                continue
            todo.append(key)
            modules.append(_KERNELS[dtype][name].module)
    if modules:
        workers = LOAD_WORKERS if max_workers is None else max_workers
        wp.force_load(device=wp.device_from_torch(tdev), modules=modules, max_workers=workers)
        _LOADED.update(todo)
    return [(name, dtype) for _, dtype, name in todo]


def _ensure_loaded(kinds, like, merge: bool = False) -> None:
    """Load, in one parallel call, the kernels a launch on ``like``'s device and dtype is about to use."""
    names = {k for k in kinds if k is not None}
    if "frustum" in names:
        names.add("absmax")
    if merge:
        names.add("merge")
    dev = str(like.device) if like.device.type != "cuda" or like.device.index is not None else str(
        _torch_device(like.device))
    if all((dev, like.dtype, name) in _LOADED for name in names):
        return
    load_kernels(like.device, kinds=sorted(names), dtypes=(like.dtype,), extras=())


def kernel_modules(kinds=None, dtypes=(torch.float64, torch.float32), extras=("absmax", "merge", "mm_probe")) -> list:
    """The Warp modules of the stage's kernels (one per kernel and float type)."""
    names = list(KINDS if kinds is None else kinds) + list(extras)
    return [_KERNELS[dtype][name].module for dtype in dtypes for name in names]


def compile_cache(arch, kinds=None, dtypes=(torch.float64, torch.float32)) -> list:
    """Compile the stage's kernels for a CUDA architecture ahead of time, without a device.

    Writes Warp's kernel cache (``warp.config.kernel_cache_dir``, which the
    ``WARP_CACHE_PATH`` environment variable sets) for the compute
    capability ``arch`` (80 for an A100), as CUBIN and as PTX, in the
    layout a load reads: a later process on a device of that architecture
    finds every kernel compiled and compiles nothing. Meant for a container
    image's build step (research repository issue 85: a fresh CUDA
    container otherwise compiles the stage on its first trace). A module's
    file names carry a hash of its kernels' source and options only, not of
    the machine that compiled it.

    Args:
        arch: A compute capability (e.g. 80) or several.
        kinds: The geometry kinds (None: every kind).
        dtypes: The float types.

    Returns:
        The paths written.
    """
    _init()
    arches = [arch] if isinstance(arch, int) else list(arch)
    paths = []
    for module in kernel_modules(kinds, dtypes):
        for use_ptx in (False, True):
            paths += wp.compile_aot_module(module, arch=arches, use_ptx=use_ptx)
    return paths


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
# Routing: what runs through the component's own intersect, and why
# ---------------------------------------------------------------------------

#: Why a component's intersection went to its own ``intersect`` instead of
#: a kernel, counted per call since the last :func:`reset_routed`.
ROUTE_KIND = "a geometry kind the kernels do not cover"
ROUTE_PLACEMENT = "a placement that carries a gradient"
ROUTE_GRADIENT = "a gradient through a kind without the tape adjoint"
ROUTE_FORWARD_AD = "a forward-mode dual tensor"
ROUTE_WIDTH = "a width at which no probed order reproduces torch's placement product"

_ROUTED: dict = {}


def routed_counts() -> dict:
    """``{reason: calls}`` of the component intersections routed to torch since the last reset."""
    return dict(_ROUTED)


def reset_routed() -> None:
    """Forget the routing and launch counts (the torch backend does this at the start of a trace)."""
    _ROUTED.clear()
    _LAUNCHED.clear()


def _route(reason: str) -> None:
    _ROUTED[reason] = _ROUTED.get(reason, 0) + 1


_LAUNCHED: dict = {}


def launch_counts() -> dict:
    """``{kind: calls}`` of the component intersections that went through a kernel since the last reset.

    Counted on the host when a launch is issued: a CUDA-graph replay repeats
    recorded launches without counting them again.
    """
    return dict(_LAUNCHED)


def _launched(kind: str) -> None:
    _LAUNCHED[kind] = _LAUNCHED.get(kind, 0) + 1


# ---------------------------------------------------------------------------
# Which components the kernels take, and their parameters
# ---------------------------------------------------------------------------

_REGISTRY: list = []


def _registry() -> list:
    """``(geometry class, kind, methods that must be the class's own)``, most specific first."""
    if not _REGISTRY:
        from optiland.nonsequential.components.geometry.analytic.annulus import (  # noqa: PLC0415
            AnnularPlaneGeometry,
        )
        from optiland.nonsequential.components.geometry.analytic.asphere import (  # noqa: PLC0415
            EvenAsphereGeometry,
            OddAsphereGeometry,
            _AsphereGeometry,
        )
        from optiland.nonsequential.components.geometry.analytic.conic import (  # noqa: PLC0415
            ConicGeometry,
        )
        from optiland.nonsequential.components.geometry.analytic.frustum import (  # noqa: PLC0415
            CylindricalFrustumGeometry,
        )
        from optiland.nonsequential.components.geometry.analytic.lenslet_array import (  # noqa: PLC0415
            LensletArrayGeometry,
        )
        from optiland.nonsequential.components.geometry.analytic.plane import (  # noqa: PLC0415
            FinitePlaneGeometry,
            PlaneGeometry,
        )
        from optiland.nonsequential.components.geometry.analytic.sphere import (  # noqa: PLC0415
            SphereGeometry,
        )
        from optiland.nonsequential.components.geometry.analytic.spherical_cavity import (  # noqa: PLC0415
            SphericalCavityGeometry,
        )

        asphere_methods = (
            "ray_intersect", "_refine", "_scan", "_scan_region", "_evaluate", "_poly",
            "_normal_local", "_curvature", "_sync_base",
        )
        _REGISTRY.extend([
            (SphericalCavityGeometry, "cavity", ("ray_intersect", "_on_wall", "_root_valid")),
            (EvenAsphereGeometry, "asphere", asphere_methods),
            (OddAsphereGeometry, "asphere", asphere_methods),
            (LensletArrayGeometry, "lenslet", ("ray_intersect", "_z_slab", "_cap_edge_sag")),
            (ConicGeometry, "conic",
             ("ray_intersect", "_root_valid", "_normal_local", "_curvature", "_quadratic_roots")),
            (FinitePlaneGeometry, "finite_plane", ("ray_intersect",)),
            (PlaneGeometry, "plane", ("ray_intersect",)),
            (AnnularPlaneGeometry, "annulus", ("ray_intersect",)),
            (SphereGeometry, "sphere", ("ray_intersect",)),
            (CylindricalFrustumGeometry, "frustum", ("ray_intersect",)),
        ])
        _REGISTRY.append((_AsphereGeometry, None, ()))
    return _REGISTRY


#: Every kind the kernels cover.
KINDS = ("cavity", "conic", "plane", "finite_plane", "annulus", "sphere", "frustum", "lenslet", "asphere")


def kind_of(component) -> str | None:
    """The kernel kind of ``component``, or None when it keeps its own intersect.

    A component is covered when ``BaseComponent.intersect`` is its own and its
    geometry is one of the analytic kinds (a subclass counts when it overrides
    none of the methods the intersection calls). A lenslet array whose cap is
    not a plain conic, a degenerate (zero-height) frustum and an odd/even
    asphere subclass that changes its flag are left to their own intersect.
    """
    from optiland.nonsequential.components.base import BaseComponent  # noqa: PLC0415

    if type(component).intersect is not BaseComponent.intersect:
        return None
    geometry = getattr(component, "geometry", None)
    if geometry is None:
        return None
    gtype = type(geometry)
    for cls, kind, methods in _registry():
        if not isinstance(geometry, cls):
            continue
        if kind is None or not all(getattr(gtype, m) is getattr(cls, m) for m in methods):
            return None
        if kind == "lenslet":
            from optiland.nonsequential.components.geometry.analytic.conic import (  # noqa: PLC0415
                ConicGeometry,
            )

            if type(geometry._cap) is not ConicGeometry:
                return None
        if kind == "frustum" and _frustum_degenerate(geometry):
            return None
        return kind
    return None


def _frustum_degenerate(geometry) -> bool:
    from optiland.nonsequential._utils import as_float  # noqa: PLC0415
    from optiland.nonsequential.components.geometry.analytic.frustum import (  # noqa: PLC0415
        _DEGENERACY_K,
    )

    h_val = as_float(geometry.z_back - geometry.z_front)
    h_scale = max(abs(as_float(geometry.z_front)), abs(as_float(geometry.z_back)), 1.0)
    return bool(abs(h_val) < _DEGENERACY_K * np.spacing(h_scale))


def _is_tensor(v) -> bool:
    return isinstance(v, torch.Tensor)


_DTYPE_CONSTANTS: dict = {}


def _dtype_constants(dtype: torch.dtype) -> dict:
    """The dtype's tolerance constants, as the torch stage forms them (on the CPU; exact at both precisions)."""
    cached = _DTYPE_CONSTANTS.get(dtype)
    if cached is None:
        ones = torch.ones((), dtype=dtype)
        np_dtype = _NP_FLOAT[dtype]
        cached = {
            "floor": float(_tol.radicand_floor(ones)),
            "tiny": float(_tol.tiny_for(ones)),
            "rmin": float(_tol.radicand_min(ones)),
            "dz_min": float(8 * np.spacing(np_dtype(1.0))),
        }
        _DTYPE_CONSTANTS[dtype] = cached
    return cached


def _recip(value: float, dtype: torch.dtype) -> float:
    """``1 / value`` rounded once in the working dtype, as torch forms the reciprocal of a Python number."""
    np_dtype = _NP_FLOAT[dtype]
    return float(np_dtype(1.0) / np_dtype(value))


def _params(kind: str, geometry, dtype: torch.dtype):
    """``(gp, gi, tab)`` of one component: the scalars (Python floats, or tensors that carry a
    gradient for the tape kinds), the integers, and the table (floats), each value what the
    torch stage's Python expression gives before it meets a tensor."""
    k = _dtype_constants(dtype)
    floor, tiny = k["floor"], k["tiny"]
    if kind == "cavity":
        radius = geometry.radius
        inv = 1.0 if _is_tensor(radius) else _recip(radius, dtype)
        tab = []
        for port in geometry.ports:
            ax, ay, az = port.unit_axis
            tab += [ax, ay, az, port.cos_half_angle]
        return [radius, radius**2, floor, inv], [len(geometry.ports)], tab
    if kind == "conic":
        c = geometry._curvature()
        K = geometry.conic
        return [c, 1.0 + K, geometry.aperture_radius**2, (1.0 + K) * c, (1.0 + K) * c**2, -c, floor, tiny], [], []
    if kind == "plane":
        return [k["dz_min"]], [], []
    if kind == "finite_plane":
        if geometry.aperture_radius is not None:
            return [k["dz_min"], geometry.aperture_radius**2, 0.0, 0.0], [1], []
        return [k["dz_min"], 0.0, geometry.width / 2.0, geometry.height / 2.0], [0], []
    if kind == "annulus":
        return [k["dz_min"], geometry.z_offset, tiny, geometry.inner_radius**2, geometry.outer_radius**2], [], []
    if kind == "sphere":
        radius = geometry.radius
        has_ap = geometry.aperture_radius is not None
        ap = geometry.aperture_radius if has_ap else 0.0
        return [radius, radius**2, floor, _recip(radius, dtype), ap], [int(has_ap)], []
    if kind == "frustum":
        from optiland.nonsequential.components.geometry.analytic.frustum import (  # noqa: PLC0415
            _DEGENERACY_K,
        )

        h = geometry.z_back - geometry.z_front
        slope = (geometry.r_back - geometry.r_front) / h
        return [geometry.r_front, slope, geometry.z_front, geometry.z_back, tiny, float(_DEGENERACY_K)], [], []
    if kind == "lenslet":
        g = geometry
        px, py = g.pitch_x, g.pitch_y
        z_lo, z_hi = g._z_slab()
        c = g._cap._curvature()
        K = g.conic
        kp = 1.0 + K
        gp = [
            px, py, px / 2.0, py / 2.0,
            -(g.num_x * px) / 2.0, (g.num_x * px) / 2.0, -(g.num_y * py) / 2.0, (g.num_y * py) / 2.0,
            z_lo, z_hi, c, kp, kp * c, (1.0 + K) * c**2, -c, floor, tiny,
            _recip(px, dtype), _recip(py, dtype),
        ]
        return gp, [g.num_x, g.num_y], list(g._sag_offsets_flat)
    if kind == "asphere":
        g = geometry
        c = g._curvature()
        K = g.conic
        a, z_lo, z_hi = g._scan_region()
        coeffs = list(g.coefficients)
        n = len(coeffs)
        m = g.scan_samples
        tab = coeffs + [(i + 1) * coeffs[i] for i in range(n)] + [j / m for j in range(m + 1)]
        gp = [
            c, (1.0 + K) * c**2, (1.0 + K) * c, 1.0 + K, -c, g.aperture_radius**2,
            floor, k["rmin"], tiny, g.guard_eta, float(g.residual_k), a * a, z_lo, z_hi,
        ]
        return gp, [n, int(g._odd), g.max_iterations, m], tab
    raise ValueError(f"unknown kind {kind!r}")


def _param_tensors(kind: str, geometry) -> list:
    """The geometry's parameters that are tensors (a gradient's route)."""
    names = {
        "cavity": ("radius",),
        "conic": ("radius", "conic", "aperture_radius"),
        "plane": (),
        "finite_plane": ("width", "height", "aperture_radius"),
        "annulus": ("inner_radius", "outer_radius", "z_offset"),
        "sphere": ("radius", "aperture_radius"),
        "frustum": ("r_front", "r_back", "z_front", "z_back"),
        "lenslet": ("pitch_x", "pitch_y", "radius", "conic"),
        "asphere": ("radius", "conic", "aperture_radius"),
    }[kind]
    out = [getattr(geometry, name, None) for name in names]
    if kind == "asphere":
        coeffs = geometry.coefficients
        out += [coeffs] if _is_tensor(coeffs) else list(coeffs)
    return [v for v in out if _is_tensor(v)]


def _values_key(values) -> tuple:
    return tuple(("t", id(v)) if _is_tensor(v) else float(v) for v in values)


def _component_inputs(component, kind: str, like: torch.Tensor):
    """Placement vector, (gp, gi, tab) tensors of one component, cached on it.

    Constant inputs are built once per trace and dtype/device and kept while
    the placement's resident pair and the parameter values stay the same (the
    CUDA-graph replay records their addresses); a tape kind's parameter that
    carries a gradient is rebuilt at every call so it stays on the graph.
    """
    from optiland.nonsequential.components.base import _resident_transform  # noqa: PLC0415

    t_be, r_be = _resident_transform(component)
    key = (like.dtype, str(like.device))
    cache = getattr(component, "_warp_stage_cache", None)
    if cache is None or cache.get("key") != key or cache.get("t_be") is not t_be or cache.get("r_be") is not r_be:
        cache = {"key": key, "t_be": t_be, "r_be": r_be}
        component._warp_stage_cache = cache
    xf = cache.get("xf")
    if xf is None:
        xf = torch.cat([t_be.reshape(3), r_be.reshape(9)]).to(dtype=like.dtype, device=like.device).contiguous()
        if not xf.requires_grad:
            cache["xf"] = xf
    gp_values, gi_values, tab_values = _params(kind, component.geometry, like.dtype)
    if kind not in TAPE_KINDS:
        gp_values = [float(v) for v in gp_values]
        tab_values = [float(v) for v in tab_values]
    np_dtype = _NP_FLOAT[like.dtype]
    itkey = (tuple(int(v) for v in gi_values), tuple(float(v) for v in tab_values))
    if cache.get("itkey") != itkey:
        cache["itkey"] = itkey
        cache["gi"] = torch.as_tensor(np.asarray(gi_values or [0], dtype=np.int32), device=like.device)
        cache["tab"] = torch.as_tensor(
            np.asarray(tab_values or [0.0], dtype=np.float64).astype(np_dtype), device=like.device
        )
    if any(_is_tensor(v) for v in gp_values):
        gp = torch.cat([
            v.to(dtype=like.dtype, device=like.device).reshape(1)
            if _is_tensor(v)
            else torch.tensor([float(v)], dtype=like.dtype, device=like.device)
            for v in gp_values
        ])
    else:
        gkey = tuple(float(v) for v in gp_values)
        if cache.get("gkey") != gkey:
            cache["gkey"] = gkey
            cache["gp"] = torch.as_tensor(np.asarray(gkey, dtype=np.float64).astype(np_dtype), device=like.device)
        gp = cache["gp"]
    return xf, gp, cache["gi"], cache["tab"]


def _placeholders(like: torch.Tensor) -> dict:
    """One-element arrays for the kernel arguments a launch does not use, cached per dtype and device."""
    key = (like.dtype, str(like.device))
    cached = _PLACEHOLDERS.get(key)
    if cached is None:
        cached = {
            "f1": torch.zeros(1, dtype=like.dtype, device=like.device),
            "f13": torch.zeros((1, 3), dtype=like.dtype, device=like.device),
            "b1": torch.zeros(1, dtype=torch.bool, device=like.device),
            "i1": torch.zeros(1, dtype=torch.int32, device=like.device),
        }
        _PLACEHOLDERS[key] = cached
    return cached


_PLACEHOLDERS: dict = {}


# ---------------------------------------------------------------------------
# Launch, the custom operators and the autograd function
# ---------------------------------------------------------------------------


def _launch_kwargs(like: torch.Tensor, dim) -> dict:
    kwargs = {"dim": dim}
    if like.device.type == "cuda":
        kwargs["stream"] = wp.stream_from_torch(torch.cuda.current_stream(like.device))
    else:
        kwargs["device"] = wp.device_from_torch(like.device)
    return kwargs


def _launch(kind, fields, alive, xf, gp, gi, tab, aux, recip, mode, comp_index, first, own, run, extra,
            *, requires_grad=False, tape=None):
    """Wrap the torch tensors as Warp arrays and launch one component's kernel on torch's stream.

    ``own`` is (t_hit, normals, hit, n_geom) for mode 0, ``run`` the running
    nearest hit for mode 1, ``extra`` (t_adv, t_local, status, steps).
    Returns the Warp arrays (inputs, outputs) so a tape can find their
    gradients.
    """
    like = fields[0]
    n = int(like.shape[0])
    kernel = _KERNELS[like.dtype][kind]
    wp_fields = [wp.from_torch(f, requires_grad=requires_grad) for f in fields]
    wp_xf = wp.from_torch(xf, requires_grad=requires_grad)
    wp_gp = wp.from_torch(gp, requires_grad=requires_grad)
    tables = [wp.from_torch(t, dtype=wp.int32) for t in matmul_orders(like.dtype, like.device).tensors]
    wp_own = [wp.from_torch(t, requires_grad=requires_grad and t.dtype.is_floating_point) for t in own]
    wp_extra = [wp.from_torch(t, requires_grad=requires_grad) for t in extra]
    wp_run = [wp.from_torch(t) for t in run]
    inputs = [
        *wp_fields, wp.from_torch(alive), wp_xf, wp_gp, wp.from_torch(gi, dtype=wp.int32), wp.from_torch(tab),
        wp.from_torch(aux), sum_order(like.dtype, like.device), int(recip), *tables,
        int(mode), int(comp_index), int(first),
    ]
    kwargs = _launch_kwargs(like, n)
    kwargs["inputs"] = inputs
    kwargs["outputs"] = [*wp_own, *wp_extra, *wp_run]
    if n > 0:
        if tape is not None:
            with tape:
                wp.launch(kernel, **kwargs)
        else:
            wp.launch(kernel, **kwargs)
    return [*wp_fields, None, wp_xf, wp_gp], [*wp_own, *wp_extra]


def _frustum_aux(fields, xf, like):
    """``be.abs(origins).max()`` of the frustum's advanced local origins, one launch (and its buffer)."""
    aux = torch.zeros(1, dtype=like.dtype, device=like.device)
    n = int(like.shape[0])
    if n > 0:
        tables = [wp.from_torch(t, dtype=wp.int32) for t in matmul_orders(like.dtype, like.device).tensors]
        kwargs = _launch_kwargs(like, n)
        kwargs["inputs"] = [*(wp.from_torch(f) for f in fields), wp.from_torch(xf),
                            sum_order(like.dtype, like.device), *tables]
        kwargs["outputs"] = [wp.from_torch(aux)]
        wp.launch(_KERNELS[like.dtype]["absmax"], **kwargs)
    return aux


@torch.library.custom_op(
    "optiland_nsq::intersect_fused",
    mutates_args=("t_run", "n_run", "g_run", "idx_run", "t_adv", "t_local", "status", "steps"),
)
def _op_fused(
    kind: str, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, L: torch.Tensor, M: torch.Tensor,
    N: torch.Tensor, alive: torch.Tensor, xf: torch.Tensor, gp: torch.Tensor, gi: torch.Tensor,
    tab: torch.Tensor, recip: int, comp_index: int, first: int,
    t_run: torch.Tensor, n_run: torch.Tensor, g_run: torch.Tensor, idx_run: torch.Tensor,
    t_adv: torch.Tensor, t_local: torch.Tensor, status: torch.Tensor, steps: torch.Tensor,
) -> None:
    fields = [x, y, z, L, M, N]
    _launched(kind)
    ph = _placeholders(x)
    aux = _frustum_aux(fields, xf, x) if kind == "frustum" else ph["f1"]
    own = (ph["f1"], ph["f13"], ph["b1"], ph["f13"])
    _launch(kind, fields, alive, xf, gp, gi, tab, aux, recip, MODE_SELECT, comp_index, first, own,
            (t_run, n_run, g_run, idx_run), (t_adv, t_local, status, steps))


@_op_fused.register_fake
def _(kind, x, y, z, L, M, N, alive, xf, gp, gi, tab, recip, comp_index, first,
      t_run, n_run, g_run, idx_run, t_adv, t_local, status, steps):
    return None


@torch.library.custom_op("optiland_nsq::merge_hit", mutates_args=("t_run", "n_run", "g_run", "idx_run"))
def _op_merge(
    t_c: torch.Tensor, normals_c: torch.Tensor, hit_c: torch.Tensor, n_geom_c: torch.Tensor,
    comp_index: int, first: int,
    t_run: torch.Tensor, n_run: torch.Tensor, g_run: torch.Tensor, idx_run: torch.Tensor,
) -> None:
    n = int(t_c.shape[0])
    if n == 0:
        return
    kwargs = _launch_kwargs(t_c, n)
    kwargs["inputs"] = [wp.from_torch(t_c), wp.from_torch(normals_c), wp.from_torch(hit_c),
                        wp.from_torch(n_geom_c), int(comp_index), int(first)]
    kwargs["outputs"] = [wp.from_torch(t) for t in (t_run, n_run, g_run)] + [wp.from_torch(idx_run, dtype=wp.int32)]
    wp.launch(_KERNELS[t_c.dtype]["merge"], **kwargs)


@_op_merge.register_fake
def _(t_c, normals_c, hit_c, n_geom_c, comp_index, first, t_run, n_run, g_run, idx_run):
    return None


def _own_outputs(n: int, like: torch.Tensor):
    return (
        torch.empty(n, dtype=like.dtype, device=like.device),
        torch.empty((n, 3), dtype=like.dtype, device=like.device),
        torch.empty(n, dtype=torch.bool, device=like.device),
        torch.empty((n, 3), dtype=like.dtype, device=like.device),
    )


def _extra_outputs(n: int, like: torch.Tensor, kind: str):
    ph = _placeholders(like)
    st = torch.empty(n, dtype=like.dtype, device=like.device) if kind == "asphere" else ph["f1"]
    steps = torch.empty(n, dtype=like.dtype, device=like.device) if kind == "asphere" else ph["f1"]
    return (
        torch.empty(n, dtype=like.dtype, device=like.device),
        torch.empty(n, dtype=like.dtype, device=like.device),
        st,
        steps,
    )


@torch.library.custom_op("optiland_nsq::intersect_component", mutates_args=())
def _op_intersect(
    kind: str, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, L: torch.Tensor, M: torch.Tensor,
    N: torch.Tensor, alive: torch.Tensor, xf: torch.Tensor, gp: torch.Tensor, gi: torch.Tensor,
    tab: torch.Tensor, recip: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    fields = [x, y, z, L, M, N]
    n = int(x.shape[0])
    _launched(kind)
    ph = _placeholders(x)
    aux = _frustum_aux(fields, xf, x) if kind == "frustum" else ph["f1"]
    own = _own_outputs(n, x)
    extra = _extra_outputs(n, x, "asphere")
    run = (ph["f1"], ph["f13"], ph["f13"], ph["i1"])
    _launch(kind, fields, alive, xf, gp, gi, tab, aux, recip, MODE_OWN, 0, 0, own, run, extra)
    return (*own, *extra)


@_op_intersect.register_fake
def _(kind, x, y, z, L, M, N, alive, xf, gp, gi, tab, recip):
    n = x.shape[0]
    return (*_own_outputs(n, x), *_extra_outputs(n, x, "asphere"))


class _IntersectFunction(torch.autograd.Function):
    """A tape kind's kernel with its adjoint: the forward launch recorded on a Warp tape."""

    @staticmethod
    def forward(ctx, kind, recip, x, y, z, L, M, N, alive, xf, gp, gi, tab):
        fields = [t.detach().contiguous() for t in (x, y, z, L, M, N)]
        like = fields[0]
        n = int(like.shape[0])
        ph = _placeholders(like)
        own = _own_outputs(n, like)
        extra = _extra_outputs(n, like, kind)
        run = (ph["f1"], ph["f13"], ph["f13"], ph["i1"])
        tape = wp.Tape()
        _launched(kind)
        wp_in, wp_out = _launch(
            kind, fields, alive, xf.detach().contiguous(), gp.detach().contiguous(), gi, tab, ph["f1"],
            recip, MODE_OWN, 0, 0, own, run, extra, requires_grad=True, tape=tape,
        )
        ctx.tape = tape
        ctx.wp_in = wp_in
        ctx.wp_out = wp_out
        ctx.mark_non_differentiable(own[2])
        return (*own, extra[0], extra[1])

    @staticmethod
    def backward(ctx, g_t, g_n, g_hit, g_ng, g_adv, g_loc):
        grads = {}
        # wp_out: t_hit normals hit n_geom t_adv t_local status steps
        for arr, g in zip(ctx.wp_out[:6], (g_t, g_n, None, g_ng, g_adv, g_loc), strict=True):
            if g is None or not arr.requires_grad:
                continue
            grads[arr] = wp.from_torch(g.contiguous())
        if grads:
            ctx.tape.backward(grads=grads)
        wp_in = ctx.wp_in  # x y z L M N alive(None) xf gp
        out = [None, None]
        out += [wp.to_torch(wp_in[k].grad).clone() for k in range(6)]
        out.append(None)
        out.append(wp.to_torch(wp_in[7].grad).clone())
        out.append(wp.to_torch(wp_in[8].grad).clone())
        out += [None, None]
        ctx.tape.zero()
        return tuple(out)


def _fields(rays) -> list:
    return [rays.x, rays.y, rays.z, rays.L, rays.M, rays.N]


def _forward_ad_module():
    try:
        from torch.autograd import forward_ad  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return None
    return forward_ad


def _forward_ad_active(tensors) -> bool:
    forward_ad = _forward_ad_module()
    if forward_ad is None or getattr(forward_ad, "_current_level", -1) < 0:
        return False
    return any(_is_tensor(t) and forward_ad.unpack_dual(t).tangent is not None for t in tensors)


def _route_reason(component, kind, rays) -> str | None:
    """Why this component's intersection must run through its own intersect, or None."""
    if kind is None:
        return ROUTE_KIND
    like = rays.x
    if not matmul_orders(like.dtype, like.device).covers(int(like.shape[0])):
        return ROUTE_WIDTH
    from optiland.nonsequential.components.base import _resident_transform  # noqa: PLC0415

    t_be, r_be = _resident_transform(component)
    fields = _fields(rays)
    params = _param_tensors(kind, component.geometry)
    if _forward_ad_active([*fields, t_be, r_be, *params]):
        return ROUTE_FORWARD_AD
    if torch.is_grad_enabled():
        if (_is_tensor(t_be) and t_be.requires_grad) or (_is_tensor(r_be) and r_be.requires_grad):
            return ROUTE_PLACEMENT
        wants = any(f.requires_grad for f in fields) or any(p.requires_grad for p in params)
        if wants and kind not in TAPE_KINDS:
            return ROUTE_GRADIENT
    return None


def _needs_tape(kind, rays, gp) -> bool:
    return torch.is_grad_enabled() and kind in TAPE_KINDS and (
        any(f.requires_grad for f in _fields(rays)) or gp.requires_grad
    )


def _recip_flag(kind, component, like) -> int:
    if kind == "cavity" and _is_tensor(component.geometry.radius):
        return 0
    if kind in ("cavity", "sphere", "lenslet"):
        return scalar_division(like.dtype, like.device)
    return 0


def _set_side_state(component, kind, extra) -> None:
    component._local_root = (extra[0], extra[1])
    if kind == "asphere":
        component.geometry.last_status = extra[2]
        component.geometry.last_steps = extra[3]


def intersect_component(component, kind: str, rays, t_min=None):
    """One component's intersection through its kernel: ``BaseComponent.intersect``'s result.

    Also sets the component's ``_local_root`` exactly as its own
    ``intersect`` does, for ``advance_to_hit`` (and an asphere's
    ``last_status`` and ``last_steps``). A call the kernels cannot reproduce
    (see :func:`routed_counts`) runs the component's own ``intersect``.

    Args:
        component: A component :func:`kind_of` accepted.
        kind: Its kind.
        rays: The ray bundle.
        t_min: Unused; the kernel forms the per-ray accept threshold itself
            (kept for the earlier signature).

    Returns:
        ``(t, normals, hit_mask, n_geom)``.
    """
    reason = _route_reason(component, kind, rays)
    if reason is not None:
        _route(reason)
        return component.intersect(rays)
    like = rays.x
    _ensure_loaded((kind,), like)
    xf, gp, gi, tab = _component_inputs(component, kind, like)
    recip = _recip_flag(kind, component, like)
    if _needs_tape(kind, rays, gp):
        t, normals, hit, n_geom, t_adv, t_local = _IntersectFunction.apply(
            kind, recip, *_fields(rays), rays.alive, xf, gp, gi, tab
        )
        component._local_root = (t_adv, t_local)
        return t, normals, hit, n_geom
    # Strided fields are passed as they are (a Warp array carries its
    # strides): the ray state is often a column of an (N, 3) product, and a
    # copy per field would be three more launches per component.
    t, normals, hit, n_geom, t_adv, t_local, st, steps = torch.ops.optiland_nsq.intersect_component(
        kind, *_fields(rays), rays.alive, xf, gp, gi, tab, recip,
    )
    _set_side_state(component, kind, (t_adv, t_local, st, steps))
    return t, normals, hit, n_geom


def accept_threshold(rays):
    """The per-ray accept threshold of ``BaseComponent.intersect`` (the kernels form it themselves)."""
    from optiland.nonsequential.components.base import coordinate_magnitude  # noqa: PLC0415

    return _tol.accept_t_min(coordinate_magnitude(rays))


def _torch_select(rays, components):
    """``ArrayBackend.intersect_scene``'s running select, each component through
    :func:`intersect_component` (the gradient route: the select stays torch's)."""
    from optiland.nonsequential.ray_bundle import backend_int_full  # noqa: PLC0415

    n = rays.num_rays
    t_min = be.ones(n) * be.inf
    hit_normals = be.zeros((n, 3))
    hit_n_geom = be.zeros((n, 3))
    comp_indices = backend_int_full((n,), -1, like=rays.x, bits=32)
    for i, comp in enumerate(components):
        kind = kind_of(comp)
        if kind is None:
            _route(ROUTE_KIND)
            t_c, normals_c, hit_c, n_geom_c = comp.intersect(rays)
        else:
            t_c, normals_c, hit_c, n_geom_c = intersect_component(comp, kind, rays)
        better = hit_c & (t_c < t_min)
        t_min = be.where(better, t_c, t_min)
        hit_normals = be.where(better[:, None], normals_c, hit_normals)
        hit_n_geom = be.where(better[:, None], n_geom_c, hit_n_geom)
        comp_indices = be.where(better, backend_int_full((n,), i, like=rays.x, bits=32), comp_indices)
    return t_min, hit_normals, comp_indices, hit_n_geom


def intersect_scene(rays, components):
    """``ArrayBackend.intersect_scene`` as one launch per component, the select fused into each.

    Every covered component's kernel folds its hit into the running nearest
    hit (distance, both normals, component index) in place, in component
    order, with the torch stage's comparison (``hit & (t < t_min)``), so the
    result is ``intersect_scene``'s bit for bit; a component routed to its
    own intersect is folded in by one small launch. In gradient mode the
    select is torch's (:func:`_torch_select`), so the derivative reaches the
    winning component's outputs.
    """
    like = rays.x
    n = int(rays.num_rays)
    if not components or n == 0:
        return _torch_select(rays, components)
    if torch.is_grad_enabled() and (
        any(f.requires_grad for f in _fields(rays)) or _forward_ad_active(_fields(rays))
    ):
        return _torch_select(rays, components)
    kinds = [kind_of(comp) for comp in components]
    reasons = [_route_reason(comp, kind, rays) for comp, kind in zip(components, kinds, strict=True)]
    if any(r in (ROUTE_PLACEMENT, ROUTE_GRADIENT, ROUTE_FORWARD_AD) for r in reasons) or (
        getattr(_forward_ad_module(), "_current_level", -1) >= 0
    ):
        return _torch_select(rays, components)
    # The scene's kernels, compiled in parallel on the first bounce that
    # meets them (nothing to do afterwards).
    _ensure_loaded(
        [k for k, r in zip(kinds, reasons, strict=True) if r is None], like,
        merge=any(r is not None for r in reasons),
    )
    t_run = torch.empty(n, dtype=like.dtype, device=like.device)
    n_run = torch.empty((n, 3), dtype=like.dtype, device=like.device)
    g_run = torch.empty((n, 3), dtype=like.dtype, device=like.device)
    idx_run = torch.empty(n, dtype=torch.int32, device=like.device)
    run = (t_run, n_run, g_run, idx_run)
    for i, (comp, kind, reason) in enumerate(zip(components, kinds, reasons, strict=True)):
        first = int(i == 0)
        if reason is not None:
            t_c, normals_c, hit_c, n_geom_c = comp.intersect(rays)
            if torch.is_grad_enabled() and any(t.requires_grad for t in (t_c, normals_c, n_geom_c)):
                # Its outputs carry a gradient the fused select would drop:
                # the whole select runs in torch instead.
                return _torch_select(rays, components)
            _route(reason)
            torch.ops.optiland_nsq.merge_hit(t_c, normals_c, hit_c, n_geom_c, i, first, *run)
            continue
        xf, gp, gi, tab = _component_inputs(comp, kind, like)
        if torch.is_grad_enabled() and gp.requires_grad:
            return _torch_select(rays, components)
        extra = _extra_outputs(n, like, kind)
        torch.ops.optiland_nsq.intersect_fused(
            kind, *_fields(rays), rays.alive, xf, gp, gi, tab, _recip_flag(kind, comp, like), i, first,
            *run, *extra,
        )
        _set_side_state(comp, kind, extra)
    return t_run, n_run, idx_run, g_run


__all__ = [
    "KINDS",
    "MM_ORDERS",
    "SUPPORTED_DEVICE_TYPES",
    "TAPE_KINDS",
    "accept_threshold",
    "availability",
    "intersect_component",
    "intersect_scene",
    "compile_cache",
    "kernel_modules",
    "kind_of",
    "launch_counts",
    "load_kernels",
    "matmul_orders",
    "prepare",
    "reset_routed",
    "routed_counts",
    "scalar_division",
    "sum_order",
]
