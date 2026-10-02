"""The reflective interaction as one Warp launch per component (prototype; the third seam's second stage).

:meth:`~optiland.nonsequential.components.reflective.ReflectiveComponent.interact`
moves every ray that hit the component onto its hit point (rebuilt in the
component's frame), reflects it specularly, weights it by the reflectance,
routes a ``scatter_fraction`` of the hits into a Lambertian lobe (the lobe's
direction from Malley's method in a frame carried by the surface), books what
the surface removes in the energy ledger, counts the bounce, and pushes the
outgoing origin off the surface: about 180 dispatched torch operations per
component and bounce (research repository issue 2, the profile of the
integrating sphere). This module computes all of it in one Warp launch. What
stays in torch is what is not elementwise or not the interaction's own:

* the generator's draws (``rng.uniform``, by the generator the trace uses, so
  the draws are its bits; the Warp generator draws each in one launch);
* the three sums of the ledger's bookings and their compensated additions into
  the surface's tallies (a reduction's order is torch's; the kernel writes the
  per-ray terms the sums read, ``where(hit, flux * fraction, 0)``).

**Same numbers.** Every expression is the torch statement's, in its order and
operator form, with the orders torch rounds in taken from
:mod:`optiland.nonsequential.stage_warp` (the placement product per row, the
three-term sum, ``python_number / tensor`` as a reciprocal product, a Python
number cast to the working dtype on the host, negation as a sign flip). Two of
torch's functions are checked at load on the device (:func:`prepare`): the
power ``x ** 0.5`` must be the square root (torch's power kernel takes that
exponent to ``sqrt``), and the sine and cosine of the lobe's azimuth must be
the kernel's own. Where they are not -- the CPU, whose torch evaluates them
with a vectorised approximation -- the trace computes ``cos(phi)`` and
``sin(phi)`` with torch and hands them to the kernel (three more operations),
which is the reported ``trig`` mode.

**What it covers.** A :class:`ReflectiveComponent` whose reflectance is a
number (or an unpolarized coating's constant), with no lobe (a mirror) or a
:class:`LambertianBSDF` with a numeric reflectance and transmissive fraction,
and a numeric ``scatter_fraction``. Every other case is routed to the
component's own ``interact``, counted by reason (:func:`routed_counts`): a
Stokes trace, a gradient through any input, a thin-film or wavelength-
dependent reflectance, another lobe, a width at which the placement product's
order is not probed.

Use it through the torch backend, ``TorchBackend(interact_kernel="warp")``,
on CUDA with Warp, like the intersection stage.
"""



import math
from typing import Any

import numpy as np

import torch
import warp as wp

import optiland.backend as be
from optiland.nonsequential import _tol
from optiland.nonsequential import stage_warp as sw

#: The device types on which the torch backend uses the kernel.
SUPPORTED_DEVICE_TYPES: tuple[str, ...] = ("cuda",)

# Bit-pattern copysign (torch.copysign), on the CPU and on CUDA alike: Warp's
# CPU runtime has no copysign of its own.
_COPYSIGN64 = (
    "union { double d; unsigned long long u; } x, y; x.d = a; y.d = b; "
    "x.u = (x.u & 0x7fffffffffffffffULL) | (y.u & 0x8000000000000000ULL); return x.d;"
)
_COPYSIGN32 = (
    "union { float f; unsigned int u; } x, y; x.f = a; y.f = b; "
    "x.u = (x.u & 0x7fffffffu) | (y.u & 0x80000000u); return x.f;"
)


@wp.func_native(_COPYSIGN64, "")
def _copysign64(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native(_COPYSIGN32, "")
def _copysign32(a: wp.float32, b: wp.float32) -> wp.float32: ...


#: _tol.origin_offset's multiple of the ulp: (k_delta / 2.0).
_OFFSET_HALF_K = float(_tol.DEFAULT_OFFSET_K_DELTA) / 2.0


def _kernel(fn, name: str, bits: int):
    """``fn`` as a Warp kernel in a module of its own, with the stage's options (no contraction, CUBIN, no adjoint)."""
    module = f"{__name__}.{name}_{bits}"
    wp.set_module_options({"fuse_fp": False, "enable_backward": False, "cuda_output": "cubin"}, module=module)
    return wp.kernel(fn, module=wp.get_module(module))


def _make_kernel(FT, bits, copysign, IT, ibits):
    f = sw._FUNCS[bits]
    _sum3, _rdiv, _row_code = f["sum3"], f["rdiv"], f["row_code"]
    _to_local, _dir_local, _to_global_normal = f["to_local"], f["dir_local"], f["to_global_normal"]
    _max_nan, ulp, neg = f["max_nan"], f["ulp"], f["neg"]

    @wp.func
    def _frame_fwd(x: FT, y: FT, z: FT, xf: wp.array(dtype=FT), code: int):
        # ``n @ frame`` (the lobe's frame: the placement's rotation).
        return _dir_local(x, y, z, xf, code)

    def k(
        x: wp.array(dtype=FT), y: wp.array(dtype=FT), z: wp.array(dtype=FT),
        L: wp.array(dtype=FT), M: wp.array(dtype=FT), N: wp.array(dtype=FT),
        flux: wp.array(dtype=FT), bounce: wp.array(dtype=IT),
        t: wp.array(dtype=FT), normals: wp.array2d(dtype=FT), n_geom: wp.array2d(dtype=FT),
        hit: wp.array(dtype=wp.bool),
        t_adv: wp.array(dtype=FT), t_local: wp.array(dtype=FT), has_root: int,
        xf: wp.array(dtype=FT),
        sum_order: int,
        mm_small: wp.array3d(dtype=int), mm_meta: wp.array(dtype=int), mm_tail: wp.array3d(dtype=int),
        # the reflectance and the lobe
        refl: FT, tiny: FT, lobe: int, tau: FT, use_tau: int, sf_det: FT, w_s: FT, w_ns: FT,
        rv: FT, two_pi: FT, offset_k: FT,
        u_lobe: wp.array(dtype=FT), r1: wp.array(dtype=FT), r2: wp.array(dtype=FT), u_sc: wp.array(dtype=FT),
        trig_in: int, cos_in: wp.array(dtype=FT), sin_in: wp.array(dtype=FT),
        # outputs
        xo: wp.array(dtype=FT), yo: wp.array(dtype=FT), zo: wp.array(dtype=FT),
        Lo: wp.array(dtype=FT), Mo: wp.array(dtype=FT), No: wp.array(dtype=FT),
        fo: wp.array(dtype=FT), bo: wp.array(dtype=IT),
        loss1: wp.array(dtype=FT), resid: wp.array(dtype=FT), loss2: wp.array(dtype=FT),
    ):
        i = wp.tid()
        n = x.shape[0]
        h = hit[i]
        xi = x[i]
        yi = y[i]
        zi = z[i]
        Li = L[i]
        Mi = M[i]
        Ni = N[i]

        # -- advance_to_hit_in_frame ------------------------------------------
        ti = t[i]
        t_safe = FT(0.0)
        if h:
            t_safe = ti
        xg = xi + t_safe * Li
        yg = yi + t_safe * Mi
        zg = zi + t_safe * Ni
        if has_root == 1:
            ta = t_adv[i]
            tl = t_local[i]
            usable = h and (ta + tl == ti)
            adv = FT(0.0)
            loc = FT(0.0)
            if usable:
                adv = ta
                loc = tl
            code_f = _row_code(i, n, 0, mm_small, mm_meta, mm_tail)
            code_b = _row_code(i, n, 1, mm_small, mm_meta, mm_tail)
            dlx, dly, dlz = _dir_local(Li, Mi, Ni, xf, code_f)
            plx, ply, plz = _to_local(xi, yi, zi, xf, code_f)
            pax = plx + adv * dlx
            pay = ply + adv * dly
            paz = plz + adv * dlz
            hlx = pax + loc * dlx
            hly = pay + loc * dly
            hlz = paz + loc * dlz
            hgx, hgy, hgz = _to_global_normal(hlx, hly, hlz, xf, code_b)
            hgx = hgx + xf[0]
            hgy = hgy + xf[1]
            hgz = hgz + xf[2]
            if usable:
                xg = hgx
                yg = hgy
                zg = hgz
        if h:
            xi = xg
            yi = yg
            zi = zg

        # -- the specular reflection ------------------------------------------
        nx = normals[i, 0]
        ny = normals[i, 1]
        nz = normals[i, 2]
        raw_dot = _sum3(Li * nx, Mi * ny, Ni * nz, sum_order)
        tw = FT(2.0) * raw_dot
        rx = Li - tw * nx
        ry = Mi - tw * ny
        rz = Ni - tw * nz
        norm_r = wp.sqrt(_sum3(rx * rx, ry * ry, rz * rz, sum_order))
        den = norm_r + tiny
        rx = rx / den
        ry = ry / den
        rz = rz / den
        ox = Li
        oy = Mi
        oz = Ni
        if h:
            ox = rx
            oy = ry
            oz = rz

        # -- the reflectance: book (1 - R), weight by R -------------------------
        f0 = flux[i]
        l1 = FT(0.0)
        if h:
            l1 = f0 * (FT(1.0) - refl)
        loss1[i] = l1
        g = FT(1.0)
        if h:
            g = refl
        f1 = f0 * g
        fout = f1

        if lobe == 1:
            # -- scatter_branch ------------------------------------------------
            scat = h and (u_sc[i] < sf_det)
            sf_gate = w_ns
            if scat:
                sf_gate = w_s
            rs = FT(0.0)
            if h:
                rs = f1 * (FT(1.0) - sf_gate)
            resid[i] = rs
            g2 = FT(1.0)
            if h:
                g2 = sf_gate
            f2 = f1 * g2
            # -- the Lambertian lobe (only its scattered rays are read) ---------
            bsdf_gate = FT(1.0)
            if scat:
                hx = nx
                hy = ny
                hz = nz
                if use_tau == 1:
                    if u_lobe[i] < tau:
                        hx = neg(nx)
                        hy = neg(ny)
                        hz = neg(nz)
                # Malley's method
                cos_t = wp.sqrt(r2[i])
                sin_t = wp.sqrt(FT(1.0) - r2[i])
                cphi = FT(0.0)
                sphi = FT(0.0)
                if trig_in == 1:
                    cphi = cos_in[i]
                    sphi = sin_in[i]
                else:
                    phi = two_pi * r1[i]
                    cphi = wp.cos(phi)
                    sphi = wp.sin(phi)
                lx = sin_t * cphi
                ly = sin_t * sphi
                lz = cos_t
                # _orthonormal_basis(n @ frame), the two vectors @ frame.T
                code_f = _row_code(i, n, 0, mm_small, mm_meta, mm_tail)
                code_b = _row_code(i, n, 1, mm_small, mm_meta, mm_tail)
                ax, ay, az = _frame_fwd(hx, hy, hz, xf, code_f)
                sgn = copysign(FT(1.0), az)
                a = _rdiv(FT(-1.0), sgn + az)
                b = (ax * ay) * a
                tlx = FT(1.0) + ((sgn * ax) * ax) * a
                tly = sgn * b
                tlz = neg(sgn) * ax
                blx = b
                bly = sgn + (ay * ay) * a
                blz = neg(ay)
                tx, ty, tz = _to_global_normal(tlx, tly, tlz, xf, code_b)
                bx, by, bz = _to_global_normal(blx, bly, blz, xf, code_b)
                sx = (lx * tx + ly * bx) + lz * hx
                sy = (lx * ty + ly * by) + lz * hy
                sz = (lx * tz + ly * bz) + lz * hz
                nrm = wp.sqrt(_sum3(sx * sx, sy * sy, sz * sz, sum_order))
                ox = sx / nrm
                oy = sy / nrm
                oz = sz / nrm
                bsdf_gate = rv
            l2 = FT(0.0)
            if h:
                l2 = f2 * (FT(1.0) - bsdf_gate)
            loss2[i] = l2
            fout = f2 * bsdf_gate

        # -- the bounce and the origin offset -----------------------------------
        b_out = bounce[i]
        if h:
            b_out = b_out + IT(1)
        gx = n_geom[i, 0]
        gy = n_geom[i, 1]
        gz = n_geom[i, 2]
        dot = (ox * gx + oy * gy) + oz * gz
        mag = wp.abs(_max_nan(_max_nan(wp.abs(xi), wp.abs(yi)), wp.abs(zi)))
        if not wp.isnan(mag):
            if mag < FT(1.0):
                mag = FT(1.0)
        delta = offset_k * ulp(mag)
        signed = delta
        if dot < FT(0.0):
            signed = neg(delta)
        step = FT(0.0)
        if h:
            step = signed
        xo[i] = xi + step * gx
        yo[i] = yi + step * gy
        zo[i] = zi + step * gz
        Lo[i] = ox
        Mo[i] = oy
        No[i] = oz
        fo[i] = fout
        bo[i] = b_out

    return _kernel(k, f"reflective_i{ibits}", bits)


#: One kernel per float type and per integer type of the bounce counter.
_KERNELS = {
    (ft, it): _make_kernel(wft, bits, cs, wit, ibits)
    for ft, wft, bits, cs in ((torch.float64, wp.float64, 64, _copysign64), (torch.float32, wp.float32, 32, _copysign32))
    for it, wit, ibits in ((torch.int32, wp.int32, 32), (torch.int64, wp.int64, 64))
}
_INT_TYPES = (torch.int32, torch.int64)


def _tally_k(FT):
    @wp.func
    def _add(dev: wp.array(dtype=FT), comp: wp.array(dtype=FT), term: FT):
        # Tally.add of a device term (Neumaier): new = old + term; the error
        # two_sum_error(old, term, new) into the compensation; the term into
        # the total -- each the torch statement's rounded operation.
        old = dev[0]
        new = old + term
        err = (term - new) + old
        if wp.abs(old) >= wp.abs(term):
            err = (old - new) + term
        comp[0] = comp[0] + err
        dev[0] = old + term

    def k(
        t1: wp.array(dtype=FT), t2: wp.array(dtype=FT), t3: wp.array(dtype=FT),
        d1: wp.array(dtype=FT), c1: wp.array(dtype=FT), d2: wp.array(dtype=FT), c2: wp.array(dtype=FT),
        lobe: int,
    ):
        # The bookings in the torch stage's order: the coating loss, then
        # (with a lobe) the sampling residual and the lobe's loss.
        _add(d1, c1, t1[0])
        if lobe == 1:
            _add(d2, c2, t2[0])
            _add(d1, c1, t3[0])

    return k


#: The ledger's compensated additions of one interaction, one launch.
_TALLY = {
    torch.float64: _kernel(_tally_k(wp.float64), "tally", 64),
    torch.float32: _kernel(_tally_k(wp.float32), "tally", 32),
}
_WP_FLOAT = {torch.float64: wp.float64, torch.float32: wp.float32}
_NP_FLOAT = {torch.float64: np.float64, torch.float32: np.float32}


# ---------------------------------------------------------------------------
# What torch's functions do on this device (probed at load)
# ---------------------------------------------------------------------------

_PROBES: dict = {}


def _probe_k(FT):
    def k(phi: wp.array(dtype=FT), c: wp.array(dtype=FT), s: wp.array(dtype=FT)):
        i = wp.tid()
        c[i] = wp.cos(phi[i])
        s[i] = wp.sin(phi[i])

    return k


_TRIG_PROBE = {
    torch.float64: _kernel(_probe_k(wp.float64), "trig_probe", 64),
    torch.float32: _kernel(_probe_k(wp.float32), "trig_probe", 32),
}

#: Draws the trig probe is made of: every value the lobe's azimuth takes for
#: a uniform on a regular grid of this many steps, plus as many uniforms of
#: the generator's own form (the top bits of a 32-bit integer).
TRIG_PROBE_SIZE = 1 << 20


def probes(dtype: torch.dtype, device: Any) -> dict:
    """What torch computes on ``device`` for the two functions the kernel evaluates itself.

    ``sqrt_pow``: ``x ** 0.5`` is ``torch.sqrt(x)`` bit for bit on a million
    values spanning the exponent range (torch's power kernel takes the
    exponent 0.5 to its square root). ``trig``: Warp's ``cos`` and ``sin`` of
    ``(2 pi) * u`` equal torch's on the device for :data:`TRIG_PROBE_SIZE`
    grid values and as many random uniforms of the generator's form; where
    they do not (torch on the CPU evaluates them with a vectorised
    approximation), the trace hands torch's values to the kernel. Cached per
    dtype and device.
    """
    tdev = sw._torch_device(device)
    key = (dtype, str(tdev))
    cached = _PROBES.get(key)
    if cached is not None:
        return cached
    sw._init()
    ivt = torch.int64 if dtype == torch.float64 else torch.int32
    with torch.no_grad():
        gen = torch.Generator(device=tdev)
        gen.manual_seed(20261002)
        x = torch.exp(torch.empty(1 << 20, dtype=torch.float64, device=tdev).uniform_(-30.0, 30.0, generator=gen))
        x = x.to(dtype)
        sqrt_pow = bool((torch.sqrt(x).view(ivt) == (x**0.5).view(ivt)).all())
        bits = 32 if dtype == torch.float64 else 24
        grid = torch.arange(TRIG_PROBE_SIZE, dtype=torch.float64, device=tdev) / TRIG_PROBE_SIZE
        rnd = torch.randint(0, 1 << bits, (TRIG_PROBE_SIZE,), generator=gen, device=tdev, dtype=torch.int64)
        u = torch.cat([grid, rnd.to(torch.float64) * 2.0**-bits]).to(dtype)
        phi = 2.0 * math.pi * u
        c = torch.empty_like(phi)
        s = torch.empty_like(phi)
        kwargs = sw._launch_kwargs(phi, int(phi.shape[0]))
        kwargs["inputs"] = [wp.from_torch(phi)]
        kwargs["outputs"] = [wp.from_torch(c), wp.from_torch(s)]
        wp.launch(_TRIG_PROBE[dtype], **kwargs)
        trig = bool((torch.cos(phi).view(ivt) == c.view(ivt)).all() and (torch.sin(phi).view(ivt) == s.view(ivt)).all())
    result = {"sqrt_pow": sqrt_pow, "trig_in_kernel": trig}
    _PROBES[key] = result
    return result


# ---------------------------------------------------------------------------
# Loading, availability, routing
# ---------------------------------------------------------------------------

_prepared: set = set()


def kernel_modules(dtypes=(torch.float64, torch.float32)) -> list:
    """The Warp modules of this stage (one per float type, and the trig probe's)."""
    return (
        [_KERNELS[(d, i)].module for d in dtypes for i in _INT_TYPES]
        + [_TRIG_PROBE[d].module for d in dtypes]
        + [_TALLY[d].module for d in dtypes]
    )


def compile_cache(arch, dtypes=(torch.float64, torch.float32)) -> list:
    """Compile this stage's kernels for a CUDA architecture ahead of time (see :func:`stage_warp.compile_cache`)."""
    sw._init()
    arches = [arch] if isinstance(arch, int) else list(arch)
    paths = []
    for module in kernel_modules(dtypes):
        paths += wp.compile_aot_module(module, arch=arches, use_ptx=False)
    return paths


def prepare(device: Any) -> None:
    """Load the kernels and probe torch's orders and functions on ``device``, before any bounce."""
    tdev = sw._torch_device(device)
    if str(tdev) in _prepared:
        return
    sw.prepare(tdev)
    # The probe's kernels here; the interaction's kernel loads on the first
    # bounce that runs it (an eager one, before a replay records), so only
    # the float and integer types a trace uses compile.
    block_dim = 1 if tdev.type == "cpu" else None
    wp.force_load(device=wp.device_from_torch(tdev), modules=[_TRIG_PROBE[d].module for d in _TRIG_PROBE],
                  block_dim=block_dim, max_workers=sw.LOAD_WORKERS)
    for dtype in (torch.float64, torch.float32):
        probes(dtype, tdev)
    _prepared.add(str(tdev))


def availability(device: Any) -> str | None:
    """Why the kernel cannot serve ``device``, or None when it can."""
    tdev = torch.device(device)
    if tdev.type not in SUPPORTED_DEVICE_TYPES:
        return (
            f"the Warp interaction kernel runs on {', '.join(SUPPORTED_DEVICE_TYPES)} "
            f"only; the device is {tdev.type}"
        )
    try:
        sw._init()
        if tdev.type == "cuda" and not wp.is_cuda_available():
            return "Warp sees no CUDA device"
        prepare(tdev)
    except Exception as exc:  # noqa: BLE001 - any failure means "use the torch stage"
        return f"Warp could not load its kernels ({type(exc).__name__})"
    for dtype in (torch.float64, torch.float32):
        if sw.matmul_orders(dtype, tdev).refusal is not None:
            return "the kernel cannot follow torch's placement product here"
    return None


ROUTE_KIND = "a component or lobe the kernel does not cover"
ROUTE_POLARIZATION = "a Stokes trace"
ROUTE_GRADIENT = "a gradient through the interaction"
ROUTE_WIDTH = "a width at which no probed order reproduces torch's placement product"
ROUTE_POW = "a device whose power x ** 0.5 is not its square root"

_ROUTED: dict = {}
_LAUNCHED: dict = {}


def routed_counts() -> dict:
    """``{reason: calls}`` of the interactions routed to the component's own ``interact`` since the last reset."""
    return dict(_ROUTED)


def launch_counts() -> dict:
    """``{"reflective": calls}`` of the interactions that went through the kernel since the last reset."""
    return dict(_LAUNCHED)


def reset_routed() -> None:
    """Forget the counts (the torch backend does this at the start of a trace)."""
    _ROUTED.clear()
    _LAUNCHED.clear()


def _route(reason: str) -> None:
    _ROUTED[reason] = _ROUTED.get(reason, 0) + 1


def _plain_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coverage(component) -> dict | None:
    """The component's constants as the kernel takes them, or None when it is not covered."""
    from optiland.coatings import BaseCoating  # noqa: PLC0415
    from optiland.nonsequential.bsdf.lambertian import LambertianBSDF  # noqa: PLC0415
    from optiland.nonsequential.components.reflective import ReflectiveComponent  # noqa: PLC0415

    if type(component).interact is not ReflectiveComponent.interact:
        return None
    refl = component.reflectance
    if callable(getattr(refl, "evaluate", None)):
        return None
    if isinstance(refl, BaseCoating):
        refl = refl.reflectance
    if not _plain_number(refl):
        return None
    bsdf = component.bsdf
    out = {"refl": float(refl), "lobe": 0, "tau": 0.0, "rv": 1.0, "sf": 1.0}
    if bsdf is None:
        return out
    if type(bsdf) is not LambertianBSDF:
        return None
    if not _plain_number(bsdf.reflectance_value) or not _plain_number(bsdf.transmissive_fraction):
        return None
    if not _plain_number(component.scatter_fraction):
        return None
    out.update(lobe=1, tau=float(bsdf._tau), rv=float(bsdf.reflectance_value), sf=float(component.scatter_fraction))
    return out


def _route_reason(component, rays, t, normals, hit_mask, bsdf_ir, n_geom) -> str | None:
    cov = _coverage(component)
    if cov is None or (cov["lobe"] == 0 and bsdf_ir.kind != "none") or (cov["lobe"] == 1 and bsdf_ir.kind == "none"):
        return ROUTE_KIND
    if rays.pol_q is not None:
        return ROUTE_POLARIZATION
    like = rays.x
    if rays.bounce.dtype not in _INT_TYPES:
        return ROUTE_KIND
    if torch.is_grad_enabled():
        from optiland.nonsequential.components.base import _resident_transform  # noqa: PLC0415

        t_be, r_be = _resident_transform(component)
        tensors = (rays.x, rays.y, rays.z, rays.L, rays.M, rays.N, rays.flux, t, normals, n_geom, t_be, r_be)
        if any(sw._is_tensor(v) and v.requires_grad for v in tensors):
            return ROUTE_GRADIENT
    if sw._forward_ad_module() is not None and getattr(sw._forward_ad_module(), "_current_level", -1) >= 0:
        return ROUTE_GRADIENT
    if not sw.matmul_orders(like.dtype, like.device).covers(int(like.shape[0])):
        return ROUTE_WIDTH
    if not probes(like.dtype, like.device)["sqrt_pow"]:
        return ROUTE_POW
    return None


def _constants(component, cov: dict, like: torch.Tensor) -> tuple:
    """The kernel's scalar arguments, each cast on the host as torch casts the Python number."""
    from optiland.nonsequential.components.sampling_support import _SF_EPS  # noqa: PLC0415

    ft = _NP_FLOAT[like.dtype]
    sf = cov["sf"]
    sf_det = min(max(sf, _SF_EPS), 1.0 - _SF_EPS)
    w_s = sf / sf_det
    w_ns = (1.0 - sf) / (1.0 - sf_det)
    tiny = _tol.tiny_for(like)
    return (
        ft(cov["refl"]), ft(tiny), int(cov["lobe"]), ft(cov["tau"]), int(cov["tau"] > 0.0), ft(sf_det),
        ft(w_s), ft(w_ns), ft(cov["rv"]), ft(2.0 * math.pi), ft(_OFFSET_HALF_K),
    )


@torch.library.custom_op(
    "optiland_nsq::interact_reflective",
    mutates_args=("xo", "yo", "zo", "Lo", "Mo", "No", "fo", "bo", "loss1", "resid", "loss2"),
)
def _op_reflective(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, L: torch.Tensor, M: torch.Tensor, N: torch.Tensor,
    flux: torch.Tensor, bounce: torch.Tensor, t: torch.Tensor, normals: torch.Tensor, n_geom: torch.Tensor,
    hit: torch.Tensor, t_adv: torch.Tensor, t_local: torch.Tensor, has_root: int, xf: torch.Tensor,
    consts: list[float], ints: list[int],
    u_lobe: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, u_sc: torch.Tensor, trig_in: int,
    cos_in: torch.Tensor, sin_in: torch.Tensor,
    xo: torch.Tensor, yo: torch.Tensor, zo: torch.Tensor, Lo: torch.Tensor, Mo: torch.Tensor, No: torch.Tensor,
    fo: torch.Tensor, bo: torch.Tensor, loss1: torch.Tensor, resid: torch.Tensor, loss2: torch.Tensor,
) -> None:
    n = int(x.shape[0])
    if n == 0:
        return
    ft = _NP_FLOAT[x.dtype]
    refl, tiny, sf_det, w_s, w_ns, rv, two_pi, offset_k, tau = (ft(v) for v in consts)
    lobe, use_tau = ints
    tables = [wp.from_torch(tt, dtype=wp.int32) for tt in sw.matmul_orders(x.dtype, x.device).tensors]
    kwargs = sw._launch_kwargs(x, n)
    kwargs["inputs"] = [
        *(wp.from_torch(f) for f in (x, y, z, L, M, N, flux)), wp.from_torch(bounce),
        wp.from_torch(t), wp.from_torch(normals), wp.from_torch(n_geom), wp.from_torch(hit),
        wp.from_torch(t_adv), wp.from_torch(t_local), int(has_root), wp.from_torch(xf),
        sw.sum_order(x.dtype, x.device), *tables,
        refl, tiny, int(lobe), tau, int(use_tau), sf_det, w_s, w_ns, rv, two_pi, offset_k,
        *(wp.from_torch(a) for a in (u_lobe, r1, r2, u_sc)), int(trig_in), wp.from_torch(cos_in),
        wp.from_torch(sin_in),
    ]
    kwargs["outputs"] = [
        *(wp.from_torch(o) for o in (xo, yo, zo, Lo, Mo, No, fo)), wp.from_torch(bo),
        wp.from_torch(loss1), wp.from_torch(resid), wp.from_torch(loss2),
    ]
    wp.launch(_KERNELS[(x.dtype, bounce.dtype)], **kwargs)


@_op_reflective.register_fake
def _(x, y, z, L, M, N, flux, bounce, t, normals, n_geom, hit, t_adv, t_local, has_root, xf, consts, ints,
      u_lobe, r1, r2, u_sc, trig_in, cos_in, sin_in, xo, yo, zo, Lo, Mo, No, fo, bo, loss1, resid, loss2):
    return None


def _placement(component, like: torch.Tensor) -> torch.Tensor:
    """The component's placement as the twelve numbers the kernels take (``stage_warp``'s layout), cached on it."""
    from optiland.nonsequential.components.base import _resident_transform  # noqa: PLC0415

    t_be, r_be = _resident_transform(component)
    cache = getattr(component, "_warp_interact_xf", None)
    key = (like.dtype, str(like.device))
    if cache is not None and cache[0] == key and cache[1] is t_be and cache[2] is r_be:
        return cache[3]
    xf = torch.cat([t_be.reshape(3), r_be.reshape(9)]).to(dtype=like.dtype, device=like.device).contiguous()
    component._warp_interact_xf = (key, t_be, r_be, xf)
    return xf


@torch.library.custom_op("optiland_nsq::tally_add", mutates_args=("d1", "c1", "d2", "c2"))
def _op_tally(
    t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, d1: torch.Tensor, c1: torch.Tensor, d2: torch.Tensor,
    c2: torch.Tensor, lobe: int,
) -> None:
    kwargs = sw._launch_kwargs(t1, 1)
    kwargs["inputs"] = [*(wp.from_torch(v.reshape(1)) for v in (t1, t2, t3, d1, c1, d2, c2)), int(lobe)]
    wp.launch(_TALLY[t1.dtype], **kwargs)


@_op_tally.register_fake
def _(t1, t2, t3, d1, c1, d2, c2, lobe):
    return None


def _device_tally(tally, like: torch.Tensor) -> bool:
    """Whether ``tally`` adds in place on the device in ``like``'s dtype (after its first term)."""
    dev, comp = tally._dev, tally._dev_comp
    return (
        sw._is_tensor(dev) and sw._is_tensor(comp) and dev.dtype == like.dtype and comp.dtype == like.dtype
        and dev.shape == () and comp.shape == () and not dev.requires_grad and dev.device == like.device
    )


def _book(component, sums, lobe: int, like: torch.Tensor) -> None:
    """Add the interaction's bookings to the component's tallies: one launch, or Tally.add on a first term."""
    coat = component._tally("_coating_loss")
    res = component._tally("_sampling_residual")
    if _device_tally(coat, like) and (not lobe or _device_tally(res, like)):
        d2, c2 = (res._dev, res._dev_comp) if lobe else (coat._dev, coat._dev_comp)
        t2, t3 = (sums[1], sums[2]) if lobe else (sums[0], sums[0])
        torch.ops.optiland_nsq.tally_add(sums[0], t2, t3, coat._dev, coat._dev_comp, d2, c2, int(lobe))
        return
    coat.add(sums[0])
    if lobe:
        res.add(sums[1])
        coat.add(sums[2])


def interact_component(component, rays, t, normals, hit_mask, rng, bsdf_ir, n_geom, sampling=None):
    """``ReflectiveComponent.interact`` for one component, through the kernel where it is covered.

    The same in-place updates of the bundle, the same bookings into the
    component's ledger tallies (one sum each, as ``masked_sum`` forms it), and
    the same draws from ``rng``; a call the kernel does not cover runs the
    component's own ``interact`` and is counted by reason.
    """
    reason = _route_reason(component, rays, t, normals, hit_mask, bsdf_ir, n_geom)
    if reason is not None:
        _route(reason)
        component.interact(rays, t, normals, hit_mask, rng, bsdf_ir, n_geom, sampling=sampling)
        return
    from optiland.nonsequential.components.base import _resident_transform  # noqa: PLC0415
    from optiland.nonsequential.rng import EventSlot  # noqa: PLC0415

    like = rays.x
    n = int(like.shape[0])
    cov = _coverage(component)
    consts = _constants(component, cov, like)
    refl, tiny, lobe, tau, use_tau, sf_det, w_s, w_ns, rv, two_pi, offset_k = consts
    ph = sw._placeholders(like)
    # The draws, by the trace's own generator and in the order the torch
    # stage draws them (each keyed by ray, bounce and slot, so the order does
    # not move a value).
    u_lobe = r1 = r2 = u_sc = ph["f1"]
    cos_in = sin_in = ph["f1"]
    trig_in = 0
    if lobe:
        ray_id_key, bounce_key = rays.ray_id, rays.bounce
        if use_tau:
            u_lobe = rng.uniform(ray_id_key, bounce_key, EventSlot.BSDF_LOBE_BRANCH)
        r1 = rng.uniform(ray_id_key, bounce_key, EventSlot.BSDF_U1)
        r2 = rng.uniform(ray_id_key, bounce_key, EventSlot.BSDF_U2)
        u_sc = rng.uniform(ray_id_key, bounce_key, EventSlot.SCATTER_BRANCH)
        if not probes(like.dtype, like.device)["trig_in_kernel"]:
            # torch's own cos and sin of the azimuth, as the lobe forms it.
            phi = 2.0 * be.pi * r1
            cos_in, sin_in = be.cos(phi), be.sin(phi)
            trig_in = 1
    xf = _placement(component, like)
    root = getattr(component, "_local_root", None)
    has_root = int(
        root is not None and type(root[0]) is type(t) and root[0].shape == t.shape
    )
    t_adv, t_local = (root[0], root[1]) if has_root else (ph["f1"], ph["f1"])
    out = [torch.empty(n, dtype=like.dtype, device=like.device) for _ in range(7)]
    bo = torch.empty(n, dtype=rays.bounce.dtype, device=like.device)
    books = [torch.empty(n, dtype=like.dtype, device=like.device) for _ in range(3 if lobe else 1)]
    loss1 = books[0]
    resid, loss2 = (books[1], books[2]) if lobe else (ph["f1"], ph["f1"])
    _LAUNCHED["reflective"] = _LAUNCHED.get("reflective", 0) + 1
    torch.ops.optiland_nsq.interact_reflective(
        rays.x, rays.y, rays.z, rays.L, rays.M, rays.N, rays.flux, rays.bounce, t, normals, n_geom,
        hit_mask, t_adv, t_local, has_root, xf,
        [float(v) for v in (refl, tiny, sf_det, w_s, w_ns, rv, two_pi, offset_k, tau)], [int(lobe), int(use_tau)],
        u_lobe, r1, r2, u_sc, trig_in, cos_in, sin_in,
        *out, bo, loss1, resid, loss2,
    )
    rays.x, rays.y, rays.z, rays.L, rays.M, rays.N, rays.flux = out
    rays.bounce = bo
    # The ledger, as book_loss / book_residual / book_lobe add to it:
    # masked_sum is be.sum(where(hit, flux * fraction, 0)), the kernel's terms.
    sums = [be.sum(loss1)] + ([be.sum(resid), be.sum(loss2)] if lobe else [])
    _book(component, sums, lobe, like)


__all__ = [
    "SUPPORTED_DEVICE_TYPES",
    "availability",
    "compile_cache",
    "interact_component",
    "kernel_modules",
    "launch_counts",
    "prepare",
    "probes",
    "reset_routed",
    "routed_counts",
]
