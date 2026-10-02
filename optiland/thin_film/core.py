"""Thin film optics core functions.

This provides core functions for thin film optics calculations using the
transfer matrix method (TMM).

Time convention (the research repository's issue 78; its theory chapters 04
section 4.4 and 06 section 6.0). Fields vary as ``exp(-i omega t)``: a complex
index is ``n + i k`` with ``k >= 0`` for an absorbing medium, a layer's matrix
is ``[[cos d, -i sin d / eta], [-i eta sin d, cos d]]``, and the root of
``N cos theta = sqrt(N^2 - (n0 sin theta0)^2)`` is the one with a non-negative
imaginary part, a wave that decays (or, beyond the critical angle of a
lossless medium, is evanescent with ``N cos theta = +i kappa``) away from the
interface. One convention for the layers, the absorbing media and the
evanescent root: a coated face used beyond its critical angle (frustrated
total internal reflection) is in it too. The reflection ``r`` is the
admittance form ``(eta0 B - C) / (eta0 B + C)`` (for p the negative of the
Fresnel ``r_p``) and ``t = 2 eta0 / (eta0 B + C)`` is the ratio of tangential
fields. Until 2026-10-01 the layers and the absorbing index were in the
``exp(+i omega t)`` convention while the evanescent root was the
``exp(-i omega t)`` one, and ``t`` was conjugated; the powers were right in
every case, and they are unchanged.

Corentin Nannini, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial
    from optiland.thin_film import ThinFilmStack

Array: TypeAlias = Any  # be.ndarray
PolSP = Literal["s", "p"]

#: What a zero characteristic denominator is replaced by in :func:`_tmm_coh`.
_DENOM_FLOOR = 1e-30 + 0j

#: That floor as a 0-dim tensor, one per (dtype, device), built on first use.
_DENOM_FLOOR_TENSORS: dict[tuple[str, str], Any] = {}


def _denominator_floor(like: Array) -> Any:
    """The denominator floor beside ``like``: a resident tensor, or the bare number.

    ``be.where(mask, 1e-30 + 0j, denom)`` builds the scalar into a tensor with
    ``torch.tensor`` at every call, which on a device is a host-to-device copy
    per evaluation -- per ray bounce when a non-sequential trace evaluates a
    coating -- and an operation a CUDA graph capture refuses. The torch
    backend's ``where`` builds exactly this tensor (the scalar in ``like``'s
    dtype, on its device); it is built once here and kept. A NumPy array gets
    the bare number, so the NumPy path is unchanged.

    Args:
        like: The denominator the floor is selected into.

    Returns:
        The floor, as ``like``'s kind of value.
    """
    if not hasattr(like, "detach"):
        return _DENOM_FLOOR
    key = (str(like.dtype), str(like.device))
    floor = _DENOM_FLOOR_TENSORS.get(key)
    if floor is None:
        import torch  # noqa: PLC0415

        floor = torch.tensor(_DENOM_FLOOR, dtype=like.dtype, device=like.device)
        _DENOM_FLOOR_TENSORS[key] = floor
    return floor


def _complex_index(material: BaseMaterial, wavelength_um: float | Array) -> Array:
    if be.get_backend() == "torch" and hasattr(wavelength_um, "detach"):
        # Avoid material cache-key conversion that may require tensor->numpy bridge.
        n = material._calculate_n(wavelength_um)
        k = material._calculate_k(wavelength_um)
    else:
        n = material.n(wavelength_um)
        k = material.k(wavelength_um)
    n = be.atleast_1d(n)
    k = be.atleast_1d(k)
    return be.to_complex(n) + 1j * be.to_complex(k)


#: The level below which a negative imaginary part of the radicand
#: ``w = N^2 - s0^2`` is rounding noise, relative to ``|N|^2 + |s0|^2``. It is
#: the research repository's ``knsrt.analytic.ROOT_NOISE_RELATIVE`` (its
#: chapter 06 section 6.0, "The branch of the root"), mirrored here: the
#: material library's agreement band for published rounded pole forms is
#: 1e-9, so an extinction coefficient a record carries below that band (a
#: page's ``k = -2.6e-11``, say) is data rounding, not gain. Evaluation
#: rounding alone is at most ``(sqrt(5) + 1) u`` of the same scale.
ROOT_NOISE_RELATIVE = 1.0e-9


def _snell_cos(n0, theta0, n):
    """Transmitted angle cosine with forward-branch selection.

    Calculation follows 'Thin-Film Optical Filters, Fifth Edition, Macleod,
    Hugh Angus CRC Press, Ch2.6.

    The root is taken with ``be.csqrt``: in an absorbing layer the argument's
    imaginary part (``-2 n k``) is small beside its real part, and on a backend
    whose native complex root loses such a part (Apple's ``mps``; see
    ``exact_complex_sqrt``) the layer's attenuation would vanish. Where the
    native root is exact, ``be.csqrt`` is that root, unchanged.

    The branch (the research repository's chapter 06 section 6.0): the
    ``exp(-i omega t)`` root of ``w = N^2 - s0^2`` is the limit of the
    principal root of ``w + i 0``, which lies in the first quadrant; where
    ``Im w`` is positive (a lossy medium) it is the decaying root. A negative
    ``Im w`` no larger in magnitude than :data:`ROOT_NOISE_RELATIVE`
    ``* (|N|^2 + |s0|^2)`` is rounding noise of a lossless medium and is
    taken as zero, which gives the forward root ``Re > 0`` below the critical
    angle and ``+i kappa`` beyond it. Until 2026-10-02 any negative ``Im w``
    flipped the root, which turned a forward wave into a backward one for a
    record with ``k`` of order ``-1e-11``. A negative ``Im w`` beyond that
    level (a gain medium, outside the passive theory) keeps the earlier rule.
    Where ``Im w >= 0`` or is exactly ``-0.0`` every value is the one the
    earlier rule gave, bit for bit (the same operations selected by
    ``where``).

    Args:
        n0 (complex): Incident medium complex refractive index.
        theta0 (float): Angle of incidence in radians.
        n (complex): Medium complex refractive index.

    Returns:
        complex: Transmitted angle cosine.
    """
    nr = n.real
    k = n.imag
    s0 = n0 * be.sin(theta0)
    w = nr**2 - k**2 - s0**2 + 2j * nr * k
    w_im = be.imag(w)
    noise = ROOT_NOISE_RELATIVE * (nr**2 + k**2 + be.real(s0 * be.conj(s0)))
    w = be.where((w_im < 0) & (-w_im <= noise), be.real(w) + 0j, w)
    root = be.csqrt(w)
    # exp(-i omega t): the root with Im >= 0. The principal root already is,
    # except beyond the critical angle of a lossless medium, where the
    # radicand is a negative real whose imaginary zero may carry either sign,
    # and for a gain medium beyond the noise level.
    root = be.where(be.imag(root) < 0, -root, root)
    return root / n


def _admittance(n: complex, cos_t: complex, pol: PolSP):
    """Admittance η = sqrt(ε/μ) * n * cos(θ) for s and p polarizations.

    Calculation follows 'Thin-Film Optical Filters, Fifth Edition, Macleod,
    Hugh Angus CRC Press, Ch2.6.

    Args:
        n (complex): Medium complex refractive index.
        cos_t (complex): Cosine of the angle of propagation in the medium.
        pol (PolSP): Polarization state ('s' or 'p').

    Returns:
        complex: Optical admittance.
    """
    sqrt_eps_mu = 0.002654418729832701370374020517935  # S
    eta_s = sqrt_eps_mu * n * cos_t

    if pol == "s":
        return eta_s
    elif pol == "p":
        eta_p = sqrt_eps_mu**2 * (n.real + 1j * n.imag) ** 2 / eta_s
        return eta_p
    else:
        raise ValueError("Invalid polarization state")


def _tmm_coh(
    stack: ThinFilmStack, wavelength_um, theta0_rad, pol: PolSP, reverse=None
):
    """Compute the reflection and transmission coefficients for a thin film stack.

    Calculation is vectorized over wavelength and angle of incidence.
    Based on Abelès Matrix.

    ``reverse`` (optional, a boolean array broadcastable with the inputs) marks
    the elements that meet the stack from its substrate side: for those the
    stack is evaluated reversed -- the substrate as the incident medium, the
    layers in the opposite order, the incident medium as the substrate --
    with ``theta0_rad`` the angle in the substrate. Where ``reverse`` is
    False every value is the one ``reverse=None`` gives, bit for bit (the
    same operations on the same operands, selected with ``where``).

    Ref:
        - Chap 13. Polarized Light and Optical Systems, Russell
          A. Chipman, Wai-Sze Tiffany Lam, and Garam Young.
        - F. Abelès, Researches sur la propagation des ondes électromagnétiques
          sinusoïdales dans les milieus stratifies. Applications aux couches
          minces, Ann. Phys. Paris, 12ième Series 5 (1950): 596–640.
        - Chap 2. Thin-Film Optical Filters, Fifth Edition, Macleod, Hugh Angus
          CRC Press.

    Args:
        stack (ThinFilmStack): The thin film stack to compute.
        wavelength_um (float | Array): Wavelength(s) in microns.
        theta0_rad (float | Array): Angle(s) of incidence in radians.
        pol (PolSP): Polarization state ('s' or 'p').

    Returns:
        tuple[Array, Array, Array, Array, Array]: (r, t, R, T, A) where:
            r: Complex reflection coefficient.
            t: Complex transmission coefficient.
            R: Reflectance.
            T: Transmittance.
            A: Absorptance.
    """
    n0 = _complex_index(stack.incident_material, wavelength_um)
    ns = _complex_index(stack.substrate_material, wavelength_um)
    if reverse is not None:
        n0, ns = be.where(reverse, ns, n0), be.where(reverse, n0, ns)
    cos0 = _snell_cos(n0, theta0_rad, n0)
    coss = _snell_cos(n0, theta0_rad, ns)
    eta0 = _admittance(n0, cos0, pol)
    etas = _admittance(ns, coss, pol)

    # Id initial matrix
    A = be.to_complex(be.ones_like(eta0))
    B = be.to_complex(be.zeros_like(eta0))
    C = be.to_complex(be.zeros_like(eta0))
    D = be.to_complex(be.ones_like(eta0))

    layers = list(stack.layers)
    for j, layer in enumerate(layers):
        n_l = layer.n_complex(wavelength_um)
        if reverse is None:
            cos_l = _snell_cos(n0, theta0_rad, n_l)
            delta = layer.phase_thickness(wavelength_um, cos_l, n_l)
        else:
            mirror = layers[len(layers) - 1 - j]
            n_l = be.where(reverse, mirror.n_complex(wavelength_um), n_l)
            cos_l = _snell_cos(n0, theta0_rad, n_l)
            ones = be.ones_like(be.real(cos_l))
            d = be.where(
                reverse, ones * mirror.thickness_um, ones * layer.thickness_um
            )
            delta = (2 * be.pi / wavelength_um) * n_l * d * cos_l
        eta_l = _admittance(n_l, cos_l, pol)
        c = be.cos(delta)
        s = be.sin(delta)
        i = -1j  # exp(-i omega t)
        mA = c
        mB = i * (s / eta_l)
        mC = i * (eta_l * s)
        mD = c
        A, B, C, D = A * mA + B * mC, A * mB + B * mD, C * mA + D * mC, C * mB + D * mD

    denom = eta0 * (A + etas * B) + C + etas * D
    denom = be.where(be.abs(denom) == 0, _denominator_floor(denom), denom)

    r = (eta0 * A + eta0 * etas * B - C - etas * D) / denom
    t = (2 * eta0) / denom

    R = (r * be.conj(r)).real
    T = (t * be.conj(t)).real * etas.real / eta0.real
    abso = 1 - R - T
    # Absorption can also be obtained with :
    # abso = (1 - R) * (1 - etas.real / ((A + etas * B) * (C + etas * D).conj()).real)
    return r, t, R, T, abso
