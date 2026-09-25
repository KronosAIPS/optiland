"""Thin film optics core functions.

This provides core functions for thin film optics calculations using the
transfer matrix method (TMM).

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


def _snell_cos(n0, theta0, n):
    """Transmitted angle cosine with forward-branch selection.

    Calculation follows 'Thin-Film Optical Filters, Fifth Edition, Macleod,
    Hugh Angus CRC Press, Ch2.6.

    The root is taken with ``be.csqrt``: in an absorbing layer the argument's
    imaginary part (``-2 n k``) is small beside its real part, and on a backend
    whose native complex root loses such a part (Apple's ``mps``; see
    ``exact_complex_sqrt``) the layer's attenuation would vanish. Where the
    native root is exact, ``be.csqrt`` is that root, unchanged.

    Args:
        n0 (complex): Incident medium complex refractive index.
        theta0 (float): Angle of incidence in radians.
        n (complex): Medium complex refractive index.

    Returns:
        complex: Transmitted angle cosine.
    """
    nr = n.real
    k = n.imag
    return be.csqrt(nr**2 - k**2 - (n0 * be.sin(theta0)) ** 2 - 2j * nr * k) / n


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
        eta_p = sqrt_eps_mu**2 * (n.real - 1j * n.imag) ** 2 / eta_s
        return eta_p
    else:
        raise ValueError("Invalid polarization state")


def _tmm_coh(stack: ThinFilmStack, wavelength_um, theta0_rad, pol: PolSP):
    """Compute the reflection and transmission coefficients for a thin film stack.

    Calculation is vectorized over wavelength and angle of incidence.
    Based on Abelès Matrix.

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
    cos0 = _snell_cos(n0, theta0_rad, n0)
    coss = _snell_cos(n0, theta0_rad, ns)
    eta0 = _admittance(n0, cos0, pol)
    etas = _admittance(ns, coss, pol)

    # Id initial matrix
    A = be.to_complex(be.ones_like(eta0))
    B = be.to_complex(be.zeros_like(eta0))
    C = be.to_complex(be.zeros_like(eta0))
    D = be.to_complex(be.ones_like(eta0))

    for layer in stack.layers:
        n_l = layer.n_complex(wavelength_um)
        cos_l = _snell_cos(n0, theta0_rad, n_l)
        eta_l = _admittance(n_l, cos_l, pol)
        delta = layer.phase_thickness(wavelength_um, cos_l, n_l)
        c = be.cos(delta)
        s = be.sin(delta)
        i = 1j
        mA = c
        mB = i * (s / eta_l)
        mC = i * (eta_l * s)
        mD = c
        A, B, C, D = A * mA + B * mC, A * mB + B * mD, C * mA + D * mC, C * mB + D * mD

    denom = eta0 * (A + etas * B) + C + etas * D
    denom = be.where(be.abs(denom) == 0, _denominator_floor(denom), denom)

    r = (eta0 * A + eta0 * etas * B - C - etas * D) / denom
    t = be.conj((2 * eta0) / denom)

    R = (r * be.conj(r)).real
    T = (t * be.conj(t)).real * etas.real / eta0.real
    abso = 1 - R - T
    # Absorption can also be obtained with :
    # abso = (1 - R) * (1 - etas.real / ((A + etas * B) * (C + etas * D).conj()).real)
    return r, t, R, T, abso
