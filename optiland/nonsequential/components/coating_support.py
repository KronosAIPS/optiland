"""Shared coating/reflectance resolution for NSQ components.

RefractiveComponent (an ``optiland.coatings`` coating on a transmissive
interface) and ReflectiveComponent (a required reflectance on a mirror) both
need to validate that a coating is unpolarized and turn it into a per-ray
array. Centralized here so the two components agree on what is accepted.

``UnpolarizedThinFilmCoating`` below is the one angle-dependent option: it
wraps an ``optiland.thin_film.ThinFilmStack`` (the same characteristic-matrix
model the sequential engine uses) and evaluates it per ray, at that ray's own
wavelength and angle of incidence, averaging the s and p reflectance the way
an unpolarized ray bundle requires. Every other coating accepted here
(``SimpleCoating``, a constant, a ``callable(wavelength_um)``) stays
wavelength-only and angle-blind, as before.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.coatings import BaseCoating
    from optiland.thin_film import ThinFilmStack


def reject_polarized_coating(coating: object, *, surface_name: str) -> None:
    """Raise if ``coating`` is a Jones-matrix (polarized) coating.

    NSQ rays carry no polarization state, so a polarized coating cannot be
    evaluated correctly. Rather than silently falling back to some scalar
    average of its Jones matrix, refuse it outright.

    ``UnpolarizedThinFilmCoating`` is not a ``BaseCoatingPolarized`` (it does
    not subclass the sequential-engine coating hierarchy at all), so it never
    trips this check -- it already performs the s/p average NSQ needs before
    a ray-facing value is produced.

    Args:
        coating: The candidate coating, or None / a plain float / a callable.
        surface_name: Name of the surface the coating is attached to, for
            the error message.

    Raises:
        NotImplementedError: If ``coating`` is a
            ``optiland.coatings.BaseCoatingPolarized`` instance.
    """
    from optiland.coatings import BaseCoatingPolarized  # noqa: PLC0415

    if isinstance(coating, BaseCoatingPolarized):
        raise NotImplementedError(
            f"Surface {surface_name!r} was given a polarized coating "
            f"({type(coating).__name__}); NSQ rays carry no polarization "
            "state, so Jones-matrix coatings cannot be evaluated. Use an "
            "unpolarized coating such as optiland.coatings.SimpleCoating, "
            "optiland.nonsequential.components.coating_support"
            ".UnpolarizedThinFilmCoating (angle- and wavelength-dependent, "
            "unpolarized thin-film stack), a constant reflectance, or a "
            "callable(wavelength_um) -> reflectance instead."
        )


class UnpolarizedThinFilmCoating:
    """Unpolarized, angle-dependent NSQ adapter around a ``ThinFilmStack``.

    ``optiland.coatings.ThinFilmCoating`` builds a Jones matrix from the same
    stack and is rejected by :func:`reject_polarized_coating` -- NSQ rays
    carry no polarization state. This adapter evaluates the stack's
    characteristic-matrix (R, T) at each ray's own wavelength and angle of
    incidence and reduces s/p to the unpolarized average the theory chapter
    uses, :math:`\\bar R=(R_s+R_p)/2` (and the matching T), which is exactly
    what ``ThinFilmStack.compute_rtRTA_elementwise(..., polarization="u")``
    already computes -- this class only adapts the call to what
    ``RefractiveComponent.interact`` has on hand (a per-ray cosine of the
    angle of incidence, not an angle in radians) and gives the dispatch in
    :func:`evaluate_transmissive_coating` an ``evaluate`` method to find.

    An empty stack (no layers) reduces to the bare unpolarized Fresnel
    reflectance of the incident/substrate interface, since the
    characteristic matrix of a layer-less stack is the identity.

    Args:
        stack: A configured ``optiland.thin_film.ThinFilmStack`` (incident
            medium, substrate, and zero or more layers).

    Note:
        Nothing here detaches from the autograd graph:
        ``compute_rtRTA_elementwise`` is built entirely out of
        ``optiland.backend`` array ops, so a stack whose layer thicknesses
        or material indices are ``torch`` tensors keeps gradients flowing
        through ``evaluate``'s (R, T) the same way the bare-Fresnel branch
        already does for ``material_front``/``material_back``.
    """

    def __init__(self, stack: ThinFilmStack) -> None:
        self.stack = stack

    def evaluate(
        self, wavelength_um: be.ndarray, cos_theta_i: be.ndarray
    ) -> tuple[be.ndarray, be.ndarray]:
        """Per-ray unpolarized (R, T) at this ray's wavelength and AOI.

        Args:
            wavelength_um: Per-ray wavelength [µm], shape (N,).
            cos_theta_i: Per-ray cosine of the angle of incidence, shape
                (N,). Clipped to [-1, 1] here (mirrors
                ``BaseCoating._compute_aoi``) since the caller's value can
                land a hair outside that range from floating-point error.

        Returns:
            ``(R, T)``, each shape (N,): :math:`(R_s+R_p)/2` and the
            matching unpolarized T, both attached to the autograd graph
            whenever the stack's parameters are.
        """
        cos_theta_i = be.clip(cos_theta_i, -1.0, 1.0)
        aoi_rad = be.arccos(cos_theta_i)
        out = self.stack.compute_rtRTA_elementwise(
            wavelength_um, aoi_rad, polarization="u"
        )
        return out["R"], out["T"]


def evaluate_transmissive_coating(
    coating: object,
    wavelength_um: be.ndarray,
    cos_theta_i: be.ndarray,
) -> tuple[be.ndarray, be.ndarray]:
    """Per-ray (R, T) for a non-None coating on a ``RefractiveComponent``.

    The dispatch point ``RefractiveComponent.interact`` hits once it knows a
    coating is attached (already checked non-polarized by
    :func:`reject_polarized_coating` at construction time):

    - A coating exposing an ``evaluate(wavelength_um, cos_theta_i)`` method
      (:class:`UnpolarizedThinFilmCoating`) is asked to compute its own
      per-ray, wavelength- and angle-dependent (R, T).
    - Anything else (``SimpleCoating``) is read via its scalar
      ``.reflectance``/``.transmittance`` attributes and broadcast to every
      ray regardless of wavelength or angle -- unchanged from before this
      adapter existed.

    Args:
        coating: The attached coating (never None -- the caller only enters
            this branch when ``self.coating is not None``).
        wavelength_um: Per-ray wavelength [µm], shape (N,).
        cos_theta_i: Per-ray cosine of the angle of incidence, shape (N,).

    Returns:
        ``(R_used, T_used)``, each shape (N,), matching ``wavelength_um``.
    """
    evaluate = getattr(coating, "evaluate", None)
    if callable(evaluate):
        return evaluate(wavelength_um, cos_theta_i)
    return (
        be.ones_like(wavelength_um) * float(coating.reflectance),
        be.ones_like(wavelength_um) * float(coating.transmittance),
    )


def resolve_reflectance(
    reflectance: float | Callable[[be.ndarray], be.ndarray] | BaseCoating,
    wavelength: be.ndarray,
) -> be.ndarray:
    """Turn a mirror's ``reflectance`` spec into a per-ray array.

    Args:
        reflectance: A constant, a ``callable(wavelength_um) -> reflectance``,
            or an unpolarized ``optiland.coatings.BaseCoating`` (read via its
            ``.reflectance`` attribute, e.g. ``SimpleCoating``).
        wavelength: Per-ray wavelength [µm], shape (N,); used only to build
            the broadcast shape for a constant/coating reflectance.

    Returns:
        Per-ray reflectance, shape (N,).
    """
    from optiland.coatings import BaseCoating  # noqa: PLC0415

    if isinstance(reflectance, BaseCoating):
        return be.ones_like(wavelength) * float(reflectance.reflectance)
    if callable(reflectance):
        return be.ones_like(wavelength) * be.array(reflectance(wavelength))
    if getattr(reflectance, "requires_grad", False):
        # A constant that carries a gradient stays attached (docs/theory/
        # 09_differentiation.md R-09-2: a coating reflectance is attached);
        # float() here detached it silently, which the dead-parameter raise
        # of the parameter register found. A plain number takes the line
        # below, unchanged.
        return be.ones_like(wavelength) * reflectance.to(
            dtype=wavelength.dtype, device=wavelength.device
        )
    return be.ones_like(wavelength) * float(reflectance)
