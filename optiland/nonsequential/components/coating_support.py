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

    The scalar mode reads a coating's R and T, and the Stokes mode the s and
    p terms and relative phases of a thin-film stack or a coating table
    (``polarization.coating_sp``); neither reads a Jones matrix. Rather than
    silently falling back to some scalar average of it, refuse it outright.

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
            f"({type(coating).__name__}); the non-sequential engine reads R "
            "and T, or in its Stokes mode the s and p terms of a thin-film "
            "stack or a table, never a Jones matrix, so Jones-matrix "
            "coatings cannot be evaluated. Use an "
            "unpolarized coating such as optiland.coatings.SimpleCoating, "
            "optiland.nonsequential.components.coating_support"
            ".UnpolarizedThinFilmCoating (angle- and wavelength-dependent, "
            "unpolarized thin-film stack), a constant reflectance, or a "
            "callable(wavelength_um) -> reflectance instead."
        )


class UnpolarizedThinFilmCoating:
    """Unpolarized, angle-dependent NSQ adapter around a ``ThinFilmStack``.

    ``optiland.coatings.ThinFilmCoating`` builds a Jones matrix from the same
    stack and is rejected by :func:`reject_polarized_coating`. This adapter
    evaluates the stack's
    characteristic-matrix (R, T) at each ray's own wavelength and angle of
    incidence and reduces s/p to the unpolarized average the theory chapter
    uses, :math:`\\bar R=(R_s+R_p)/2` (and the matching T), which is exactly
    what ``ThinFilmStack.compute_rtRTA_elementwise(..., polarization="u")``
    already computes -- this class only adapts the call to what
    ``RefractiveComponent.interact`` has on hand (a per-ray cosine of the
    angle of incidence, not an angle in radians) and gives the dispatch in
    :func:`evaluate_transmissive_coating` an ``evaluate`` method to find.

    In the engine's Stokes mode a refractive face reads the same stack's s and
    p terms and relative phases instead (``polarization.coating_sp``), and its
    scalar (R, T) are their means, the values :meth:`evaluate` returns, bit
    for bit.

    An empty stack (no layers) reduces to the bare unpolarized Fresnel
    reflectance of the incident/substrate interface, since the
    characteristic matrix of a layer-less stack is the identity.

    The stack describes the interface as its incident medium sees it. A ray
    arriving from the substrate side is evaluated on the reversed stack (the
    research repository's issue 83): the refractive component matches the
    stack's two media to its own front and back once
    (:func:`coating_incident_is_front`) and passes the per-ray side to
    :meth:`evaluate`.

    Args:
        stack: A configured ``optiland.thin_film.ThinFilmStack`` (incident
            medium, substrate, and zero or more layers).
        incident_side: Which side of the component the stack's incident
            medium is on: ``"front"``, ``"back"``, or ``"auto"`` (the
            default: matched by index, see :func:`coating_incident_is_front`).

    Note:
        Nothing here detaches from the autograd graph:
        ``compute_rtRTA_elementwise`` is built entirely out of
        ``optiland.backend`` array ops, so a stack whose layer thicknesses
        or material indices are ``torch`` tensors keeps gradients flowing
        through ``evaluate``'s (R, T) the same way the bare-Fresnel branch
        already does for ``material_front``/``material_back``.
    """

    def __init__(self, stack: ThinFilmStack, incident_side: str = "auto") -> None:
        self.stack = stack
        self.incident_side = _check_incident_side(incident_side)

    def media(self):
        """``(incident, substrate)``: the stack's two media (``BaseMaterial``)."""
        return self.stack.incident_material, self.stack.substrate_material

    def evaluate(
        self,
        wavelength_um: be.ndarray,
        cos_theta_i: be.ndarray,
        from_substrate: be.ndarray | None = None,
    ) -> tuple[be.ndarray, be.ndarray]:
        """Per-ray unpolarized (R, T) at this ray's wavelength and AOI.

        Args:
            wavelength_um: Per-ray wavelength [µm], shape (N,).
            cos_theta_i: Per-ray cosine of the angle of incidence, shape
                (N,), in the medium the ray arrives from. Clipped to [-1, 1]
                here (mirrors ``BaseCoating._compute_aoi``) since the
                caller's value can land a hair outside that range from
                floating-point error.
            from_substrate: Optional per-ray boolean mask of the rays that
                arrive from the stack's substrate side (the research
                repository's issue 83). Those are evaluated on the reversed
                stack: the substrate as the incident medium, the layers in
                the opposite order, at their own angle in the substrate.
                ``None`` (the default, and what a caller with no side
                information passes) evaluates every ray from the incident
                medium, as before.

        Returns:
            ``(R, T)``, each shape (N,): :math:`(R_s+R_p)/2` and the
            matching unpolarized T, both attached to the autograd graph
            whenever the stack's parameters are.
        """
        cos_theta_i = be.clip(cos_theta_i, -1.0, 1.0)
        aoi_rad = be.arccos(cos_theta_i)
        out = self.stack.compute_rtRTA_elementwise(
            wavelength_um, aoi_rad, polarization="u", reverse=from_substrate
        )
        return out["R"], out["T"]


#: The values ``incident_side`` takes on a side-aware coating.
INCIDENT_SIDES = ("auto", "front", "back")


def _check_incident_side(value: str) -> str:
    if value not in INCIDENT_SIDES:
        raise ValueError(f"incident_side must be one of {INCIDENT_SIDES}, got {value!r}")
    return value


def _index_at(material, wavelength_um: float) -> float:
    """A material's real index at one wavelength, as a Python float (host side).

    ``None`` (the engine's vacuum) is 1. Accepts an ``NSQMaterial`` (its
    ``optiland_material``) or an ``optiland.materials.BaseMaterial``.
    """
    inner = getattr(material, "optiland_material", material)
    if inner is None:
        return 1.0
    import numpy as np  # noqa: PLC0415

    value = inner.n(wavelength_um)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value, dtype=float).ravel()[0])


def coating_incident_is_front(
    coating: object, material_front: object, material_back: object
) -> bool | None:
    """Whether a side-aware coating's incident medium is the component's front.

    A coating that describes one side of an interface (a thin-film stack, a
    table) has its own incident medium and substrate, and a refractive
    component has its own front and back media; the two are matched once,
    when the coating is attached, never per ray (the research repository's
    issue 83). ``coating.incident_side`` decides when it is ``"front"`` or
    ``"back"``. With ``"auto"`` (the default) the indices are compared at
    the stack's reference wavelength (0.55 um when it has none): the
    incident medium is the front when
    ``|n_front - n_inc| + |n_back - n_sub| <= |n_front - n_sub| + |n_back - n_inc|``,
    so a stack written from its outer medium inwards is placed right on
    either face of a lens; a tie (a stack between two equal media) keeps
    the front, which is what every coating was before the side existed.

    Args:
        coating: The attached coating.
        material_front, material_back: The component's two media.

    Returns:
        True or False for a side-aware coating (one with ``media()``);
        None for a side-blind one (a ``SimpleCoating``, a constant).
    """
    media = getattr(coating, "media", None)
    if not callable(media):
        return None
    side = getattr(coating, "incident_side", "auto")
    if side == "front":
        return True
    if side == "back":
        return False
    incident, substrate = media()
    stack = getattr(coating, "stack", None)
    wl = getattr(stack, "reference_wl_um", None) or getattr(
        coating, "reference_wavelength_um", None
    ) or 0.55
    wl = float(wl)
    n_inc, n_sub = _index_at(incident, wl), _index_at(substrate, wl)
    n_f, n_b = _index_at(material_front, wl), _index_at(material_back, wl)
    return abs(n_f - n_inc) + abs(n_b - n_sub) <= abs(n_f - n_sub) + abs(n_b - n_inc)


def coating_holds_beyond_critical(coating: object) -> bool:
    """Whether a coating's own R and T hold beyond the critical angle (issue 96).

    A side-aware coating (one with ``media()``: a thin-film stack, a table)
    describes the whole interface, its two media included, so its reflectance
    beyond the bare interface's critical angle is its own: a lossless stack
    gives ``R = 1`` there to rounding, an absorbing layer in the evanescent
    field gives ``R < 1`` (frustrated total internal reflection, the research
    repository's chapter 06 section 6.15). A side-blind coating (a
    ``SimpleCoating``, a constant) states one ``R`` and ``T`` for the
    transmitting regime and says nothing about total internal reflection, so
    the bare interface's ``R = 1`` stands for it.

    Args:
        coating: The attached coating, or None.

    Returns:
        True for a side-aware coating, False otherwise.
    """
    return coating is not None and callable(getattr(coating, "media", None))


def from_substrate_mask(incident_is_front: bool | None, entering_back):
    """Per ray: does it arrive from the coating's substrate side?

    ``entering_back`` is the component's own mask of the rays on its front
    side (travelling from ``material_front`` into ``material_back``).

    Returns:
        A boolean mask, or None for a side-blind coating.
    """
    if incident_is_front is None:
        return None
    return ~entering_back if incident_is_front else entering_back


def evaluate_transmissive_coating(
    coating: object,
    wavelength_um: be.ndarray,
    cos_theta_i: be.ndarray,
    from_substrate: be.ndarray | None = None,
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
        from_substrate: Optional per-ray mask of the rays arriving from the
            coating's substrate side (:func:`from_substrate_mask`); passed
            to a side-aware coating, ignored by a side-blind one.

    Returns:
        ``(R_used, T_used)``, each shape (N,), matching ``wavelength_um``.
    """
    evaluate = getattr(coating, "evaluate", None)
    if callable(evaluate):
        if from_substrate is not None:
            return evaluate(wavelength_um, cos_theta_i, from_substrate=from_substrate)
        return evaluate(wavelength_um, cos_theta_i)
    return (
        be.ones_like(wavelength_um) * float(coating.reflectance),
        be.ones_like(wavelength_um) * float(coating.transmittance),
    )


def resolve_reflectance(
    reflectance: float | Callable[[be.ndarray], be.ndarray] | BaseCoating,
    wavelength: be.ndarray,
    cos_theta_i: be.ndarray | None = None,
) -> be.ndarray:
    """Turn a mirror's ``reflectance`` spec into a per-ray array.

    Args:
        reflectance: A constant, a ``callable(wavelength_um) -> reflectance``,
            an unpolarized ``optiland.coatings.BaseCoating`` (read via its
            ``.reflectance`` attribute, e.g. ``SimpleCoating``), or an
            :class:`UnpolarizedThinFilmCoating` -- a dielectric or metal
            mirror as a thin-film stack (a bare absorbing substrate is a
            stack with no layers), evaluated at each ray's wavelength and
            angle of incidence as ``(R_s + R_p) / 2``.
        wavelength: Per-ray wavelength [µm], shape (N,); used only to build
            the broadcast shape for a constant/coating reflectance.
        cos_theta_i: Per-ray ``|cos theta_i|``; read only for a thin-film
            stack, which needs it.

    Returns:
        Per-ray reflectance, shape (N,).
    """
    from optiland.coatings import BaseCoating  # noqa: PLC0415

    evaluate = getattr(reflectance, "evaluate", None)
    if callable(evaluate):
        if cos_theta_i is None:
            raise ValueError(
                "a thin-film mirror is evaluated at each ray's angle of incidence; "
                "pass cos_theta_i"
            )
        return evaluate(wavelength, cos_theta_i)[0]
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
