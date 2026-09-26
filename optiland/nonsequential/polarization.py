"""Stokes polarization for the non-sequential engine: conventions and math.

The minimal polarization of the research repository's issue 5: every ray keeps
its scalar flux ``w`` as the Stokes ``I`` and, when polarization is on, carries
the *reduced* Stokes state ``(q, u, v) = (Q, U, V) / I`` and a unit reference
axis ``e`` perpendicular to its direction ``k``. This module holds the
conventions and the arithmetic; it touches no ray bundle and no component.

Conventions (stated here once, at the API boundary; R-06-1)
-----------------------------------------------------------

* **Basis.** ``e`` is the ``x = p`` axis of the ray's Stokes frame, ``s = k x e``
  its ``y`` axis, and ``(p, s, k)`` is right-handed (``p x s = k``). At an
  interface ``p`` lies in the plane of incidence and ``s = k x n / |k x n|``.
* **Stokes vector.** ``S = (|E_p|^2 + |E_s|^2, |E_p|^2 - |E_s|^2,
  2 Re(E_p E_s*), -2 Im(E_p E_s*))``: ``Q > 0`` is ``p``-dominant, ``U > 0`` is
  linear polarization at +45 degrees from ``p`` towards ``s``, and the sign of
  ``V`` is the one the quarter-wave retarder fixes: a fast axis at +45 degrees
  takes ``(1, 1, 0, 0)`` to ``(1, 0, 0, -1)``.
* **Time-harmonic convention.** Fields vary as ``exp(-i omega t)``, a complex
  index is ``n + i k`` with ``k >= 0`` for an absorbing medium, and an
  evanescent (totally reflected) wave takes the root with
  ``cos(theta_t) = +i kappa``. The Fresnel amplitudes are
  ``r_s = (n1 cos_i - n2 cos_t) / (n1 cos_i + n2 cos_t)`` and
  ``r_p = (n2 cos_i - n1 cos_t) / (n2 cos_i + n1 cos_t)``. This is the
  convention of the catalogue's analytic reference (its
  ``fresnel_amplitude_rs_rp``): beyond the critical angle at 1.5 -> 1.0 and
  45 degrees, ``arg(r_p r_s*) = -36.8699`` degrees. Section 6.1 of the
  research repository's theory chapter 06 prints the same magnitudes with the
  opposite sign, which is the other time convention; the chapter does not
  state its own, and the catalogue grades against the analytic module.
* **Frame rotation.** Turning the reference axis from ``e`` to ``a`` (both
  perpendicular to ``k``) by the angle ``psi`` measured about ``k`` applies
  ``M_rot(psi)`` of chapter 06 section 6.5: ``q' = cos2psi q + sin2psi u``,
  ``u' = -sin2psi q + cos2psi u``, ``v' = v``. The doubled angle is formed
  algebraically from ``cos psi = e . a`` and ``sin psi = (e x a) . k``; no
  trigonometric function is called.
* **Interface form.** Every element this module builds is, in its own frame,
  ``[[m00, m01, 0, 0], [m01, m00, 0, 0], [0, 0, m22, m23], [0, 0, -m23, m22]]``
  (:class:`InterfaceMueller`): a Fresnel reflection or transmission, a linear
  diattenuator or polarizer with its axis on ``p``, a linear retarder with its
  fast axis on ``p``. Applied to the reduced state it multiplies the flux by
  ``g = m00 + m01 q`` and returns the normalised output state.

Why the reduced form. ``I`` stays the engine's ``flux``, so the one ledger of
R-06-2 is literal: Beer-Lambert, roulette, split weights and detectors keep
acting on ``flux`` alone, and a trace with polarization off carries no extra
field and does no extra operation. With an unpolarized state (``q = 0``) the
flux factor ``g`` is ``m00`` exactly, the scalar coefficient, bit for bit.

Every function works on whatever array library the arguments are (NumPy or
torch, float32 or float64) through ``optiland.backend``, on real arrays only.
Guards follow the double-``where`` rule of the theory's chapter 09 (R-09-10):
the input of a division or a square root is masked, never its output alone.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import optiland.backend as be
from optiland.nonsequential import _tol

#: The run-level polarization modes (R-06-9). ``"off"`` is the scalar engine
#: exactly as it was; ``"stokes"`` carries the reduced Stokes state.
MODES: tuple[str, ...] = ("off", "stokes")

#: The six optional per-ray fields of :class:`~optiland.nonsequential
#: .ray_bundle.NSQRayBundle` that hold the state when polarization is on.
POL_FIELDS: tuple[str, ...] = ("pol_q", "pol_u", "pol_v", "pol_ex", "pol_ey", "pol_ez")


#: The environment variable a backend built without an explicit
#: ``polarization`` reads (``off`` or ``stokes``; unset is ``off``), so a
#: harness that builds its own backends -- a catalogue runner, a notebook that
#: calls ``scene.trace()`` -- can be run in Stokes mode unchanged, as
#: ``OPTILAND_NSQ_COMPILE_STEP`` does for the compiled step.
POLARIZATION_ENV = "OPTILAND_NSQ_POLARIZATION"


def mode_for_backend(value: object) -> str:
    """A backend's polarization mode: its explicit argument, or the environment's.

    Args:
        value: The backend's ``polarization`` argument; ``None`` reads
            :data:`POLARIZATION_ENV`.

    Returns:
        ``"off"`` or ``"stokes"``.
    """
    if value is None:
        import os  # noqa: PLC0415

        env = os.environ.get(POLARIZATION_ENV, "").strip()
        return normalise_mode(env) if env else "off"
    return normalise_mode(value)


def normalise_mode(value: object) -> str:
    """The polarization mode as one of :data:`MODES`.

    Args:
        value: ``"off"`` or ``"stokes"``; ``False``/``None`` mean ``"off"`` and
            ``True`` means ``"stokes"``.

    Returns:
        ``"off"`` or ``"stokes"``.

    Raises:
        ValueError: For anything else.
    """
    if value is None or value is False:
        return "off"
    if value is True:
        return "stokes"
    if isinstance(value, str) and value.strip().lower() in MODES:
        return value.strip().lower()
    raise ValueError(f"polarization must be one of {MODES} (or a bool), got {value!r}")


class InterfaceMueller(NamedTuple):
    """The four distinct elements of an interface-form Mueller matrix.

    ``M = [[m00, m01, 0, 0], [m01, m00, 0, 0], [0, 0, m22, m23],
    [0, 0, -m23, m22]]`` in the element's own frame (``p`` on the plane of
    incidence, the polarizer's transmission axis or the retarder's fast axis).
    """

    m00: object
    m01: object
    m22: object
    m23: object


# ---------------------------------------------------------------------------
# Small vector helpers on component arrays
# ---------------------------------------------------------------------------


def _dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def _masked_sqrt(x):
    """``sqrt(x)`` for ``x > 0``; zero, with a finite derivative, where ``x <= 0``."""
    positive = x > 0
    root = be.where(positive, x, be.ones_like(x)) ** 0.5
    return be.where(positive, root, be.zeros_like(x))


def degeneracy_tolerance(like):
    """The ``|k x n|`` below which a plane of incidence is not defined (R-06-4).

    ``sqrt(eps)`` of the working dtype (``eps = 2 u``): 1.49e-8 at float64,
    3.45e-4 at float32. Below it the two interface eigen-polarizations are
    degenerate to within the dtype's own resolution of ``R_s - R_p`` (which
    goes as ``sin^2 theta``), so any frame is right and the rotation is
    skipped rather than a vanishing vector normalised.

    Args:
        like: An array or tensor in the working dtype.

    Returns:
        The tolerance as a 0-d array or tensor in ``like``'s dtype.
    """
    return (2.0 * _tol.unit_roundoff(like)) ** 0.5


# ---------------------------------------------------------------------------
# Frame rotation
# ---------------------------------------------------------------------------


def rotation_2psi(e, a, k):
    """``(cos 2psi, sin 2psi)`` of the rotation taking reference axis ``e`` to ``a``.

    ``psi`` is measured about ``k``: ``cos psi = e . a``,
    ``sin psi = (e x a) . k``; then ``cos 2psi = c^2 - s^2`` and
    ``sin 2psi = 2 c s``. All three vectors are given as ``(x, y, z)`` tuples
    of per-ray component arrays; ``e`` and ``a`` are unit vectors
    perpendicular to the unit ``k``.

    Returns:
        ``(c2, s2)``, per-ray arrays.
    """
    c = _dot(*e, *a)
    s = _dot(*_cross(*e, *a), *k)
    return c * c - s * s, 2.0 * c * s


def rotate(q, u, c2, s2):
    """Apply ``M_rot(psi)`` to the reduced state: returns ``(q', u')``; ``v`` is unchanged."""
    return c2 * q + s2 * u, c2 * u - s2 * q


# ---------------------------------------------------------------------------
# Applying an interface-form element
# ---------------------------------------------------------------------------


def apply_interface(q, u, v, m: InterfaceMueller):
    """Apply an interface-form Mueller matrix to the reduced state.

    ``S' = M S`` with ``S = I (1, q, u, v)``: the flux factor is
    ``g = m00 + m01 q`` and the output reduced state is
    ``((m01 + m00 q), (m22 u + m23 v), (m22 v - m23 u)) / g``. Where
    ``g = 0`` (a crossed ideal polarizer, a totally reflected ray's transmitted
    branch) the output state is set to zero behind a double ``where``: the
    divisor is replaced by one before dividing, so no infinity is formed on
    either pass.

    Args:
        q, u, v: The reduced input state, per ray, in the element's frame.
        m: The element, per ray or broadcastable.

    Returns:
        ``(g, q', u', v')``.
    """
    g = m.m00 + m.m01 * q
    positive = g > 0
    safe = be.where(positive, g, be.ones_like(g))
    zero = be.zeros_like(g)
    q2 = be.where(positive, (m.m01 + m.m00 * q) / safe, zero)
    u2 = be.where(positive, (m.m22 * u + m.m23 * v) / safe, zero)
    v2 = be.where(positive, (m.m22 * v - m.m23 * u) / safe, zero)
    return g, q2, u2, v2


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------


def reflection_mueller(rs, rp, tir=None, tir_phase=None) -> InterfaceMueller:
    """Fresnel reflection (chapter 06 section 6.4) from real amplitudes.

    Below the critical angle ``r_s`` and ``r_p`` are real, so
    ``m00 = (r_s^2 + r_p^2) / 2`` -- the same expression, in the same order,
    as the scalar engine's reflectance, so an unpolarized ray's flux factor
    is the scalar one bit for bit -- ``m01 = (r_p^2 - r_s^2) / 2``,
    ``m22 = r_p r_s`` and ``m23 = 0``. Where ``tir`` is set the moduli are one
    and the relative phase is kept (R-06-5): ``m00 = 1``, ``m01 = 0``,
    ``(m22, m23) = (Re, Im)(r_p r_s*)`` from ``tir_phase``.

    Args:
        rs, rp: Real Fresnel amplitudes (the engine's own), per ray.
        tir: Optional total-internal-reflection mask.
        tir_phase: ``(re, im)`` of ``r_p r_s*`` on the ``tir`` lanes
            (:func:`tir_relative_phase`); required with ``tir``.

    Returns:
        The reflection's :class:`InterfaceMueller`.
    """
    m00 = 0.5 * (rs**2 + rp**2)
    m01 = 0.5 * (rp**2 - rs**2)
    m22 = rp * rs
    m23 = be.zeros_like(m22)
    if tir is None:
        return InterfaceMueller(m00, m01, m22, m23)
    x_re, x_im = tir_phase
    return InterfaceMueller(
        be.where(tir, be.ones_like(m00), m00),
        be.where(tir, be.zeros_like(m01), m01),
        be.where(tir, x_re, m22),
        be.where(tir, x_im, m23),
    )


def tir_relative_phase(n1, n2, cos_i, sin2_t, tir):
    """``(Re, Im)`` of ``r_p r_s*`` beyond the critical angle, in real arithmetic.

    With ``cos_t = +i kappa``, ``kappa = sqrt(sin^2 theta_t - 1)``:
    ``r_s = (A - iB) / (A + iB)``, ``A = n1 cos_i``, ``B = n2 kappa``, and
    ``r_p = (C - iD) / (C + iD)``, ``C = n2 cos_i``, ``D = n1 kappa``. Each is
    ``(X^2 - Y^2 - 2iXY) / (X^2 + Y^2)``, so

    ``Re(r_p r_s*) = [(C^2 - D^2)(A^2 - B^2) + 4ABCD] / den``,
    ``Im(r_p r_s*) = [2AB(C^2 - D^2) - 2CD(A^2 - B^2)] / den``,
    ``den = (A^2 + B^2)(C^2 + D^2)``.

    At 1.5 -> 1.0 and 45 degrees this is ``exp(-i 36.8699 deg)``, the
    analytic reference's value. Lanes outside ``tir`` take ``kappa`` from a
    masked radicand and return a value the caller discards.

    Args:
        n1, n2: Indices of the incident and the far medium, per ray.
        cos_i: ``|cos theta_i|``, per ray.
        sin2_t: ``(n1 / n2)^2 sin^2 theta_i``, per ray.
        tir: The total-internal-reflection mask.

    Returns:
        ``(re, im)``, per ray.
    """
    radicand = be.where(tir, sin2_t - 1.0, be.ones_like(sin2_t))
    kappa = be.where(tir, radicand**0.5, be.zeros_like(sin2_t))
    A = n1 * cos_i
    B = n2 * kappa
    C = n2 * cos_i
    D = n1 * kappa
    a2 = A * A - B * B
    c2 = C * C - D * D
    den = (A * A + B * B) * (C * C + D * D)
    den = den + _tol.tiny_for(den)
    re = (c2 * a2 + 4.0 * A * B * C * D) / den
    im = (2.0 * A * B * c2 - 2.0 * C * D * a2) / den
    return re, im


def transmission_mueller(Ts, Tp, m00=None, phase=None) -> InterfaceMueller:
    """Fresnel (or coated) transmission from the power transmittances.

    ``m00 = (T_s + T_p) / 2`` (or the caller's own scalar ``T``, so the
    unpolarized flux factor is the scalar engine's bit for bit),
    ``m01 = (T_p - T_s) / 2``, and
    ``(m22, m23) = sqrt(T_s T_p) (cos Delta, sin Delta)`` with
    ``Delta = arg(t_p t_s*)`` -- zero for a bare dielectric. The square root
    is taken of a masked radicand, so a lane with ``T_s T_p = 0`` (total
    internal reflection) has a zero element and a finite derivative.

    Args:
        Ts, Tp: Power transmittances, per ray.
        m00: Optional ``m00`` to use instead of ``(Ts + Tp) / 2``.
        phase: Optional ``(cos Delta, sin Delta)``; default no phase.

    Returns:
        The transmission's :class:`InterfaceMueller`.
    """
    if m00 is None:
        m00 = 0.5 * (Ts + Tp)
    m01 = 0.5 * (Tp - Ts)
    amp = _masked_sqrt(Ts * Tp)
    if phase is None:
        return InterfaceMueller(m00, m01, amp, be.zeros_like(amp))
    cos_d, sin_d = phase
    return InterfaceMueller(m00, m01, amp * cos_d, amp * sin_d)


def diattenuator_mueller(tx, ty) -> InterfaceMueller:
    """Linear diattenuator with power transmittances ``tx`` on ``p`` and ``ty`` on ``s``.

    Jones ``diag(sqrt(tx), sqrt(ty))``: ``m00 = (tx + ty) / 2``,
    ``m01 = (tx - ty) / 2``, ``m22 = sqrt(tx ty)``, ``m23 = 0``. An ideal
    linear polarizer is ``tx = 1``, ``ty = 0``; a finite extinction ratio is
    ``ty > 0``. ``tx`` and ``ty`` may be Python floats or arrays.
    """
    if isinstance(tx, float) and isinstance(ty, float):
        return InterfaceMueller(0.5 * (tx + ty), 0.5 * (tx - ty), (tx * ty) ** 0.5, 0.0)
    return InterfaceMueller(
        0.5 * (tx + ty), 0.5 * (tx - ty), _masked_sqrt(tx * ty), 0.0 * tx
    )


def polarizer_mueller(like, extinction=0.0) -> InterfaceMueller:
    """Linear polarizer with its transmission axis on ``p``, per ray.

    Args:
        like: A per-ray array giving the shape, library and dtype.
        extinction: Power transmittance of the blocked axis (0 is ideal).

    Returns:
        ``diattenuator_mueller(1, extinction)`` broadcast to ``like``.
    """
    one = be.ones_like(like)
    return diattenuator_mueller(one, one * extinction)


def retarder_mueller(cos_delta, sin_delta) -> InterfaceMueller:
    """Linear retarder with its fast axis on ``p`` and retardance ``delta``.

    ``m00 = 1``, ``m01 = 0``, ``m22 = cos delta``, ``m23 = -sin delta``: the
    matrix of the analytic reference's ``retarder_stokes`` at a fast axis of
    zero, so that a quarter wave with the fast axis at +45 degrees takes
    ``(1, 1, 0, 0)`` to ``(1, 0, 0, -1)``.

    Args:
        cos_delta, sin_delta: Per-ray cosine and sine of the retardance.
    """
    return InterfaceMueller(
        be.ones_like(cos_delta), be.zeros_like(cos_delta), cos_delta, -sin_delta
    )


def depolarize(q, u, v, kappa):
    """``diag(1, kappa, kappa, kappa)`` (chapter 06 section 6.7): ``I`` is unchanged."""
    return kappa * q, kappa * u, kappa * v


# ---------------------------------------------------------------------------
# State and frame helpers
# ---------------------------------------------------------------------------


def degree_of_polarization(q, u, v):
    """``sqrt(q^2 + u^2 + v^2)``, the degree of polarization of a reduced state."""
    return (q * q + u * u + v * v) ** 0.5


def realizability_excess(q, u, v):
    """``q^2 + u^2 + v^2 - 1``: positive only for an unphysical state (R-06-7)."""
    return q * q + u * u + v * v - 1.0


def birth_axis(kx, ky, kz):
    """The reference axis a newly born ray gets: lab ``x`` projected perpendicular to ``k``.

    ``e = normalize(a - (a . k) k)`` with ``a = x``; where ``k`` is within
    :func:`degeneracy_tolerance` of ``x`` the lab ``y`` axis is used instead.
    For a ray along ``z`` this is ``e = x`` exactly.

    Args:
        kx, ky, kz: Unit direction components, per ray.

    Returns:
        ``(ex, ey, ez)``, per ray.
    """
    one = be.ones_like(kx)
    ex_x, ex_y, ex_z = one - kx * kx, -kx * ky, -kx * kz
    ey_x, ey_y, ey_z = -ky * kx, one - ky * ky, -ky * kz
    nx2 = ex_x * ex_x + ex_y * ex_y + ex_z * ex_z
    tol = degeneracy_tolerance(kx)
    use_x = nx2 > tol * tol
    ax = be.where(use_x, ex_x, ey_x)
    ay = be.where(use_x, ex_y, ey_y)
    az = be.where(use_x, ex_z, ey_z)
    n2 = ax * ax + ay * ay + az * az
    inv = 1.0 / be.where(n2 > 0, n2, one) ** 0.5
    return ax * inv, ay * inv, az * inv


def _detached_zeros_like(x):
    """Zeros of ``x``'s library, dtype, device and shape, with no gradient flag."""
    if be.is_torch_tensor(x):
        import torch  # noqa: PLC0415

        return torch.zeros_like(x, requires_grad=False)
    return np.zeros_like(x)


def prepare_bundle(rays) -> None:
    """Give a freshly generated, device-placed bundle its polarization state, in place.

    Called once per batch by the trace loop when polarization is on, after the
    backend has placed the bundle on its library and device. A bundle whose
    source set no state is unpolarized, ``(q, u, v) = (0, 0, 0)``, with the
    reference axis :func:`birth_axis` of its direction. A state a source did
    set is moved onto the ray state's library and dtype. ``flux`` is not
    touched: it is the Stokes ``I`` as it stands.

    Args:
        rays: An :class:`~optiland.nonsequential.ray_bundle.NSQRayBundle`.
    """
    if rays.pol_q is None:
        rays.pol_q = _detached_zeros_like(rays.flux)
        rays.pol_u = _detached_zeros_like(rays.flux)
        rays.pol_v = _detached_zeros_like(rays.flux)
        rays.pol_ex, rays.pol_ey, rays.pol_ez = birth_axis(rays.L, rays.M, rays.N)
        return
    for name in POL_FIELDS:
        value = getattr(rays, name)
        if be.is_torch_tensor(rays.flux) and not be.is_torch_tensor(value):
            import torch  # noqa: PLC0415

            value = torch.as_tensor(
                np.asarray(value), dtype=rays.flux.dtype, device=rays.flux.device
            )
        elif not be.is_torch_tensor(rays.flux):
            value = np.asarray(value, dtype=np.asarray(rays.flux).dtype)
        setattr(rays, name, value)


def transport_axis(e, k):
    """Re-project a reference axis perpendicular to a new direction and normalise it.

    For an element that changes ``k`` without a plane of incidence (a
    paraxial lens, a transmissive detector crossing): ``e' = normalize(e -
    (e . k) k)``. The Stokes state is unchanged.

    Args:
        e: ``(ex, ey, ez)`` per-ray components.
        k: ``(kx, ky, kz)``, the new unit direction.

    Returns:
        ``(ex', ey', ez')``.
    """
    ek = _dot(*e, *k)
    px, py, pz = e[0] - ek * k[0], e[1] - ek * k[1], e[2] - ek * k[2]
    n2 = px * px + py * py + pz * pz
    inv = 1.0 / be.where(n2 > 0, n2, be.ones_like(n2)) ** 0.5
    return px * inv, py * inv, pz * inv


# ---------------------------------------------------------------------------
# The thin-film adapter: s and p coefficients in this module's convention
# ---------------------------------------------------------------------------


class SPCoefficients(NamedTuple):
    """What a coated (or bare, or metallic) interface gives a Mueller matrix.

    Attributes:
        Rs, Rp, Ts, Tp: Power reflectances and transmittances, per ray.
        xr_re, xr_im: ``Re`` and ``Im`` of ``r_p r_s*`` in this module's
            convention (the reflection's ``m22`` and ``m23``).
        xt_cos, xt_sin: ``cos`` and ``sin`` of ``Delta = arg(t_p t_s*)``
            (``(1, 0)`` where either transmittance is zero).
        phase_valid: Per-ray mask, False where the thin-film module's phases
            are not in either time convention (a coated interface beyond the
            critical angle; see :func:`thin_film_sp`). The power terms are
            right everywhere.
    """

    Rs: object
    Rp: object
    Ts: object
    Tp: object
    xr_re: object
    xr_im: object
    xt_cos: object
    xt_sin: object
    phase_valid: object

    def reflection(self) -> InterfaceMueller:
        """The reflection element: ``m00 = (R_s + R_p) / 2`` and the phase terms."""
        return InterfaceMueller(
            0.5 * (self.Rs + self.Rp), 0.5 * (self.Rp - self.Rs), self.xr_re, self.xr_im
        )

    def transmission(self) -> InterfaceMueller:
        """The transmission element: ``m00 = (T_s + T_p) / 2`` and ``sqrt(T_s T_p) e^{i Delta}``."""
        return transmission_mueller(self.Ts, self.Tp, phase=(self.xt_cos, self.xt_sin))


def _material_nk(material, wavelength_um):
    """Real ``n`` and ``k`` of a thin-film stack's material at the rays' wavelengths."""
    if be.get_backend() == "torch" and hasattr(wavelength_um, "detach"):
        n = material._calculate_n(wavelength_um)
        k = material._calculate_k(wavelength_um)
    else:
        n = material.n(wavelength_um)
        k = material.k(wavelength_um)
    return be.atleast_1d(n), be.atleast_1d(k)


def thin_film_sp(stack, wavelength_um, cos_theta_i) -> SPCoefficients:
    """The fork's thin-film module's s and p results, in this module's convention.

    ``ThinFilmStack.compute_rtRTA_elementwise(..., "s" | "p")`` gives ``R`` and
    ``T`` right to about 1e-15 at float64 on bare, totally reflecting,
    metallic and coated interfaces. Its complex reflection amplitudes need two
    corrections before they are Mueller phase terms, each pinned by a test
    (``tests/nonsequential/test_nsq_polarization_thin_film.py``):

    1. **The sign of ``r_p``.** The module's ``p`` admittance is
       ``eta_p = n / cos theta``, which gives ``r_p`` with the opposite sign to
       the Fresnel convention of the catalogue's analytic reference; ``r_p`` is
       negated.
    2. **The time convention.** The module's layer matrices and its complex
       index (entered through ``n - i k``) are in the ``exp(+i omega t)``
       convention, so for a stack with at least one layer, and for an
       absorbing substrate or incident medium, its reflection phase is the
       complex conjugate of this module's; it is conjugated. For a bare,
       lossless interface below the critical angle both amplitudes are real
       and nothing changes; beyond it the module's evanescent root is this
       module's (``cos_t = +i kappa``) and nothing is conjugated.

    Its transmission amplitudes carry a conjugation of their own, and their
    relative phase ``arg(t_p t_s*)`` agrees with an independent
    ``exp(-i omega t)`` characteristic-matrix calculation as it is.

    One case is outside both conventions: a stack **with layers** met beyond
    the critical angle of its substrate (a coated face used in total internal
    reflection). There the module mixes the ``exp(+i omega t)`` layers with
    the ``exp(-i omega t)`` evanescent root, and its relative phase matches
    neither convention (measured: -34.00 degrees against -29.22 degrees at
    60 degrees for one quarter-wave layer of 1.38 between 1.5 and 1.0). Those
    lanes are flagged in ``phase_valid``; the power terms stay right.

    Args:
        stack: A configured ``optiland.thin_film.ThinFilmStack``.
        wavelength_um: Per-ray wavelength [um].
        cos_theta_i: Per-ray ``|cos theta_i|`` in the incident medium.

    Returns:
        :class:`SPCoefficients`, per ray, real arrays in the working dtype.
    """
    cos_theta_i = be.clip(cos_theta_i, -1.0, 1.0)
    aoi = be.arccos(cos_theta_i)
    s = stack.compute_rtRTA_elementwise(wavelength_um, aoi, polarization="s")
    p = stack.compute_rtRTA_elementwise(wavelength_um, aoi, polarization="p")
    rs, rp = s["r"], -p["r"]
    x_r = rp * be.conj(rs)

    n0, k0 = _material_nk(stack.incident_material, wavelength_um)
    ns, ks = _material_nk(stack.substrate_material, wavelength_um)
    absorbing = (k0 > 0) | (ks > 0)
    everywhere = be.ones_like(ks) > 0
    if stack.layers:
        conjugate = everywhere
        sin2_t = (n0 / ns) ** 2 * (1.0 - cos_theta_i**2)
        phase_valid = ~((sin2_t >= 1.0) & ~absorbing)
    else:
        conjugate = absorbing
        phase_valid = everywhere
    re = be.real(x_r)
    im = be.where(conjugate, -be.imag(x_r), be.imag(x_r))

    x_t = p["t"] * be.conj(s["t"])
    mag = be.abs(x_t)
    nonzero = mag > 0
    safe = be.where(nonzero, mag, be.ones_like(mag))
    xt_cos = be.where(nonzero, be.real(x_t) / safe, be.ones_like(mag))
    xt_sin = be.where(nonzero, be.imag(x_t) / safe, be.zeros_like(mag))
    return SPCoefficients(
        s["R"], p["R"], s["T"], p["T"], re, im, xt_cos, xt_sin, phase_valid
    )


# ---------------------------------------------------------------------------
# Full 4x4 matrices, for tests and reports (NumPy, float64)
# ---------------------------------------------------------------------------


def mueller_matrix(m: InterfaceMueller) -> np.ndarray:
    """The 4x4 matrix of scalar interface-form elements, as a NumPy array."""
    m00, m01, m22, m23 = (float(np.asarray(x)) for x in m)
    return np.array(
        [
            [m00, m01, 0.0, 0.0],
            [m01, m00, 0.0, 0.0],
            [0.0, 0.0, m22, m23],
            [0.0, 0.0, -m23, m22],
        ]
    )


def rotation_matrix(psi_rad: float) -> np.ndarray:
    """``M_rot(psi)`` of chapter 06 section 6.5, as a NumPy array."""
    c, s = np.cos(2.0 * psi_rad), np.sin(2.0 * psi_rad)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, s, 0.0], [0.0, -s, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
