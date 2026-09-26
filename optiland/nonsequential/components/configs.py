"""Config dataclasses for NSQ compound components.

SurfaceConfig, InteractionType, LensConfig, DoubletConfig, MirrorConfig,
PrismConfig, ParaxialLensConfig.

Kramer Harrison, 2026
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.materials.nsq_material import NSQMaterial


class InteractionType(enum.Enum):
    """Optical interaction type for a single surface.

    Attributes:
        REFRACTIVE: Surface refracts (and optionally reflects via Fresnel).
        REFLECTIVE: Surface reflects only; no transmission.
        ABSORBING: Surface absorbs all incident rays.
    """

    REFRACTIVE = "refractive"
    REFLECTIVE = "reflective"
    ABSORBING = "absorbing"


@dataclass
class SurfaceConfig:
    """Optional per-surface overrides within a compound component.

    All fields default to ``None``, meaning "use the compound-level default."
    When a field is set, it overrides the compound's default for that surface.

    Attributes:
        bsdf: Custom BSDF for this surface.  Routes rays through the scatter
            model instead of the surface's specular/refractive behaviour.
        scatter_fraction: Probability in [0, 1] that a ray striking this
            surface is routed through ``bsdf`` rather than following the
            specular or refractive path.  The default of 1.0 sends every ray
            to the BSDF, turning the surface into a pure diffuser.  Set it
            below 1 to model a partially scattering surface, e.g. 0.1 for a
            surface that scatters a tenth of the light and transmits the
            rest.  Ignored when ``bsdf`` is None.
        coating: An ``optiland.coatings.BaseCoating`` for a refractive
            surface (e.g. an AR coating). When set, its reflectance/
            transmittance replace the bare Fresnel calculation, so NSQ and
            the sequential engine agree on R. Must be an unpolarized coating
            (``SimpleCoating``, or a
            ``coating_support.UnpolarizedThinFilmCoating`` wrapping a
            ``ThinFilmStack`` for an angle- and wavelength-dependent R/T); a
            ``BaseCoatingPolarized`` instance (Jones-matrix based --
            ``FresnelCoating``, ``ThinFilmCoating``, ...) raises
            ``NotImplementedError`` rather than being silently degraded to
            its scalar average. Ignored on absorbing surfaces.
        aperture_radius: Semi-diameter override [mm].  Overrides the aperture
            computed from the compound config.
        interaction: Force a specific interaction type on this surface.
        reflectance: Required when ``interaction`` selects
            ``InteractionType.REFLECTIVE``: a constant in [0, 1], a
            wavelength-dependent ``callable(wavelength_um) -> reflectance``,
            or an unpolarized ``BaseCoating``. See
            ``ReflectiveComponent`` -- constructing a reflective surface
            without one raises rather than defaulting to a perfect mirror.
    """

    bsdf: BaseBSDF | None = None
    scatter_fraction: float = 1.0
    coating: object | None = None  # optiland.coatings.BaseCoating
    aperture_radius: float | None = None
    interaction: InteractionType | None = None
    reflectance: object | None = None  # float | Callable | BaseCoating


@dataclass
class LensConfig:
    """Configuration for a single-element refractive lens.

    The lens assembles up to four physical surfaces:

    1. **Front face** -- refractive, conic.
    2. **Back face** -- refractive, conic.
    3. **Edge** -- cylindrical frustum, absorbing by default.
    4. **Rim** -- annular plane, absorbing; only created when
       ``front_aperture_radius != back_aperture_radius``.

    Attributes:
        r1: Front vertex radius of curvature [mm].  Positive = centre of
            curvature on +z side.
        r2: Back vertex radius of curvature [mm].
        thickness: Centre thickness of the lens [mm].
        material: Glass name (e.g. ``'N-BK7'``) or a ready-made
            :class:`~optiland.nonsequential.materials.NSQMaterial` instance.
        front_aperture_radius: Semi-diameter of the front face [mm].
        back_aperture_radius: Semi-diameter of the back face [mm].
            Defaults to ``front_aperture_radius`` when ``None``.
        conic1: Conic constant of the front face (0 = sphere).
        conic2: Conic constant of the back face (0 = sphere).
        front: Per-surface overrides for the front face.
        back: Per-surface overrides for the back face.
        edge: Per-surface overrides for the edge (barrel) surface.
        rim: Per-surface overrides for the rim annulus (only used when
            apertures differ).
        coefficients1: Polynomial coefficients of the front face; empty for
            a conic face. Non-empty makes the face an asphere (the geometry
            kinds ``even_asphere``/``odd_asphere``, KronosNSRT issue 30):
            entry ``i`` multiplies ``r^(2 (i + 1))``, or ``r^(i + 1)`` when
            ``odd1`` is set.
        coefficients2: The same for the back face.
        odd1: The front face's polynomial is odd.
        odd2: The back face's polynomial is odd.
    """

    r1: float
    r2: float
    thickness: float
    material: str | NSQMaterial
    front_aperture_radius: float
    back_aperture_radius: float | None = None
    conic1: float = 0.0
    conic2: float = 0.0
    front: SurfaceConfig | None = None
    back: SurfaceConfig | None = None
    edge: SurfaceConfig | None = None
    rim: SurfaceConfig | None = None
    coefficients1: tuple = ()
    coefficients2: tuple = ()
    odd1: bool = False
    odd2: bool = False


@dataclass
class DoubletConfig:
    """Configuration for a cemented achromatic doublet.

    Surfaces in order (front -> back): front face, cemented interface, back
    face, edge.

    Attributes:
        r1: Front radius of curvature [mm].
        r2: Cemented interface radius of curvature [mm].
        r3: Back radius of curvature [mm].
        thickness1: Thickness of the crown element [mm].
        thickness2: Thickness of the flint element [mm].
        material1: Crown element glass name or NSQMaterial.
        material2: Flint element glass name or NSQMaterial.
        aperture_radius: Common semi-diameter for all surfaces [mm].
        conic1: Conic constant of the front face.
        conic2: Conic constant of the cemented interface.
        conic3: Conic constant of the back face.
        front: Per-surface overrides for the front face.
        cemented: Per-surface overrides for the cemented interface.
        back: Per-surface overrides for the back face.
        edge: Per-surface overrides for the edge surface.
    """

    r1: float
    r2: float
    r3: float
    thickness1: float
    thickness2: float
    material1: str | NSQMaterial
    material2: str | NSQMaterial
    aperture_radius: float
    conic1: float = 0.0
    conic2: float = 0.0
    conic3: float = 0.0
    front: SurfaceConfig | None = None
    cemented: SurfaceConfig | None = None
    back: SurfaceConfig | None = None
    edge: SurfaceConfig | None = None


@dataclass
class MirrorConfig:
    """Configuration for a single reflective mirror surface.

    Attributes:
        radius: Vertex radius of curvature [mm].  Negative = concave when
            oriented with the normal pointing toward +z.
        reflectance: Mirror reflectance: a constant in [0, 1], a
            wavelength-dependent ``callable(wavelength_um) -> reflectance``,
            or an unpolarized ``optiland.coatings.BaseCoating`` (e.g.
            ``SimpleCoating``). Required -- there is no implicit
            perfect-mirror default: a mirror built without specifying
            how much light it reflects is a modelling bug, not a 100%
            reflector. Overridden per-surface by ``surface.reflectance``.
        conic: Conic constant (0 = sphere, -1 = paraboloid, etc.).
        aperture_radius: Semi-diameter [mm].
        surface: Per-surface overrides (e.g. to attach a scatter BSDF).
        coefficients: Polynomial coefficients; empty for a conic mirror.
            Non-empty makes the surface an asphere (KronosNSRT issue 30):
            entry ``i`` multiplies ``r^(2 (i + 1))``, or ``r^(i + 1)`` when
            ``odd`` is set.
        odd: The polynomial is odd.
    """

    radius: float
    reflectance: object  # float | Callable | BaseCoating
    conic: float = 0.0
    aperture_radius: float = 25.0
    surface: SurfaceConfig | None = None
    coefficients: tuple = ()
    odd: bool = False


@dataclass
class PrismConfig:
    """Configuration for a prism or wedge: two plane faces at an apex angle.

    The frame is the placement's own. The apex edge is the local y axis;
    the principal section is the local x-z plane; the apex angle ``A`` opens
    toward local -x and is bisected by it. The **front** (entrance) face
    runs from the apex edge to ``(x, z) = (-L cos(A/2), -L sin(A/2))``, the
    **back** (exit) face to ``(-L cos(A/2), +L sin(A/2))``, and the **base**
    joins the two far edges in the plane ``x = -L cos(A/2)``. A ray
    travelling along +z below the apex enters the front face, leaves the
    back face and is deviated toward -x, the base; at minimum deviation the
    ray inside the glass travels along local z, parallel to the base.

    Surfaces, in order:

    1. **Front face** -- refractive, a ``length`` x ``face_length`` plane.
    2. **Back face** -- refractive, the same.
    3. **Base** -- absorbing by default (``base`` overrides it, e.g. to
       ``InteractionType.REFRACTIVE`` for a polished base); omitted when
       ``open_base`` is set.

    The two triangular end faces are open: a ray in the principal section
    never reaches them, and one that does leaves the glass unrefracted.

    Attributes:
        apex_angle_deg: Apex angle ``A`` [deg], in (0, 180).
        face_length: ``L``, from the apex edge to the base along each face
            [mm].
        length: Extent along the apex edge [mm].
        material: Glass name or a ready-made ``NSQMaterial``.
        open_base: Leave the base out (a wedge open at the back).
        front: Per-surface overrides for the entrance face.
        back: Per-surface overrides for the exit face.
        base: Per-surface overrides for the base.
    """

    apex_angle_deg: float
    face_length: float
    length: float
    material: str | NSQMaterial
    open_base: bool = False
    front: SurfaceConfig | None = None
    back: SurfaceConfig | None = None
    base: SurfaceConfig | None = None


@dataclass
class ParaxialLensConfig:
    """Configuration for an ideal thin (paraxial) lens.

    A plane of no thickness at the placement's local z = 0 that deflects
    every ray crossing it inside ``aperture_radius`` by the paraxial
    thin-lens law in its slopes, ``u' = u - h / f`` in x and in y, and
    passes it on with its weight unchanged: no Fresnel reflection, no
    absorption, no aberration. It images every object point to one image
    point exactly, with magnification ``m = s' / s`` and Lagrange invariant
    conserved, which is what the paraxial bookkeeping of
    ``docs/theory/11_validation_catalogue.md`` 11.4.15 is written for.

    Attributes:
        focal_length: ``f`` [mm]; positive converges, for a ray crossing
            in either direction.
        aperture_radius: Clear semi-diameter [mm].
        stop_radius: When set, an absorbing annulus from ``aperture_radius``
            out to this radius in the lens plane, so the lens is also the
            system's stop. ``None`` (default) leaves the plane outside the
            clear aperture open.
    """

    focal_length: float
    aperture_radius: float
    stop_radius: float | None = None
