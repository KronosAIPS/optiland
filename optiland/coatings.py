"""Coatings Module

The coatings module contains classes for modeling optical coatings.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.jones import (
    BaseJones,
    JonesFresnel,
    JonesLinearPolarizer,
    JonesLinearRetarder,
)
from optiland.materials import BaseMaterial
from optiland.thin_film import ThinFilmStack

if TYPE_CHECKING:
    from optiland.rays import RealRays


class BaseCoating(ABC):
    """Base class for coatings.

    This class defines the basic structure and behavior of a coating.

    Methods:
        interact: Performs an interaction with the coating.
        reflect: Abstract method to handle reflection interaction with the coating.
        transmit: Abstract method to handle transmission interaction with the coating.

    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseCoating._registry[cls.__name__] = cls

    def interact(
        self,
        rays: RealRays,
        reflect: bool = False,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Performs an interaction with the coating.

        Args:
            rays (RealRays): The rays incident on the coating.
            reflect (bool, optional): Flag indicating whether to perform
                reflection (True) or transmission (False). Defaults to False.
            nx (be.ndarray, optional): The x-component of the surface normal vectors.
            ny (be.ndarray, optional): The y-component of the surface normal vectors.
            nz (be.ndarray, optional): The z-component of the surface normal vectors.

        Returns:
            rays (RealRays): The rays after the interaction.

        """
        if reflect:
            return self.reflect(rays, nx, ny, nz)
        return self.transmit(rays, nx, ny, nz)

    def _compute_aoi(
        self, rays: RealRays, nx: be.ndarray, ny: be.ndarray, nz: be.ndarray
    ) -> be.ndarray:
        """Computes the angle of incidence for the given rays and surface normals.

        Args:
            rays (RealRays): The incident rays.
            nx (be.ndarray): The x-component of the surface normal vectors at each ray's
                intersection point.
            ny (be.ndarray): The y-component of the surface normal vectors at each ray's
                intersection point.
            nz (be.ndarray): The z-component of the surface normal vectors at each ray's
                intersection point.

        Returns:
            be.ndarray: The angle of incidence for each ray.

        """
        dot = be.abs(nx * rays.L0 + ny * rays.M0 + nz * rays.N0)
        dot = be.clip(dot, -1, 1)  # required due to numerical precision
        return be.arccos(dot)

    @abstractmethod
    def reflect(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Abstract method to handle reflection interaction.

        Args:
            rays (RealRays): The rays incident on the coating.
            nx (be.ndarray, optional): The x-component of the surface normal vectors.
            ny (be.ndarray, optional): The y-component of the surface normal vectors.
            nz (be.ndarray, optional): The z-component of the surface normal vectors.

        Returns:
            RealRays: The rays after reflection.

        """
        # pragma: no cover

    @abstractmethod
    def transmit(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Abstract method to handle transmission interaction.

        Args:
            rays (RealRays): The rays incident on the coating.
            nx (be.ndarray, optional): The x-component of the surface normal vectors.
            ny (be.ndarray, optional): The y-component of the surface normal vectors.
            nz (be.ndarray, optional): The z-component of the surface normal vectors.

        Returns:
            RealRays: The rays after transmission.

        """
        # pragma: no cover

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover
        """Converts the coating to a dictionary.

        Returns:
            dict: The dictionary representation of the coating.

        """
        return {
            "type": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseCoating:
        """Creates a coating from a dictionary.

        Args:
            data (dict): The dictionary representation of the coating.

        Returns:
            BaseCoating: The coating created from the dictionary.

        """
        coating_type = data["type"]
        return cls._registry[coating_type].from_dict(data)


class SimpleCoating(BaseCoating):
    """A simple coating class that represents a coating with given transmittance
    and reflectance.

    Args:
        transmittance (float): The transmittance of the coating.
        reflectance (float, optional): The reflectance of the coating.
            Defaults to 0.

    Attributes:
        transmittance (float): The transmittance of the coating.
        reflectance (float): The reflectance of the coating.
        absorptance (float): The absorptance of the coating, calculated
            as 1 - reflectance - transmittance.

    Methods:
        reflect(rays: RealRays, nx: be.ndarray = None, ny: be.ndarray = None,
            nz: be.ndarray = None) -> RealRays: Reflects the rays based on the
            reflectance of the coating.
        transmit(rays: RealRays, nx: be.ndarray = None, ny: be.ndarray = None,
            nz: be.ndarray = None) -> RealRays: Transmits the rays based on the
            transmittance of the coating.

    """

    def __init__(self, transmittance: float, reflectance: float = 0):
        self.transmittance = transmittance
        self.reflectance = reflectance
        self.absorptance = 1 - reflectance - transmittance

    def reflect(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Reflects the rays based on the reflectance of the coating.

        Args:
            rays (RealRays): The rays incident on the coating.
            nx (be.ndarray, optional): The x-component of the surface normal vectors.
            ny (be.ndarray, optional): The y-component of the surface normal vectors.
            nz (be.ndarray, optional): The z-component of the surface normal vectors.

        Returns:
            RealRays: The rays after reflection.

        """
        rays.i = rays.i * self.reflectance
        return rays

    def transmit(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Transmits the rays through the coating by multiplying their intensity
        with the transmittance.

        Args:
            rays (RealRays): The rays incident on the coating.
            nx (be.ndarray, optional): The x-component of the surface normal vectors.
            ny (be.ndarray, optional): The y-component of the surface normal vectors.
            nz (be.ndarray, optional): The z-component of the surface normal vectors.

        Returns:
            RealRays: The rays after transmission.

        """
        rays.i = rays.i * self.transmittance
        return rays

    def to_dict(self) -> dict[str, Any]:
        """Converts the coating to a dictionary.

        Returns:
            dict: The dictionary representation of the coating.

        """
        return {
            "type": self.__class__.__name__,
            "transmittance": self.transmittance,
            "reflectance": self.reflectance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimpleCoating:
        """Creates a coating from a dictionary.

        Args:
            data (dict): The dictionary representation of the coating.

        Returns:
            BaseCoating: The coating created from the dictionary.

        """
        return cls(data["transmittance"], data["reflectance"])


class BaseCoatingPolarized(BaseCoating, ABC):
    """A base class for polarized coatings.

    This class inherits from the `BaseCoating` class and the `ABC`
    (Abstract Base Class) module. Any subclass must implement the `jones`
    property to provide the Jones matrix model for the coating.

    Methods:
        reflect(rays, nx, ny, nz): Reflects the rays off the coating.
        transmit(rays, nx, ny, nz): Transmits the rays through the coating.

    """

    @property
    @abstractmethod
    def jones(self) -> BaseJones:
        """The Jones matrix model associated with the coating."""
        pass  # pragma: no cover

    def reflect(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Reflects the rays off the coating.

        Args:
            rays (RealRays): The rays to be reflected.
            nx (be.ndarray, optional): The x-component of the surface normal vector.
            ny (be.ndarray, optional): The y-component of the surface normal vector.
            nz (be.ndarray, optional): The z-component of the surface normal vector.

        Returns:
            RealRays: The updated rays after reflection.

        """
        aoi = self._compute_aoi(rays, nx, ny, nz)
        jones = self.jones.calculate_matrix(rays, reflect=True, aoi=aoi)
        rays.update(jones)
        return rays

    def transmit(
        self,
        rays: RealRays,
        nx: be.ndarray = None,
        ny: be.ndarray = None,
        nz: be.ndarray = None,
    ) -> RealRays:
        """Transmits the rays through the coating.

        Args:
            rays (RealRays): The rays to be transmitted.
            nx (be.ndarray, optional): The x-component of the surface normal vector.
            ny (be.ndarray, optional): The y-component of the surface normal vector.
            nz (be.ndarray, optional): The z-component of the surface normal vector.

        Returns:
            RealRays: The updated rays after transmission through a surface.

        """
        aoi = self._compute_aoi(rays, nx, ny, nz)
        jones = self.jones.calculate_matrix(rays, reflect=False, aoi=aoi)
        rays.update(jones)
        return rays

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover
        """Converts the coating to a dictionary.

        Returns:
            dict: The dictionary representation of the coating.

        """
        return {
            "type": self.__class__.__name__,
            "material_pre": self.material_pre.to_dict(),
            "material_post": self.material_post.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> BaseCoatingPolarized:  # pragma: no cover
        """Creates a coating from a dictionary.

        Args:
            data (dict): The dictionary representation of the coating.

        Returns:
            BaseCoating: The coating created from the dictionary.

        """
        return cls(data["material_pre"], data["material_post"])


class FresnelCoating(BaseCoatingPolarized):
    """Represents a Fresnel coating for polarized light.

    This class inherits from the BaseCoatingPolarized class and provides
    interaction functionality for polarized light with uncoated surfaces.
    In general, this updates ray intensities based on the Fresnel equations
    on a surface.

    Attributes:
        material_pre (str): The material before the coating.
        material_post (str): The material after the coating.
        jones (JonesFresnel): The JonesFresnel object, which calculates the
            Jones matrices for given ray properties.

    """

    def __init__(self, material_pre: BaseMaterial, material_post: BaseMaterial):
        self.material_pre = material_pre
        self.material_post = material_post

        self._jones = JonesFresnel(material_pre, material_post)

    @property
    def jones(self) -> JonesFresnel:
        return self._jones

    def to_dict(self) -> dict[str, Any]:
        """Converts the coating to a dictionary.

        Returns:
            dict: The dictionary representation of the coating.

        """
        return {
            "type": self.__class__.__name__,
            "material_pre": self.material_pre.to_dict(),
            "material_post": self.material_post.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FresnelCoating:
        """Creates a coating from a dictionary.

        Args:
            data (dict): The dictionary representation of the coating.

        Returns:
            BaseCoating: The coating created from the dictionary.

        """
        return cls(
            BaseMaterial.from_dict(data["material_pre"]),
            BaseMaterial.from_dict(data["material_post"]),
        )


class PolarizerCoating(BaseCoatingPolarized):
    """Represents a linear polarizer coating.

    Args:
        axis (tuple | list | be.ndarray): A 3D vector representing the transmission
            axis in global coordinates. Defaults to [1.0, 0.0, 0.0] (horizontal).
    """

    def __init__(
        self,
        axis: tuple[float, float, float] | list[float] | be.ndarray = (1.0, 0.0, 0.0),
    ):
        self.axis = axis
        self._jones = JonesLinearPolarizer(axis)

    @property
    def jones(self) -> JonesLinearPolarizer:
        return self._jones

    def to_dict(self) -> dict[str, Any]:
        """Converts the coating to a dictionary."""
        return {
            "type": self.__class__.__name__,
            "axis": list(self.axis) if not isinstance(self.axis, list) else self.axis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolarizerCoating:
        """Creates a coating from a dictionary."""
        return cls(axis=data.get("axis", (1.0, 0.0, 0.0)))


class RetarderCoating(BaseCoatingPolarized):
    """Represents a linear retarder coating.

    Args:
        retardance (float): The retardance of the coating in radians.
        axis (tuple | list | be.ndarray): A 3D vector representing the fast axis
            in global coordinates. Defaults to [1.0, 0.0, 0.0] (horizontal).
    """

    def __init__(
        self,
        retardance: float,
        axis: tuple[float, float, float] | list[float] | be.ndarray = (1.0, 0.0, 0.0),
    ):
        self.retardance = retardance
        self.axis = axis
        self._jones = JonesLinearRetarder(retardance, axis)

    @property
    def jones(self) -> JonesLinearRetarder:
        return self._jones

    def to_dict(self) -> dict[str, Any]:
        """Converts the coating to a dictionary."""
        return {
            "type": self.__class__.__name__,
            "retardance": self.retardance,
            "axis": list(self.axis) if not isinstance(self.axis, list) else self.axis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetarderCoating:
        """Creates a coating from a dictionary."""
        return cls(
            retardance=data["retardance"], axis=data.get("axis", (1.0, 0.0, 0.0))
        )


class JonesThinFilm(BaseJones):
    """Jones matrix generator for a thin-film stack.

    Builds diagonal Jones matrices in the s/p basis using thin-film r/t
    amplitude coefficients, in :class:`~optiland.jones.JonesFresnel`'s form
    (the research repository's issue 74): an empty stack gives
    ``JonesFresnel``'s matrices. The thin-film module's coefficients are the
    admittance form's: its ``r_p`` is the negative of the Fresnel ``r_p``
    (``eta_p = n / cos theta``), and its ``t`` is the ratio of the tangential
    fields. The reflected p entry is therefore the module's ``r_p`` itself
    (``JonesFresnel`` puts ``-r_p`` there), and the transmitted p entry is the
    module's ``t_p`` times ``cos theta_0 / cos theta_sub``, the ratio of the p
    field to its tangential component on the two sides.

    Args:
        stack: ThinFilmStack configured with incident/substrate and layers.
        wavelength_nm: Optional wavelength override (nm); if None uses rays.w (µm)
        converted.
        aoi_override_rad: Optional AOI override (radians); if None uses computed AOI.
    """

    def __init__(self, stack: ThinFilmStack):
        self.stack = stack

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ) -> be.ndarray:
        # wavelengths: rays.w is in microns in Optiland
        wl_um = be.atleast_1d(rays.w)
        th = be.atleast_1d(aoi if aoi is not None else be.zeros_like(rays.w))

        # Compute s/p amplitudes per-ray; expect broadcasting over (N,)
        r_s, t_s, _, _ = self._coeffs_amp(wl_um, th, pol="s", reflect=reflect)
        r_p, t_p, _, _ = self._coeffs_amp(wl_um, th, pol="p", reflect=reflect)

        # Filled in place into a complex array, as JonesFresnel does: the torch
        # backend's ``stack`` casts to its real working dtype, which dropped
        # the imaginary part of every coefficient (a total internal
        # reflection's phase, a coated face's) on that backend.
        jones = be.to_complex(be.zeros((be.size(r_s), 3, 3)))
        if reflect:
            # the module's r_p is -r_p(Fresnel); JonesFresnel's entry is -r_p(Fresnel)
            jones[:, 0, 0] = r_s
            jones[:, 1, 1] = r_p
            jones[:, 2, 2] = -1
        else:
            jones[:, 0, 0] = t_s
            jones[:, 1, 1] = t_p * self._p_field_ratio(wl_um, th)
            jones[:, 2, 2] = 1
        return jones

    def _p_field_ratio(self, wl_um: be.ndarray, th_rad: be.ndarray) -> be.ndarray:
        """``cos theta_0 / cos theta_sub``: p field over its tangential part, per ray."""
        from optiland.thin_film.core import _complex_index, _snell_cos  # noqa: PLC0415

        n0 = _complex_index(self.stack.incident_material, wl_um)
        ns = _complex_index(self.stack.substrate_material, wl_um)
        return _snell_cos(n0, th_rad, n0) / _snell_cos(n0, th_rad, ns)

    def _coeffs_amp(
        self, wl_um: be.ndarray, th_rad: be.ndarray, pol: str, reflect: bool
    ) -> tuple[be.ndarray, be.ndarray, be.ndarray, be.ndarray]:
        # Use internal helpers returning amplitudes from the stack TMM
        # We compute on per-ray vectors so shapes are (N,)
        out = self.stack.compute_rtRTA_elementwise(wl_um, th_rad, pol)
        r, t = out["r"], out["t"]
        R, T = out["R"], out["T"]
        return r, t, R, T


class ThinFilmCoating(BaseCoatingPolarized):
    """Polarized coating that applies a thin-film stack to rays.

    This class mirrors FresnelCoating but uses a ThinFilmStack to compute the
    s/p amplitude coefficients and builds a Jones matrix per ray via JonesThinFilm.

    Args:
        material_pre: Material before the stack (incident medium of the stack).
        material_post: Material after the stack (substrate of the stack).
        layers: Optional list of (material, thickness_nm, name) to build the stack.
    """

    def __init__(
        self,
        material_pre: BaseMaterial,
        material_post: BaseMaterial,
        layers: list[tuple[BaseMaterial, float, str | None]] | None = None,
    ):
        self.material_pre = material_pre
        self.material_post = material_post
        self.stack = ThinFilmStack(material_pre, material_post)
        if layers:
            for mat, thickness_nm, name in layers:
                self.stack.add_layer_nm(mat, thickness_nm, name)
        self._jones = JonesThinFilm(self.stack)

    @property
    def jones(self) -> JonesThinFilm:
        """The Jones matrix model associated with the thin-film coating."""
        return self._jones

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover
        return {
            "type": self.__class__.__name__,
            "material_pre": self.material_pre.to_dict(),
            "material_post": self.material_post.to_dict(),
            "layers": [
                {
                    "material": layer.material.to_dict(),
                    "thickness_nm": layer.thickness_um * 1000.0,
                    "name": layer.name,
                }
                for layer in self.stack.layers
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThinFilmCoating:  # pragma: no cover
        mats = []
        for d in data.get("layers", []):
            mats.append(
                (
                    BaseMaterial.from_dict(d["material"]),
                    d["thickness_nm"],
                    d.get("name"),
                )
            )
        return cls(
            BaseMaterial.from_dict(data["material_pre"]),
            BaseMaterial.from_dict(data["material_post"]),
            mats,
        )


# ---------------------------------------------------------------------------
# A coating given as a table against wavelength and angle of incidence
# ---------------------------------------------------------------------------

#: The power grids of a coating table, named as the shared library's
#: ``kmat.surface_optics.CoatingTable`` names them: reflectance and
#: transmittance (fractions 0..1, not amplitudes) for s and p.
TABLE_POWER_FIELDS = ("r_s", "r_p", "t_s", "t_p")

#: The engine's two optional phase grids (degrees): ``phase_r_deg`` is
#: ``arg(r_p r_s*)`` and ``phase_t_deg`` is ``arg(t_p t_s*)``, in the
#: ``exp(-i omega t)`` convention with the Fresnel sign of ``r_p`` (the research
#: repository's chapter 04 section 4.2). Not in the library's schema at its
#: tag v0.5.5.
TABLE_PHASE_FIELDS = ("phase_r_deg", "phase_t_deg")

#: Tolerance on a power value outside 0..1 and on R + T > 1, as the library's
#: validator uses.
_TABLE_ENERGY_TOL = 1e-6


def _host_grid(name, values, rows, cols):
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(values, dtype=float)
    if arr.shape != (rows, cols):
        raise ValueError(
            f"{name} must be shaped (len(wavelength_nm), len(angle_deg)) = "
            f"({rows}, {cols}); got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} holds a value that is not finite")
    return arr


def _ideal_or(material):
    from optiland.materials import IdealMaterial  # noqa: PLC0415

    if material is None or isinstance(material, int | float):
        return IdealMaterial(1.0 if material is None else float(material))
    return material


class TabulatedCoating(BaseCoating):
    """A coating given as a table of R and T against wavelength and angle.

    The table's fields and units are the shared library's coating-table record
    (``kmat.surface_optics.CoatingTable``), so a library record lowers without
    translation (:meth:`from_table`): ``wavelength_nm`` (strictly increasing,
    nanometres), ``angle_deg`` (strictly increasing, 0 to 90 degrees of
    incidence), and the grids ``r_s``, ``r_p``, ``t_s``, ``t_p``, each shaped
    ``(len(wavelength_nm), len(angle_deg))``: the power reflectance and
    transmittance (fractions of the incident power) for s and p. The
    absorptance is the complement, ``A = 1 - R - T``. Two optional grids of the
    same shape carry what the Stokes mode needs beyond powers:
    ``phase_r_deg = arg(r_p r_s*)`` and ``phase_t_deg = arg(t_p t_s*)`` in
    degrees, in the ``exp(-i omega t)`` convention with the Fresnel sign of
    ``r_p`` (``r_p = (n2 cos_i - n1 cos_t) / (n2 cos_i + n1 cos_t)``). The
    absolute phases of the four amplitudes do not enter a Mueller matrix; the
    two relative phases are all of it.

    **What each mode reads, and what is refused.**

    * Scalar mode reads ``(R_s + R_p) / 2`` and ``(T_s + T_p) / 2``. A table
      must give both powers of both polarizations; one with a reflectance and
      no transmittance (or the reverse) is refused at construction unless
      ``lossless=True`` states ``T = 1 - R`` (or ``R = 1 - T``) for it.
    * Stokes mode reads the four powers and both phase grids:
      ``M01 = (R_p - R_s) / 2``, ``(M22, M23) = sqrt(R_s R_p)
      (cos, sin)(phase_r)``, and the transmission alike. A table without
      ``phase_r_deg`` and ``phase_t_deg`` is refused when a Stokes trace meets
      it (:meth:`sp` raises); its scalar use is unaffected.

    **Interpolation.** Bilinear in ``(wavelength_nm, angle_deg)``: the angle of
    incidence in degrees is recovered from the ray's cosine, the cell found by
    a sorted search on each axis, and each grid interpolated as
    ``lerp(lerp(v00, v01, w_a), lerp(v10, v11, w_a), w_l)`` with
    ``lerp(a, b, w) = a + w (b - a)``, so a ray exactly on a node reads the
    node's value. Outside the table each coordinate is clamped to the
    table's edge (constant extrapolation). The phases are interpolated as
    their cosine and sine, then normalised, so a phase never wraps through
    the interpolation. The error bound on one cell, for a grid ``f`` with
    continuous second derivatives, at a point at distances ``a_0, a_1`` from
    the cell's wavelength nodes and ``b_0, b_1`` from its angle nodes::

        |f - L f| <= (a_0 a_1 / 2) max|d2f/dlambda2| + (b_0 b_1 / 2) max|d2f/dtheta2|

    the maxima taken over the cell; at the cell's centre that is
    ``h_l^2 / 8 max|f_ll| + h_a^2 / 8 max|f_aa|``. From the table alone, the
    second differences ``D2_l f = f[i+1] - 2 f[i] + f[i-1]`` estimate
    ``h_l^2 f_ll``, so ``(max|D2_l f| + max|D2_a f|) / 8`` over the
    neighbouring nodes estimates the bound at a cell's centre
    (:meth:`interpolation_error_estimate`); it is an estimate, not a bound,
    where the second derivative changes within a cell.

    **Which side the table describes.** The table is the interface as the
    incident medium (``incident_material``) sees it, light arriving from it
    towards the substrate (``substrate_material``), as a thin-film stack is.
    A refractive component matches the two media to its front and back once
    (``incident_side``: ``"auto"`` by index, or ``"front"``/``"back"``). A ray
    arriving from the substrate side at ``theta_s`` reads the table at the
    Snell angle in the incident medium,
    ``sin theta_0 = (n_sub / n_inc) sin theta_s``, by reciprocity: its
    transmittance is the table's ``T(theta_0)``, exactly, for any linear
    reciprocal coating; its reflectance is taken as ``R(theta_0)``, which is
    exact for a lossless coating and an approximation where the coating
    absorbs (an absorbing stack reflects differently from its two sides,
    issue 83); its relative phases are ``phase_t`` and
    ``2 phase_t - phase_r`` (the Stokes relations of a lossless coating).
    Beyond the critical angle from the substrate (``sin theta_0 >= 1``) the
    ray reads ``R = 1``, ``T = 0``. A table of an absorbing coating seen from
    both sides needs a second table for the far side, attached as the
    coating of that side.

    Device residency: the grids are uploaded once per (dtype, device) on
    first use and kept; inside the bounce loop the lookup is backend
    operations only (a sorted search, gathers and arithmetic), with no
    host read.

    Args:
        wavelength_nm: Wavelength nodes [nm], strictly increasing.
        angle_deg: Angle-of-incidence nodes [deg], strictly increasing, 0..90.
        r_s, r_p, t_s, t_p: Power grids, ``(n_wavelength, n_angle)``.
        phase_r_deg, phase_t_deg: Optional relative-phase grids [deg].
        lossless: Fill a missing transmittance as ``1 - R`` (or a missing
            reflectance as ``1 - T``) per polarization.
        substrate_material: The medium behind the coating (an
            ``optiland.materials.BaseMaterial`` or an index). Required: a
            table describes one side of an interface.
        incident_material: The medium the table's light arrives from (a
            material or an index); vacuum (1.0) by default, the library's
            ``"air"`` default.
        incident_side: ``"auto"``, ``"front"`` or ``"back"``.
        name: An optional label.
    """

    def __init__(
        self,
        wavelength_nm,
        angle_deg,
        r_s=None,
        r_p=None,
        t_s=None,
        t_p=None,
        *,
        phase_r_deg=None,
        phase_t_deg=None,
        lossless: bool = False,
        substrate_material=None,
        incident_material=None,
        incident_side: str = "auto",
        name: str = "",
    ):
        import numpy as np  # noqa: PLC0415

        from optiland.nonsequential.components.coating_support import (  # noqa: PLC0415
            _check_incident_side,
        )

        wl = np.asarray(wavelength_nm, dtype=float).ravel()
        ang = np.asarray(angle_deg, dtype=float).ravel()
        for label, axis in (("wavelength_nm", wl), ("angle_deg", ang)):
            if axis.size < 1 or not np.all(np.isfinite(axis)):
                raise ValueError(f"{label} needs at least one finite node")
            if np.any(np.diff(axis) <= 0):
                raise ValueError(f"{label} must be strictly increasing")
        if np.any(wl <= 0):
            raise ValueError("wavelength_nm must be positive")
        if np.any(ang < 0) or np.any(ang > 90):
            raise ValueError("angle_deg must lie in 0..90 degrees")
        rows, cols = wl.size, ang.size
        grids = {}
        for field, value in zip(TABLE_POWER_FIELDS, (r_s, r_p, t_s, t_p), strict=True):
            if value is not None:
                arr = _host_grid(field, value, rows, cols)
                if np.any(arr < -_TABLE_ENERGY_TOL) or np.any(arr > 1 + _TABLE_ENERGY_TOL):
                    raise ValueError(f"{field} holds a value outside 0..1")
                grids[field] = arr
        for pol in ("s", "p"):
            r, t = f"r_{pol}", f"t_{pol}"
            if r in grids and t in grids:
                if np.any(grids[r] + grids[t] > 1 + _TABLE_ENERGY_TOL):
                    raise ValueError(f"{r} + {t} exceeds 1: a coating does not create power")
                continue
            if lossless and (r in grids or t in grids):
                have, missing = (r, t) if r in grids else (t, r)
                grids[missing] = 1.0 - grids[have]
                continue
            absent = [f for f in (r, t) if f not in grids]
            raise ValueError(
                f"the table gives no {' and no '.join(absent)}: the scalar mode reads "
                f"R and T of both polarizations; give the grid, or pass lossless=True "
                f"to take T = 1 - R (or R = 1 - T)"
            )
        phases = {}
        for field, value in zip(TABLE_PHASE_FIELDS, (phase_r_deg, phase_t_deg), strict=True):
            if value is not None:
                phases[field] = _host_grid(field, value, rows, cols)
        if len(phases) == 1:
            raise ValueError("give both phase_r_deg and phase_t_deg, or neither")
        if substrate_material is None:
            raise ValueError(
                "a coating table describes one side of an interface: give its "
                "substrate_material (and incident_material when it is not vacuum)"
            )
        self.wavelength_nm = wl
        self.angle_deg = ang
        self.grids = grids
        self.phases = phases
        self.lossless = bool(lossless)
        self.incident_material = _ideal_or(incident_material)
        self.substrate_material = _ideal_or(substrate_material)
        self.incident_side = _check_incident_side(incident_side)
        self.name = name
        self._resident: dict = {}

    # ----- construction from the shared library's records -----
    @classmethod
    def from_table(cls, table, **kwargs) -> TabulatedCoating:
        """Lower a ``kmat.surface_optics.CoatingTable`` (or its ``arrays()`` dict).

        The table's fields are this class's arguments by name, so nothing is
        translated: ``cls(**table.arrays(), **kwargs)``. ``kwargs`` carries
        what the record type does not (the media, ``lossless``, the phases).
        """
        arrays = table.arrays() if hasattr(table, "arrays") else dict(table)
        return cls(**arrays, **kwargs)

    @classmethod
    def from_record(cls, record, **kwargs) -> TabulatedCoating:
        """Lower a ``kmat.surface_optics.CoatingRecord`` of ``kind="table"``.

        The record names its media (``incident_medium``, ``substrate``); this
        engine takes materials, so the caller passes ``substrate_material``
        (and ``incident_material``) resolved through the family's material
        source. The record's ``name`` becomes the label.
        """
        if getattr(record, "kind", None) != "table" or getattr(record, "table", None) is None:
            raise ValueError("from_record takes a coating record of kind 'table'")
        kwargs.setdefault("name", getattr(record, "name", ""))
        return cls.from_table(record.table, **kwargs)

    # ----- the side-aware coating protocol (coating_support) -----
    def media(self):
        """``(incident, substrate)`` materials."""
        return self.incident_material, self.substrate_material

    @property
    def has_phases(self) -> bool:
        """Whether the table carries the phase grids the Stokes mode needs."""
        return bool(self.phases)

    # ----- device residency -----
    def _channels(self) -> list:
        import numpy as np  # noqa: PLC0415

        chans = [self.grids[f] for f in TABLE_POWER_FIELDS]
        if self.phases:
            pr = np.radians(self.phases["phase_r_deg"])
            pt = np.radians(self.phases["phase_t_deg"])
            chans += [np.cos(pr), np.sin(pr), np.cos(pt), np.sin(pt)]
        return chans

    def _host_tables(self):
        """Axes padded to two nodes, and the channels as one ``(nw * na, C)`` table."""
        import numpy as np  # noqa: PLC0415

        wl, ang = self.wavelength_nm, self.angle_deg
        chans = np.stack(self._channels(), axis=-1)  # (nw, na, C)
        if wl.size == 1:
            wl = np.array([wl[0], wl[0] + 1.0])
            chans = np.concatenate([chans, chans], axis=0)
        if ang.size == 1:
            ang = np.array([ang[0], ang[0] + 1.0])
            chans = np.concatenate([chans, chans], axis=1)
        return wl, ang, chans.reshape(wl.size * ang.size, chans.shape[-1])

    def _tables(self, like):
        """The resident tables beside ``like`` (one upload per dtype and device)."""
        if hasattr(like, "detach"):
            key = (str(like.dtype), str(like.device))
        else:
            key = ("numpy", str(like.dtype))
        tables = self._resident.get(key)
        if tables is None:
            wl, ang, values = self._host_tables()
            if hasattr(like, "detach"):
                import torch  # noqa: PLC0415

                def up(a):
                    return torch.as_tensor(a, dtype=like.dtype, device=like.device)
            else:
                import numpy as np  # noqa: PLC0415

                def up(a):
                    return np.asarray(a, dtype=like.dtype)

            tables = {
                "wl": up(wl), "ang": up(ang), "values": up(values),
                "wl_range": (float(wl[0]), float(wl[-1])),
                "ang_range": (float(ang[0]), float(ang[-1])),
                "na": int(ang.size),
            }
            self._resident[key] = tables
        return tables

    # ----- the lookup -----
    @staticmethod
    def _cell(nodes, x, lo: float, hi: float):
        """Lower node index and weight of ``x`` (clamped to ``[lo, hi]``) on ``nodes``."""
        m = int(nodes.shape[0])
        xc = be.clip(x, lo, hi)
        if hasattr(xc, "detach"):
            import torch  # noqa: PLC0415

            i = torch.searchsorted(nodes, xc.contiguous(), right=True) - 1
            i = torch.clamp(i, 0, m - 2)
        else:
            import numpy as np  # noqa: PLC0415

            i = np.clip(np.searchsorted(nodes, xc, side="right") - 1, 0, m - 2)
        x0 = nodes[i]
        x1 = nodes[i + 1]
        return i, (xc - x0) / (x1 - x0)

    def _material_n(self, material, wavelength_um):
        if be.get_backend() == "torch" and hasattr(wavelength_um, "detach"):
            return material._calculate_n(wavelength_um)
        return material.n(wavelength_um)

    def lookup(self, wavelength_um, cos_theta_i, from_substrate=None) -> dict:
        """Every grid at each ray's wavelength and angle, by bilinear interpolation.

        Args:
            wavelength_um: Per-ray wavelength [um].
            cos_theta_i: Per-ray ``|cos theta|`` in the medium the ray arrives
                from.
            from_substrate: Optional per-ray mask of the rays arriving from the
                substrate side (read at the Snell angle in the incident
                medium; see the class docstring).

        Returns:
            ``{"r_s", "r_p", "t_s", "t_p"}`` per ray, and with phase grids
            ``"xr_cos", "xr_sin", "xt_cos", "xt_sin"`` (normalised).
        """
        c = be.clip(be.abs(cos_theta_i), 0.0, 1.0)
        far_tir = None
        if from_substrate is not None:
            n_inc = self._material_n(self.incident_material, wavelength_um)
            n_sub = self._material_n(self.substrate_material, wavelength_um)
            ratio = n_sub / n_inc
            s2_0 = ratio * ratio * (1.0 - c * c)
            far_tir = from_substrate & (s2_0 >= 1.0)
            w0 = be.where(far_tir, be.zeros_like(s2_0), 1.0 - s2_0)
            c = be.where(from_substrate, w0**0.5, c)
        theta = be.arccos(c) * (180.0 / be.pi)
        x = wavelength_um * 1000.0
        t = self._tables(theta)
        il, wl = self._cell(t["wl"], x, *t["wl_range"])
        ia, wa = self._cell(t["ang"], theta, *t["ang_range"])
        na = t["na"]
        v = t["values"]
        k00 = il * na + ia
        v00, v01 = v[k00], v[k00 + 1]
        v10, v11 = v[k00 + na], v[k00 + na + 1]
        wa_c, wl_c = wa[:, None], wl[:, None]
        lo = v00 + wa_c * (v01 - v00)
        hi = v10 + wa_c * (v11 - v10)
        f = lo + wl_c * (hi - lo)
        out = {name: f[:, j] for j, name in enumerate(TABLE_POWER_FIELDS)}
        if self.phases:
            for j, name in ((4, "xr"), (6, "xt")):
                cc, ss = f[:, j], f[:, j + 1]
                norm = (cc * cc + ss * ss) ** 0.5
                ok = norm > 0
                safe = be.where(ok, norm, be.ones_like(norm))
                out[f"{name}_cos"] = be.where(ok, cc / safe, be.ones_like(norm))
                out[f"{name}_sin"] = be.where(ok, ss / safe, be.zeros_like(norm))
            if from_substrate is not None:
                # far side, lossless Stokes relations: phase_r' = 2 phase_t - phase_r
                ct, st = out["xt_cos"], out["xt_sin"]
                c2t, s2t = ct * ct - st * st, 2.0 * ct * st
                cr, sr = out["xr_cos"], out["xr_sin"]
                out["xr_cos"] = be.where(from_substrate, c2t * cr + s2t * sr, cr)
                out["xr_sin"] = be.where(from_substrate, s2t * cr - c2t * sr, sr)
        if far_tir is not None:
            one, zero = be.ones_like(c), be.zeros_like(c)
            for name in ("r_s", "r_p"):
                out[name] = be.where(far_tir, one, out[name])
            for name in ("t_s", "t_p"):
                out[name] = be.where(far_tir, zero, out[name])
        return out

    def evaluate(self, wavelength_um, cos_theta_i, from_substrate=None):
        """Per-ray unpolarized ``(R, T)``: ``(R_s + R_p) / 2`` and ``(T_s + T_p) / 2``."""
        g = self.lookup(wavelength_um, cos_theta_i, from_substrate)
        return 0.5 * (g["r_s"] + g["r_p"]), 0.5 * (g["t_s"] + g["t_p"])

    def sp(self, wavelength_um, cos_theta_i, from_substrate=None):
        """The Stokes mode's s and p terms (``polarization.SPCoefficients``).

        Raises:
            ValueError: When the table has no phase grids.
        """
        if not self.phases:
            raise ValueError(
                f"coating table {self.name!r} has no phase_r_deg and phase_t_deg: "
                "a Stokes trace needs the relative phases of r and t (arg(r_p r_s*), "
                "arg(t_p t_s*)); trace it in scalar mode or give the phase grids"
            )
        from optiland.nonsequential.polarization import SPCoefficients  # noqa: PLC0415

        g = self.lookup(wavelength_um, cos_theta_i, from_substrate)
        amp = (be.clip(g["r_s"] * g["r_p"], 0.0, None)) ** 0.5
        return SPCoefficients(
            g["r_s"], g["r_p"], g["t_s"], g["t_p"],
            amp * g["xr_cos"], amp * g["xr_sin"], g["xt_cos"], g["xt_sin"],
            be.ones_like(amp) > 0,
        )

    def interpolation_error_estimate(self) -> dict:
        """The bound's estimate from the table's own second differences, per grid.

        ``(max|D2_l f| + max|D2_a f|) / 8``: the bilinear error at a cell's
        centre when each grid's second derivative is constant over the cell
        (a node's second difference ``f[i+1] - 2 f[i] + f[i-1]`` is
        ``h^2 f''`` there). A grid with fewer than three nodes on an axis
        contributes nothing on that axis.
        """
        import numpy as np  # noqa: PLC0415

        out = {}
        for field in TABLE_POWER_FIELDS:
            f = self.grids[field]
            d_l = np.abs(np.diff(f, n=2, axis=0)).max() if f.shape[0] >= 3 else 0.0
            d_a = np.abs(np.diff(f, n=2, axis=1)).max() if f.shape[1] >= 3 else 0.0
            out[field] = float(d_l + d_a) / 8.0
        return out

    # ----- the sequential engine -----
    def _rays_RT(self, rays, nx, ny, nz):
        aoi = self._compute_aoi(rays, nx, ny, nz)
        return self.evaluate(be.atleast_1d(rays.w), be.cos(aoi))

    def reflect(self, rays, nx=None, ny=None, nz=None):
        """Scale the rays' intensity by the table's unpolarized reflectance."""
        R, _ = self._rays_RT(rays, nx, ny, nz)
        rays.i = rays.i * R
        return rays

    def transmit(self, rays, nx=None, ny=None, nz=None):
        """Scale the rays' intensity by the table's unpolarized transmittance."""
        _, T = self._rays_RT(rays, nx, ny, nz)
        rays.i = rays.i * T
        return rays

    # ----- JSON -----
    def to_dict(self) -> dict[str, Any]:
        """The table, its media and its options; field names as the library's."""
        d: dict[str, Any] = {
            "type": self.__class__.__name__,
            "wavelength_nm": self.wavelength_nm.tolist(),
            "angle_deg": self.angle_deg.tolist(),
        }
        for field in TABLE_POWER_FIELDS:
            d[field] = self.grids[field].tolist()
        for field in TABLE_PHASE_FIELDS:
            if field in self.phases:
                d[field] = self.phases[field].tolist()
        d["incident_material"] = self.incident_material.to_dict()
        d["substrate_material"] = self.substrate_material.to_dict()
        d["incident_side"] = self.incident_side
        d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabulatedCoating:
        """Rebuild from :meth:`to_dict`'s form."""
        kwargs = {
            k: data[k]
            for k in ("wavelength_nm", "angle_deg", *TABLE_POWER_FIELDS, *TABLE_PHASE_FIELDS)
            if k in data
        }
        return cls(
            **kwargs,
            incident_material=BaseMaterial.from_dict(data["incident_material"]),
            substrate_material=BaseMaterial.from_dict(data["substrate_material"]),
            incident_side=data.get("incident_side", "auto"),
            name=data.get("name", ""),
        )
