"""Physical constraints shared by OSLO import and export."""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.fileio.oslo.constants import DEFAULT_WAVELENGTHS_UM

if TYPE_CHECKING:
    from optiland.optic import Optic


def validate_object_na(optic: Optic, value: float) -> None:
    """Require a finite object and a forward cone smaller than a hemisphere."""
    if optic.object_surface.is_infinite:
        raise ValueError("OSLO NAO requires a finite object distance")
    wavelength = (
        optic.primary_wavelength if optic.wavelengths else DEFAULT_WAVELENGTHS_UM[0]
    )
    n_object = float(optic.object_surface.material_post.n(wavelength).item())
    # NA = n*sin(theta) (OSLO Program Reference p. 120). Equality needs
    # extended ray aiming, which is outside the supported launch mapping.
    if not 0 < value < n_object:
        raise ValueError("OSLO NAO requires 0 < NA < the object refractive index")
