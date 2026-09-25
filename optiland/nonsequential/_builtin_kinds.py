"""The engine's own kinds, registered through :mod:`optiland.nonsequential.kinds`.

Every builder, JSON writer and reader and IR lowering below is the code that
used to sit in an ``isinstance`` chain of :mod:`~optiland.nonsequential.scene`,
:mod:`~optiland.nonsequential.serialization` or
:mod:`~optiland.nonsequential.ir.lower`, moved here unchanged, so the JSON and
the IR the built-in kinds produce are exactly what they were. A new kind added
inside the engine registers here; one added from outside registers the same
way through the ``optiland.nonsequential`` entry-point group.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from optiland.nonsequential import kinds


def _to_float(value: Any) -> float:
    from optiland.nonsequential.serialization import _to_float as f  # noqa: PLC0415

    return f(value)


def _material_out(mat: Any) -> Any:
    from optiland.nonsequential.serialization import (  # noqa: PLC0415
        _serialize_material,
    )

    return _serialize_material(mat)


def _material_in(d: Any) -> Any:
    from optiland.nonsequential.serialization import (  # noqa: PLC0415
        _deserialize_material,
    )

    return _deserialize_material(d)


def _resolve_total_flux(config: Any) -> Any:
    from optiland.nonsequential.scene import _resolve_total_flux as f  # noqa: PLC0415

    return f(config)



# ---------------------------------------------------------------------------
# Builders of the built-in sources and detectors (named functions, so the
# ignored-config audit of the test suite can read their source)
# ---------------------------------------------------------------------------


def build_point_source(cs: Any, config: Any) -> Any:
    """``PointSourceConfig`` -> ``PointSource``."""
    from optiland.nonsequential.sources.point import PointSource  # noqa: PLC0415

    return PointSource(
        cs=cs,
        spectrum=config.spectrum,
        total_flux=_resolve_total_flux(config),
        half_angle_deg=config.half_angle_deg,
        medium=getattr(config, "medium", None),
    )


def build_collimated_source(cs: Any, config: Any) -> Any:
    """``CollimatedSourceConfig`` -> ``CollimatedSource``."""
    from optiland.nonsequential.sources.collimated import (  # noqa: PLC0415
        CollimatedSource,
    )

    return CollimatedSource(
        cs=cs,
        spectrum=config.spectrum,
        total_flux=_resolve_total_flux(config),
        aperture_radius=config.aperture_radius,
        profile=config.profile,
        gaussian_sigma=config.gaussian_sigma,
        medium=getattr(config, "medium", None),
    )


def build_extended_source(cs: Any, config: Any) -> Any:
    """``ExtendedSourceConfig`` -> ``ExtendedSource``."""
    from optiland.nonsequential.sources.extended import (  # noqa: PLC0415
        ExtendedSource,
    )

    return ExtendedSource(
        cs=cs,
        spectrum=config.spectrum,
        total_flux=_resolve_total_flux(config),
        width=config.width,
        height=config.height,
        aperture_radius=config.aperture_radius,
        half_angle_deg=config.half_angle_deg,
        medium=getattr(config, "medium", None),
    )


def build_irradiance_detector(cs: Any, config: Any) -> Any:
    """``IrradianceDetectorConfig`` -> ``IrradianceDetector``."""
    from optiland.nonsequential.detectors.irradiance import (  # noqa: PLC0415
        IrradianceDetector,
    )

    return IrradianceDetector(
        cs=cs,
        width=config.width,
        height=config.height,
        num_pixels_x=config.num_pixels_x,
        num_pixels_y=config.num_pixels_y,
        splat=config.splat,
        splat_sigma=config.splat_sigma,
        absorb=config.absorb,
        side=config.side,
        reflection_bins=config.reflection_bins,
    )


def build_spectral_detector(cs: Any, config: Any) -> Any:
    """``SpectralDetectorConfig`` -> ``SpectralDetector``."""
    import optiland.backend as be  # noqa: PLC0415
    from optiland.nonsequential.detectors.spectral import (  # noqa: PLC0415
        SpectralDetector,
    )

    wl_bins = be.linspace(config.wl_min, config.wl_max, config.num_bins + 1)
    return SpectralDetector(
        cs=cs,
        width=config.width,
        height=config.height,
        num_pixels_x=config.num_pixels_x,
        num_pixels_y=config.num_pixels_y,
        wavelength_bins=wl_bins,
        splat=config.splat,
        splat_sigma=config.splat_sigma,
        absorb=config.absorb,
    )


def build_far_field_detector(cs: Any, config: Any) -> Any:
    """``FarFieldDetectorConfig`` -> ``FarFieldDetector``."""
    from optiland.nonsequential.detectors.far_field import (  # noqa: PLC0415
        FarFieldDetector,
    )

    return FarFieldDetector(
        cs=cs,
        theta_max_deg=90.0,
        num_bins_theta=config.num_theta,
        num_bins_phi=config.num_phi,
        absorb=config.absorb,
        side=config.side,
        reflection_bins=config.reflection_bins,
    )


def build_hemisphere_detector(cs: Any, config: Any) -> Any:
    """``HemisphereDetectorConfig`` -> ``HemisphereDetector``."""
    from optiland.nonsequential.detectors.hemisphere import (  # noqa: PLC0415
        HemisphereDetector,
    )

    return HemisphereDetector(
        cs=cs,
        radius=config.radius,
        num_bins_theta=config.num_theta,
        num_bins_phi=config.num_phi,
        absorb=config.absorb,
        reflection_bins=config.reflection_bins,
    )


def build_ray_database_detector(cs: Any, config: Any) -> Any:
    """``RayDatabaseConfig`` -> ``RayDatabaseDetector``."""
    from optiland.nonsequential.components.geometry.analytic.plane import (  # noqa: PLC0415
        FinitePlaneGeometry,
    )
    from optiland.nonsequential.detectors.ray_database import (  # noqa: PLC0415
        RayDatabaseDetector,
    )

    geometry = FinitePlaneGeometry(width=config.width, height=config.height)
    return RayDatabaseDetector(
        cs=cs,
        geometry=geometry,
        # 0 ("unlimited", the config default) maps to RayDatabaseDetector's
        # own None-means-unlimited convention.
        max_rays=config.max_rays if config.max_rays > 0 else None,
        absorb=config.absorb,
    )


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------


def _register_spectra() -> None:
    from optiland.nonsequential.sources.base import Spectrum  # noqa: PLC0415

    def to_dict(spectrum: Spectrum) -> dict:
        from optiland.nonsequential.serialization import _to_list  # noqa: PLC0415

        return {
            "wavelengths": _to_list(spectrum.wavelengths),
            "weights": _to_list(spectrum.weights),
        }

    def from_dict(d: dict) -> Spectrum:
        return Spectrum(
            wavelengths=np.array(d["wavelengths"], dtype=np.float64),
            weights=np.array(d["weights"], dtype=np.float64),
        )

    def lower(spectrum: Spectrum) -> dict:
        return {
            "wavelengths": np.asarray(spectrum.wavelengths, dtype=np.float64).tolist(),
            "weights": np.asarray(spectrum.weights, dtype=np.float64).tolist(),
        }

    kinds.register_spectrum(
        "lines", Spectrum, to_dict, from_dict, lower,
        description="discrete lines: wavelengths and relative weights",
    )


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

_SOURCE_ATTACHED = ("total_flux", "total_flux_lumens")


def _register_sources() -> None:
    from optiland.nonsequential.sources.collimated import (  # noqa: PLC0415
        CollimatedSource,
    )
    from optiland.nonsequential.sources.configs import (  # noqa: PLC0415
        CollimatedSourceConfig,
        ExtendedSourceConfig,
        PointSourceConfig,
    )
    from optiland.nonsequential.sources.extended import (  # noqa: PLC0415
        ExtendedSource,
    )
    from optiland.nonsequential.sources.point import PointSource  # noqa: PLC0415

    # -- point ----------------------------------------------------------------
    kinds.register_source(
        "point",
        PointSource,
        PointSourceConfig,
        build=build_point_source,
        to_dict=lambda s: {"half_angle_deg": _to_float(s.half_angle_deg)},
        from_dict=lambda d, spectrum, total_flux, medium: PointSourceConfig(
            spectrum=spectrum,
            total_flux=total_flux,
            half_angle_deg=d.get("half_angle_deg", 90.0),
            medium=medium,
        ),
        lower=lambda s: {"half_angle_deg": s.half_angle_deg},
        attached=_SOURCE_ATTACHED,
        description="a point emitting into a cone or the full sphere",
    )

    # -- collimated -----------------------------------------------------------
    kinds.register_source(
        "collimated",
        CollimatedSource,
        CollimatedSourceConfig,
        build=build_collimated_source,
        to_dict=lambda s: {
            "aperture_radius": _to_float(s.aperture_radius),
            "profile": s.profile,
            "gaussian_sigma": _to_float(s.gaussian_sigma),
        },
        from_dict=lambda d, spectrum, total_flux, medium: CollimatedSourceConfig(
            spectrum=spectrum,
            total_flux=total_flux,
            aperture_radius=d.get("aperture_radius", 1.0),
            profile=d.get("profile", "tophat"),
            gaussian_sigma=d.get("gaussian_sigma"),
            medium=medium,
        ),
        lower=lambda s: {
            "aperture_radius": s.aperture_radius,
            "profile": s.profile,
            "gaussian_sigma": s.gaussian_sigma,
        },
        attached=_SOURCE_ATTACHED,
        description="a parallel beam, top-hat or truncated Gaussian",
    )

    # -- extended -------------------------------------------------------------
    kinds.register_source(
        "extended",
        ExtendedSource,
        ExtendedSourceConfig,
        build=build_extended_source,
        to_dict=lambda s: {
            "width": _to_float(s.width),
            "height": _to_float(s.height),
            "aperture_radius": (
                _to_float(s.aperture_radius) if s.aperture_radius is not None else None
            ),
            "half_angle_deg": _to_float(s.half_angle_deg),
        },
        from_dict=lambda d, spectrum, total_flux, medium: ExtendedSourceConfig(
            spectrum=spectrum,
            total_flux=total_flux,
            width=d.get("width", 1.0),
            height=d.get("height", 1.0),
            aperture_radius=d.get("aperture_radius"),
            half_angle_deg=d.get("half_angle_deg", 90.0),
            medium=medium,
        ),
        lower=lambda s: {
            "width": s.width,
            "height": s.height,
            "aperture_radius": s.aperture_radius,
            "half_angle_deg": s.half_angle_deg,
        },
        attached=_SOURCE_ATTACHED,
        description="a rectangle or disc emitting into a cone or Lambertian",
    )


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _register_detectors() -> None:
    from optiland.nonsequential.detectors.configs import (  # noqa: PLC0415
        FarFieldDetectorConfig,
        HemisphereDetectorConfig,
        IrradianceDetectorConfig,
        RayDatabaseConfig,
        SpectralDetectorConfig,
    )
    from optiland.nonsequential.detectors.far_field import (  # noqa: PLC0415
        FarFieldDetector,
    )
    from optiland.nonsequential.detectors.hemisphere import (  # noqa: PLC0415
        HemisphereDetector,
    )
    from optiland.nonsequential.detectors.irradiance import (  # noqa: PLC0415
        IrradianceDetector,
    )
    from optiland.nonsequential.detectors.ray_database import (  # noqa: PLC0415
        RayDatabaseDetector,
    )
    from optiland.nonsequential.detectors.spectral import (  # noqa: PLC0415
        SpectralDetector,
    )

    # -- irradiance -----------------------------------------------------------
    kinds.register_detector(
        "irradiance",
        IrradianceDetector,
        IrradianceDetectorConfig,
        build=build_irradiance_detector,
        to_dict=lambda det: {
            "width": _to_float(det.width),
            "height": _to_float(det.height),
            "num_pixels_x": int(det.num_pixels_x),
            "num_pixels_y": int(det.num_pixels_y),
            "splat": det.splat,
            "splat_sigma": _to_float(det.splat_sigma),
            "absorb": bool(det.absorb),
            "side": det.side,
            "reflection_bins": int(det.reflection_bins),
        },
        from_dict=lambda d: IrradianceDetectorConfig(
            width=d["width"],
            height=d["height"],
            num_pixels_x=d.get("num_pixels_x", 256),
            num_pixels_y=d.get("num_pixels_y", 256),
            splat=d.get("splat", "bilinear"),
            splat_sigma=d.get("splat_sigma", 0.5),
            absorb=d.get("absorb", True),
            side=d.get("side", "both"),
            reflection_bins=d.get("reflection_bins", 0),
        ),
        lower=lambda det: {
            "width": det.width,
            "height": det.height,
            "num_pixels_x": det.num_pixels_x,
            "num_pixels_y": det.num_pixels_y,
            "splat": det.splat,
            "splat_sigma": det.splat_sigma,
            "side": det.side,
        },
        attached=("width", "height"),
        description="a plane of pixels booking flux (W) and irradiance (W/mm^2)",
    )

    # -- spectral -------------------------------------------------------------
    def spectral_to_dict(det) -> dict:
        wl_bins = np.asarray(det.wavelength_bins, dtype=float)
        return {
            "width": _to_float(det.width),
            "height": _to_float(det.height),
            "num_pixels_x": int(det.num_pixels_x),
            "num_pixels_y": int(det.num_pixels_y),
            "wl_min": float(wl_bins[0]),
            "wl_max": float(wl_bins[-1]),
            "num_bins": int(len(wl_bins) - 1),
            "splat": det.splat,
            "splat_sigma": _to_float(det.splat_sigma),
            "absorb": bool(det.absorb),
        }

    kinds.register_detector(
        "spectral",
        SpectralDetector,
        SpectralDetectorConfig,
        build=build_spectral_detector,
        to_dict=spectral_to_dict,
        from_dict=lambda d: SpectralDetectorConfig(
            width=d["width"],
            height=d["height"],
            num_pixels_x=d.get("num_pixels_x", 256),
            num_pixels_y=d.get("num_pixels_y", 256),
            wl_min=d.get("wl_min", 0.4),
            wl_max=d.get("wl_max", 0.7),
            num_bins=d.get("num_bins", 100),
            splat=d.get("splat", "bilinear"),
            splat_sigma=d.get("splat_sigma", 0.5),
            absorb=d.get("absorb", True),
        ),
        lower=lambda det: {
            "width": det.width,
            "height": det.height,
            "num_pixels_x": det.num_pixels_x,
            "num_pixels_y": det.num_pixels_y,
            "wavelength_bins": np.asarray(
                det.wavelength_bins, dtype=np.float64
            ).tolist(),
            "splat": det.splat,
            "splat_sigma": det.splat_sigma,
        },
        attached=(),
        description="a plane of pixels with wavelength bins",
    )

    # -- far field ------------------------------------------------------------
    kinds.register_detector(
        "far_field",
        FarFieldDetector,
        FarFieldDetectorConfig,
        build=build_far_field_detector,
        to_dict=lambda det: {
            "num_bins_theta": int(det.num_bins_theta),
            "num_bins_phi": int(det.num_bins_phi),
            "absorb": bool(det.absorb),
            "side": det.side,
            "reflection_bins": int(det.reflection_bins),
        },
        from_dict=lambda d: FarFieldDetectorConfig(
            num_theta=d.get("num_bins_theta", 90),
            num_phi=d.get("num_bins_phi", 360),
            absorb=d.get("absorb", True),
            side=d.get("side", "both"),
            reflection_bins=d.get("reflection_bins", 0),
        ),
        lower=lambda det: {
            "num_bins_theta": det.num_bins_theta,
            "num_bins_phi": det.num_bins_phi,
            "side": det.side,
        },
        attached=(),
        description="a plane binning the arriving flux by direction (W/sr)",
    )

    # -- hemisphere -----------------------------------------------------------
    kinds.register_detector(
        "hemisphere",
        HemisphereDetector,
        HemisphereDetectorConfig,
        build=build_hemisphere_detector,
        to_dict=lambda det: {
            "radius": _to_float(det.radius),
            "num_bins_theta": int(det.num_bins_theta),
            "num_bins_phi": int(det.num_bins_phi),
            "absorb": bool(det.absorb),
            "reflection_bins": int(det.reflection_bins),
        },
        from_dict=lambda d: HemisphereDetectorConfig(
            radius=d["radius"],
            num_theta=d.get("num_bins_theta", 18),
            num_phi=d.get("num_bins_phi", 36),
            absorb=d.get("absorb", True),
            reflection_bins=d.get("reflection_bins", 0),
        ),
        lower=lambda det: {
            "radius": det.radius,
            "num_bins_theta": det.num_bins_theta,
            "num_bins_phi": det.num_bins_phi,
        },
        attached=("radius",),
        description="a closed hemispherical shell binning flux by direction",
    )

    # -- ray database ---------------------------------------------------------
    kinds.register_detector(
        "ray_database",
        RayDatabaseDetector,
        RayDatabaseConfig,
        build=build_ray_database_detector,
        # RayDatabaseDetector holds a geometry object; extract width/height.
        to_dict=lambda det: {
            "width": float(getattr(det.geometry, "width", 10.0)),
            "height": float(getattr(det.geometry, "height", 10.0)),
            "absorb": bool(det.absorb),
        },
        from_dict=lambda d: RayDatabaseConfig(
            width=d["width"],
            height=d["height"],
            absorb=d.get("absorb", True),
        ),
        lower=lambda det: {
            "width": getattr(det.geometry, "width", 10.0),
            "height": getattr(det.geometry, "height", 10.0),
        },
        attached=(),
        description="a plane recording every arriving ray",
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _register_geometries() -> None:
    from optiland.nonsequential.components.geometry.analytic.annulus import (  # noqa: PLC0415
        AnnularPlaneGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.conic import (  # noqa: PLC0415
        ConicGeometry,
        ParaboloidGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.frustum import (  # noqa: PLC0415
        CylindricalFrustumGeometry,
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
    from optiland.nonsequential.components.geometry.mesh.mesh_geometry import (  # noqa: PLC0415
        MeshGeometry,
    )

    def conic_params(g) -> dict:
        return {
            "radius": g.radius,
            "conic": g.conic,
            "aperture_radius": g.aperture_radius,
        }

    kinds.register_geometry("conic", ConicGeometry, conic_params)
    # A paraboloid is a conic with the conic constant fixed at -1.
    kinds.register_geometry(
        "paraboloid", ParaboloidGeometry, conic_params, ir_kind="conic"
    )
    kinds.register_geometry(
        "plane",
        FinitePlaneGeometry,
        lambda g: {
            "width": g.width,
            "height": g.height,
            "aperture_radius": g.aperture_radius,
        },
    )
    # Infinite plane: no width/height/aperture limit.
    kinds.register_geometry(
        "infinite_plane",
        PlaneGeometry,
        lambda g: {"width": None, "height": None, "aperture_radius": None},
        ir_kind="plane",
    )
    kinds.register_geometry(
        "annulus",
        AnnularPlaneGeometry,
        lambda g: {
            "inner_radius": g.inner_radius,
            "outer_radius": g.outer_radius,
            "z_offset": g.z_offset,
        },
    )
    kinds.register_geometry(
        "frustum",
        CylindricalFrustumGeometry,
        lambda g: {
            "r_front": g.r_front,
            "r_back": g.r_back,
            "z_front": g.z_front,
            "z_back": g.z_back,
        },
    )
    kinds.register_geometry(
        "sphere",
        SphereGeometry,
        lambda g: {"radius": g.radius, "aperture_radius": g.aperture_radius},
    )
    # Ports are plain data -- axis and angular radius -- so the cavity
    # round-trips through the IR's JSON without a geometry object.
    kinds.register_geometry(
        "spherical_cavity",
        SphericalCavityGeometry,
        lambda g: {
            "radius": g.radius,
            "ports": [
                {
                    "axis": list(port.unit_axis),
                    "half_angle_deg": float(port.half_angle_deg),
                }
                for port in g.ports
            ],
        },
    )
    kinds.register_geometry(
        "mesh",
        MeshGeometry,
        lambda g: {
            "vertices": np.asarray(g.mesh.vertices, dtype=np.float64).tolist(),
            "faces": np.asarray(g.mesh.faces, dtype=np.int64).tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Scatter (BSDF)
# ---------------------------------------------------------------------------


def _register_bsdfs() -> None:
    from optiland.nonsequential.bsdf.harvey_shack import (  # noqa: PLC0415
        HarveyShackBSDF,
    )
    from optiland.nonsequential.bsdf.lambertian import LambertianBSDF  # noqa: PLC0415
    from optiland.nonsequential.bsdf.specular import SpecularBRDF  # noqa: PLC0415
    from optiland.nonsequential.bsdf.tabulated import TabulatedBSDF  # noqa: PLC0415

    kinds.register_bsdf(
        "lambertian",
        LambertianBSDF,
        lambda b: {
            "reflectance_value": b.reflectance_value,
            "transmissive_fraction": b.transmissive_fraction,
        },
    )
    kinds.register_bsdf(
        "harvey_shack",
        HarveyShackBSDF,
        lambda b: {
            "b0": b.b0,
            "l0": b.l0,
            "s": b.s,
            "transmissive_fraction": b.transmissive_fraction,
        },
    )
    kinds.register_bsdf(
        "tabulated",
        TabulatedBSDF,
        lambda b: {
            "path": str(b.path),
            "transmissive_fraction": b.transmissive_fraction,
        },
    )
    kinds.register_bsdf("specular", SpecularBRDF, lambda b: {})


# ---------------------------------------------------------------------------
# Compound components
# ---------------------------------------------------------------------------


def _register_components() -> None:
    from optiland.nonsequential.components.configs import (  # noqa: PLC0415
        DoubletConfig,
        LensConfig,
        MirrorConfig,
        ParaxialLensConfig,
        PrismConfig,
    )
    from optiland.nonsequential.components.doublet import Doublet  # noqa: PLC0415
    from optiland.nonsequential.components.lens import Lens  # noqa: PLC0415
    from optiland.nonsequential.components.mirror import Mirror  # noqa: PLC0415
    from optiland.nonsequential.components.paraxial import (  # noqa: PLC0415
        ParaxialLens,
    )
    from optiland.nonsequential.components.prism import Prism  # noqa: PLC0415

    kinds.register_component(
        "lens",
        Lens,
        LensConfig,
        build=lambda scene, name, cs, config: scene.add_lens(name, cs, config),
        to_dict=lambda c: {
            "r1": _to_float(c._config.r1),
            "r2": _to_float(c._config.r2),
            "thickness": _to_float(c._config.thickness),
            "material": _material_out(c._config.material),
            "front_aperture_radius": _to_float(c._config.front_aperture_radius),
            "back_aperture_radius": (
                _to_float(c._config.back_aperture_radius)
                if c._config.back_aperture_radius is not None
                else None
            ),
            "conic1": _to_float(c._config.conic1),
            "conic2": _to_float(c._config.conic2),
        },
        from_dict=lambda cfg: LensConfig(
            r1=cfg["r1"],
            r2=cfg["r2"],
            thickness=cfg["thickness"],
            material=_material_in(cfg["material"]),
            front_aperture_radius=cfg["front_aperture_radius"],
            back_aperture_radius=cfg.get("back_aperture_radius"),
            conic1=cfg.get("conic1", 0.0),
            conic2=cfg.get("conic2", 0.0),
        ),
        attached="*",
    )

    def mirror_to_dict(c) -> dict:
        config = c._config
        if not isinstance(config.reflectance, int | float) and not hasattr(
            config.reflectance, "numpy"
        ):
            raise TypeError(
                f"Cannot serialize mirror '{c.name}': reflectance is "
                f"{type(config.reflectance).__name__}, not a constant. "
                "Only a scalar reflectance round-trips through JSON "
                "serialization; a callable or coating reflectance must be "
                "re-attached after loading."
            )
        return {
            "radius": _to_float(config.radius),
            "reflectance": _to_float(config.reflectance),
            "conic": _to_float(config.conic),
            "aperture_radius": _to_float(config.aperture_radius),
        }

    kinds.register_component(
        "mirror",
        Mirror,
        MirrorConfig,
        build=lambda scene, name, cs, config: scene.add_mirror(name, cs, config),
        to_dict=mirror_to_dict,
        from_dict=lambda cfg: MirrorConfig(
            radius=cfg["radius"],
            reflectance=cfg["reflectance"],
            conic=cfg.get("conic", 0.0),
            aperture_radius=cfg["aperture_radius"],
        ),
        attached="*",
    )
    kinds.register_component(
        "doublet",
        Doublet,
        DoubletConfig,
        build=lambda scene, name, cs, config: scene.add_doublet(name, cs, config),
        to_dict=lambda c: {
            "r1": _to_float(c._config.r1),
            "r2": _to_float(c._config.r2),
            "r3": _to_float(c._config.r3),
            "thickness1": _to_float(c._config.thickness1),
            "thickness2": _to_float(c._config.thickness2),
            "material1": _material_out(c._config.material1),
            "material2": _material_out(c._config.material2),
            "aperture_radius": _to_float(c._config.aperture_radius),
            "conic1": _to_float(c._config.conic1),
            "conic2": _to_float(c._config.conic2),
            "conic3": _to_float(c._config.conic3),
        },
        from_dict=lambda cfg: DoubletConfig(
            r1=cfg["r1"],
            r2=cfg["r2"],
            r3=cfg["r3"],
            thickness1=cfg["thickness1"],
            thickness2=cfg["thickness2"],
            material1=_material_in(cfg["material1"]),
            material2=_material_in(cfg["material2"]),
            aperture_radius=cfg["aperture_radius"],
            conic1=cfg.get("conic1", 0.0),
            conic2=cfg.get("conic2", 0.0),
            conic3=cfg.get("conic3", 0.0),
        ),
        attached="*",
    )
    # As for a lens, per-surface overrides (SurfaceConfig) are not part of
    # the round trip.
    kinds.register_component(
        "prism",
        Prism,
        PrismConfig,
        build=lambda scene, name, cs, config: scene.add_prism(name, cs, config),
        to_dict=lambda c: {
            "apex_angle_deg": _to_float(c._config.apex_angle_deg),
            "face_length": _to_float(c._config.face_length),
            "length": _to_float(c._config.length),
            "material": _material_out(c._config.material),
            "open_base": bool(c._config.open_base),
        },
        from_dict=lambda cfg: PrismConfig(
            apex_angle_deg=cfg["apex_angle_deg"],
            face_length=cfg["face_length"],
            length=cfg["length"],
            material=_material_in(cfg["material"]),
            open_base=cfg.get("open_base", False),
        ),
        attached="*",
    )
    kinds.register_component(
        "paraxial_lens",
        ParaxialLens,
        ParaxialLensConfig,
        build=lambda scene, name, cs, config: scene.add_paraxial_lens(
            name, cs, config
        ),
        to_dict=lambda c: {
            "focal_length": _to_float(c._config.focal_length),
            "aperture_radius": _to_float(c._config.aperture_radius),
            "stop_radius": (
                _to_float(c._config.stop_radius)
                if c._config.stop_radius is not None
                else None
            ),
        },
        from_dict=lambda cfg: ParaxialLensConfig(
            focal_length=cfg["focal_length"],
            aperture_radius=cfg["aperture_radius"],
            stop_radius=cfg.get("stop_radius"),
        ),
        attached="*",
    )


def register_all() -> None:
    """Register every built-in kind (called once, by :mod:`.kinds`)."""
    _register_spectra()
    _register_sources()
    _register_detectors()
    _register_geometries()
    _register_bsdfs()
    _register_components()
