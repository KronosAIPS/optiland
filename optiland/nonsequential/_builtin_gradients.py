"""The gradient rule of every parameter of every built-in kind (T-09-2).

``docs/theory/09_differentiation.md`` section 9.13.3 of the research
repository: each registered kind declares, for every field of its config
(sources, detectors, compound components) or every argument of its
constructor (geometries, scatter models, spectra), whether a gradient-carrying
value is *attached* (with its class of chapter 09 and the stage its derivative
enters), *detached* (not a differentiable quantity: a count, a mode, a name),
or *refused* (a differentiable quantity whose derivative is not built). The
parameter register reads these rules to classify a parameter, and the kind
registry's gradient rule refuses a detached or refused field by its reason.
The test ``tests/nonsequential/test_nsq_kind_gradient_rules.py`` walks every
registered kind and fails on a field without a rule.

The ``attached`` lists the kinds were registered with are kept, and must name
exactly the attached rules of a kind that states a list (the test checks it).
"""

from __future__ import annotations

from optiland.nonsequential.kinds import attached, detached, refused

IB = "interior+boundary"
IN = "interior"
BO = "boundary-only"

# -- shared wording -------------------------------------------------------------

_FLUX = attached(IN, "the birth weight")
_FLUX_LM = attached(IN, "the birth weight, through the conversion of lumens to watts")
_SPECTRUM = refused("the wavelength is drawn by an inverse distribution on the host; a spectrum is data")
_MEDIUM = detached("a material record; its index is a material parameter, registered where a surface reads it")
_CHANGE_POINTS = attached(IB, "the source's change of variables (emission points, R-09-4)")
_CHANGE_DIRECTIONS = attached(IB, "the source's change of variables (emission directions, R-09-4)")
_COUNT = detached("a count")
_SWITCH = detached("a switch")
_MODE = detached("a choice of mode")
_SETTING = detached("a setting of the algorithm, not a design quantity")
_SURFACE_CONFIG = detached(
    "a surface's configuration (coating, scatter, aperture); its own values carry their rules"
)
_MATERIAL = detached("a material name or record; its index is a material parameter of the surfaces")
_KIND_CHOICE = detached("a choice of surface kind")
_CURVATURE = attached(IB, "the intersection and the normal (sag, curvature)")
_CONIC = attached(IB, "the intersection and the normal (sag)")
_COEFFICIENTS = attached(IB, "the intersection and the normal (sag)")
_APERTURE = attached(BO, "the clear-aperture test (a boolean)")
_EXTENT = attached(BO, "the finite-extent test (a boolean)")
_DETECTOR_EXTENT = attached(IB, "the binning coordinate (pixel pitch) and the edge")
_SPLAT_SIGMA = refused("the Gaussian splat's width is read as a number on the host")
_REFLECTION_BINS = detached("a count of reflection bins")
_SIDE = detached("a choice of the detecting side")
_ABSORB = _SWITCH

# -- sources ------------------------------------------------------------------------

SOURCES = {
    "point": {
        "spectrum": _SPECTRUM,
        "total_flux": _FLUX,
        "total_flux_lumens": _FLUX_LM,
        "half_angle_deg": _CHANGE_DIRECTIONS,
        "medium": _MEDIUM,
    },
    "collimated": {
        "spectrum": _SPECTRUM,
        "total_flux": _FLUX,
        "total_flux_lumens": _FLUX_LM,
        "aperture_radius": _CHANGE_POINTS,
        "profile": _MODE,
        "gaussian_sigma": attached(
            IB,
            "the source's change of variables (emission points, R-09-4): the truncated "
            "Gaussian's implicit reparameterisation, the default (profile_gradient='implicit'); "
            "refused at construction when the beam is built with profile_gradient='refuse'",
        ),
        "medium": _MEDIUM,
        "profile_gradient": _SWITCH,
    },
    "extended": {
        "spectrum": _SPECTRUM,
        "total_flux": _FLUX,
        "total_flux_lumens": _FLUX_LM,
        "width": _CHANGE_POINTS,
        "height": _CHANGE_POINTS,
        "aperture_radius": _CHANGE_POINTS,
        "half_angle_deg": _CHANGE_DIRECTIONS,
        "medium": _MEDIUM,
    },
    "tabulated": {
        "spectrum": _SPECTRUM,
        "polar_angles_deg": refused(
            "the table is sampled by an inverse distribution on the host; the implicit "
            "derivative of a table value is not built"
        ),
        "intensity": refused(
            "the table is sampled by an inverse distribution on the host; the implicit "
            "derivative of a table value is not built"
        ),
        "azimuth_angles_deg": refused(
            "the table is sampled by an inverse distribution on the host; the implicit "
            "derivative of a table value is not built"
        ),
        "intensity_units": _MODE,
        "total_flux": _FLUX,
        "total_flux_lumens": _FLUX_LM,
        "width": _CHANGE_POINTS,
        "height": _CHANGE_POINTS,
        "aperture_radius": _CHANGE_POINTS,
        "medium": _MEDIUM,
    },
}

# -- detectors ------------------------------------------------------------------------

_HOST_BINNED = refused("this kind bins its hits on the host; its extent has no derivative path")
_ANGLE_GRID = _COUNT

DETECTORS = {
    "irradiance": {
        "width": _DETECTOR_EXTENT,
        "height": _DETECTOR_EXTENT,
        "num_pixels_x": _COUNT,
        "num_pixels_y": _COUNT,
        "splat": _MODE,
        "splat_sigma": _SPLAT_SIGMA,
        "absorb": _ABSORB,
        "side": _SIDE,
        "reflection_bins": _REFLECTION_BINS,
        "stokes": _SWITCH,
    },
    "spectral": {
        "width": refused("the spectral detector's extent is read as a number on the host"),
        "height": refused("the spectral detector's extent is read as a number on the host"),
        "num_pixels_x": _COUNT,
        "num_pixels_y": _COUNT,
        "wl_min": refused("a wavelength bin edge, read as a number"),
        "wl_max": refused("a wavelength bin edge, read as a number"),
        "num_bins": _COUNT,
        "splat": _MODE,
        "splat_sigma": _SPLAT_SIGMA,
        "absorb": _ABSORB,
    },
    "far_field": {
        "num_theta": _ANGLE_GRID,
        "num_phi": _ANGLE_GRID,
        "absorb": _ABSORB,
        "side": _SIDE,
        "reflection_bins": _REFLECTION_BINS,
    },
    "hemisphere": {
        "radius": attached(IB, "the binning coordinate and the edge"),
        "num_theta": _ANGLE_GRID,
        "num_phi": _ANGLE_GRID,
        "absorb": _ABSORB,
        "reflection_bins": _REFLECTION_BINS,
    },
    "colorimetric": {
        "width": _DETECTOR_EXTENT,
        "height": _DETECTOR_EXTENT,
        "num_pixels_x": _COUNT,
        "num_pixels_y": _COUNT,
        "splat": _MODE,
        "splat_sigma": _SPLAT_SIGMA,
        "absorb": _ABSORB,
        "side": _SIDE,
        "reflection_bins": _REFLECTION_BINS,
        "stokes": _SWITCH,
    },
    "colorimetric_far_field": {
        "num_theta": _ANGLE_GRID,
        "num_phi": _ANGLE_GRID,
        "absorb": _ABSORB,
        "side": _SIDE,
        "reflection_bins": _REFLECTION_BINS,
    },
    "ray_database": {
        "width": _HOST_BINNED,
        "height": _HOST_BINNED,
        "max_rays": _COUNT,
        "absorb": _ABSORB,
    },
}

# -- geometries (constructor arguments) ---------------------------------------------------

_ASPHERE = {
    "radius": _CURVATURE,
    "conic": _CONIC,
    "aperture_radius": _APERTURE,
    "coefficients": _COEFFICIENTS,
    "max_iterations": _SETTING,
    "guard_eta": _SETTING,
    "residual_k": _SETTING,
    "scan_samples": _SETTING,
}

GEOMETRIES = {
    "conic": {"radius": _CURVATURE, "conic": _CONIC, "aperture_radius": _APERTURE},
    "paraboloid": {"radius": _CURVATURE, "aperture_radius": _APERTURE},
    "plane": {"width": _EXTENT, "height": _EXTENT, "aperture_radius": _APERTURE},
    "infinite_plane": {},
    "annulus": {
        "inner_radius": attached(BO, "the annulus test (a boolean)"),
        "outer_radius": attached(BO, "the annulus test (a boolean)"),
        "z_offset": attached(IB, "the intersection"),
    },
    "frustum": {
        "r_front": attached(IB, "the intersection and the normal"),
        "r_back": attached(IB, "the intersection and the normal"),
        "z_front": attached(IB, "the intersection"),
        "z_back": attached(IB, "the intersection"),
    },
    "sphere": {"radius": _CURVATURE, "aperture_radius": _APERTURE},
    "spherical_cavity": {
        "radius": _CURVATURE,
        "ports": refused("the ports' positions and radii are read as numbers on the host"),
    },
    "mesh": {"mesh": refused("a mesh's vertices are not attached in this release")},
    "even_asphere": dict(_ASPHERE),
    "odd_asphere": dict(_ASPHERE),
    "nurbs": {
        "arrays": detached("the net in array form; its control points and weights are the attached parameters"),
        "control_points": attached(IB, "the intersection and the normal (the NURBS net)"),
        "weights": attached(IB, "the intersection and the normal (the NURBS weights)"),
        **{
            name: _SETTING
            for name in (
                "cone_deg",
                "tangent_deg",
                "max_depth",
                "fill_min",
                "n_iter",
                "k_tol",
                "n_candidates",
                "n_ambiguous",
                "n_candidates_2",
                "n_ambiguous_2",
                "second_round_share",
            )
        },
    },
    "lenslet_array": {
        "pitch_x": attached(IB, "the lenslet cell frame"),
        "pitch_y": attached(IB, "the lenslet cell frame"),
        "radius": _CURVATURE,
        "conic": _CONIC,
        "num_x": _COUNT,
        "num_y": _COUNT,
        "sag_offsets": refused("the per-cell offsets are kept as a host table"),
    },
}

# -- scatter models (constructor arguments) -------------------------------------------------

# Chapter 09 section 9.14.3 of the research repository (issue 93): the branch
# is drawn with the fraction's host value and each branch's weight carries its
# share; the lobe's width and slope enter the weight through d log f at the
# drawn direction (the likelihood-ratio path), the direction staying detached.
_TRANSMISSIVE = attached(
    IN,
    "the scatter weight: each branch's share of the reflect-or-transmit split "
    "(tau / p, (1 - tau) / (1 - p)), drawn with the detached p",
)
_LOBE_SHAPE = attached(
    IN,
    "the scatter weight: w d log f / d theta at the drawn direction (the "
    "likelihood-ratio path; the direction is drawn detached)",
)

BSDFS = {
    "lambertian": {
        "reflectance_value": attached(IN, "the scatter weight"),
        "transmissive_fraction": _TRANSMISSIVE,
    },
    "harvey_shack": {
        "b0": refused(
            "zero by structure: the lobe is normalised over the reachable directions, "
            "so b0 cancels from the drawn direction and from the weight"
        ),
        "l0": _LOBE_SHAPE,
        "s": _LOBE_SHAPE,
        "transmissive_fraction": _TRANSMISSIVE,
    },
    "tabulated": {"path": detached("a file"), "transmissive_fraction": _TRANSMISSIVE},
    "specular": {},
}

# -- compound components (config fields) -------------------------------------------------------

_THICKNESS = attached(IB, "the back surface's placement, through the reference chain")

COMPONENTS = {
    "lens": {
        "r1": _CURVATURE,
        "r2": _CURVATURE,
        "thickness": _THICKNESS,
        "material": _MATERIAL,
        "front_aperture_radius": _APERTURE,
        "back_aperture_radius": _APERTURE,
        "conic1": _CONIC,
        "conic2": _CONIC,
        "front": _SURFACE_CONFIG,
        "back": _SURFACE_CONFIG,
        "edge": _SURFACE_CONFIG,
        "rim": _SURFACE_CONFIG,
        "coefficients1": _COEFFICIENTS,
        "coefficients2": _COEFFICIENTS,
        "odd1": _KIND_CHOICE,
        "odd2": _KIND_CHOICE,
    },
    "mirror": {
        "radius": _CURVATURE,
        "reflectance": attached(IN, "the reflect weight"),
        "conic": _CONIC,
        "aperture_radius": _APERTURE,
        "surface": _SURFACE_CONFIG,
        "coefficients": _COEFFICIENTS,
        "odd": _KIND_CHOICE,
        "nurbs": detached("a NURBS net given as a record; its control points and weights are the geometry's"),
    },
    "doublet": {
        "r1": _CURVATURE,
        "r2": _CURVATURE,
        "r3": _CURVATURE,
        "thickness1": _THICKNESS,
        "thickness2": _THICKNESS,
        "material1": _MATERIAL,
        "material2": _MATERIAL,
        "aperture_radius": _APERTURE,
        "conic1": _CONIC,
        "conic2": _CONIC,
        "conic3": _CONIC,
        "front": _SURFACE_CONFIG,
        "cemented": _SURFACE_CONFIG,
        "back": _SURFACE_CONFIG,
        "edge": _SURFACE_CONFIG,
        "coefficients1": _COEFFICIENTS,
        "coefficients2": _COEFFICIENTS,
        "coefficients3": _COEFFICIENTS,
        "odd1": _KIND_CHOICE,
        "odd2": _KIND_CHOICE,
        "odd3": _KIND_CHOICE,
    },
    "prism": {
        "apex_angle_deg": refused("the prism's faces are built from the host value"),
        "face_length": refused("the prism's faces are built from the host value"),
        "length": refused("the prism's faces are built from the host value"),
        "material": _MATERIAL,
        "open_base": _SWITCH,
        "front": _SURFACE_CONFIG,
        "back": _SURFACE_CONFIG,
        "base": _SURFACE_CONFIG,
    },
    "paraxial_lens": {
        "focal_length": attached(IB, "the paraxial deflection"),
        "aperture_radius": refused("an edge, and the stop's annulus is built from its host value"),
        "stop_radius": refused("an edge, and the stop's annulus is built from its host value"),
    },
    "polarizer": {
        "axis_deg": attached(IN, "the polarizing element's frame rotation"),
        "aperture_radius": _APERTURE,
        "extinction": attached(IN, "the polarizer's blocked-axis transmittance"),
    },
    "retarder": {
        "fast_axis_deg": attached(IN, "the polarizing element's frame rotation"),
        "retardance_waves": attached(IN, "the retarder's phase"),
        "aperture_radius": _APERTURE,
        "design_wavelength_um": refused("read as a number on the host"),
    },
}

# -- spectra (constructor arguments) -----------------------------------------------------------

SPECTRA = {
    "lines": {"wavelengths": _SPECTRUM, "weights": _SPECTRUM},
    "piecewise_linear": {
        "wavelengths": _SPECTRUM,
        "values": _SPECTRUM,
        "label": detached("a name"),
    },
}

RULES = {
    "source": SOURCES,
    "detector": DETECTORS,
    "geometry": GEOMETRIES,
    "bsdf": BSDFS,
    "component": COMPONENTS,
    "spectrum": SPECTRA,
}


def apply() -> None:
    """Attach the rules above to the built-in kinds (called by ``register_all``)."""
    from optiland.nonsequential import kinds  # noqa: PLC0415

    for family, by_kind in RULES.items():
        registry = kinds.registry(family)
        for name, rules in by_kind.items():
            registry.set_gradients(name, rules)
