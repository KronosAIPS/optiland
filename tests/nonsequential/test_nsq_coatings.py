"""Coating and mirror-reflectance tests for Non-Sequential Raytracing.

Covers D-2 (RefractiveComponent honoring an attached optiland.coatings
coating instead of the bare Fresnel split) and D-3 (ReflectiveComponent
requiring an explicit reflectance instead of an implicit perfect mirror), and
the angle-dependent thin-film adapter, UnpolarizedThinFilmCoating (X5): its
own (R, T) against an independent numpy characteristic-matrix evaluation,
and a full ray-traced scene against the same reference.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optiland.coatings import FresnelCoating, SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    MirrorConfig,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential.components.coating_support import (
    UnpolarizedThinFilmCoating,
)
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry
from optiland.nonsequential.components.reflective import ReflectiveComponent
from optiland.thin_film import ThinFilmStack

GREEN = 0.55


def _glass():
    return NSQMaterial.from_glass("N-BK7")


def _coated_interface_scene(coating, *, num_rays=60_000):
    """Collimated beam at normal incidence on a single coated interface.

    Returns (reflected_total_flux, transmitted_total_flux, input_flux).
    """
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-5.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN),
            total_flux=1.0,
            aperture_radius=1.0,
        ),
    )
    scene.add_component(
        "I",
        RefractiveComponent(
            cs=CoordinateSystem(z=0.0),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=_glass(),
            coating=coating,
            name="I",
        ),
    )
    scene.add_detector(
        "R",
        CoordinateSystem(z=-15.0),
        IrradianceDetectorConfig(width=10, height=10, num_pixels_x=8, num_pixels_y=8),
    )
    scene.add_detector(
        "T",
        CoordinateSystem(z=15.0),
        IrradianceDetectorConfig(width=10, height=10, num_pixels_x=8, num_pixels_y=8),
    )
    result = scene.trace(num_rays=num_rays, seed=7)
    return (
        result.detectors["R"].total_flux,
        result.detectors["T"].total_flux,
        1.0,
    )


class TestCoatingOverridesFresnel:
    def test_coating_reflectance_and_transmittance_match_flux_split(self):
        """NSQ's coating-driven R/T must agree with the SimpleCoating values,
        not the bare (uncoated) Fresnel reflectance at this interface."""
        coating = SimpleCoating(transmittance=0.85, reflectance=0.10)
        r_flux, t_flux, in_flux = _coated_interface_scene(coating)

        assert r_flux == pytest.approx(coating.reflectance * in_flux, abs=0.01)
        assert t_flux == pytest.approx(coating.transmittance * in_flux, abs=0.01)
        # Absorptance (0.05) is unaccounted-for flux, not a third detector hit.
        assert r_flux + t_flux == pytest.approx(
            (coating.reflectance + coating.transmittance) * in_flux, abs=0.01
        )

    def test_uncoated_interface_still_uses_bare_fresnel(self):
        """No coating attached -> unchanged pre-PR7 behaviour: R+T == 1."""
        r_flux, t_flux, in_flux = _coated_interface_scene(coating=None)
        assert r_flux + t_flux == pytest.approx(in_flux, abs=0.01)
        # Normal incidence, VACUUM -> N-BK7: small but nonzero Fresnel R.
        assert 0.0 < r_flux < 0.1

    def test_polarized_coating_on_refractive_surface_raises(self):
        with pytest.raises(NotImplementedError, match="polarized"):
            RefractiveComponent(
                cs=CoordinateSystem(z=0.0),
                geometry=PlaneGeometry(),
                material_front=VACUUM,
                material_back=_glass(),
                coating=FresnelCoating(None, None),
                name="I",
            )


class TestMirrorReflectanceIsRequired:
    def test_reflective_component_requires_reflectance(self):
        with pytest.raises(TypeError):
            ReflectiveComponent(cs=CoordinateSystem(z=0.0), geometry=PlaneGeometry())

    def test_reflective_component_rejects_polarized_coating(self):
        coating = FresnelCoating(None, None)
        with pytest.raises(NotImplementedError, match="polarized"):
            ReflectiveComponent(
                cs=CoordinateSystem(z=0.0),
                geometry=PlaneGeometry(),
                reflectance=coating,
            )

    def test_constant_reflectance_scales_flux(self):
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-10.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN),
                total_flux=1.0,
                aperture_radius=1.0,
            ),
        )
        scene.add_mirror(
            "M",
            CoordinateSystem(z=0.0),
            MirrorConfig(radius=np.inf, reflectance=0.5, aperture_radius=10.0),
        )
        scene.add_detector(
            "R",
            CoordinateSystem(z=-20.0),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=20_000, seed=3)
        assert result.detectors["R"].total_flux == pytest.approx(0.5, abs=0.02)

    def test_coating_reflectance_on_mirror_scales_flux(self):
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-10.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(GREEN),
                total_flux=1.0,
                aperture_radius=1.0,
            ),
        )
        scene.add_mirror(
            "M",
            CoordinateSystem(z=0.0),
            MirrorConfig(
                radius=np.inf,
                reflectance=SimpleCoating(transmittance=0.0, reflectance=0.8),
                aperture_radius=10.0,
            ),
        )
        scene.add_detector(
            "R",
            CoordinateSystem(z=-20.0),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=8, num_pixels_y=8
            ),
        )
        result = scene.trace(num_rays=20_000, seed=3)
        assert result.detectors["R"].total_flux == pytest.approx(0.8, abs=0.02)

    def test_make_surface_reflective_override_without_reflectance_raises(self):
        from optiland.nonsequential.components.configs import (
            InteractionType,
            SurfaceConfig,
        )
        from optiland.nonsequential.components.lens import _make_surface

        with pytest.raises(ValueError, match="reflectance"):
            _make_surface(
                CoordinateSystem(z=0.0),
                PlaneGeometry(),
                VACUUM,
                VACUUM,
                SurfaceConfig(interaction=InteractionType.REFLECTIVE),
                InteractionType.ABSORBING,
                name="edge",
            )


# ---------------------------------------------------------------------------
# X5: UnpolarizedThinFilmCoating -- an independent numpy characteristic-matrix
# reference, from the theory chapter's formulas (04_surfaces.md sec 4.4), not
# the library's own optiland.thin_film._tmm_coh. Both the pure-adapter tests
# below and the ray-traced ones compare against this, not against each other.
# ---------------------------------------------------------------------------
def _charmat_R_T(
    n0: float,
    n_film: float,
    n_sub: float,
    thickness_um: float,
    wavelength_um: float,
    theta_i_rad: float,
    pol: str,
) -> tuple[float, float]:
    """Single-layer characteristic-matrix (R, T) for one polarization.

    Reproduces docs/theory/04_surfaces.md sec 4.4 directly: tilted optical
    admittance eta_s = n cos(theta), eta_p = n / cos(theta), phase thickness
    delta = 2 pi n d cos(theta) / lambda, one layer matrix M, and
    r = (eta0 B - C) / (eta0 B + C), T = 4 eta0 Re(eta_sub) / |eta0 B + C|^2.
    All media here are real (lossless, propagating -- no evanescent branch
    handling needed).
    """
    sin_i = math.sin(theta_i_rad)
    cos_i = math.cos(theta_i_rad)
    cos_film = math.sqrt(1.0 - (n0 * sin_i / n_film) ** 2)
    cos_sub = math.sqrt(1.0 - (n0 * sin_i / n_sub) ** 2)

    def eta(n: float, cos_t: float) -> float:
        return n * cos_t if pol == "s" else n / cos_t

    eta0 = eta(n0, cos_i)
    eta_f = eta(n_film, cos_film)
    eta_sub = eta(n_sub, cos_sub)

    delta = 2.0 * math.pi * n_film * thickness_um * cos_film / wavelength_um
    c, s = math.cos(delta), math.sin(delta)
    m11, m12 = c, 1j * s / eta_f
    m21, m22 = 1j * eta_f * s, c

    b = m11 * 1.0 + m12 * eta_sub
    cc = m21 * 1.0 + m22 * eta_sub

    r = (eta0 * b - cc) / (eta0 * b + cc)
    R = abs(r) ** 2
    T = 4.0 * eta0 * eta_sub.real / abs(eta0 * b + cc) ** 2
    return R, T


def _charmat_unpolarized(
    n0: float,
    n_film: float,
    n_sub: float,
    thickness_um: float,
    wavelength_um: float,
    theta_i_rad: float,
) -> tuple[float, float]:
    """(R, T) averaged over s and p -- the unpolarized reduction."""
    Rs, Ts = _charmat_R_T(n0, n_film, n_sub, thickness_um, wavelength_um, theta_i_rad, "s")
    Rp, Tp = _charmat_R_T(n0, n_film, n_sub, thickness_um, wavelength_um, theta_i_rad, "p")
    return 0.5 * (Rs + Rp), 0.5 * (Ts + Tp)


# N-BK7 at the d-line, and the ideal quarter-wave AR layer for it: index
# sqrt(n_sub), one quarter wave thick at the design wavelength.
_N0 = 1.0
_N_SUB = 1.5168
_WL0 = 0.5876
_N_FILM = math.sqrt(_N0 * _N_SUB)


def _qwot_thickness_um(n_film: float, wavelength_um: float) -> float:
    """Quarter-wave optical thickness at normal incidence: lambda / (4 n)."""
    return wavelength_um / (4.0 * n_film)


def _ar_stack() -> ThinFilmStack:
    stack = ThinFilmStack(
        IdealMaterial(_N0), IdealMaterial(_N_SUB), reference_wl_um=_WL0
    )
    stack.add_layer_qwot(IdealMaterial(_N_FILM), qwot_thickness=1.0)
    return stack


class TestUnpolarizedThinFilmCoatingAdapter:
    """Pure-adapter tests: no ray tracing, just UnpolarizedThinFilmCoating.evaluate."""

    def test_empty_stack_reproduces_bare_fresnel(self):
        """No layers -> the unpolarized Fresnel reflectance of n0/n_sub alone."""
        stack = ThinFilmStack(IdealMaterial(_N0), IdealMaterial(_N_SUB))
        coating = UnpolarizedThinFilmCoating(stack)

        for theta_deg in (0.0, 30.0, 60.0):
            theta = math.radians(theta_deg)
            R, T = coating.evaluate(
                np.array([_WL0]), np.array([math.cos(theta)])
            )
            R_ref, T_ref = _charmat_unpolarized(_N0, _N0, _N_SUB, 0.0, _WL0, theta)
            assert R[0] == pytest.approx(R_ref, abs=1e-12)
            assert T[0] == pytest.approx(T_ref, abs=1e-12)

        # And the textbook normal-incidence closed form.
        R0, _ = coating.evaluate(np.array([_WL0]), np.array([1.0]))
        bare = ((_N0 - _N_SUB) / (_N0 + _N_SUB)) ** 2
        assert R0[0] == pytest.approx(bare, abs=1e-12)

    def test_quarter_wave_layer_zero_reflectance_at_design_wavelength(self):
        """R < 1e-12 at normal incidence, design wavelength -- the r1_13 case."""
        coating = UnpolarizedThinFilmCoating(_ar_stack())
        R, T = coating.evaluate(np.array([_WL0]), np.array([1.0]))
        assert R[0] < 1e-12
        assert T[0] == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("theta_deg", [30.0, 60.0])
    def test_matches_characteristic_matrix_off_axis(self, theta_deg):
        """R(theta) at 30 and 60 degrees against the independent numpy TMM."""
        theta = math.radians(theta_deg)
        coating = UnpolarizedThinFilmCoating(_ar_stack())
        R, T = coating.evaluate(np.array([_WL0]), np.array([math.cos(theta)]))

        thickness_um = _qwot_thickness_um(_N_FILM, _WL0)
        R_ref, T_ref = _charmat_unpolarized(
            _N0, _N_FILM, _N_SUB, thickness_um, _WL0, theta
        )
        assert R[0] == pytest.approx(R_ref, rel=1e-9)
        assert T[0] == pytest.approx(T_ref, rel=1e-9)

    def test_r_theta_matches_theory_chapter_percentages(self):
        """Cross-check against 04_surfaces.md sec 4.4's tabulated numbers
        (n_sub=1.52 there, not N-BK7's 1.5168) at 30 and 60 degrees."""
        n_sub = 1.52
        n_film = math.sqrt(n_sub)
        wl0 = 0.55
        thickness_um = _qwot_thickness_um(n_film, wl0)
        stack = ThinFilmStack(
            IdealMaterial(1.0), IdealMaterial(n_sub), reference_wl_um=wl0
        )
        stack.add_layer_qwot(IdealMaterial(n_film), qwot_thickness=1.0)
        coating = UnpolarizedThinFilmCoating(stack)

        # (R_s, R_p) in percent from the theory chapter's table.
        expected_pct = {30.0: (0.129148, 0.061611), 60.0: (4.932438, 0.909815)}
        for theta_deg, (rs_pct, rp_pct) in expected_pct.items():
            theta = math.radians(theta_deg)
            R, _ = coating.evaluate(np.array([wl0]), np.array([math.cos(theta)]))
            expected_r = 0.5 * (rs_pct + rp_pct) / 100.0
            assert R[0] == pytest.approx(expected_r, rel=1e-4)

    @pytest.mark.parametrize("theta_deg", [0.0, 15.0, 30.0, 45.0, 60.0])
    def test_energy_conservation_lossless(self, theta_deg):
        """R + T == 1 for a lossless film at every angle tested."""
        theta = math.radians(theta_deg)
        coating = UnpolarizedThinFilmCoating(_ar_stack())
        R, T = coating.evaluate(np.array([_WL0]), np.array([math.cos(theta)]))
        assert R[0] + T[0] == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("theta_deg", [0.0, 30.0, 60.0])
    def test_energy_conservation_with_absorption(self, theta_deg):
        """R + T + A == 1 once the film carries an extinction coefficient."""
        theta = math.radians(theta_deg)
        lossy_film = IdealMaterial(_N_FILM, k=0.01)
        stack = ThinFilmStack(
            IdealMaterial(_N0), IdealMaterial(_N_SUB), reference_wl_um=_WL0
        )
        stack.add_layer_qwot(lossy_film, qwot_thickness=1.0)
        coating = UnpolarizedThinFilmCoating(stack)

        R, T = coating.evaluate(np.array([_WL0]), np.array([math.cos(theta)]))
        out = stack.compute_rtRTA_elementwise(
            np.array([_WL0]), np.array([theta]), polarization="u"
        )
        A = out["A"]
        assert R[0] + T[0] + A[0] == pytest.approx(1.0, abs=1e-10)
        # A lossy film must actually absorb something, or the test above
        # would pass trivially with A == 0.
        assert A[0] > 1e-6

    def test_vectorized_over_a_ray_bundle_with_mixed_wavelength_and_angle(self):
        """One call, many rays: each ray's own wavelength and angle, not a
        broadcast grid."""
        coating = UnpolarizedThinFilmCoating(_ar_stack())
        wavelengths = np.array([_WL0, _WL0, 0.6, 0.5])
        thetas_deg = np.array([0.0, 30.0, 15.0, 45.0])
        cos_thetas = np.cos(np.radians(thetas_deg))

        R, T = coating.evaluate(wavelengths, cos_thetas)
        assert R.shape == (4,)
        assert T.shape == (4,)

        for i in range(4):
            R_i, T_i = coating.evaluate(wavelengths[i : i + 1], cos_thetas[i : i + 1])
            assert R[i] == pytest.approx(R_i[0], abs=1e-14)
            assert T[i] == pytest.approx(T_i[0], abs=1e-14)


class TestUnpolarizedThinFilmCoatingAutograd:
    """Gradients through the adapter when the stack's own parameters are
    torch tensors -- the torch backend stands in for a differentiable
    device throughout this branch."""

    def setup_method(self):
        import optiland.backend as be  # noqa: PLC0415

        be.set_backend("torch")
        be.set_precision("float64")

    def teardown_method(self):
        import optiland.backend as be  # noqa: PLC0415

        be.set_backend("numpy")

    def test_gradient_flows_through_layer_thickness(self):
        torch = pytest.importorskip("torch", reason="Torch not available")
        from optiland.thin_film import Layer  # noqa: PLC0415

        thickness = torch.tensor(
            _qwot_thickness_um(_N_FILM, _WL0), dtype=torch.float64, requires_grad=True
        )
        stack = ThinFilmStack(IdealMaterial(_N0), IdealMaterial(_N_SUB))
        stack.layers.append(Layer(IdealMaterial(_N_FILM), thickness))
        coating = UnpolarizedThinFilmCoating(stack)

        wl = torch.tensor([_WL0, _WL0], dtype=torch.float64)
        cos_i = torch.tensor([1.0, math.cos(math.radians(30.0))], dtype=torch.float64)
        R, T = coating.evaluate(wl, cos_i)

        assert R.requires_grad, "R lost its grad_fn -- the stack detached the thickness"
        assert T.requires_grad
        R.sum().backward()
        assert thickness.grad is not None
        assert torch.isfinite(thickness.grad).all()
        # At the design wavelength and normal incidence R is at its minimum
        # w.r.t. thickness (the whole point of a quarter-wave AR layer), so
        # the gradient of R's sum is dominated by the off-axis (30 deg) ray.
        assert thickness.grad.item() != 0.0


class TestUnpolarizedThinFilmCoatingOnRefractiveComponent:
    """Integration: the adapter attached to a real RefractiveComponent."""

    def test_not_rejected_as_polarized(self):
        """UnpolarizedThinFilmCoating is not a BaseCoatingPolarized -- this
        must not raise, unlike optiland.coatings.ThinFilmCoating."""
        RefractiveComponent(
            cs=CoordinateSystem(z=0.0),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=NSQMaterial.from_glass("N-BK7"),
            coating=UnpolarizedThinFilmCoating(_ar_stack()),
            name="I",
        )

    def _traced_reflectance(self, theta_deg: float, num_rays: int, seed: int) -> float:
        """Trace a coated vacuum/N-BK7 interface tilted by theta_deg; return
        the measured unpolarized reflectance (1 - transmitted fraction),
        with a narrow beam aimed at a tilted detector plane sharing the
        interface's own orientation (every ray in a collimated beam sees the
        same incidence angle, so a wide beam buys nothing physically)."""
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(_WL0),
                total_flux=1.0,
                aperture_radius=0.02,
            ),
        )
        rx = math.radians(theta_deg)
        comp = RefractiveComponent(
            CoordinateSystem(z=10.0, rx=rx),
            PlaneGeometry(),
            material_front=VACUUM,
            material_back=NSQMaterial(optiland_material=IdealMaterial(_N_SUB)),
            coating=UnpolarizedThinFilmCoating(_ar_stack()),
            name="front",
        )
        scene.add_component("front", comp)
        scene.add_detector(
            "T",
            CoordinateSystem(z=10.001, rx=rx),
            IrradianceDetectorConfig(
                width=500, height=500, num_pixels_x=1, num_pixels_y=1, splat="hard"
            ),
        )
        result = scene.trace(num_rays=num_rays, seed=seed, max_depth=2)
        t_sim = result.total_flux_detected / result.total_flux_in
        return 1.0 - t_sim

    @pytest.mark.parametrize("theta_deg", [30.0, 60.0])
    def test_traced_reflectance_matches_characteristic_matrix(self, theta_deg):
        theta = math.radians(theta_deg)
        thickness_um = _qwot_thickness_um(_N_FILM, _WL0)
        R_ref, _ = _charmat_unpolarized(
            _N0, _N_FILM, _N_SUB, thickness_um, _WL0, theta
        )
        measured = self._traced_reflectance(theta_deg, num_rays=300_000, seed=11)
        # Roulette-resolved branch -> Monte Carlo noise; a generous absolute
        # tolerance well above the ~1e-3 sampling noise at 3e5 rays.
        assert measured == pytest.approx(R_ref, abs=0.01)

    def test_traced_reflectance_near_zero_at_design_wavelength_normal_incidence(self):
        """The r1_13 configuration: essentially nothing reflects."""
        measured = self._traced_reflectance(0.0, num_rays=50_000, seed=5)
        assert measured < 1e-6
