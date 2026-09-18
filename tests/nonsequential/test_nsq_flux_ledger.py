"""The flux ledger closes on scenes that used to report an energy defect.

``docs/theory/10_ledger_and_diagnostics.md`` (10.1) makes every watt a
source emits land in exactly one of eight destinations, and (10.2) makes
the identity close on every realisation rather than only in expectation.
Three things were missing, each of which made the engine report a large
energy error on a physically correct scene -- which teaches a user to
ignore the diagnostic:

- a mirror below unit reflectance, and a coating whose reflectance and
  transmittance do not sum to one, removed flux with nowhere to book it;
- a transmissive detector's reading was counted as a destination even
  though the ray carried on, so the same watt was booked twice;
- Russian roulette booked the weight it killed but not the boost it handed
  the survivors, which is the naive booking of section 10.2.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQScene,
    ReflectiveComponent,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry

# Section 10.4's fail threshold for ledger closure.
_CLOSURE_FAIL = 1e-11


def _two_mirror_cavity(reflectance: float) -> NSQScene:
    """Two facing mirrors with the beam launched between them.

    Every watt ends in a coating (or, for the tail of the series, at the
    depth cap), so the scene is the sharpest possible test of the coating
    bin: leave it out and the ledger is wrong by the whole emitted flux.
    """
    scene = NSQScene()
    spec = Spectrum.monochromatic(0.55)
    scene.add_source(
        "S",
        CoordinateSystem(z=10.0),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=2.0),
    )
    scene.add_component(
        "M1",
        ReflectiveComponent(
            CoordinateSystem(z=20.0),
            PlaneGeometry(),
            reflectance=reflectance,
            name="M1",
        ),
    )
    scene.add_component(
        "M2",
        ReflectiveComponent(
            CoordinateSystem(z=5.0),
            PlaneGeometry(),
            reflectance=reflectance,
            name="M2",
        ),
    )
    # The scene needs a detector; this one is placed where no ray reaches
    # it, so it takes nothing out of the cavity.
    scene.add_detector(
        "D",
        CoordinateSystem(x=500.0, z=-500.0),
        IrradianceDetectorConfig(
            width=1, height=1, num_pixels_x=2, num_pixels_y=2
        ),
    )
    return scene


def _tapped_beam() -> NSQScene:
    """A transmissive detector reading a beam that then lands on a screen."""
    scene = NSQScene()
    spec = Spectrum.monochromatic(0.55)
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=1.0),
    )
    scene.add_detector(
        "tap",
        CoordinateSystem(z=10),
        IrradianceDetectorConfig(
            width=10, height=10, num_pixels_x=32, num_pixels_y=32, absorb=False
        ),
    )
    scene.add_detector(
        "screen",
        CoordinateSystem(z=20),
        IrradianceDetectorConfig(
            width=10, height=10, num_pixels_x=32, num_pixels_y=32
        ),
    )
    return scene


class TestCoatingBin:
    """A mirror's loss has a destination."""

    def test_two_mirror_cavity_closes(self):
        scene = _two_mirror_cavity(0.5)
        result = scene.trace(
            num_rays=100_000, seed=42, max_depth=16, backend=NumpyBackend(seed=42)
        )
        assert result.flux_conservation_error < _CLOSURE_FAIL
        # Essentially all of it: the mirrors take everything but the tail
        # the depth cap truncates.
        assert result.total_flux_coating == pytest.approx(1.0, abs=1e-3)
        assert result.total_flux_detected == pytest.approx(0.0)

    @pytest.mark.parametrize("reflectance", [0.2, 0.5, 0.9, 1.0])
    def test_closure_holds_across_reflectance(self, reflectance):
        scene = _two_mirror_cavity(reflectance)
        result = scene.trace(
            num_rays=20_000, seed=3, max_depth=12, backend=NumpyBackend(seed=3)
        )
        assert result.flux_conservation_error < _CLOSURE_FAIL

    def test_a_perfect_mirror_books_nothing(self):
        scene = _two_mirror_cavity(1.0)
        result = scene.trace(
            num_rays=20_000, seed=3, max_depth=12, backend=NumpyBackend(seed=3)
        )
        assert result.total_flux_coating == pytest.approx(0.0, abs=1e-12)

    def test_the_loss_is_reported_as_its_own_destination(self):
        scene = _two_mirror_cavity(0.5)
        result = scene.trace(
            num_rays=20_000, seed=3, max_depth=12, backend=NumpyBackend(seed=3)
        )
        # Surface absorption is a different bin: nothing here is an
        # AbsorbingComponent.
        assert result.total_flux_absorbed == pytest.approx(0.0)
        assert result.total_flux_coating > 0.9
        assert "coating_loss_flux_fraction" in result.report()


class TestTransmissiveDetectorIsNotADestination:
    """A tap reads the beam; it does not remove it from the trace."""

    def test_tapped_beam_closes(self):
        result = _tapped_beam().trace(
            num_rays=50_000, seed=1, backend=NumpyBackend(seed=1)
        )
        assert result.flux_conservation_error < _CLOSURE_FAIL

    def test_both_readings_are_still_reported(self):
        result = _tapped_beam().trace(
            num_rays=50_000, seed=1, backend=NumpyBackend(seed=1)
        )
        # The tap and the screen each read the whole beam, and
        # total_flux_detected is still the sum of what every detector read.
        assert result.detectors["tap"].total_flux_float == pytest.approx(1.0, rel=1e-6)
        assert result.detectors["screen"].total_flux_float == pytest.approx(
            1.0, rel=1e-6
        )
        assert result.total_flux_detected == pytest.approx(2.0, rel=1e-6)
        assert result.total_flux_tapped == pytest.approx(1.0, rel=1e-6)

    def test_an_absorbing_detector_taps_nothing(self):
        scene = _tapped_beam()
        scene.detectors[0].absorb = True
        result = scene.trace(num_rays=50_000, seed=1, backend=NumpyBackend(seed=1))
        assert result.total_flux_tapped == pytest.approx(0.0)
        assert result.flux_conservation_error < _CLOSURE_FAIL


class TestSamplingResidual:
    """Roulette's boost is booked, not only its kill."""

    def test_residual_carries_the_roulette_boost(self):
        # A deep cavity with an aggressive roulette threshold, so roulette
        # fires often and the naive booking would be visibly wrong.
        scene = _two_mirror_cavity(0.5)
        result = scene.trace(
            num_rays=100_000,
            seed=7,
            max_depth=24,
            min_flux_fraction=0.5,
            backend=NumpyBackend(seed=7),
        )
        assert result.total_flux_lost > 0.0, "roulette did not fire"
        assert result.flux_conservation_error < _CLOSURE_FAIL

    def test_residual_shrinks_with_ray_count(self):
        """Section 10.2: |Phi_samp| / Phi_emit falls as N grows."""
        fractions = []
        for n in (10_000, 640_000):
            result = _two_mirror_cavity(0.5).trace(
                num_rays=n,
                seed=11,
                max_depth=24,
                min_flux_fraction=0.5,
                backend=NumpyBackend(seed=11),
            )
            fractions.append(
                abs(result.diagnostics.sampling_residual_flux_fraction)
            )
        # 64x the rays; exact N^-1/2 would be 8x. Asserting only that it
        # shrinks by at least 3 leaves room for the correlation between the
        # several roulette events one ray sees (section 10.2 measures the
        # scaling exponent drifting 13% over this range).
        assert fractions[0] > 3.0 * fractions[1]


class TestTorchAgrees:
    """The same books on the device backend."""

    def test_cavity_closes_on_torch(self):
        torch = pytest.importorskip("torch", reason="Torch not available")
        assert torch is not None
        import optiland.backend as be
        from optiland.nonsequential.backends.torch_backend import TorchBackend

        be.set_backend("torch")
        be.set_precision("float64")
        try:
            result = _two_mirror_cavity(0.5).trace(
                num_rays=20_000,
                seed=42,
                max_depth=12,
                backend=TorchBackend(seed=42),
            )
        finally:
            be.set_backend("numpy")
        assert result.flux_conservation_error < _CLOSURE_FAIL
        assert result.total_flux_coating > 0.9


def _scattering_mirror(bsdf, scatter_fraction: float = 1.0) -> NSQScene:
    """A beam onto one mirror carrying ``bsdf``, and a collector around it.

    The mirror is perfectly reflective, so every watt the ledger books
    against the surface came from the lobe's weight and nothing else.
    """
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55),
            total_flux=1.0,
            aperture_radius=2.0,
        ),
    )
    scene.add_component(
        "M",
        ReflectiveComponent(
            CoordinateSystem(z=20.0),
            PlaneGeometry(),
            reflectance=1.0,
            bsdf=bsdf,
            name="M",
            scatter_fraction=scatter_fraction,
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=-40.0, rx=3.141592653589793),
        IrradianceDetectorConfig(
            width=400, height=400, num_pixels_x=8, num_pixels_y=8
        ),
    )
    return scene


def _tabulated_bsdf(tmp_path, transmissive_fraction: float = 0.0):
    """A measured lobe whose weight varies strongly with the scatter angle."""
    from optiland.nonsequential import TabulatedBSDF

    path = tmp_path / "scatter.csv"
    rows = []
    for ti in (0.0, 45.0, 90.0):
        for ts, value in ((0.0, 0.30), (45.0, 0.12), (90.0, 0.01)):
            rows.append(f"{ti},{ts},{value}")
    path.write_text("\n".join(rows) + "\n")
    return TabulatedBSDF(path, transmissive_fraction=transmissive_fraction)


class TestLobeWeightIsNotAlwaysAnAlbedo:
    """``BaseBSDF.weight_is_albedo``: which bin ``1 - weight`` belongs in.

    A lobe that draws its direction from one distribution and corrects with
    ``f / pdf`` hands one ray far more than the surface's reflectance and
    the next ray far less. Booking ``1 - weight`` as absorption then makes
    the coating bin a random variable. The flag splits it: ``1 -
    reflectance()`` is the physical loss and the rest is the event residual
    of section 10.2.
    """

    def test_the_default_says_the_weight_is_an_albedo(self):
        from optiland.nonsequential import (
            HarveyShackBSDF,
            LambertianBSDF,
            SpecularBRDF,
        )

        assert LambertianBSDF().weight_is_albedo
        assert SpecularBRDF().weight_is_albedo
        assert HarveyShackBSDF(b0=1e-3, l0=0.05, s=2.0).weight_is_albedo

    def test_the_tabulated_lobe_says_it_is_a_sampling_weight(self, tmp_path):
        assert not _tabulated_bsdf(tmp_path).weight_is_albedo

    def test_the_identity_still_closes(self, tmp_path):
        """Whichever bin the flux lands in, the eight of them still sum."""
        result = _scattering_mirror(_tabulated_bsdf(tmp_path)).trace(
            num_rays=20_000, seed=5, max_depth=8, backend=NumpyBackend(seed=5)
        )
        assert result.flux_conservation_error < _CLOSURE_FAIL

    def test_the_fluctuation_is_booked_as_a_residual_not_as_absorption(
        self, tmp_path
    ):
        """The split puts something in each bin, and the residual shrinks.

        With the flag off, every watt below would be in the coating bin and
        the residual would be exactly the roulette term. The residual's
        documented property is that it falls as the square root of the ray
        count, so measuring it at two ray counts is the check that it is
        sampling noise rather than a weight-update defect.
        """
        counts = (20_000, 320_000)
        residuals = []
        coatings = []
        for n in counts:
            result = _scattering_mirror(_tabulated_bsdf(tmp_path)).trace(
                num_rays=n, seed=5, max_depth=8, backend=NumpyBackend(seed=5)
            )
            assert result.flux_conservation_error < _CLOSURE_FAIL
            assert result.total_flux_coating > 0.0
            residuals.append(abs(result.total_flux_sampling_residual))
            coatings.append(result.total_flux_coating)
        assert residuals[0] > 0.0
        # 16x the rays, so 4x smaller if it is sampling noise. Allow a
        # generous factor: the point is that it shrinks, not its constant.
        assert residuals[1] < residuals[0] / 2.0

    def test_the_physical_bin_stops_depending_on_the_ray_count(self, tmp_path):
        """The measurement the flag exists for.

        The coating bin is what the surface absorbed, so it must converge
        immediately, not as the estimator does. Measured on this scene:
        with the lobe declaring its weight an albedo the bin reads 0.579844
        W at 20 000 rays and 0.582695 W at 1 280 000, still climbing toward
        the surface's true 0.583052; with the flag set it reads 0.583025 and
        0.583052 -- the same number at both counts, because the fluctuation
        is in the residual where it belongs.
        """
        coatings = []
        for n in (20_000, 320_000):
            result = _scattering_mirror(_tabulated_bsdf(tmp_path)).trace(
                num_rays=n, seed=5, max_depth=8, backend=NumpyBackend(seed=5)
            )
            coatings.append(result.total_flux_coating)
        # 4e-5 apart with the flag set, 3.9e-3 apart without it.
        assert coatings[1] == pytest.approx(coatings[0], rel=1e-3)
