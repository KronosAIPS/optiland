"""X7 -- a ray that leaves a surface at a grazing angle must not come back.

The self-intersection accept threshold is a *path length*: a root is
rejected when ``t`` is below ``k`` ulps of the ray's coordinate. What it has
to reject is a *perpendicular* error -- the half-ulp by which the rebuilt hit
point misses the surface, sometimes on the side the ray is leaving. A ray
leaving at ``alpha`` above the surface plane converts that perpendicular
error into a path length ``delta_perp / sin(alpha)``, so the closer the exit
is to grazing, the larger the spurious root, and past about 89 degrees from
the normal it clears the threshold and the surface accepts the ray it has
just released. It then refracts or reflects a second time.

The cure is the one ``docs/theory/07_geometry.md`` section 7.7 calls cure 2
and R-07-6 requires: the outgoing origin is offset off the surface, along
the geometric normal, into the hemisphere it leaves into. It removes the
``1 / sin(alpha)`` amplification instead of raising ``k`` to chase it --
raising ``k`` would blind the engine to a genuinely nearby surface, which is
what wave 1's k = 16384 did.

Each test here carries its own control: the same scene with
``offset_from_surface`` disabled, asserted to fail the way the measurement
found it failing.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    MirrorConfig,
    NSQMaterial,
    NSQScene,
    RayDatabaseConfig,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential import _tol as tol
from optiland.nonsequential.components.base import BaseComponent
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry
from optiland.nonsequential.materials.nsq_material import VACUUM

#: N-BK7 at the d-line, pinned as a constant so the critical angle in this
#: file is the one the arithmetic below uses, not a dispersion fit's.
N_GLASS = 1.5168
WAVELENGTH = 0.5876


@pytest.fixture(autouse=True)
def _numpy_float64():
    be.set_backend("numpy")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


@pytest.fixture
def no_offset(monkeypatch):
    """Disable the R-07-6 origin offset, for the controls."""
    monkeypatch.setattr(
        BaseComponent, "offset_from_surface", lambda self, rays, n_geom, hit: None
    )


def _incidence_for_exit(exit_deg: float) -> float:
    """Internal angle whose refracted ray leaves the glass at ``exit_deg``."""
    return math.degrees(math.asin(math.sin(math.radians(exit_deg)) / N_GLASS))


def _fresnel_t_unpolarized(theta_i_deg: float) -> float:
    """Unpolarized transmittance of the glass-to-vacuum interface."""
    ci = math.cos(math.radians(theta_i_deg))
    st = N_GLASS * math.sin(math.radians(theta_i_deg))
    if st >= 1.0:
        return 0.0
    ct = math.sqrt(1.0 - st * st)
    rs = ((N_GLASS * ci - ct) / (N_GLASS * ci + ct)) ** 2
    rp = ((N_GLASS * ct - ci) / (N_GLASS * ct + ci)) ** 2
    return 1.0 - 0.5 * (rs + rp)


def _exit_face_scene(theta_i_deg: float, aperture_radius: float = 0.02):
    """A pencil inside glass meeting a flat exit face tilted by the incidence angle.

    The same geometry the catalogue's critical-angle case uses: the beam
    travels along +z, and the interface is rotated about x so the beam meets
    it at ``theta_i_deg``. The detector sits just beyond the face, parallel
    to it, and catches whatever is transmitted.
    """
    rx = math.radians(theta_i_deg)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(WAVELENGTH),
            total_flux=1.0,
            aperture_radius=aperture_radius,
        ),
    )
    scene.add_component(
        "IF",
        RefractiveComponent(
            CoordinateSystem(z=10.0, rx=rx),
            PlaneGeometry(),
            material_front=NSQMaterial(optiland_material=IdealMaterial(n=N_GLASS)),
            material_back=VACUUM,
            name="IF",
        ),
    )
    scene.add_detector(
        "T",
        CoordinateSystem(z=10.001, rx=rx),
        IrradianceDetectorConfig(
            width=500, height=500, num_pixels_x=1, num_pixels_y=1, splat="hard"
        ),
    )
    return scene


#: The fold mirror sits this far out along +z, and is tilted 45 degrees
#: about x. The hit point is rebuilt by adding the surface's own position
#: last, so it is that position's ulp -- 1.14e-13 mm at 1000 mm in float64
#: -- that sets how far off the surface a ray can land, and therefore how
#: large a root a grazing exit makes out of it.
MIRROR_Z = 1000.0
MIRROR_TILT_DEG = 45.0


def _fold_mirror_frame() -> tuple[np.ndarray, np.ndarray]:
    """The mirror's geometric normal and one in-plane direction, in global x/y/z."""
    tilt = math.radians(MIRROR_TILT_DEG)
    normal = np.array([0.0, -math.sin(tilt), math.cos(tilt)])
    in_plane = np.array([0.0, math.cos(tilt), math.sin(tilt)])
    return normal, in_plane


def _rx_for(direction: np.ndarray) -> float:
    """Rotation about x that points a component's local +z along ``direction``."""
    return math.atan2(-direction[1], direction[2])


def _grazing_mirror_scene(grazing_deg: float, aperture_radius: float = 0.02):
    """A pencil skimming a 45-degree fold mirror ``grazing_deg`` above its plane.

    The beam descends onto the mirror at ``90 - grazing_deg`` from its
    normal and leaves at the same shallow angle on the other side, which is
    the exposure a grazing exit has with no critical angle involved.

    The tilt matters, and 45 degrees is not decoration. A surface whose
    normal lies along a coordinate axis hides the failure twice over: the
    rebuilt hit point's large addition is in the one coordinate the normal
    reads, so the sum rounds back to the surface's own position exactly and
    the residual is zero. And a surface tilted to exactly meet the beam
    hides it a third way: the cosine that makes the exit grazing projects
    the position error onto the normal by the same cosine, and the two
    cancel. A fold mirror in a real layout has neither protection.
    """
    normal, in_plane = _fold_mirror_frame()
    g = math.radians(grazing_deg)
    incoming = math.cos(g) * in_plane + math.sin(g) * normal
    outgoing = math.cos(g) * in_plane - math.sin(g) * normal
    vertex = np.array([0.0, 0.0, MIRROR_Z])
    source_at = vertex - 100.0 * incoming
    detector_at = vertex + 200.0 * outgoing

    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(
            y=float(source_at[1]), z=float(source_at[2]), rx=_rx_for(incoming)
        ),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(WAVELENGTH),
            total_flux=1.0,
            aperture_radius=aperture_radius,
        ),
    )
    scene.add_mirror(
        "M",
        CoordinateSystem(z=MIRROR_Z, rx=math.radians(MIRROR_TILT_DEG)),
        MirrorConfig(radius=0.0, reflectance=1.0, aperture_radius=200.0),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(
            y=float(detector_at[1]), z=float(detector_at[2]), rx=_rx_for(outgoing)
        ),
        RayDatabaseConfig(width=400, height=400),
    )
    return scene


def _hits_per_ray(result, surface_name: str) -> Counter:
    """How many times each ray was recorded hitting ``surface_name``."""
    events = result.ray_paths["events"]
    hits = events[
        (events["event_type"] == "hit") & (events["component_name"] == surface_name)
    ]
    return Counter(hits["ray_id"].tolist())


def _max_hits(result, surface_name: str) -> int:
    counts = _hits_per_ray(result, surface_name)
    return max(counts.values()) if counts else 0


# ---------------------------------------------------------------------------
# 1. the grazing exit
# ---------------------------------------------------------------------------


class TestGrazingExit:
    """A refracted ray leaving into air must cross its interface once."""

    @pytest.mark.parametrize("exit_deg", [85.0, 89.0, 89.638489])
    def test_one_hit_per_ray(self, exit_deg):
        theta_i = _incidence_for_exit(exit_deg)
        result = _exit_face_scene(theta_i).trace(
            num_rays=20_000, seed=7, max_depth=4, record_paths=True
        )
        assert _max_hits(result, "IF") == 1, (
            f"a ray leaving at {exit_deg} deg from the normal re-entered the "
            "interface it had just refracted through"
        )

    @pytest.mark.parametrize("exit_deg", [85.0, 89.0, 89.638489])
    def test_transmittance_matches_fresnel(self, exit_deg):
        theta_i = _incidence_for_exit(exit_deg)
        n_rays = 200_000
        result = _exit_face_scene(theta_i).trace(
            num_rays=n_rays, seed=7, max_depth=4
        )
        measured = result.total_flux_detected / result.total_flux_in
        expected = _fresnel_t_unpolarized(theta_i)
        # The branch is resolved by roulette, so the estimator's own standard
        # error is the tolerance; 4 of them, with a fixed seed.
        se = math.sqrt(expected * (1.0 - expected) / n_rays)
        assert measured == pytest.approx(expected, abs=4.0 * se)

    def test_without_the_offset_the_ray_comes_back(self, no_offset):
        """Control: the measurement this file exists for.

        At one millidegree inside the critical angle the exit is 0.36
        degrees from grazing, and the interface accepts its own outgoing
        rays a second time.
        """
        theta_i = _incidence_for_exit(89.638489)
        result = _exit_face_scene(theta_i).trace(
            num_rays=20_000, seed=7, max_depth=4, record_paths=True
        )
        counts = _hits_per_ray(result, "IF")
        repeats = sum(1 for n in counts.values() if n > 1)
        assert repeats > 100, (
            "the control is supposed to reproduce the double hit; with "
            f"only {repeats} of {len(counts)} rays affected it is not "
            "measuring what the test above fixes"
        )

    def test_without_the_offset_the_transmittance_is_low(self, no_offset):
        """Control: and the double hit costs real flux, not just a hit count."""
        theta_i = _incidence_for_exit(89.638489)
        n_rays = 200_000
        result = _exit_face_scene(theta_i).trace(num_rays=n_rays, seed=7, max_depth=4)
        measured = result.total_flux_detected / result.total_flux_in
        expected = _fresnel_t_unpolarized(theta_i)
        se = math.sqrt(expected * (1.0 - expected) / n_rays)
        assert measured < expected - 10.0 * se


# ---------------------------------------------------------------------------
# 2. the grazing incidence
# ---------------------------------------------------------------------------


class TestGrazingIncidenceOnAMirror:
    """A mirror skimmed near-tangentially reflects once, not twice.

    The reflected ray leaves at the same shallow angle it arrived at, so a
    mirror is exposed to the same failure as a refracting exit face -- and
    reaches it with no critical angle involved, purely from the layout.
    """

    @pytest.mark.parametrize("grazing_deg", [5.0, 1.0, 0.1])
    def test_one_hit_per_ray(self, grazing_deg):
        result = _grazing_mirror_scene(grazing_deg).trace(
            num_rays=20_000, seed=3, max_depth=8, record_paths=True
        )
        assert _max_hits(result, "M.surface") == 1, (
            f"a ray leaving the mirror {grazing_deg} deg above its plane "
            "struck it again"
        )

    @pytest.mark.parametrize("grazing_deg", [5.0, 1.0, 0.1])
    def test_all_the_flux_arrives_once(self, grazing_deg):
        """A perfect mirror hit once passes all of it; hit twice, still all
        of it -- so the flux is only the check that nothing else broke."""
        result = _grazing_mirror_scene(grazing_deg).trace(
            num_rays=20_000, seed=3, max_depth=8
        )
        assert result.total_flux_detected == pytest.approx(1.0, rel=1e-12)

    def test_reflected_direction_is_specular(self):
        """One reflection, not two: the direction is what tells them apart.

        A second reflection at the same mirror sends the ray back down at
        the angle it came in at, so the direction reaching the collector is
        the measurement with the most to say.
        """
        grazing_deg = 1.0
        result = _grazing_mirror_scene(grazing_deg).trace(
            num_rays=2_000, seed=3, max_depth=8
        )
        db = result.detectors["D"]
        d = np.stack([db.L, db.M, db.N], axis=1)
        assert d.shape[0] > 0
        normal, in_plane = _fold_mirror_frame()
        g = math.radians(grazing_deg)
        expected = math.cos(g) * in_plane - math.sin(g) * normal
        np.testing.assert_allclose(d, np.broadcast_to(expected, d.shape), atol=1e-12)

    def test_without_the_offset_the_mirror_is_hit_twice(self, no_offset):
        """Control: at 0.1 degrees above the plane the mirror re-catches its
        own outgoing rays, and they then bounce along it."""
        result = _grazing_mirror_scene(0.1).trace(
            num_rays=20_000, seed=3, max_depth=8, record_paths=True
        )
        counts = _hits_per_ray(result, "M.surface")
        repeats = sum(1 for n in counts.values() if n > 1)
        assert repeats > 100, (
            "the control is supposed to reproduce the repeat hit; with only "
            f"{repeats} of {len(counts)} rays affected it is not measuring "
            "what the test above fixes"
        )


# ---------------------------------------------------------------------------
# 3. the offset itself
# ---------------------------------------------------------------------------


class TestOffsetConstant:
    """R-07-6's bound, and its relation to the accept threshold."""

    def test_k_delta_is_the_documented_one(self):
        assert tol.DEFAULT_OFFSET_K_DELTA == 32

    def test_accept_multiplier_is_unchanged(self):
        """The offset is the cure; k stays where X1 measured it."""
        assert tol.DEFAULT_ACCEPT_K == 16

    @pytest.mark.parametrize("magnitude", [1.0, 10.0, 50.0, 1e3, 1e6])
    def test_offset_is_half_k_delta_ulps(self, magnitude):
        delta = float(tol.origin_offset(np.float64(magnitude)))
        assert delta == pytest.approx(16.0 * np.spacing(magnitude), rel=1e-12)

    @pytest.mark.parametrize("magnitude", [1.0, 10.0, 50.0, 1e3, 1e6])
    def test_offset_clears_the_residual_and_matches_the_threshold(self, magnitude):
        """The two are complements: the offset lands on the threshold's edge.

        The rebuilt hit point sits within one ulp of the surface, so an
        offset of 16 ulps is outside the error ball; and because it equals
        ``accept_t_min`` at the same coordinate, a ray that does turn back
        toward the surface is still rejected by the threshold.
        """
        delta = float(tol.origin_offset(np.float64(magnitude)))
        assert delta > np.spacing(magnitude)
        assert delta == pytest.approx(
            float(tol.accept_t_min(np.float64(magnitude))), rel=1e-12
        )

    def test_magnitude_floor_applies(self):
        """A hit at the coordinate origin still gets a non-zero offset."""
        assert float(tol.origin_offset(np.float64(0.0))) == pytest.approx(
            16.0 * np.spacing(1.0), rel=1e-12
        )


# ---------------------------------------------------------------------------
# 4. the transmissive detector
# ---------------------------------------------------------------------------


def _grazing_transmissive_detector_scene(
    grazing_deg: float, aperture_radius: float = 0.02
):
    """A pencil crossing a transmissive detector plane at a grazing angle.

    The exposure X7 section 6 named and left open. A detector with
    ``absorb=False`` reads the beam and lets it through, so the ray carries
    on from the point the loop puts it at and the next bounce tests the same
    plane again -- the grazing-exit failure, with no interface involved.

    The plane is the fold mirror's: 45 degrees about x at ``MIRROR_Z``, for
    the reasons ``_grazing_mirror_scene`` gives (an axis-aligned normal or a
    plane tilted to meet the beam both hide the residual). The beam
    approaches from the ``-normal`` side at ``grazing_deg`` above the plane
    and leaves on the other side at the same angle, undeviated.
    """
    normal, in_plane = _fold_mirror_frame()
    g = math.radians(grazing_deg)
    direction = math.cos(g) * in_plane + math.sin(g) * normal
    vertex = np.array([0.0, 0.0, MIRROR_Z])
    source_at = vertex - 100.0 * direction
    collector_at = vertex + 200.0 * direction

    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(
            y=float(source_at[1]), z=float(source_at[2]), rx=_rx_for(direction)
        ),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(WAVELENGTH),
            total_flux=1.0,
            aperture_radius=aperture_radius,
        ),
    )
    scene.add_detector(
        "TAP",
        CoordinateSystem(z=MIRROR_Z, rx=math.radians(MIRROR_TILT_DEG)),
        IrradianceDetectorConfig(
            width=400,
            height=400,
            num_pixels_x=1,
            num_pixels_y=1,
            splat="hard",
            absorb=False,
        ),
    )
    scene.add_detector(
        "END",
        CoordinateSystem(
            y=float(collector_at[1]),
            z=float(collector_at[2]),
            rx=_rx_for(direction),
        ),
        IrradianceDetectorConfig(
            width=400, height=400, num_pixels_x=1, num_pixels_y=1, splat="hard"
        ),
    )
    return scene


@pytest.fixture
def no_detector_offset(monkeypatch):
    """Disable the origin offset on detectors only, for the control."""
    from optiland.nonsequential.detectors.base import BaseDetector

    monkeypatch.setattr(
        BaseDetector, "offset_from_surface", lambda self, rays, n_geom, hit: None
    )


class TestGrazingCrossingOfATransmissiveDetector:
    """A ray crossing a tap at a grazing angle is recorded once."""

    @pytest.mark.parametrize("grazing_deg", [5.0, 1.0, 0.1])
    def test_recorded_once_per_ray(self, grazing_deg):
        n_rays = 20_000
        result = _grazing_transmissive_detector_scene(grazing_deg).trace(
            num_rays=n_rays, seed=3, max_depth=8
        )
        tap = result.detectors["TAP"]
        assert tap.num_rays_hit == n_rays, (
            f"a ray crossing the tap {grazing_deg} deg above its plane was "
            f"recorded {tap.num_rays_hit / n_rays:.3f} times on average"
        )
        # And the tap read the whole beam exactly once, in watts.
        assert float(tap.total_flux) == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize("grazing_deg", [5.0, 1.0, 0.1])
    def test_the_beam_still_arrives_undeviated(self, grazing_deg):
        """A tap must not change what reaches the collector behind it."""
        result = _grazing_transmissive_detector_scene(grazing_deg).trace(
            num_rays=20_000, seed=3, max_depth=8
        )
        assert float(result.detectors["END"].total_flux) == pytest.approx(
            1.0, rel=1e-12
        )
        # The tap's reading is booked separately so the ledger does not
        # count the same watt twice.
        assert result.total_flux_tapped == pytest.approx(1.0, rel=1e-12)
        assert result.flux_conservation_error < 1e-12

    def test_without_the_offset_the_tap_reads_twice(self, no_detector_offset):
        """Control: the failure this fixes, measured rather than assumed."""
        n_rays = 20_000
        result = _grazing_transmissive_detector_scene(0.1).trace(
            num_rays=n_rays, seed=3, max_depth=8
        )
        tap = result.detectors["TAP"]
        extra = tap.num_rays_hit - n_rays
        assert extra > 100, (
            "the control is supposed to reproduce the repeat reading; with "
            f"only {extra} extra readings of {n_rays} rays it is not "
            "measuring what the test above fixes"
        )
