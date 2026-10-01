"""Volume tests for Non-Sequential Raytracing (PR6).

Covers watertightness validation, the ray-parity orientation check, CSG
union/intersection/difference scoping, and that Lens/Doublet are actually
validated as Volumes at construction.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential.components.absorbing import AbsorbingComponent
from optiland.nonsequential.components.configs import DoubletConfig, LensConfig
from optiland.nonsequential.components.doublet import Doublet
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.geometry.analytic.frustum import (
    CylindricalFrustumGeometry,
)
from optiland.nonsequential.components.lens import Lens
from optiland.nonsequential.components.refractive import RefractiveComponent
from optiland.nonsequential.components.volume import NonWatertightVolumeError, Volume
from optiland.nonsequential.materials.nsq_material import VACUUM, NSQMaterial


def _glass():
    return NSQMaterial.from_glass("N-BK7")


class TestLensAndDoubletBuildVolumes:
    def test_simple_lens_builds_one_volume(self):
        lens = Lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50, r2=-50, thickness=5, material="N-BK7", front_aperture_radius=12.5
            ),
        )
        assert lens._volume.name == "L"
        assert len(lens._volume.boundary) == 3  # front, back, edge

    def test_lens_with_rim_builds_one_volume(self):
        lens = Lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50,
                r2=-50,
                thickness=5,
                material="N-BK7",
                front_aperture_radius=12.5,
                back_aperture_radius=10.0,
            ),
        )
        assert len(lens._volume.boundary) == 4  # front, back, edge, rim

    def test_doublet_builds_two_volumes_sharing_the_cemented_interface(self):
        d = Doublet(
            "D",
            CoordinateSystem(z=0),
            DoubletConfig(
                r1=50,
                r2=-40,
                r3=-100,
                thickness1=5,
                thickness2=3,
                material1="N-BK7",
                material2="N-SF5",
                aperture_radius=10.0,
            ),
        )
        assert len(d._volumes) == 2
        crown, flint = d._volumes
        assert crown.name == "D.crown"
        assert flint.name == "D.flint"
        cemented = next(s for s in d.surfaces if s.name == "D.cemented")
        assert cemented in crown.boundary
        assert cemented in flint.boundary
        # 5 physical surfaces total, not duplicated for being shared.
        assert len(d.surfaces) == 5
        assert len({s.name for s in d.surfaces}) == 5


class TestDeepMeniscusVolume:
    """Issue #13, item 1: the closed-volume check's interior point used to
    be the mean of the rim points, which lies outside a deep meniscus (both
    faces curving the same way, closely spaced) on the axis -- refusing any
    semi-diameter above a limit far inside the beam the element actually
    needs to pass. The fix takes the interior point on the axis between the
    two vertices instead (inside the glass by construction whenever the
    centre thickness is positive), falling back to the rim-point mean only
    when fewer than two conic vertex surfaces bound the volume.

    E2 of the Canon EF 50mm f/1.8 II (patent JP-S62-087922 example 1,
    ``cases/lenses/canon_ef50_f18_ii.yaml`` in KronosNSRT): r1=21.51,
    r2=40.31, thickness=4.35 mm, a positive meniscus. Before the fix, the
    rim-mean interior point failed the ray-parity check above about 10.76 mm
    of semi-diameter (docs/build/Z2_lens_ghosts.md section 4.1); this lens
    needs 13.53 mm.
    """

    def test_deep_meniscus_beyond_the_old_rim_mean_limit_builds(self):
        lens = Lens(
            "E2",
            CoordinateSystem(z=0),
            LensConfig(
                r1=21.51,
                r2=40.31,
                thickness=4.35,
                material=_glass(),
                front_aperture_radius=13.53,
            ),
        )
        assert lens._volume.name == "E2"

    def test_vertex_axis_candidate_used_before_rim_mean(self):
        """The on-axis vertex-mean candidate -- not the rim-point mean --
        is what makes the check pass here: a semi-diameter well beyond the
        rim-mean's old ~10.76 mm limit still builds.
        """
        from optiland.nonsequential.components.volume import (
            _vertex_axis_candidate,
        )

        lens = Lens(
            "E2",
            CoordinateSystem(z=0),
            LensConfig(
                r1=21.51,
                r2=40.31,
                thickness=4.35,
                material=_glass(),
                front_aperture_radius=13.53,
            ),
        )
        vertex_mid = _vertex_axis_candidate(lens._volume.boundary)
        assert vertex_mid is not None
        # On-axis, halfway between the front vertex (z=0) and back vertex
        # (z=4.35): inside the glass by construction, independent of the
        # rim -- unlike the rim-point mean this replaces as the primary
        # candidate.
        import numpy as np  # noqa: PLC0415

        np.testing.assert_allclose(vertex_mid, [0.0, 0.0, 2.175], atol=1e-9)


class TestWatertightnessCatchesGaps:
    def test_lens_missing_its_edge_is_not_watertight(self):
        """Front and back faces alone leave an open annular gap between
        their rims -- exactly the leak watertightness exists to catch."""
        front = RefractiveComponent(
            cs=CoordinateSystem(z=0),
            geometry=ConicGeometry(50.0, 0.0, 12.5),
            material_front=VACUUM,
            material_back=_glass(),
            name="front",
        )
        back = RefractiveComponent(
            cs=CoordinateSystem(z=5.0),
            geometry=ConicGeometry(-50.0, 0.0, 12.5),
            material_front=_glass(),
            material_back=VACUUM,
            name="back",
        )
        with pytest.raises(NonWatertightVolumeError, match="not watertight"):
            Volume(name="broken", boundary=[front, back], interior=_glass())

    def test_mismatched_edge_radius_is_not_watertight(self):
        """An edge whose radius doesn't match the faces' aperture leaves a gap."""
        front = RefractiveComponent(
            cs=CoordinateSystem(z=0),
            geometry=ConicGeometry(50.0, 0.0, 12.5),
            material_front=VACUUM,
            material_back=_glass(),
            name="front",
        )
        back = RefractiveComponent(
            cs=CoordinateSystem(z=5.0),
            geometry=ConicGeometry(-50.0, 0.0, 12.5),
            material_front=_glass(),
            material_back=VACUUM,
            name="back",
        )
        edge = AbsorbingComponent(
            cs=CoordinateSystem(z=0),
            geometry=CylindricalFrustumGeometry(
                r_front=12.5, r_back=12.5, z_front=0.0, z_back=5.0
            ),
            name="edge",
        )
        # Deliberately wrong: the true rim radius is not exactly 12.5 once
        # sag is accounted for at a curved front face -- but to construct an
        # unambiguous gap, offset the edge radius outright.
        edge.geometry.r_front = 13.0
        edge.geometry.r_back = 13.0
        with pytest.raises(NonWatertightVolumeError, match="not watertight"):
            Volume(name="broken", boundary=[front, back, edge], interior=_glass())

    def test_empty_boundary_raises(self):
        with pytest.raises(NonWatertightVolumeError):
            Volume(name="empty", boundary=[], interior=_glass())


class TestCsgScope:
    def test_union_concatenates_boundaries(self):
        lens = Lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50, r2=-50, thickness=5, material="N-BK7", front_aperture_radius=12.5
            ),
        )
        other = RefractiveComponent(
            cs=CoordinateSystem(z=100),
            geometry=ConicGeometry(10.0, 0.0, 1.0),
            material_front=VACUUM,
            material_back=_glass(),
            name="extra",
        )
        merged = Volume.union(lens._volume, [other])
        assert len(merged) == len(lens._volume.boundary) + 1
        assert other in merged

    def test_intersection_not_implemented(self):
        with pytest.raises(NotImplementedError, match="boolean surface evaluator"):
            Volume.intersection()

    def test_difference_not_implemented(self):
        with pytest.raises(NotImplementedError, match="boolean surface evaluator"):
            Volume.difference()


class TestVolumeIntegratesWithTracing:
    """A rebuilt Lens/Doublet still traces correctly end to end."""

    def test_singlet_still_focuses(self):
        from optiland.nonsequential import (
            CollimatedSourceConfig,
            IrradianceDetectorConfig,
            NSQScene,
            Spectrum,
        )

        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-80),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=5.0,
            ),
        )
        scene.add_lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50, r2=-50, thickness=5, material="N-BK7", front_aperture_radius=12.5
            ),
        )
        scene.add_detector(
            "D",
            CoordinateSystem(z=100),
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=32, num_pixels_y=32
            ),
        )
        result = scene.trace(num_rays=2_000, seed=1)
        assert result.detectors["D"].total_flux > 0.5

    def test_doublet_still_transmits(self):
        from optiland.nonsequential import (
            CollimatedSourceConfig,
            IrradianceDetectorConfig,
            NSQScene,
            Spectrum,
        )

        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-80),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=5.0,
            ),
        )
        scene.add_doublet(
            "D",
            CoordinateSystem(z=0),
            DoubletConfig(
                r1=50,
                r2=-40,
                r3=-100,
                thickness1=5,
                thickness2=3,
                material1="N-BK7",
                material2="N-SF5",
                aperture_radius=10.0,
            ),
        )
        scene.add_detector(
            "D_det",
            CoordinateSystem(z=100),
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=32, num_pixels_y=32
            ),
        )
        result = scene.trace(num_rays=2_000, seed=1)
        assert result.detectors["D_det"].total_flux > 0.5


# -- KronosNSRT issue 80: the rim tolerance follows the working precision ------

#: The decentred, tilted singlet of the interior-gradient tests
#: (``test_nsq_interior_gradients._LENS``): its back rim sits near z = 55 mm,
#: where float32's spacing is 3.8e-6 mm, and the absolute 1e-6 mm tolerance
#: refused it at float32.
_FAR_SINGLET = {"x": 0.5, "y": -0.3, "z": 50.0, "rx": 0.01, "ry": -0.02, "rz": 0.0}


@pytest.fixture
def _torch_backend():
    torch = pytest.importorskip("torch", reason="Torch not available")  # noqa: F841
    import optiland.backend as be  # noqa: PLC0415

    yield be
    be.set_backend("numpy")
    be.set_precision("float64")


def _far_singlet():
    return Lens(
        "L",
        CoordinateSystem(**_FAR_SINGLET),
        LensConfig(
            r1=60.0,
            r2=float("inf"),
            thickness=5.0,
            material="N-BK7",
            front_aperture_radius=12.0,
        ),
    )


def _open_lens(cs_front, cs_back):
    """Front and back faces with no edge: an annular gap of several mm."""
    front = RefractiveComponent(
        cs=cs_front,
        geometry=ConicGeometry(50.0, 0.0, 12.5),
        material_front=VACUUM,
        material_back=_glass(),
        name="front",
    )
    back = RefractiveComponent(
        cs=cs_back,
        geometry=ConicGeometry(-50.0, 0.0, 12.5),
        material_front=_glass(),
        material_back=VACUUM,
        name="back",
    )
    return [front, back]


class TestRimToleranceFollowsThePrecision:
    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_far_singlet_is_accepted(self, _torch_backend, precision):
        _torch_backend.set_backend("torch")
        _torch_backend.set_precision(precision)
        lens = _far_singlet()
        assert len(lens._volume.boundary) == 3

    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_open_volume_far_from_the_origin_is_refused(self, _torch_backend, precision):
        _torch_backend.set_backend("torch")
        _torch_backend.set_precision(precision)
        cs_front = CoordinateSystem(**_FAR_SINGLET)
        cs_back = CoordinateSystem(z=5.0, reference_cs=cs_front)
        with pytest.raises(NonWatertightVolumeError, match="not watertight"):
            Volume(name="open", boundary=_open_lens(cs_front, cs_back), interior=_glass())

    def test_derived_term_scales_with_the_unit_roundoff(self, _torch_backend):
        """The numerical term is linear in u: float32 over float64 is 2**29, up to
        the float32 storage of the placement's own values (a relative 2**-24)."""
        from optiland.nonsequential.components.volume import (  # noqa: PLC0415
            _rim_error_bound,
            _rim_local_points,
        )

        bounds = {}
        for precision in ("float32", "float64"):
            _torch_backend.set_backend("torch")
            _torch_backend.set_precision(precision)
            lens = _far_singlet()
            bounds[precision] = [
                _rim_error_bound(c, _rim_local_points(c)) for c in lens.surfaces
            ]
        for b32, b64 in zip(bounds["float32"], bounds["float64"]):
            assert b32 / b64 == pytest.approx(2.0**29, rel=4.0 * 2.0**-24)

    def test_float64_tolerance_is_the_floor(self):
        """At float64 the derived term sits orders below the 1e-6 mm floor, so the
        verdicts are those of the floor alone (the rule before issue 80)."""
        import numpy as np  # noqa: PLC0415

        from optiland.nonsequential.components.volume import (  # noqa: PLC0415
            WATERTIGHT_TOL,
            _rim_error_bound,
            _rim_local_points,
        )

        lens = _far_singlet()
        terms = [_rim_error_bound(c, _rim_local_points(c)) for c in lens.surfaces]
        assert np.sqrt(3.0) * 2.0 * max(terms) < WATERTIGHT_TOL * 1e-6
