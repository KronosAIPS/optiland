"""A coated face met beyond its critical angle (the research repository's issue 96).

Until this change a refractive face forced ``R = 1, T = 0`` wherever the bare
interface reflected totally, whatever its coating. An absorbing layer in the
evanescent field absorbs (frustrated total internal reflection), so the face
reflects less than one. Now a side-aware coating (a thin-film stack, a table)
keeps its own reflectance there, the ray reflects deterministically with that
weight, and the absorptance is booked as coating loss. A bare face and a
side-blind coating (a stated R and T) still reflect exactly one.

The independent route is the Airy sum of one layer between two media (the
research repository's chapter 06 section 6.15):
``r = (r01 + r12 e^{2 i beta}) / (1 + r01 r12 e^{2 i beta})``,
``beta = 2 pi d N1 cos theta_1 / lambda``, every ``N cos theta`` the root with
a non-negative imaginary part (``exp(-i omega t)``), written here without a
characteristic matrix.

Tolerances. The engine's stack is about 60 complex operations per polarization
(the characteristic-matrix product of one layer, the admittances, the
quotient), each within a few u of its operands, and the reflectance here is
conditioned by at most about 10 on its intermediates; 1e-12 relative at
float64 leaves a factor of more than ten on 600 u = 6.7e-14, and the
detector, the ledger and the weight add three roundings. At float32 the same
count is 600 u32 = 3.6e-5; the float32 test asserts 2e-4.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import SimpleCoating, TabulatedCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential import polarization as P
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.coating_support import (
    UnpolarizedThinFilmCoating,
    coating_holds_beyond_critical,
)
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.nonsequential.materials import VACUUM
from optiland.thin_film import ThinFilmStack

WL = 0.55
N_GLASS = 1.52
METAL = complex(3.13, 4.33)  # a metal-like absorbing layer (chromium's order in the visible)
METAL_NM = 10.0
MGF2 = 1.38
QW_UM = WL / (4.0 * MGF2)
ANGLES = [45.0, 60.0, 75.0]  # all beyond asin(1 / 1.52) = 41.14 degrees


def _glass():
    return NSQMaterial(optiland_material=IdealMaterial(N_GLASS))


def _stack(layer: complex, thickness_um: float) -> ThinFilmStack:
    """Written from the glass: the glass is the incident medium, vacuum the substrate."""
    st = ThinFilmStack(IdealMaterial(N_GLASS), IdealMaterial(1.0))
    st.add_layer(IdealMaterial(layer.real, layer.imag), thickness_um)
    return st


def airy_R(n0: float, N1: complex, d_um: float, ns: float, theta_deg: float, pol: str) -> float:
    """One layer by the Airy sum (no matrix): the independent route."""
    s0 = n0 * math.sin(math.radians(theta_deg))

    def ncos(N):
        c = cmath.sqrt(complex(N) * complex(N) - s0 * s0)
        if c.imag < 0 or (c.imag == 0 and c.real < 0):
            c = -c
        return c

    def eta(N):
        return ncos(N) if pol == "s" else complex(N) ** 2 / ncos(N)

    e0, e1, e2 = eta(n0), eta(N1), eta(ns)
    r01 = (e0 - e1) / (e0 + e1)
    r12 = (e1 - e2) / (e1 + e2)
    ph = cmath.exp(2j * (2.0 * math.pi * d_um * ncos(N1) / WL))
    r = (r01 + r12 * ph) / (1.0 + r01 * r12 * ph)
    return abs(r) ** 2


def airy_unpol(N1, d_um, theta_deg):
    return 0.5 * (
        airy_R(N_GLASS, N1, d_um, 1.0, theta_deg, "s")
        + airy_R(N_GLASS, N1, d_um, 1.0, theta_deg, "p")
    )


def _scene(theta_deg: float, coating, split: bool = False):
    """A beam inside the glass meets the coated face (front: glass, back: vacuum)."""
    t = math.radians(theta_deg)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(y=-50.0 * math.sin(t), z=-50.0 * math.cos(t), rx=-t),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(WL), total_flux=1.0, aperture_radius=0.5
        ),
    )
    face = RefractiveComponent(
        CoordinateSystem(z=0.0), PlaneGeometry(), _glass(), VACUUM, coating=coating, name="face"
    )
    scene.add_component("face", face)
    for name, z in (("reflected", -100.0), ("transmitted", 100.0)):
        scene.add_detector(
            name,
            CoordinateSystem(z=z),
            IrradianceDetectorConfig(
                width=4000, height=4000, num_pixels_x=1, num_pixels_y=1, splat="hard"
            ),
        )
    if split:
        scene.sampling_policy = SamplingPolicy(
            split_depth=2, split_budget=8.0, rr_start_flux=1e-16
        )
    return scene, face


def _fractions(result, face):
    phi = float(result.total_flux_in)
    return (
        float(result.detectors["reflected"].total_flux_float) / phi,
        float(result.detectors["transmitted"].total_flux_float) / phi,
        float(face.coating_loss) / phi,
        float(face.sampling_residual) / phi,
    )


@pytest.fixture(autouse=True)
def _numpy_float64():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


class TestWhichCoatingsHoldBeyondCritical:
    def test_side_aware_and_side_blind(self):
        assert coating_holds_beyond_critical(UnpolarizedThinFilmCoating(_stack(METAL, 0.01)))
        assert not coating_holds_beyond_critical(SimpleCoating(0.9, 0.05))
        assert not coating_holds_beyond_critical(None)


class TestAbsorbingLayer:
    @pytest.mark.parametrize("split", [False, True])
    @pytest.mark.parametrize("theta_deg", ANGLES)
    def test_reflects_the_stack_and_books_the_rest(self, theta_deg, split):
        coating = UnpolarizedThinFilmCoating(_stack(METAL, METAL_NM / 1000.0))
        scene, face = _scene(theta_deg, coating, split=split)
        result = scene.trace(num_rays=64, seed=1, max_depth=2)
        refl, trans, loss, resid = _fractions(result, face)
        ref = airy_unpol(METAL, METAL_NM / 1000.0, theta_deg)
        assert ref < 0.45  # the defect read 1 here
        assert refl == pytest.approx(ref, rel=1e-12)
        assert trans == 0.0
        assert loss == pytest.approx(1.0 - ref, rel=1e-12)
        assert abs(resid) <= 1e-15
        assert abs(result.flux_conservation_error) <= 1e-12

    def test_stack_transmittance_is_zero_beyond_the_cutoff(self):
        """The R + T the face reflects is R itself: the stack's T vanishes on every lane."""
        c = UnpolarizedThinFilmCoating(_stack(METAL, METAL_NM / 1000.0))
        th = np.radians(ANGLES)
        R, T = c.evaluate(np.full(3, WL), np.cos(th))
        assert np.all(np.asarray(T) == 0.0)

    @pytest.mark.parametrize("precision,rel", [("float64", 1e-12), ("float32", 2e-4)])
    def test_torch_legs(self, precision, rel):
        pytest.importorskip("torch")
        from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: PLC0415

        be.set_backend("torch")
        be.set_device("cpu")
        be.grad_mode.disable()
        be.set_precision(precision)
        for theta_deg in ANGLES:
            coating = UnpolarizedThinFilmCoating(_stack(METAL, METAL_NM / 1000.0))
            scene, face = _scene(theta_deg, coating)
            result = scene.trace(
                num_rays=64, seed=1, max_depth=2, backend=TorchBackend(seed=1)
            )
            refl, trans, loss, _ = _fractions(result, face)
            ref = airy_unpol(METAL, METAL_NM / 1000.0, theta_deg)
            assert refl == pytest.approx(ref, rel=rel)
            assert trans == 0.0
            assert loss == pytest.approx(1.0 - ref, rel=rel)


class TestLosslessAndBare:
    @pytest.mark.parametrize("theta_deg", ANGLES)
    def test_lossless_layer_reflects_one_to_rounding(self, theta_deg):
        coating = UnpolarizedThinFilmCoating(_stack(complex(MGF2, 0.0), QW_UM))
        scene, face = _scene(theta_deg, coating)
        result = scene.trace(num_rays=64, seed=1, max_depth=2)
        refl, trans, loss, _ = _fractions(result, face)
        assert abs(refl - 1.0) <= 64 * 2.0**-53
        assert trans == 0.0
        assert abs(loss) <= 64 * 2.0**-53

    @pytest.mark.parametrize("coating", [None, SimpleCoating(0.3, 0.6)])
    def test_bare_face_and_side_blind_coating_reflect_exactly_one(self, coating):
        scene, face = _scene(60.0, coating)
        result = scene.trace(num_rays=64, seed=1, max_depth=2)
        refl, trans, loss, resid = _fractions(result, face)
        assert refl == 1.0 and trans == 0.0 and loss == 0.0 and resid == 0.0

    def test_below_the_cutoff_nothing_changes(self):
        """At 30 degrees no lane is totally reflected: the split tree reads R and T as before."""
        coating = UnpolarizedThinFilmCoating(_stack(METAL, METAL_NM / 1000.0))
        scene, face = _scene(30.0, coating, split=True)
        result = scene.trace(num_rays=64, seed=1, max_depth=2)
        refl, trans, loss, _ = _fractions(result, face)
        R, T = coating.evaluate(np.array([WL]), np.array([math.cos(math.radians(30.0))]))
        assert refl == pytest.approx(float(R[0]), rel=1e-14)
        assert trans == pytest.approx(float(T[0]), rel=1e-14)
        assert loss == pytest.approx(1.0 - float(R[0]) - float(T[0]), rel=1e-12)


class TestTable:
    def test_table_beyond_the_cutoff_from_its_incident_side(self):
        """A table written from the glass: beyond the cutoff its own R is read, at a node."""
        stack = _stack(METAL, METAL_NM / 1000.0)
        wl_nm = np.array([500.0, 550.0, 600.0])
        ang = np.arange(0.0, 90.0, 5.0)
        grids = {k: np.zeros((wl_nm.size, ang.size)) for k in ("r_s", "r_p", "t_s", "t_p")}
        for i, w in enumerate(wl_nm):
            for pol in ("s", "p"):
                out = stack.compute_rtRTA_elementwise(
                    np.full(ang.size, w / 1000.0), np.radians(ang), polarization=pol
                )
                grids[f"r_{pol}"][i] = np.asarray(out["R"])
                grids[f"t_{pol}"][i] = np.clip(np.asarray(out["T"]), 0.0, None)
        table = TabulatedCoating(
            wl_nm, ang, **grids, substrate_material=1.0, incident_material=N_GLASS
        )
        scene, face = _scene(60.0, table)
        result = scene.trace(num_rays=64, seed=1, max_depth=2)
        refl, trans, loss, _ = _fractions(result, face)
        ref = airy_unpol(METAL, METAL_NM / 1000.0, 60.0)
        assert refl == pytest.approx(ref, rel=1e-12)
        assert trans == 0.0
        assert loss == pytest.approx(1.0 - ref, rel=1e-12)


class TestStokes:
    @pytest.mark.parametrize("theta_deg", ANGLES)
    def test_s_and_p_inputs_read_their_own_reflectance(self, theta_deg):
        """A ray polarized along s (the x axis here) reflects R_s; along p, R_p."""
        d = METAL_NM / 1000.0
        for q_x, pol in ((1.0, "s"), (-1.0, "p")):
            coating = UnpolarizedThinFilmCoating(_stack(METAL, d))
            scene, face = _scene(theta_deg, coating)
            P.set_source_polarization(
                scene, "S", stokes=(1.0, q_x, 0.0, 0.0), reference_axis=(1.0, 0.0, 0.0)
            )
            result = scene.trace(
                num_rays=16, seed=1, max_depth=2,
                backend=NumpyBackend(seed=1, polarization="stokes"),
            )
            refl, _, loss, _ = _fractions(result, face)
            ref = airy_R(N_GLASS, METAL, d, 1.0, theta_deg, pol)
            assert refl == pytest.approx(ref, rel=1e-11)
            assert loss == pytest.approx(1.0 - ref, rel=1e-11)

    def test_unpolarized_stokes_equals_scalar_bit_for_bit(self):
        out = []
        for mode in ("off", "stokes"):
            coating = UnpolarizedThinFilmCoating(_stack(METAL, METAL_NM / 1000.0))
            scene, face = _scene(60.0, coating)
            result = scene.trace(
                num_rays=16, seed=1, max_depth=2, backend=NumpyBackend(seed=1, polarization=mode)
            )
            out.append(_fractions(result, face))
        assert out[0] == out[1]
