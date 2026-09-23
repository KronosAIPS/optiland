"""Absorbed power per component and the deposition tally.

Every expected number is a closed form:

* two absorbing slabs in one beam: the per-component bulk books sum to the scene total, and a tally's map plus
  what fell outside its grid equals its component's book (bookkeeping identities, round-off tolerance);
* a slab whose faces carry a lossless anti-reflection coating (R = 0, T = 1): no ray is reflected, so the
  deposited power density is exactly ``alpha I(r) exp(-alpha z)`` with ``I(r)`` the source's truncated
  Gaussian. Rays are drawn at random positions, so each (r, z) bin holds ``N p_i`` rays on average and its
  relative Monte Carlo error is ``sqrt((1 - p_i) / (N p_i))``; every bin must lie within 4.5 of those standard
  errors and the chi-square over the bins within its own 4.5-sigma band;
* a slab with bare faces: the incoherent-slab closed form ``A = (1-R)(1-tau)(1+R tau)/(1 - R^2 tau^2)``,
  ``tau = exp(-alpha L)``, ``R = ((n-1)/(n+1))^2``, within 5 standard errors of the Fresnel branch sampling;
* a slab whose coatings absorb (R = 0, T = 0.99 per face): the coating book of the component is
  ``F (1 - T) + F T (1 - T)`` exactly, since no ray is reflected and no branch is drawn.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optiland.coatings import SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQMaterial,
    NSQScene,
    Spectrum,
    SurfaceConfig,
)

WL = 1.07          # um
N_GLASS = 1.45
THICK = 3.0        # mm
W = 7.5            # 1/e^2 beam radius, mm
APERTURE = 15.0    # source aperture, mm (2 w)
FLUX = 4000.0      # W


def _k(alpha_per_mm: float) -> float:
    """Extinction coefficient for an attenuation coefficient in 1/mm (alpha = 4 pi k / lambda)."""
    return alpha_per_mm * 1e-3 * WL / (4.0 * math.pi)


def _scene(alpha_per_mm: float, coating=None, second_slab: bool = False) -> NSQScene:
    glass = NSQMaterial(optiland_material=IdealMaterial(n=N_GLASS, k=_k(alpha_per_mm)))
    face = SurfaceConfig(coating=coating) if coating is not None else None
    scene = NSQScene()
    kw = {"front": face, "back": face} if face is not None else {}
    scene.add_lens("window", CoordinateSystem(z=0.0),
                   LensConfig(r1=float("inf"), r2=float("inf"), thickness=THICK, material=glass,
                              front_aperture_radius=18.5, back_aperture_radius=18.5, **kw))
    if second_slab:
        scene.add_lens("second", CoordinateSystem(z=20.0),
                       LensConfig(r1=float("inf"), r2=float("inf"), thickness=THICK, material=glass,
                                  front_aperture_radius=18.5, back_aperture_radius=18.5))
    scene.add_source("laser", CoordinateSystem(z=-20.0), CollimatedSourceConfig(
        spectrum=Spectrum.monochromatic(WL), total_flux=FLUX,
        aperture_radius=APERTURE, profile="gaussian", gaussian_sigma=W / 2.0))
    scene.add_detector("back", CoordinateSystem(z=60.0), IrradianceDetectorConfig(width=60.0, height=60.0,
                                                                                  num_pixels_x=16, num_pixels_y=16))
    return scene


def test_component_books_sum_to_the_scene_total_and_the_tally_to_its_book():
    scene = _scene(0.1, second_slab=True)
    scene.add_deposition_tally("win", "window", kind="rz", bounds=(0.0, 20.0, 0.0, THICK), shape=(20, 6))
    res = scene.trace(num_rays=40_000, seed=3, max_depth=32)
    books = res.absorbed_by_component
    assert set(books) == {"window", "second", "ambient", "unassigned"}
    bulk = sum(b["bulk"] for b in books.values())
    assert bulk == pytest.approx(res.total_flux_bulk_absorbed, rel=1e-12)
    assert books["window"]["bulk"] > 0.0 and books["second"]["bulk"] > 0.0
    assert books["ambient"]["bulk"] == 0.0 and books["unassigned"]["bulk"] == 0.0
    dep = res.deposition["win"]
    assert dep.booked == pytest.approx(books["window"]["bulk"], rel=1e-12)
    assert dep.total + dep.outside == pytest.approx(dep.booked, rel=1e-12)
    assert dep.outside == 0.0
    assert dep.power.shape == (20, 6) and dep.density.shape == (20, 6)


def test_the_deposition_map_is_alpha_times_the_irradiance():
    alpha = 0.01                                   # 1/mm
    nr, nz, r_max = 10, 6, 15.0
    scene = _scene(alpha, coating=SimpleCoating(transmittance=1.0, reflectance=0.0))
    scene.add_deposition_tally("win", "window", kind="rz", bounds=(0.0, r_max, 0.0, THICK), shape=(nr, nz))
    n_rays = 200_000
    res = scene.trace(num_rays=n_rays, seed=11, max_depth=16)
    dep = res.deposition["win"]
    r_edges, z_edges = dep.edges
    # the truncated Gaussian's flux fraction in each annulus, and the power a ray deposits in each z slice
    s2 = (W / 2.0) ** 2
    norm = 1.0 - math.exp(-APERTURE ** 2 / (2.0 * s2))
    cdf = (1.0 - np.exp(-np.minimum(r_edges, APERTURE) ** 2 / (2.0 * s2))) / norm
    p = np.diff(cdf)                               # probability that a ray lands in each annulus
    slice_fraction = np.exp(-alpha * z_edges[:-1]) - np.exp(-alpha * z_edges[1:])
    expected = FLUX * np.outer(p, slice_fraction)  # W per (r, z) bin
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = expected * np.sqrt((1.0 - p[:, None]) / (n_rays * p[:, None]))
    used = p > 0
    z = (dep.power[used] - expected[used]) / sigma[used]
    assert np.max(np.abs(z)) < 4.5, np.round(z, 2)
    # the z slices of one annulus share their rays, so the independent test is per annulus
    ring = dep.power.sum(axis=1)[used]
    ring_exp = expected.sum(axis=1)[used]
    ring_sig = ring_exp * np.sqrt((1.0 - p[used]) / (n_rays * p[used]))
    chi2 = float(np.sum(((ring - ring_exp) / ring_sig) ** 2))
    dof = int(used.sum())
    assert abs(chi2 - dof) < 4.5 * math.sqrt(2.0 * dof), chi2
    # density is power over the annulus volume
    vol = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)[:, None] * np.diff(z_edges)[None, :]
    np.testing.assert_allclose(dep.density, dep.power / vol, rtol=1e-12)
    # the total the tally saw equals the closed form exactly for a lossless coating (no branch is drawn)
    assert dep.total == pytest.approx(FLUX * (1.0 - math.exp(-alpha * THICK)), rel=1e-9)


def test_a_bare_slab_absorbs_the_incoherent_slab_closed_form():
    alpha = 0.1
    scene = _scene(alpha)
    n_rays = 200_000
    res = scene.trace(num_rays=n_rays, seed=1, max_depth=32)
    R = ((N_GLASS - 1.0) / (N_GLASS + 1.0)) ** 2
    tau = math.exp(-alpha * THICK)
    closed = FLUX * (1.0 - R) * (1.0 - tau) * (1.0 + R * tau) / (1.0 - R * R * tau * tau)
    got = res.absorbed_by_component["window"]["bulk"]
    # the Fresnel branch is drawn per ray: the reflected share at the front face is binomial, sd sqrt(R(1-R)/N)
    sd = FLUX * (1.0 - tau) * math.sqrt(R * (1.0 - R) / n_rays)
    assert abs(got - closed) < 5.0 * sd, (got, closed, sd)


def test_a_lossy_coating_is_booked_to_its_component():
    T = 0.99
    scene = _scene(0.0, coating=SimpleCoating(transmittance=T, reflectance=0.0))
    res = scene.trace(num_rays=20_000, seed=5, max_depth=16)
    book = res.absorbed_by_component["window"]
    assert book["bulk"] == 0.0
    assert book["coating"] == pytest.approx(FLUX * ((1.0 - T) + T * (1.0 - T)), rel=1e-9)
    assert book["coating"] == pytest.approx(res.total_flux_coating, rel=1e-12)


def test_a_tally_names_a_component_the_scene_holds():
    scene = _scene(0.1)
    with pytest.raises(ValueError, match="no component named"):
        scene.add_deposition_tally("x", "nowhere", kind="rz", bounds=(0, 1, 0, 1), shape=(1, 1))
    with pytest.raises(ValueError, match="kind"):
        scene.add_deposition_tally("x", "window", kind="polar", bounds=(0, 1, 0, 1), shape=(1, 1))
    with pytest.raises(ValueError, match="bin counts"):
        scene.add_deposition_tally("x", "window", kind="xyz", bounds=(0, 1, 0, 1), shape=(1, 1))


def test_an_xyz_tally_holds_the_same_power_as_the_book():
    scene = _scene(0.1, coating=SimpleCoating(transmittance=1.0, reflectance=0.0))
    scene.add_deposition_tally("box", "window", kind="xyz", bounds=(-16, 16, -16, 16, 0, THICK), shape=(8, 8, 3))
    res = scene.trace(num_rays=20_000, seed=2, max_depth=16)
    dep = res.deposition["box"]
    assert dep.power.shape == (8, 8, 3)
    assert dep.total + dep.outside == pytest.approx(res.absorbed_by_component["window"]["bulk"], rel=1e-12)
    # symmetric beam on a symmetric grid: the four quadrants hold about the same power
    q = [dep.power[:4, :4].sum(), dep.power[4:, :4].sum(), dep.power[:4, 4:].sum(), dep.power[4:, 4:].sum()]
    assert max(q) / min(q) < 1.1
