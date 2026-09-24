"""Numerical launch caches must retain array-valued snapshot precision."""

import pickle
from collections.abc import Iterator
from contextlib import ExitStack

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays.ray_aiming.cached import CachedRayAimer
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer
from optiland.rays.ray_aiming.robust import RobustRayAimer

from .utils import assert_allclose


@pytest.fixture
def numerical_backend(set_test_backend: None) -> Iterator[None]:
    """Exercise real numerical cache hits with both Torch gradient controls off."""
    with ExitStack() as stack:
        stack.callback(be.set_precision, f"float{be.get_precision()}")
        be.set_precision("float64")
        if be.get_backend() == "torch":
            import torch

            grad_mode = be.grad_mode
            stack.callback(setattr, grad_mode, "requires_grad", grad_mode.requires_grad)
            grad_mode.disable()
            stack.enter_context(torch.no_grad())
        yield


def _plane_system() -> Optic:
    """Use an air-space first stop so the launch and NA cone are analytic."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=100.0, material=IdealMaterial(1.0))
    optic.surfaces.add(
        index=1, thickness=10.0, material=IdealMaterial(1.0), is_stop=True
    )
    optic.surfaces.add(index=2, material=IdealMaterial(1.0))
    optic.set_aperture("objectNA", be.array(0.05))
    optic.fields.set_type("object_height")
    optic.fields.add(y=be.array(2.125))
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _snapshot(optic: Optic) -> tuple:
    return (
        optic.surfaces.to_dict(),
        optic.fields.to_dict(),
        optic.wavelengths.to_dict(),
        optic.aperture.to_dict(),
        optic.ray_tracer.ray_aiming_config,
    )


def _assert_launch(launch: tuple, na: float, height: float, hy: float) -> None:
    stop_y = 100.0 * na / np.sqrt(1.0 - na**2)
    x, y, z, L, M, N = launch
    assert_allclose(x - z * L / N, 0.0, rtol=0, atol=1e-12)
    assert_allclose(y - z * M / N, stop_y, rtol=0, atol=1e-12)
    origin_y = height * hy
    direction = np.array([0.0, stop_y - origin_y, 100.0])
    direction /= np.linalg.norm(direction)
    for actual, expected in zip(
        launch, (0.0, origin_y, -100.0, *direction), strict=True
    ):
        assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "aimer_type", [ParaxialRayAimer, IterativeRayAimer, RobustRayAimer]
)
def test_unchanged_array_snapshot_is_an_exact_hit(
    numerical_backend: None, aimer_type: type
) -> None:
    """Repeated snapshots and real solver calls retain exact numerical reuse."""
    optic = _plane_system()
    aimer = CachedRayAimer(optic, aimer_type(optic, tol=1e-12))
    inputs = ((0.0, 0.7), 0.55, (0.0, 1.0))
    snapshot = pickle.dumps(_snapshot(optic))
    original = aimer.aim_rays(*inputs)
    _assert_launch(original, 0.05, 2.125, 0.7)
    assert pickle.dumps(_snapshot(optic)) == snapshot
    assert aimer.aim_rays(*inputs) is original
    assert pickle.dumps(_snapshot(optic)) == snapshot


@pytest.mark.parametrize(
    ("aimer_type", "dependency"),
    [
        (ParaxialRayAimer, "aperture"),
        (IterativeRayAimer, "aperture"),
        (RobustRayAimer, "aperture"),
        # Robust field cases are excluded: its independent PupilMapCache still
        # rounds field fingerprints, which is outside this wrapper's scope.
        (ParaxialRayAimer, "field"),
        (IterativeRayAimer, "field"),
    ],
)
@pytest.mark.parametrize("in_place", [False, True], ids=["replace", "in_place"])
def test_sub_repr_snapshot_change_matches_current_physical_launch(
    numerical_backend: None, aimer_type: type, dependency: str, in_place: bool
) -> None:
    """Distinct array values with identical repr must not return an old launch."""
    optic = _plane_system()
    aimer = CachedRayAimer(optic, aimer_type(optic, tol=1e-12))
    hy = 0.0 if dependency == "aperture" else 0.7
    inputs = ((0.0, hy), 0.55, (0.0, 1.0))
    original = aimer.aim_rays(*inputs)
    assert aimer.aim_rays(*inputs) is original
    _assert_launch(original, 0.05, 2.125, hy)
    text = str(_snapshot(optic))
    payload = pickle.dumps(_snapshot(optic))
    assert pickle.dumps(_snapshot(optic)) == payload

    delta = 1e-5 if be.get_backend() == "torch" else 1e-10
    na, height = 0.05, 2.125
    if dependency == "aperture":
        na += delta
        if in_place:
            optic.aperture.value[...] += delta
        else:
            optic.set_aperture("objectNA", be.array(na))
    else:
        height += delta
        if in_place:
            optic.fields[0].y[...] += delta
        else:
            optic.fields[0].y = be.array(height)

    assert str(_snapshot(optic)) == text
    assert pickle.dumps(_snapshot(optic)) != payload
    current_payload = pickle.dumps(_snapshot(optic))
    assert pickle.dumps(_snapshot(optic)) == current_payload
    fresh = aimer_type(optic, tol=1e-12).aim_rays(*inputs)
    _assert_launch(fresh, na, height, hy)
    changed_component = 4 if dependency == "aperture" else 1
    assert not np.allclose(
        be.to_numpy(original[changed_component]),
        be.to_numpy(fresh[changed_component]),
        rtol=0,
        atol=1e-12,
    )

    current = aimer.aim_rays(*inputs)
    _assert_launch(current, na, height, hy)
    for actual, expected in zip(current, fresh, strict=True):
        assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert current is not original
    assert aimer.aim_rays(*inputs) is current
