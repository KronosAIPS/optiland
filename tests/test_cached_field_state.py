"""Full-launch caches must not freeze field-dependent launch constraints."""

from unittest.mock import patch

import numpy as np
import pytest

import optiland.backend as be
from optiland.fields import FieldGroup
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures.radial import RadialAperture
from optiland.rays import RealRays
from optiland.rays.ray_aiming.cached import CachedRayAimer
from optiland.rays.ray_aiming.iterative import IterativeRayAimer

from .test_cached_aimer import numerical_backend  # noqa: F401
from .utils import assert_allclose


def _field_system(infinite: bool, field_type: str) -> Optic:
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf if infinite else 100.0)
    optic.surfaces.add(
        index=1, radius=50.0, thickness=5.0, material=IdealMaterial(1.5)
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=15.0)
    optic.surfaces.add(index=3, thickness=40.0, is_stop=True)
    optic.surfaces.add(index=4)
    optic.surfaces[3].aperture = RadialAperture(r_max=4.0)
    optic.set_aperture("float_by_stop_size", 8.0)
    optic.fields.set_type(field_type)
    optic.fields.add(y=0.0)
    optic.fields.add(y=1.0)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _assert_stop_target(optic: Optic, launch: tuple, pupil: tuple) -> None:
    rays = RealRays(
        *(be.copy(value) for value in launch), intensity=1.0, wavelength=0.55
    )
    start = 1 if optic.object_surface.is_infinite else 0
    for index in range(start, optic.surfaces.stop_index + 1):
        optic.surfaces[index].trace(rays)
    optic.surfaces[optic.surfaces.stop_index].geometry.cs.localize(rays)
    assert_allclose(rays.x, pupil[0] * 4.0, atol=1e-9, rtol=0.0)
    assert_allclose(rays.y, pupil[1] * 4.0, atol=1e-9, rtol=0.0)
    assert be.all(be.isfinite(rays.x))
    assert be.all(rays.i > 0.0)


@pytest.mark.parametrize("mode", ["iterative", "robust"])
@pytest.mark.parametrize(
    "infinite,field_type",
    [(False, "object_height"), (False, "angle"), (True, "angle")],
    ids=["finite-height", "finite-angle", "infinite-angle"],
)
@pytest.mark.parametrize("change", ["replace", "scale"])
@pytest.mark.usefixtures("numerical_backend")
def test_field_change_matches_cold_launch(
    mode: str, infinite: bool, field_type: str, change: str
) -> None:
    optic = _field_system(infinite, field_type)
    optic.ray_tracer.set_aiming(mode, cache=True, max_iter=30, tol=1e-10)
    config = optic.ray_tracer.ray_aiming_config
    generator = optic.ray_tracer.ray_generator
    generator.set_ray_aiming(**config)
    warm = generator.aimer
    assert isinstance(warm, CachedRayAimer)
    inputs = ((0.25, 1.0), 0.55, (0.2, -0.3))
    original = warm.aim_rays(*inputs)
    assert warm.aim_rays(*inputs) is original
    _assert_stop_target(optic, original, inputs[2])

    if change == "replace":
        replacement = FieldGroup()
        replacement.set_type(field_type)
        replacement.add(y=0.0)
        replacement.add(y=3.0)
        optic.fields = replacement
    else:
        optic.fields[1].y *= 3.0

    with patch.object(
        warm.wrapped_aimer, "aim_rays", wraps=warm.wrapped_aimer.aim_rays
    ) as delegate:
        warmed = warm.aim_rays(*inputs)
        delegate.assert_called_once_with(*inputs)
        assert warm.aim_rays(*inputs) is warmed
        delegate.assert_called_once()
    assert warm.wrapped_aimer.last_report.converged
    generator.set_ray_aiming(**config)
    cold = generator.aimer
    fresh = cold.aim_rays(*inputs)
    assert cold.wrapped_aimer.last_report.converged
    for actual, expected in zip(warmed, fresh, strict=True):
        assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
    _assert_stop_target(optic, warmed, inputs[2])
    _assert_stop_target(optic, fresh, inputs[2])

    # An old full launch still hits the stop, but encodes the previous field.
    control = IterativeRayAimer(optic, max_iter=30, tol=1e-10)
    stale = control.aim_rays(*inputs, initial_guess=original)
    assert control.last_report.converged
    _assert_stop_target(optic, stale, inputs[2])
    fixed = slice(3, 6) if infinite else slice(0, 3)
    assert any(
        not np.allclose(be.to_numpy(old), be.to_numpy(new), atol=1e-6, rtol=0.0)
        for old, new in zip(stale[fixed], fresh[fixed], strict=True)
    )
