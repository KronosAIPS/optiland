"""Per-trace running totals are compensated, so they do not depend on the batch size (issue 25).

KronosNSRT issue 25: the running totals of a trace (the ``Tally`` objects of
the loop and the ledger, the far-field detector's flux, an absorber's flux)
added one partial per batch in plain floating point, so their value depended
on the batch size: 2.6e-12 relative at batch size 1 on the catalogue's
diffuser, against the 1e-12 that section 10.3 of the theory allows an
unordered reduction. They now keep the rounding error of every addition
(Neumaier's variant of Kahan summation) and report the sum plus it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    AnnularPlaneGeometry,
    CollimatedSourceConfig,
    FarFieldDetectorConfig,
    HarveyShackBSDF,
    MirrorConfig,
    NSQScene,
    Spectrum,
    SurfaceConfig,
)
from optiland.nonsequential._tally import (
    Tally,
    accumulate_compensated,
    two_sum_error,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.absorbing import AbsorbingComponent

torch = pytest.importorskip("torch")
from optiland.nonsequential.backends.torch_backend import TorchBackend  # noqa: E402

U64 = 2.0**-53


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _terms(n: int = 20_000) -> list[float]:
    rng = np.random.default_rng(4)
    # Near-equal positive terms, the shape of a per-batch partial.
    return list(0.1 + 1e-3 * rng.random(n))


class TestTheCompensatedSum:
    def test_two_sum_error_is_exact(self):
        rng = np.random.default_rng(1)
        for _ in range(1000):
            a, b = rng.normal(size=2) * 10.0 ** rng.integers(-8, 8, size=2)
            s = a + b
            e = two_sum_error(a, b, s)
            assert math.fsum([a, b, -s, -e]) == 0.0

    def test_host_tally_is_within_an_ulp_of_the_exact_sum(self):
        terms = _terms()
        tally = Tally()
        plain = 0.0
        for x in terms:
            tally.add(x)
            plain += x
        exact = math.fsum(terms)
        assert abs(tally.value() - exact) <= abs(exact) * U64
        # The control: the plain sum of the same terms is many ulps off.
        assert abs(plain - exact) > 10 * abs(exact) * U64

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    def test_device_tally_is_compensated_in_place(self, dtype):
        terms = _terms()
        tally = Tally()
        tally.add(torch.tensor(terms[0], dtype=dtype))
        dev, comp = tally._dev, tally._dev_comp
        ptrs = (dev.data_ptr(), comp.data_ptr())
        for x in terms[1:]:
            tally.add(torch.tensor(x, dtype=dtype))
        # Added into in place from the first term to the last (a recorded
        # bounce writes the memory it recorded).
        assert tally._dev is dev and tally._dev_comp is comp
        assert (dev.data_ptr(), comp.data_ptr()) == ptrs
        exact = math.fsum(float(torch.tensor(x, dtype=dtype)) for x in terms)
        u = U64 if dtype == torch.float64 else 2.0**-24
        assert abs(tally.value() - exact) <= 2 * abs(exact) * u

    def test_integer_tally_is_unchanged(self):
        tally = Tally(is_int=True)
        for _ in range(10):
            tally.add(3)
            tally.add(torch.tensor(2))
        assert tally.value() == 50 and isinstance(tally.value(), int)
        assert tally._dev_comp is None

    def test_accumulate_compensated_on_host_and_device(self):
        terms = _terms(5_000)
        exact = math.fsum(terms)
        total, comp = 0.0, 0.0
        for x in terms:
            total, comp = accumulate_compensated(total, comp, x)
        assert abs((total + comp) - exact) <= abs(exact) * U64
        total, comp = 0.0, 0.0
        first = None
        for x in terms:
            total, comp = accumulate_compensated(
                total, comp, torch.tensor(x, dtype=torch.float64)
            )
            first = first or (total, comp)
        assert total is first[0] and comp is first[1]
        assert abs(float(total + comp) - exact) <= abs(exact) * U64

    def test_gradient_flows_through_the_sum(self):
        w = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
        tally = Tally()
        for k in range(5):
            tally.add(w * (k + 1))
        tally._dev.backward()
        assert float(w.grad) == pytest.approx(15.0)


def _scene() -> NSQScene:
    """A stop, a lossy scattering mirror and a far-field collector: every kind of total."""
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=0.7, aperture_radius=5.0
        ),
    )
    scene.add_component(
        "stop",
        AbsorbingComponent(
            cs=CoordinateSystem(z=20.0),
            geometry=AnnularPlaneGeometry(inner_radius=3.0, outer_radius=20.0),
            name="stop",
        ),
    )
    scene.add_mirror(
        "M1",
        CoordinateSystem(z=100),
        MirrorConfig(
            radius=0.0, reflectance=0.9, aperture_radius=50.0,
            surface=SurfaceConfig(
                bsdf=HarveyShackBSDF(b0=1e-3, l0=0.05, s=2.0), scatter_fraction=0.7
            ),
        ),
    )
    scene.add_detector(
        "FF", CoordinateSystem(z=-60), FarFieldDetectorConfig(num_theta=16, num_phi=32)
    )
    return scene


_TOTALS = (
    "total_flux_coating",
    "total_flux_absorbed",
    "total_flux_detected",
    "total_flux_escaped",
)


def _totals(kind: str, num_rays: int, batch_size: int) -> list[float]:
    if kind == "numpy":
        backend = NumpyBackend(seed=3)
    else:
        be.set_backend("torch")
        be.set_precision("float64")
        backend = TorchBackend(seed=3, alive_check_every=0)
    result = _scene().trace(
        num_rays=num_rays, seed=3, max_depth=8, batch_size=batch_size, backend=backend
    )
    return [getattr(result, k) for k in _TOTALS] + [
        float(result.detectors["FF"].total_flux)
    ]


@pytest.mark.parametrize(("kind", "num_rays"), [("numpy", 3000), ("torch", 1000)])
def test_totals_are_invariant_to_the_batch_size(kind, num_rays):
    """Section 10.3: within 1e-12 relative at batch sizes 1, 7 and the full count.

    The tighter bound (16 u) is what the compensation actually delivers: at
    batch size 1 the total is the exactly rounded sum of the weights, and at
    the full count one pairwise partial. The plain per-batch sum missed it by
    about 440 u on the absorbed flux at batch size 1 on numpy (measured on
    the engine before this change, 3000 rays).
    """
    full = _totals(kind, num_rays, num_rays)
    assert full[1] > 0 and full[0] > 0 and full[2] > 0
    for batch_size in (1, 7):
        got = _totals(kind, num_rays, batch_size)
        for name, x, ref in zip((*_TOTALS, "FF.total_flux"), got, full):
            scale = abs(ref) if ref else 1.0
            rel = abs(x - ref) / scale
            assert rel <= 1e-12, (batch_size, name, x, ref)
            assert rel <= 16 * U64, (batch_size, name, x, ref, rel / U64)


def test_the_compensation_replays():
    """The compensated adds run under the replay's emulate mode: no host transfer, no rebinding.

    ``graph_replay="emulate"`` runs the bounce that would be recorded under a
    check that refuses every host read and upload and every accumulator
    rebound after the capture; its numbers are the eager fixed-width ones.
    """
    be.set_backend("torch")
    be.set_precision("float64")
    eager = _scene().trace(
        num_rays=8192, seed=3, max_depth=8, batch_size=4096,
        backend=TorchBackend(seed=3, alive_check_every=0, compact_every=0),
    )
    emulated = _scene().trace(
        num_rays=8192, seed=3, max_depth=8, batch_size=4096,
        backend=TorchBackend(seed=3, alive_check_every=0, graph_replay="emulate"),
    )
    # The control: both batches were handed to the (emulated) replay.
    assert emulated.environment["graph_replay_batches"] == 2, emulated.environment
    for name in _TOTALS:
        assert getattr(emulated, name) == getattr(eager, name), name
    assert float(emulated.detectors["FF"].total_flux) == float(
        eager.detectors["FF"].total_flux
    )
