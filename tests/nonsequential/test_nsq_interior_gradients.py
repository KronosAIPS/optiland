"""The interior-gradient contract: the register, attached placements, the dead-parameter raise.

``docs/theory/09_differentiation.md`` of the research repository, requirements
R-09-2 (the register), R-09-3 (positions and orientations are attached),
R-09-5 (a dead parameter raises) and tests T-09-2, T-09-3 and T-09-12; the
research repository's issue 31.

How the finite differences are kept honest
-------------------------------------------
Every comparison is autograd against the fourth-order central difference

    f'(p) ~ (-f(p + 2h) + 8 f(p + h) - 8 f(p - h) + f(p - 2h)) / (12 h)

at float64 on the CPU, with common random numbers: every evaluation uses the
same seed, so each ray draws the same numbers at every point of the stencil.
The draws are kept fixed further by the scenes themselves: no ray comes near
an aperture edge, a detector edge or a pixel centre of the 2 x 2 bilinear
detector (the loss below is then linear in every landing position), mirrors
reflect with probability one, and the lens scenes fix the Fresnel branch
probability (``SamplingPolicy(reflect_prob=1e-6)``) so no branch decision can
flip between the stencil's points. The loss is therefore a smooth function of
the parameter at every stencil point, and the difference quotient converges.

The loss is the flux-weighted landing centroid ``x + y / 2`` on a 2 x 2
bilinear detector: inside the square between the four pixel centres a
bilinear splat reproduces a linear function exactly, so
``sum(data * x_centre) / sum(data)`` is the flux-weighted mean landing
coordinate, with no pixel discretisation in it.

Tolerance, derived rather than chosen: the stencil's roundoff error is about
``1.5 * eps_f / h``, with ``eps_f`` the loss's own rounding error. A bound on
``eps_f`` is ``K * u * |f|`` with ``u = 2**-53`` and ``K = 1e4`` operations (a
generous count: about a hundred floating-point operations per ray per bounce,
eight bounces, and a 2,000-term sum whose rounding grows at most linearly).
The truncation error ``h**4 |f^(5)| / 30`` is below 1e-10 absolute at the steps
used (the loss is linear in a translation of the source or the detector and a
tangent of a small angle times a lever of about 100 mm for a rotation). The
test's relative tolerance is ``max(1e-7, 1.5 * K * u * |f| / (h * |f'|))``,
which is 1e-7 for every case here, a thousand times below the 1e-4 T-09-3 asks
for. The agreement measured on the development machine (Apple silicon, CPU,
float64, 2,000 rays, seed 3) was between 1e-14 and 1e-10 relative.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.ir.scene_ir import SamplingPolicy
from optiland.nonsequential.parameter_register import (
    BOUNDARY_ONLY,
    INTERIOR,
    INTERIOR_BOUNDARY,
    UNCLASSIFIED,
    DeadParameterError,
    ParameterRefused,
    ParameterRegister,
)

_NUM_RAYS = 2_000
_SEED = 3
_U = 2.0**-53
_K_OPS = 1e4


@pytest.fixture(autouse=True)
def _torch_float64():
    be.set_backend("torch")
    be.set_precision("float64")
    yield
    be.set_backend("numpy")


def _g(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def _centroid(data, width: float):
    """Flux-weighted landing ``x + y / 2`` on a 2 x 2 bilinear detector of side ``width``."""
    q = width / 4.0
    xc = torch.tensor([-q, q, -q, q], dtype=torch.float64)
    yc = torch.tensor([-q, -q, q, q], dtype=torch.float64)
    s = data.sum()
    return (data * xc).sum() / s + 0.5 * (data * yc).sum() / s


def _detector(scene, cs, width):
    scene.add_detector(
        "D1",
        cs,
        IrradianceDetectorConfig(
            width=width, height=width, num_pixels_x=2, num_pixels_y=2, splat="bilinear"
        ),
    )


def _source(scene, cs, radius=2.0, flux=1.0):
    scene.add_source(
        "S1",
        cs,
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=flux, aperture_radius=radius
        ),
    )


# -- the scenes: nominal placements and builders -------------------------------

_DETECTOR = {"x": 0.3, "y": -0.2, "z": 100.0, "rx": 0.02, "ry": -0.01, "rz": 0.05}
_SOURCE = {"x": 0.4, "y": -0.3, "z": 0.0, "rx": -0.02, "ry": 0.03, "rz": 0.0}
_MIRROR = {"x": 0.0, "y": 0.0, "z": 100.0, "rx": 0.02, "ry": 0.2, "rz": 0.0}
_LENS = {"x": 0.5, "y": -0.3, "z": 50.0, "rx": 0.01, "ry": -0.02, "rz": 0.0}


def _detector_scene(p):
    """An oblique collimated beam onto a displaced, tilted detector."""
    scene = NSQScene()
    _source(scene, CoordinateSystem(rx=-0.02, ry=0.03))
    _detector(scene, CoordinateSystem(**p), 40.0)
    return scene, 40.0


def _source_scene(p):
    """A displaced, tilted collimated source onto a fixed detector."""
    scene = NSQScene()
    _source(scene, CoordinateSystem(**p))
    _detector(scene, CoordinateSystem(z=100.0), 40.0)
    return scene, 40.0


def _mirror_scene(p):
    """A tilted concave mirror folding the beam onto a detector beside the source."""
    scene = NSQScene()
    _source(scene, CoordinateSystem())
    scene.add_mirror(
        "M1",
        CoordinateSystem(**p),
        MirrorConfig(radius=-800.0, reflectance=1.0, aperture_radius=25.0),
    )
    _detector(scene, CoordinateSystem(x=-25.0, z=40.0), 20.0)
    return scene, 20.0


def _lens_scene(p, thickness=5.0):
    """A decentred, tilted plano-convex singlet; the Fresnel branch probability fixed."""
    scene = NSQScene()
    _source(scene, CoordinateSystem(), radius=3.0)
    scene.add_lens(
        "L1",
        CoordinateSystem(**p),
        LensConfig(
            r1=60.0,
            r2=float("inf"),
            thickness=thickness,
            material="N-BK7",
            front_aperture_radius=12.0,
        ),
    )
    _detector(scene, CoordinateSystem(z=150.0), 40.0)
    scene.sampling_policy = SamplingPolicy(reflect_prob=1e-6)
    return scene, 40.0


def _trace(scene, **kw):
    return scene.trace(num_rays=_NUM_RAYS, seed=_SEED, max_depth=8, **kw)


def _placement_loss(build, nominal, key):
    def loss(value):
        p = dict(nominal)
        p[key] = value
        scene, width = build(p)
        return _centroid(_trace(scene).detectors["D1"].data, width)

    return loss


def _check_against_fd4(loss, value: float, h: float) -> float:
    """Assert autograd equals the fourth-order central difference; return the relative gap."""
    param = _g(value)
    f = loss(param)
    assert f.requires_grad, "the parameter never reached the autograd graph"
    (grad,) = torch.autograd.grad(f, param)
    ad = grad.item()
    assert np.isfinite(ad)
    with torch.no_grad():
        fp2, fp1, fm1, fm2 = (float(loss(value + m * h)) for m in (2, 1, -1, -2))
    fd = (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)
    assert fd != 0.0, "the loss does not move: the comparison would mean nothing"
    tol = max(1e-7, 1.5 * _K_OPS * _U * abs(f.detach().item()) / (h * abs(fd)))
    rel = abs(ad - fd) / abs(fd)
    assert rel < tol, f"autograd {ad:.12e} vs FD4 {fd:.12e}: relative {rel:.2e} > {tol:.1e}"
    return rel


def _step(key: str) -> float:
    return 1e-2 if key in ("x", "y", "z") else 1e-3


# -- T-09-3: positions are live, and right --------------------------------------


class TestPlacementGradients:
    """Autograd of the landing centroid against FD4, one rigid-transform parameter at a time."""

    @pytest.mark.parametrize("key", ["x", "y", "z", "rx", "ry", "rz"])
    def test_detector_placement(self, key):
        """All six rigid-transform parameters of a detector (T-09-3)."""
        _check_against_fd4(
            _placement_loss(_detector_scene, _DETECTOR, key), _DETECTOR[key], _step(key)
        )

    @pytest.mark.parametrize("key", ["x", "y", "z", "rx", "ry"])
    def test_source_placement(self, key):
        """A source's position and tilt (its rotation about its own axis is a symmetry of a round beam)."""
        _check_against_fd4(
            _placement_loss(_source_scene, _SOURCE, key), _SOURCE[key], _step(key)
        )

    @pytest.mark.parametrize("key", ["z", "rx", "ry"])
    def test_mirror_tilt_and_shift(self, key):
        """Tilt a mirror (and move it along its axis): the folded beam's landing point."""
        _check_against_fd4(
            _placement_loss(_mirror_scene, _MIRROR, key), _MIRROR[key], _step(key)
        )

    @pytest.mark.parametrize("key", ["x", "y", "z", "rx", "ry"])
    def test_lens_shift_and_tilt(self, key):
        """Shift and tilt a lens: every sub-surface's transform shares the one placement."""
        _check_against_fd4(
            _placement_loss(_lens_scene, _LENS, key), _LENS[key], _step(key)
        )

    def test_lens_thickness_through_the_reference_chain(self):
        """The back surface sits at ``z = thickness`` in the front's frame: a chained placement."""

        def loss(t):
            scene, width = _lens_scene(dict(_LENS), thickness=t)
            return _centroid(_trace(scene).detectors["D1"].data, width)

        _check_against_fd4(loss, 5.0, 1e-2)

    def test_source_translation_moves_the_centroid_one_for_one(self):
        """An independent route: a translated source moves every landing point by the same amount."""
        loss = _placement_loss(_source_scene, _SOURCE, "x")
        param = _g(_SOURCE["x"])
        (grad,) = torch.autograd.grad(loss(param), param)
        assert grad.item() == pytest.approx(1.0, rel=1e-12)


# -- attaching moves no value ----------------------------------------------------


def _ledger(result):
    return (
        result.total_flux_detected,
        result.total_flux_escaped,
        result.total_flux_absorbed,
        result.total_flux_sampling_residual,
        result.num_rays_escaped,
    )


class TestAttachingMovesNoValue:
    """The attached transform's value is the host build's, to the bit (the regression rule)."""

    @pytest.mark.parametrize(
        ("build", "nominal"),
        [
            (_detector_scene, _DETECTOR),
            (_source_scene, _SOURCE),
            (_mirror_scene, _MIRROR),
            (_lens_scene, _LENS),
        ],
        ids=["detector", "source", "mirror", "lens"],
    )
    def test_forward_values_bit_identical(self, build, nominal):
        plain_scene, _ = build(dict(nominal))
        attached_scene, _ = build({k: _g(v) for k, v in nominal.items()})
        plain = _trace(plain_scene)
        attached = _trace(attached_scene)
        a = plain.detectors["D1"].data
        b = attached.detectors["D1"].data
        assert b.requires_grad and not a.requires_grad
        assert torch.equal(a, b.detach())
        assert _ledger(plain) == _ledger(attached)

    def test_nothing_attached_means_no_register(self):
        scene, _ = _lens_scene(dict(_LENS))
        result = _trace(scene)
        assert not hasattr(result, "parameter_register")
        assert "gradient_boundary_term" not in result.environment
        assert not ParameterRegister.from_scene(scene)


# -- T-09-2: the register --------------------------------------------------------


def _every_kind_scene():
    """A scene holding one parameter of every kind the register classifies."""
    scene = NSQScene()
    _source(scene, CoordinateSystem(x=_g(0.1)), radius=3.0, flux=_g(1.0))
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50.0, rx=_g(0.01)),
        LensConfig(
            r1=_g(60.0),
            r2=float("inf"),
            thickness=_g(5.0),
            material="N-BK7",
            front_aperture_radius=12.0,
            conic1=_g(-0.5),
        ),
    )
    scene.add_mirror(
        "M1",
        CoordinateSystem(z=150.0, ry=_g(0.2)),
        MirrorConfig(radius=-800.0, reflectance=_g(0.9), aperture_radius=25.0),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(x=-20.0, z=_g(90.0)),
        IrradianceDetectorConfig(
            width=_g(30.0), height=30.0, num_pixels_x=8, num_pixels_y=8, splat="bilinear"
        ),
    )
    return scene


_EXPECTED_ROWS = {
    ("S1", "cs.x"): INTERIOR_BOUNDARY,
    ("S1", "total_flux"): INTERIOR,
    ("L1.front", "cs.rx"): INTERIOR_BOUNDARY,
    ("L1.front", "geometry.radius"): INTERIOR_BOUNDARY,
    ("L1.front", "geometry.conic"): INTERIOR_BOUNDARY,
    ("L1.back", "cs.z"): INTERIOR_BOUNDARY,  # the thickness, through the reference chain
    ("M1.surface", "cs.ry"): INTERIOR_BOUNDARY,
    ("M1.surface", "reflectance"): INTERIOR,
    ("D1", "cs.z"): INTERIOR_BOUNDARY,
    ("D1", "width"): INTERIOR_BOUNDARY,
}


class TestRegister:
    """What the register holds, and that it matches the code (T-09-2)."""

    def test_every_kind_is_registered_and_classified(self):
        scene = _every_kind_scene()
        register = ParameterRegister.from_scene(scene)
        got = {(r["owner"], r["name"]): r["gradient_class"] for r in register.rows()}
        assert got == _EXPECTED_ROWS
        assert UNCLASSIFIED not in got.values()
        for row in register.rows():
            assert row["boundary_term"] == (
                "none" if row["gradient_class"] == INTERIOR else "absent"
            )
            assert "tensor" not in row
        # The lens placement is one tensor shared by three sub-surfaces.
        assert register.find("L1.front", "cs.rx").also_owned_by == ["L1.back", "L1.edge"]

    def test_register_is_on_the_result_and_every_entry_is_live(self):
        """Each registered parameter has a finite, non-zero gradient of the detector image."""
        scene = _every_kind_scene()
        result = _trace(scene)
        register = result.parameter_register
        assert result.environment["gradient_boundary_term"] == "absent"
        data = result.detectors["D1"].data
        weights = torch.arange(data.numel(), dtype=data.dtype)
        loss = (data * weights).sum()
        grads = torch.autograd.grad(loss, [e.tensor for e in register])
        for entry, g in zip(register, grads, strict=True):
            assert g is not None and torch.isfinite(g), (entry.owner, entry.name)
            assert g.item() != 0.0, (entry.owner, entry.name)

    def test_the_same_scene_traced_twice_registers_the_same_parameters(self):
        """Caches filled by a trace are not mistaken for parameters by the next one."""
        scene = _every_kind_scene()
        first = [(r["owner"], r["name"]) for r in _trace(scene).parameter_register.rows()]
        second = [(r["owner"], r["name"]) for r in _trace(scene).parameter_register.rows()]
        assert first == second


# -- T-09-12: a dead parameter raises --------------------------------------------


class TestDeadParameterRaise:
    """A parameter with no live path to the outputs raises, with its reason (R-09-5)."""

    def test_detached_by_contract_boundary_only(self):
        """A clear-aperture radius enters only an in-or-out test: detached by contract."""
        scene, _ = _lens_scene(dict(_LENS))
        scene.component_registry.get("L1").surfaces[0].geometry.aperture_radius = _g(12.0)
        with pytest.raises(DeadParameterError) as info:
            _trace(scene)
        (owner, name, stage, reason) = info.value.dead[0]
        assert (owner, name) == ("L1.front", "geometry.aperture_radius")
        assert "in-or-out test" in stage or "aperture" in stage
        assert "detached by contract" in reason
        assert len(info.value.dead) == 1
        register = ParameterRegister.from_scene(scene)
        assert register.find("L1.front", "geometry.aperture_radius").gradient_class == BOUNDARY_ONLY

    def test_a_surface_no_ray_reaches(self):
        """A mirror beside the beam: its placement and reflectance cannot influence the tallies."""
        scene, _ = _detector_scene(dict(_DETECTOR))
        scene.add_mirror(
            "M9",
            CoordinateSystem(x=200.0, z=_g(80.0)),
            MirrorConfig(radius=-500.0, reflectance=_g(0.9), aperture_radius=10.0),
        )
        with pytest.raises(DeadParameterError) as info:
            _trace(scene)
        dead = {(o, n): r for o, n, _s, r in info.value.dead}
        assert set(dead) == {("M9.surface", "cs.z"), ("M9.surface", "reflectance")}
        for reason in dead.values():
            assert "cannot influence the tallies" in reason
            assert "no ray reached M9.surface" in reason
        # The finished trace is on the error, for a caller that expected the zero.
        assert info.value.result is not None
        assert info.value.result.total_flux_detected > 0.0

    def test_no_output_depends_on_it(self):
        """A placement on the graph of nothing the trace returns: raised, not a None gradient."""
        scene, _ = _detector_scene(dict(_DETECTOR))
        spare = _g(1.0)
        # A parameter hung on the source that no computation reads.
        scene.sources[0].unused_design_variable = spare
        with pytest.raises(DeadParameterError) as info:
            _trace(scene)
        ((owner, name, _stage, reason),) = info.value.dead
        assert (owner, name) == ("S1", "unused_design_variable")
        assert "no output of the trace depends on it" in reason

    def test_forward_only_evaluation_does_not_raise(self):
        """Under torch.no_grad nothing is recorded and nothing is claimed: no raise."""
        scene, _ = _detector_scene(dict(_DETECTOR))
        scene.add_mirror(
            "M9",
            CoordinateSystem(x=200.0, z=_g(80.0)),
            MirrorConfig(radius=-500.0, reflectance=1.0, aperture_radius=10.0),
        )
        with torch.no_grad():
            result = _trace(scene)
        assert len(result.parameter_register) == 1

    def test_a_backend_without_autograd_refuses(self):
        """R-09-3: the NumPy backend refuses a gradient-carrying parameter instead of detaching it."""
        scene, _ = _detector_scene({**_DETECTOR, "z": _g(100.0)})
        be.set_backend("numpy")
        with pytest.raises(ParameterRefused, match="D1:cs.z"):
            _trace(scene)


# -- the graph replay and its emulation -------------------------------------------


class TestGraphReplayWithAttachedPlacement:
    """A replayed bounce records no autograd graph: refused in gradient mode, clean without it."""

    @staticmethod
    def _scene(attached: bool):
        p = dict(_LENS)
        if attached:
            p = {k: _g(v) for k, v in p.items()}
        scene, _ = _lens_scene(p)
        return scene

    @staticmethod
    def _replay_trace(scene):
        from optiland.nonsequential.backends.torch_backend import TorchBackend

        backend = TorchBackend(seed=_SEED, graph_replay="emulate")
        return scene.trace(
            num_rays=4_096, seed=_SEED, max_depth=8, batch_size=4_096, backend=backend
        )

    def test_refused_in_gradient_mode(self):
        from optiland.nonsequential.backends.graph_replay import GraphReplayUnavailable

        with pytest.raises(GraphReplayUnavailable, match="carries a gradient"):
            self._replay_trace(self._scene(attached=True))

    def test_emulated_capture_with_attached_placement_has_no_host_transfer(self):
        """The attached transform adds no host transfer to the bounce a capture would record.

        Run forward-only (no graph is recorded, so the replay may proceed):
        the emulated capture's host-transfer check passes, a batch was
        replayed, and the values equal the same emulated trace of the
        detached scene to the bit.
        """
        with torch.no_grad():
            attached = self._replay_trace(self._scene(attached=True))
            plain = self._replay_trace(self._scene(attached=False))
        assert attached.environment["graph_replay"] == "emulate"
        assert attached.environment["graph_replay_batches"] == 1
        assert torch.equal(attached.detectors["D1"].data, plain.detectors["D1"].data)
        assert _ledger(attached) == _ledger(plain)


# -- T-09-6: forward mode equals reverse mode --------------------------------------


_ALL_PLACEMENTS = [
    *(("detector", _detector_scene, _DETECTOR, k) for k in ("x", "y", "z", "rx", "ry", "rz")),
    *(("source", _source_scene, _SOURCE, k) for k in ("x", "y", "z", "rx", "ry")),
    *(("mirror", _mirror_scene, _MIRROR, k) for k in ("z", "rx", "ry")),
    *(("lens", _lens_scene, _LENS, k) for k in ("x", "y", "z", "rx", "ry")),
]


class TestForwardModeCrossCheck:
    """One trace in forward mode against the reverse-mode gradient (R-09-9, T-09-6).

    Forward mode here is ``torch.autograd.forward_ad``: the placement is a
    dual tensor, the register recognises its tangent as a derivative to
    attach, and the landing centroid comes out with its directional
    derivative. The two modes compute the same derivative with the same
    operations in a different order (and forward mode, being forward-only,
    runs with compaction on), so they differ by rounding only. The bound is
    the one the finite-difference tolerance above uses for the loss's own
    rounding, ``K * u`` relative with ``K = 1e4`` operations and
    ``u = 2**-53``: about 1.1e-12. Chapter 09's T-09-6 asks for ``64 u``
    (7.1e-15); measured on the development machine (Apple silicon, CPU,
    float64, 2,000 rays, seed 3), 13 of these 19 cases are within it and the
    worst, the lens tilts, are at about 240 u and 950 u.
    """

    @pytest.mark.parametrize(
        ("build", "nominal", "key"),
        [case[1:] for case in _ALL_PLACEMENTS],
        ids=[f"{case[0]}-{case[3]}" for case in _ALL_PLACEMENTS],
    )
    def test_forward_equals_reverse(self, build, nominal, key):
        from torch.autograd import forward_ad

        loss = _placement_loss(build, nominal, key)
        param = _g(nominal[key])
        (reverse,) = torch.autograd.grad(loss(param), param)
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(
                torch.tensor(nominal[key], dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
            )
            tangent = forward_ad.unpack_dual(loss(dual)).tangent
        assert tangent is not None, "the forward-mode tangent never reached the output"
        rel = abs(tangent.item() - reverse.item()) / abs(reverse.item())
        assert rel < _K_OPS * _U, f"forward {tangent.item():.15e} reverse {reverse.item():.15e}"
