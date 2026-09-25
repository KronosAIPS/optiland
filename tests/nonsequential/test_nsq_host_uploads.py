"""No host-to-device copy inside the bounce loop, on the device backend.

The companion of ``test_nsq_host_reads.py``, in the other direction.
``docs/theory/12_gpu_mapping.md`` R-12-8 asks for scene data to be resolved
once per scene change, not per bounce. A bare Python number that an
``optiland.backend`` operation turns into a tensor (``be.where(m, -1.0,
1.0)``, ``be.maximum(x, 0.0)``) is built with ``torch.tensor`` on every call:
on a device that is a host-to-device copy per call, so per bounce -- a launch
and a synchronising copy each, and an operation a CUDA graph cannot capture
(issue 61 of the research repository). Such constants are held on the device
instead (``optiland.nonsequential._utils.resident_scalar``).

On the CPU there is no device, so the instrument counts the calls that would
be copies on one: every tensor built from host data -- ``torch.tensor``,
``torch.as_tensor``, ``torch.from_numpy`` or ``torch.asarray`` given
something that is not already a tensor. The loop is bracketed by the
backend's own ``intersect_scene``, as in the read counter, and only *steady*
bounces are judged: those that start and end on the bundle object the
previous bounce ran on, so a once-per-trace upload (a glass table, a
detector's bin edges) and a once-per-batch one (the fresh batch itself) are
not mistaken for per-bounce costs. The traces run at a fixed width
(``alive_check_every=0``), the configuration a CUDA graph captures.
"""

from __future__ import annotations

import collections
import traceback

import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# ruff: noqa: E402
import optiland.backend as be
from optiland.nonsequential import (
    VACUUM,
    AnnularPlaneGeometry,
    CollimatedSourceConfig,
    FarFieldDetectorConfig,
    HarveyShackBSDF,
    IrradianceDetectorConfig,
    LambertianBSDF,
    LensConfig,
    MirrorConfig,
    NSQMaterial,
    NSQScene,
    ReflectiveComponent,
    RefractiveComponent,
    SpectralDetectorConfig,
    Spectrum,
    SphereGeometry,
    SphericalCavityGeometry,
    SphericalPort,
    SurfaceConfig,
)
from optiland.nonsequential.backends.array_backend import ArrayBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.components.absorbing import AbsorbingComponent

#: The torch entry points that build a tensor from host data.
_UPLOADS = ("tensor", "as_tensor", "from_numpy", "asarray")


class _UploadCounter:
    """Count and attribute tensors built from host data during a trace."""

    def __init__(self) -> None:
        self._orig: dict[str, object] = {}
        self._orig_intersect = None
        self._interval: collections.Counter[str] = collections.Counter()
        self._last_bundle = None
        self._previous_was_same = False
        self._seen: set[str] = set()
        self.steady_bounces = 0
        self.recurring: collections.Counter[str] = collections.Counter()

    def _site(self) -> str:
        for frame in reversed(traceback.extract_stack(limit=25)[:-2]):
            if "/tests/" in frame.filename:
                continue
            if "/nonsequential/" in frame.filename:
                short = frame.filename.split("/nonsequential/")[-1]
                return f"{short}:{frame.lineno} {frame.name}"
        return "outside the engine"

    def __enter__(self) -> _UploadCounter:
        for name in _UPLOADS:
            orig = getattr(torch, name)
            self._orig[name] = orig

            def wrapper(data, *a, _orig=orig, **k):
                if not isinstance(data, torch.Tensor) and self._last_bundle is not None:
                    self._interval[self._site()] += 1
                return _orig(data, *a, **k)

            setattr(torch, name, wrapper)
        self._orig_intersect = ArrayBackend.intersect_scene

        def bracketed(backend, rays, *a, **k):
            same = rays is self._last_bundle
            # Steady: the bounce that just ended ran on the bundle the one
            # before it ran on, and the next one runs on it too. As in the
            # read counter, a site is counted from its second appearance, so
            # a constant built once on first use is not a per-bounce cost.
            if same and self._previous_was_same:
                self.steady_bounces += 1
                for site in self._interval:
                    if site in self._seen:
                        self.recurring[site] += 1
            self._seen.update(self._interval)
            self._interval = collections.Counter()
            self._previous_was_same = same
            self._last_bundle = rays
            return self._orig_intersect(backend, rays, *a, **k)

        ArrayBackend.intersect_scene = bracketed
        return self

    def __exit__(self, *exc) -> None:
        for name, orig in self._orig.items():
            setattr(torch, name, orig)
        ArrayBackend.intersect_scene = self._orig_intersect


def _collimated(scene: NSQScene, z: float = 0.0, radius: float = 5.0) -> None:
    scene.add_source(
        "S1",
        CoordinateSystem(z=z),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55),
            total_flux=1.0,
            aperture_radius=radius,
        ),
    )


def _singlet_with_stop() -> NSQScene:
    """A singlet (conic faces, frustum edge, medium stack) behind an annular stop."""
    scene = NSQScene()
    _collimated(scene)
    scene.add_component(
        "stop",
        AbsorbingComponent(
            cs=CoordinateSystem(z=20.0),
            geometry=AnnularPlaneGeometry(inner_radius=4.0, outer_radius=20.0),
        ),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50),
        LensConfig(
            r1=100.0,
            r2=-100.0,
            thickness=5.0,
            material="N-BK7",
            front_aperture_radius=12.5,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=150),
        IrradianceDetectorConfig(
            width=20, height=20, num_pixels_x=32, num_pixels_y=32, splat="bilinear"
        ),
    )
    return scene


def _ball_lens() -> NSQScene:
    """A glass sphere: the full-sphere geometry and its normal flip."""
    scene = NSQScene()
    _collimated(scene, radius=3.0)
    scene.add_component(
        "ball",
        RefractiveComponent(
            CoordinateSystem(z=30.0),
            SphereGeometry(5.0),
            VACUUM,
            NSQMaterial.from_glass("N-BK7"),
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=60),
        IrradianceDetectorConfig(width=40, height=40, num_pixels_x=16, num_pixels_y=16),
    )
    return scene


def _scattering() -> NSQScene:
    """A Harvey-Shack mirror: part of its hits go into the lobe (the scatter branch)."""
    scene = NSQScene()
    _collimated(scene)
    scene.add_detector(
        "TAP",
        CoordinateSystem(z=40),
        SpectralDetectorConfig(
            width=200, height=200, num_pixels_x=8, num_pixels_y=8,
            wl_min=0.4, wl_max=0.7, num_bins=4, splat="bilinear", absorb=False,
        ),
    )
    scene.add_mirror(
        "M1",
        CoordinateSystem(z=100),
        MirrorConfig(
            radius=0.0, reflectance=1.0, aperture_radius=50.0,
            surface=SurfaceConfig(
                bsdf=HarveyShackBSDF(b0=1e-3, l0=0.05, s=2.0), scatter_fraction=0.7
            ),
        ),
    )
    scene.add_detector(
        "FF", CoordinateSystem(z=-60), FarFieldDetectorConfig(num_theta=16, num_phi=32)
    )
    return scene


def _sphere_cavity() -> NSQScene:
    """An integrating sphere: the cavity wall's normal flip and a Lambertian lobe."""
    scene = NSQScene()
    scene.add_component(
        "wall",
        ReflectiveComponent(
            CoordinateSystem(),
            SphericalCavityGeometry(
                50.0, [SphericalPort.from_area_fraction((0.0, 0.0, -1.0), 0.02)]
            ),
            reflectance=0.95,
            bsdf=LambertianBSDF(reflectance_value=1.0),
            scatter_fraction=0.9,
            name="wall",
        ),
    )
    _collimated(scene, z=-40.0, radius=2.5)
    # Outside the port: what leaves the sphere through it.
    scene.add_detector(
        "exit",
        CoordinateSystem(z=-80.0),
        IrradianceDetectorConfig(width=60, height=60, num_pixels_x=8, num_pixels_y=8),
    )
    return scene


def _quarter_wave_ar():
    """The catalogue's single-layer coating (r1_13): an ideal quarter wave."""
    from optiland.materials import IdealMaterial
    from optiland.thin_film import ThinFilmStack

    stack = ThinFilmStack(
        incident_material=IdealMaterial(1.0),
        substrate_material=IdealMaterial(1.5168),
        reference_wl_um=0.5876,
    )
    stack.add_layer_qwot(IdealMaterial(1.2315843454672522), qwot_thickness=1.0)
    return stack


def _ten_layer_mirror():
    """The catalogue's multilayer stack (r1_23): five high/low quarter-wave pairs."""
    from optiland.materials import IdealMaterial
    from optiland.thin_film import ThinFilmStack

    stack = ThinFilmStack(
        incident_material=IdealMaterial(1.0),
        substrate_material=IdealMaterial(1.5),
        reference_wl_um=0.55,
    )
    for _ in range(5):
        stack.add_layer_qwot(IdealMaterial(2.32))
        stack.add_layer_qwot(IdealMaterial(1.38))
    return stack


def _coated_window(stack_fn):
    """A window whose front face carries a thin-film stack, evaluated every bounce."""

    def build() -> NSQScene:
        from optiland.materials import IdealMaterial
        from optiland.nonsequential import PlaneGeometry
        from optiland.nonsequential.components.coating_support import (
            UnpolarizedThinFilmCoating,
        )

        stack = stack_fn()
        substrate = NSQMaterial(optiland_material=stack.substrate_material)
        scene = NSQScene()
        _collimated(scene, z=-50.0, radius=2.5)
        scene.add_component(
            "front",
            RefractiveComponent(
                CoordinateSystem(z=0.0),
                PlaneGeometry(),
                VACUUM,
                substrate,
                coating=UnpolarizedThinFilmCoating(stack),
                name="front",
            ),
        )
        scene.add_component(
            "back",
            RefractiveComponent(
                CoordinateSystem(z=10.0),
                PlaneGeometry(),
                NSQMaterial(optiland_material=IdealMaterial(1.5168)),
                VACUUM,
                name="back",
            ),
        )
        for name, z in (("reflected", -100.0), ("transmitted", 50.0)):
            scene.add_detector(
                name,
                CoordinateSystem(z=z),
                IrradianceDetectorConfig(
                    width=40, height=40, num_pixels_x=4, num_pixels_y=4
                ),
            )
        return scene

    return build


_SCENES = {
    "singlet_with_stop": (_singlet_with_stop, 12),
    "ball_lens": (_ball_lens, 12),
    "scattering": (_scattering, 8),
    "sphere_cavity": (_sphere_cavity, 24),
    "quarter_wave_coated_window": (_coated_window(_quarter_wave_ar), 12),
    "ten_layer_coated_window": (_coated_window(_ten_layer_mirror), 12),
}


def _trace_and_count(scene_fn, max_depth: int) -> _UploadCounter:
    be.set_backend("torch")
    be.set_precision("float64")
    try:
        scene = scene_fn()
        with _UploadCounter() as counter:
            scene.trace(
                num_rays=8_192,
                seed=3,
                max_depth=max_depth,
                batch_size=4_096,
                backend=TorchBackend(seed=3, alive_check_every=0),
            )
        return counter
    finally:
        be.set_backend("numpy")


class TestNoHostUploadPerBounce:
    """R-12-8, issue 61: a fixed-width bounce builds no tensor from host data."""

    @pytest.mark.parametrize("scene_name", sorted(_SCENES))
    def test_a_steady_bounce_uploads_nothing(self, scene_name):
        scene_fn, depth = _SCENES[scene_name]
        counter = _trace_and_count(scene_fn, depth)
        # The control: a trace whose bundle changed every bounce would judge
        # nothing and pass vacuously.
        assert counter.steady_bounces >= 6, counter.steady_bounces
        assert dict(counter.recurring) == {}, (
            f"per-bounce host-to-device copies remain in {scene_name}: "
            f"{dict(counter.recurring)}"
        )

    def test_the_instrument_sees_a_bare_number(self, monkeypatch):
        """The control: a per-bounce ``be.where`` on bare numbers is seen."""
        from optiland.nonsequential.components.geometry.analytic import sphere

        original = sphere.SphereGeometry.ray_intersect

        def with_an_upload(self, origins, directions, eps=None):
            be.where(directions[:, 2] > 0, -1.0, 1.0)
            return original(self, origins, directions, eps)

        monkeypatch.setattr(sphere.SphereGeometry, "ray_intersect", with_an_upload)
        counter = _trace_and_count(_ball_lens, 12)
        # Attributed to the engine frame that called the patched method; the
        # patch itself lives in this file, which the attribution skips.
        assert counter.recurring, "the instrument missed a per-bounce upload"
        assert all(n >= counter.steady_bounces - 1 for n in counter.recurring.values())

    def test_the_constants_are_the_values_the_bare_numbers_were(self):
        """A resident constant has the operand's dtype and device, as before."""
        from optiland.nonsequential._utils import resident_scalar

        class _Owner:
            pass

        owner = _Owner()
        for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
            like = torch.zeros(3, dtype=dtype)
            c = resident_scalar(owner, "k", -1, like)
            assert c.dtype == dtype and c.shape == () and c.device == like.device
            assert c.item() == -1
            # Built once: the same object on the next call.
            assert resident_scalar(owner, "k", -1, like) is c
        # A host array gets the number back untouched.
        import numpy as np

        assert resident_scalar(owner, "k", -1.0, np.zeros(3)) == -1.0

    def test_the_thin_film_floor_is_the_value_where_built(self):
        """The coating's denominator floor: the tensor ``be.where`` built, once."""
        import numpy as np

        from optiland.thin_film.core import _denominator_floor

        for dtype in (torch.complex64, torch.complex128):
            like = torch.zeros(3, dtype=dtype)
            floor = _denominator_floor(like)
            built = torch.tensor(1e-30 + 0j, dtype=dtype, device=like.device)
            assert floor.dtype == dtype and floor.shape == ()
            assert floor.device == like.device
            assert torch.equal(torch.view_as_real(floor), torch.view_as_real(built))
            assert _denominator_floor(like) is floor
        # NumPy is handed the bare number, as before.
        assert _denominator_floor(np.zeros(3, dtype=complex)) == 1e-30 + 0j
