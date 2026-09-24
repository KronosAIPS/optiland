"""Soundness tests for NSQMaterial's n()/k() identity memo.

The memo (``NSQMaterial._n_memo``/``_k_memo``, added alongside the fix for
``test_nsq_host_reads.py``) reuses a material's last n()/k() result whenever
it is called again with the *same* wavelength array object, to avoid paying
``optiland.materials.BaseMaterial``'s per-call host synchronization on every
bounce. Its correctness rests on two facts, each pinned by a test here:

1. A cached result must not outlive the material parameters that produced
   it. ``ArrayBackend.trace()`` clears every material's memo at the start of
   every trace (``NSQMaterial.reset_memo()``), so a material changed between
   two traces that happen to reuse the same wavelength array object cannot
   read back a stale result -- ``TestMemoClearedAcrossTraces``.
2. The memo's hit test (object identity) is only sound because nsq never
   writes into a ray bundle's wavelength array in place -- every update
   (masking, compaction, backend promotion) replaces it with a fresh array.
   ``TestWavelengthArrayNeverMutatedInPlace`` pins that assumption directly,
   so a future change that started mutating it in place would fail here
   before it could silently stale the memo.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry


def _fresnel_normal_incidence_transmittance(n1: float, n2: float) -> float:
    """Unpolarized Fresnel transmittance at normal incidence: T = 1 - R."""
    r = (n1 - n2) / (n1 + n2)
    return 1.0 - r * r


def _single_interface_scene(index: float) -> tuple[NSQScene, NSQMaterial]:
    """One flat vacuum -> IdealMaterial(index) interface, on-axis.

    A collimated on-axis beam meets the interface square-on, so the
    transmitted ray direction never bends (Snell's law at zero incidence),
    but the Fresnel transmittance ``1 - ((n1-n2)/(n1+n2))**2`` still
    depends on ``index`` -- the flux behind the interface is therefore a
    clean, deterministic-geometry probe of whether ``n()`` is reading the
    material's *current* index. A big, absorbing detector right behind the
    interface catches essentially all of the transmitted flux; the
    reflected flux heads back toward the source and escapes the scene, so
    ``total_flux_detected / total_flux_in`` is a direct Monte Carlo estimate
    of the transmittance.
    """
    material = NSQMaterial(optiland_material=IdealMaterial(n=index))
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0
        ),
    )
    scene.add_component(
        "I1",
        RefractiveComponent(
            cs=CoordinateSystem(z=10.0),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=material,
            name="I1",
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=10.5),
        IrradianceDetectorConfig(
            width=40, height=40, num_pixels_x=8, num_pixels_y=8, splat="bilinear"
        ),
    )
    return scene, material


class TestMemoClearedAcrossTraces:
    """R1 hazard 1: a material change between two traces must be seen."""

    def test_changed_index_changes_the_second_traces_transmittance(self):
        scene, material = _single_interface_scene(index=1.5)
        n_rays = 200_000

        result_1 = scene.trace(num_rays=n_rays, seed=7)
        flux_1 = result_1.total_flux_detected

        # Change the material's own index between the two traces -- direct
        # parameter reassignment, the same route an optimizer step or a
        # notebook cell would use.
        material.optiland_material.index = np.array([2.0])

        result_2 = scene.trace(num_rays=n_rays, seed=7)
        flux_2 = result_2.total_flux_detected

        expected_1 = _fresnel_normal_incidence_transmittance(1.0, 1.5)
        expected_2 = _fresnel_normal_incidence_transmittance(1.0, 2.0)
        assert expected_1 == pytest.approx(0.96, abs=1e-6)
        assert expected_2 == pytest.approx(8.0 / 9.0, abs=1e-6)

        # Each trace's measured transmittance tracks its own index -- not
        # the other's -- well outside Monte Carlo noise at this ray count
        # (binomial standard error here is ~9e-4).
        assert flux_1 == pytest.approx(expected_1, abs=0.01)
        assert flux_2 == pytest.approx(expected_2, abs=0.01)
        assert flux_2 < flux_1 - 0.03, (
            "the second trace did not follow the changed index -- the memo "
            f"may be serving a stale n(): flux_1={flux_1}, flux_2={flux_2}, "
            f"expected roughly {expected_1} then {expected_2}"
        )

    def test_trace_resets_every_materials_memo(self):
        """The structural half of the fix, checked directly.

        The physics test above is a real end-to-end regression check, but
        by itself it is not a discriminating one: ``Spectrum.sample()``
        allocates a fresh wavelength array on every ``trace()`` call, so
        under the public API the two traces essentially never reuse the
        same array object regardless of whether ``ArrayBackend.trace()``
        resets anything -- checked directly, this test still passed with
        the reset loop in ``ArrayBackend.trace()`` removed. What must
        actually be pinned is that ``ArrayBackend.trace()`` calls
        ``NSQMaterial.reset_memo()`` for every material the scene holds, on
        every trace, which is what makes the physics test's guarantee hold
        in general rather than by the accident of fresh allocation. Spies
        on ``NSQMaterial.reset_memo`` and asserts it is called for both
        materials this scene holds (the vacuum in front, the glass behind),
        each trace.
        """
        scene, material = _single_interface_scene(index=1.5)
        calls: list[int] = []
        orig_reset = NSQMaterial.reset_memo

        def spy_reset(self):
            calls.append(id(self))
            return orig_reset(self)

        NSQMaterial.reset_memo = spy_reset
        try:
            scene.trace(num_rays=1_000, seed=1)
            first_trace_calls = list(calls)
            calls.clear()
            scene.trace(num_rays=1_000, seed=1)
            second_trace_calls = list(calls)
        finally:
            NSQMaterial.reset_memo = orig_reset

        for label, trace_calls in (
            ("first", first_trace_calls),
            ("second", second_trace_calls),
        ):
            assert id(material) in trace_calls, (
                f"{label} trace did not reset the glass's memo"
            )
            assert id(VACUUM) in trace_calls, (
                f"{label} trace did not reset the front medium's memo"
            )

    def test_reset_memo_directly_drops_the_cached_value(self):
        """The primitive itself, isolated from the scene/trace machinery."""
        material = NSQMaterial(optiland_material=IdealMaterial(n=1.5))
        wl = np.full(1000, 0.55)

        n_before = np.asarray(material.n(wl))
        assert np.all(n_before == 1.5)

        # Without a reset, the memo (correctly) still serves the old value
        # for the *same* wavelength object -- this is the fast path working
        # as designed, not a bug.
        material.optiland_material.index = np.array([2.0])
        n_stale = np.asarray(material.n(wl))
        assert np.all(n_stale == 1.5), (
            "sanity check on the memo itself: expected the unreset memo to "
            "still serve the pre-change value"
        )

        material.reset_memo()
        n_after = np.asarray(material.n(wl))
        assert np.all(n_after == 2.0)


class TestWavelengthArrayNeverMutatedInPlace:
    """R1 hazard 2: the memo's soundness assumption, pinned directly."""

    def test_wavelength_object_is_stable_and_unmutated_across_bounces(self):
        """Every array NSQMaterial.n() sees keeps the same identity and
        content from the moment it is first seen to the end of the trace.

        Wraps ``NSQMaterial.n`` for the duration of one multi-bounce torch
        trace and records, per call, the wavelength array's object id and a
        bounce sequence number (via the same ``ArrayBackend.intersect_scene``
        bracketing ``test_nsq_host_reads.py`` uses). Asserts two things:
        a steady bounce (one that runs on the same ray-state buffer as the
        bounce before and after it) hands ``n()`` the *same* wavelength
        object as the previous bounce did, and that object's content -- a
        host-side snapshot taken the first time it is seen, compared against
        a fresh host-side read at the end of the trace -- never changes.
        The comparison itself reads the host; it is the test doing that,
        not the traced code.
        """
        torch = pytest.importorskip("torch", reason="Torch not available")

        import optiland.backend as be
        from optiland.nonsequential.backends.array_backend import ArrayBackend
        from optiland.nonsequential.backends.torch_backend import TorchBackend

        scene, _material = _single_interface_scene(index=1.5)

        calls: list[tuple[int, int]] = []  # (bundle_seq, id(wavelength))
        first_snapshot: dict[int, torch.Tensor] = {}
        live_ref: dict[int, torch.Tensor] = {}

        orig_n = NSQMaterial.n

        def wrapped_n(self, wavelength_um):
            if isinstance(wavelength_um, torch.Tensor):
                oid = id(wavelength_um)
                calls.append((bundle_seq[0], oid))
                if oid not in first_snapshot:
                    first_snapshot[oid] = wavelength_um.detach().clone()
                    live_ref[oid] = wavelength_um  # keep it alive by identity
            return orig_n(self, wavelength_um)

        bundle_seq = [0]
        last_bundle = [None]
        orig_intersect = ArrayBackend.intersect_scene

        def bracketed_intersect(backend, rays, *a, **k):
            if rays is not last_bundle[0]:
                bundle_seq[0] += 1
                last_bundle[0] = rays
            return orig_intersect(backend, rays, *a, **k)

        be.set_backend("torch")
        be.set_precision("float64")
        NSQMaterial.n = wrapped_n
        ArrayBackend.intersect_scene = bracketed_intersect
        try:
            scene.trace(
                num_rays=100_000,
                seed=11,
                max_depth=8,
                batch_size=25_000,
                backend=TorchBackend(seed=11, alive_check_every=0),
            )
        finally:
            NSQMaterial.n = orig_n
            ArrayBackend.intersect_scene = orig_intersect
            be.set_backend("numpy")

        assert calls, "NSQMaterial.n() was never called -- the probe scene is wrong"

        # Content: every wavelength object n() ever saw is unchanged from
        # its first sighting to the end of the trace. [host read: the test]
        for oid, snapshot in first_snapshot.items():
            current = live_ref[oid]
            assert torch.equal(current, snapshot), (
                f"wavelength array {oid} changed content during the trace -- "
                "the memo's no-in-place-mutation assumption is broken"
            )

        # Identity: within a run of consecutive calls that share a bundle
        # sequence number (a "steady" stretch -- no compaction/new batch
        # landed between them), the wavelength object id must not change.
        by_bundle: dict[int, set[int]] = {}
        for seq, oid in calls:
            by_bundle.setdefault(seq, set()).add(oid)
        multi_object_bundles = {
            seq: oids for seq, oids in by_bundle.items() if len(oids) > 1
        }
        assert not multi_object_bundles, (
            "more than one wavelength array object was seen within a single "
            f"bundle generation: {multi_object_bundles}"
        )
        assert len(by_bundle) >= 3, (
            f"only {len(by_bundle)} distinct bundle generations seen -- too "
            "few to say anything about steady-bounce stability"
        )
