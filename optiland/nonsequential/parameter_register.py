"""The parameter register and the attached placements of the interior gradient.

What this module is for (``docs/theory/09_differentiation.md`` of the research
repository, requirements R-09-2, R-09-3 and R-09-5; the research repository's
issue 31):

- **The register.** :class:`ParameterRegister` lists every tensor a user
  attached to a scene -- a tensor with ``requires_grad=True`` on a placement
  (the ``x``, ``y``, ``z``, ``rx``, ``ry``, ``rz`` of a coordinate system and of
  its reference chain), a surface's radius, conic constant or aperture, a
  material or coating parameter, a source's flux, a detector's extent -- with
  its owner, its name and its gradient class from chapter 09 (below). The trace
  fills it when it uploads the scene and checks it after the loop.
- **Attached placements.** Before this module a placement was read through
  :func:`~optiland.nonsequential.components.base._get_transform`, which copies
  it to the host as NumPy float64, so a surface, detector or source position
  could not be optimised although the rest of the interior gradient was exact.
  :func:`attach_placement` and :func:`attach_source_placement` now build the
  transform a second time from the attached tensors, on the device, in the
  working dtype, with differentiable operations (the rotation from the
  coordinate system's own three angles, ``Rz @ Ry @ Rx``, composed down the
  reference chain), and join the two so that the **value** is the detached host
  build, to the bit, and the **derivative** is the device build's:

      T = T_host + (T_dev - sg(T_dev))

  with ``sg`` the stop-gradient (``detach``). The bracket is exactly zero in
  value, so attaching a placement moves no forward number, whatever precision
  or device (the regression rule: every catalogue value is unchanged with or
  without a gradient); its derivative is the derivative of the device build.
  The host build stays the value's owner, so the rounding policy of the
  research repository's issue 22 (rotation matrices built in float64 on the
  host and rounded once to the working dtype) applies to an attached placement
  exactly as to a detached one when it lands; the tangent is formed in the
  working dtype on the device, where a float32 derivative carries float32
  rounding (about 1e-7 relative), which is the derivative's own error budget
  and not a forward difference.
- **The dead-parameter raise.** After a trace in which autograd is enabled,
  :meth:`ParameterRegister.check` walks the autograd graph of every output of
  the trace (detector buffers and results, surface ledgers) and raises
  :class:`DeadParameterError` for every registered parameter the graph does
  not reach, naming its owner, its name, the stage at which its path ends and
  the reason the register knows: *detached by contract* (a parameter whose
  only effect is a boundary term, or one chapter 09 detaches), or *cannot
  influence the tallies in this scene* (no ray reached its owner, or none of
  the reached paths depends on it). A silent zero is what R-09-5 forbids.

Gradient classes (chapter 09 sections 9.2, 9.3 and 9.7):

``interior``
    The pathwise (interior) gradient is the whole derivative: the parameter
    moves no discontinuity. Source flux, reflectances, transmittances,
    absorptances, scatter fractions.
``interior+boundary``
    The pathwise gradient is exact on the interior, and a boundary term also
    exists that the pathwise gradient misses (an aperture edge, an occluder's
    silhouette, a detector edge, the onset of total internal reflection, a
    swap of the nearest hit), chapter 09 section 9.3's table. Placements,
    curvatures, conic constants, thicknesses, refractive indices, detector
    extents. The boundary term is not computed (the research repository's
    issue 3); a gradient result says so (``boundary_term="absent"``).
``boundary-only``
    The parameter enters only an in-or-out test: a clear aperture's radius, a
    finite plane's width and height, an annulus' radii. Its pathwise gradient
    is structurally zero; if it reaches no output it is dead by contract.
``detached``
    Detached by contract (chapter 09 R-09-2): source sampling geometry until
    its change of variables lands (R-09-4). The constructors of those kinds
    refuse a gradient-carrying value today, so the class is listed in
    :data:`CONTRACT` and seldom meets a live tensor.

Nothing in this module runs when no tensor in the scene requires a gradient:
the register is empty, :func:`attach_placement` returns the host transform it
is given, and every number of the trace is the one it always was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.ray_bundle import NSQRayBundle

#: The gradient classes of chapter 09, in the order the module docstring
#: gives them.
INTERIOR = "interior"
INTERIOR_BOUNDARY = "interior+boundary"
BOUNDARY_ONLY = "boundary-only"
DETACHED = "detached"
UNCLASSIFIED = "unclassified"

#: The rigid-transform fields of a coordinate system.
PLACEMENT_FIELDS = ("x", "y", "z", "rx", "ry", "rz")

#: Where a placement's gradient enters the trace, by owner kind.
_PLACEMENT_STAGE = {
    "surface": "the surface's rigid transform (intersection, normal, local frame)",
    "detector": "the detector's rigid transform (intersection, binning coordinate)",
    "source": "the source's rigid transform (emission point and direction)",
}

#: Parameter name -> (gradient class, stage at which the path enters or ends).
#: Names are the attribute's last component. The first match wins; a name not
#: listed is ``unclassified``, which the register's test forbids for the
#: built-in kinds.
CONTRACT: dict[str, tuple[str, str]] = {
    # surface shape: through the intersection and the normal
    "radius": (INTERIOR_BOUNDARY, "the intersection and the normal (sag, curvature)"),
    "conic": (INTERIOR_BOUNDARY, "the intersection and the normal (sag)"),
    "focal_length": (INTERIOR_BOUNDARY, "the paraxial deflection"),
    "pitch_x": (INTERIOR_BOUNDARY, "the lenslet cell frame"),
    "pitch_y": (INTERIOR_BOUNDARY, "the lenslet cell frame"),
    "z_offset": (INTERIOR_BOUNDARY, "the intersection"),
    "z_front": (INTERIOR_BOUNDARY, "the intersection"),
    "z_back": (INTERIOR_BOUNDARY, "the intersection"),
    "r_front": (INTERIOR_BOUNDARY, "the intersection and the normal"),
    "r_back": (INTERIOR_BOUNDARY, "the intersection and the normal"),
    # clear apertures and finite extents: an in-or-out test only
    "aperture_radius": (BOUNDARY_ONLY, "the clear-aperture test (a boolean)"),
    "inner_radius": (BOUNDARY_ONLY, "the annulus test (a boolean)"),
    "outer_radius": (BOUNDARY_ONLY, "the annulus test (a boolean)"),
    # radiometric weights: interior only
    "total_flux": (INTERIOR, "the birth weight"),
    "scatter_fraction": (INTERIOR, "the scatter branch weight"),
    "reflectance": (INTERIOR, "the reflect weight"),
    "reflectance_value": (INTERIOR, "the scatter weight"),
    "transmittance": (INTERIOR, "the transmit weight"),
    "absorptance": (INTERIOR, "the absorbed weight"),
    # materials: Fresnel, Snell and Beer-Lambert; the critical angle moves
    "n": (INTERIOR_BOUNDARY, "Fresnel, Snell and Beer-Lambert"),
    "k": (INTERIOR_BOUNDARY, "Beer-Lambert and Fresnel"),
    "index": (INTERIOR_BOUNDARY, "Fresnel and Snell"),
}

#: Parameters a detector carries: its extent is a binning coordinate
#: (attached through a bilinear or Gaussian splat) and an edge.
_DETECTOR_CONTRACT: dict[str, tuple[str, str]] = {
    "width": (INTERIOR_BOUNDARY, "the binning coordinate (pixel pitch) and the edge"),
    "height": (INTERIOR_BOUNDARY, "the binning coordinate (pixel pitch) and the edge"),
    "radius": (INTERIOR_BOUNDARY, "the binning coordinate and the edge"),
}

#: Parameters of a surface geometry whose finite extent is only an edge.
_GEOMETRY_EXTENT = {"width", "height"}

#: Source-geometry parameters: detached by contract until the change of
#: variables of chapter 09 section 9.7 lands (R-09-4). Their constructors
#: refuse a gradient-carrying value; the table is here so the register can
#: name the reason if one ever arrives.
_SOURCE_CONTRACT: dict[str, tuple[str, str]] = {
    "aperture_radius": (DETACHED, "source sampling on the host (R-09-4 pending)"),
    "gaussian_sigma": (DETACHED, "source sampling on the host (R-09-4 pending)"),
    "half_angle_deg": (DETACHED, "source sampling on the host (R-09-4 pending)"),
    "width": (DETACHED, "source sampling on the host (R-09-4 pending)"),
    "height": (DETACHED, "source sampling on the host (R-09-4 pending)"),
}

#: Attribute names never walked for parameters: back references and scene
#: plumbing, not parameters.
_SKIP_ATTRS = {"cs", "reference_cs", "spectrum", "name"}

#: How deep the walk follows nested objects of the engine (component ->
#: material -> the shared library's material -> its coefficients).
_MAX_DEPTH = 3


class DeadParameterError(RuntimeError):
    """A registered parameter reached no output of the trace (R-09-5).

    Attributes:
        dead: The dead parameters, each ``(owner, name, stage, reason)``.
        result: The finished :class:`SimulationResult` of the trace, so a
            caller that expected the zero can still read it.
    """

    def __init__(self, dead: list[tuple[str, str, str, str]], result=None) -> None:
        self.dead = dead
        self.result = result
        lines = [
            f"{owner}: {name} -- its path ends at {stage}; {reason}"
            for owner, name, stage, reason in dead
        ]
        super().__init__(
            "parameters requested for differentiation reach no output of the "
            "trace (a gradient of None, not a zero; docs/theory/"
            "09_differentiation.md R-09-5):\n  " + "\n  ".join(lines)
        )


class ParameterRefused(RuntimeError):
    """A parameter requires a gradient on a backend that cannot give one (R-09-3)."""


@dataclass
class RegisteredParameter:
    """One attached tensor of the scene.

    Attributes:
        owner: The scene name of the object holding it (a surface such as
            ``"L1.front"``, a detector, a source), or ``"(material)"`` for a
            shared material reached from several surfaces.
        name: Its dotted attribute path from the owner (``"cs.rx"``,
            ``"cs.reference_cs.z"``, ``"geometry.radius"``).
        tensor: The tensor itself (a leaf or not).
        kind: ``"surface"``, ``"detector"`` or ``"source"``.
        gradient_class: One of the classes of the module docstring.
        stage: Where its gradient path enters the trace.
        also_owned_by: Other owners that hold the same tensor (a lens's
            front and edge share one coordinate system).
    """

    owner: str
    name: str
    tensor: Any
    kind: str
    gradient_class: str
    stage: str
    also_owned_by: list[str] = field(default_factory=list)

    @property
    def boundary_term(self) -> str:
        """``"absent"`` where a boundary term exists and is not computed, else ``"none"``."""
        if self.gradient_class in (INTERIOR_BOUNDARY, BOUNDARY_ONLY):
            return "absent"
        return "none"

    def row(self) -> dict[str, Any]:
        """The register's row for this parameter, without the tensor."""
        return {
            "owner": self.owner,
            "name": self.name,
            "kind": self.kind,
            "gradient_class": self.gradient_class,
            "boundary_term": self.boundary_term,
            "stage": self.stage,
            "shape": tuple(getattr(self.tensor, "shape", ())),
            "also_owned_by": list(self.also_owned_by),
        }


def _requires_grad(value: Any) -> bool:
    return bool(getattr(value, "requires_grad", False)) and hasattr(value, "grad_fn")


def _is_engine_object(value: Any) -> bool:
    module = type(value).__module__ or ""
    return module.startswith("optiland") and not isinstance(value, type)


def placement_tensors(cs) -> list[tuple[str, Any]]:
    """The gradient-carrying fields of a coordinate system and its reference chain.

    Args:
        cs: A :class:`~optiland.coordinate_system.CoordinateSystem`, or None.

    Returns:
        ``[(dotted name, tensor)]``, e.g. ``("cs.rx", t)`` or
        ``("cs.reference_cs.z", t)``, in chain order.
    """
    out: list[tuple[str, Any]] = []
    prefix = "cs"
    seen: set[int] = set()
    while cs is not None and id(cs) not in seen:
        seen.add(id(cs))
        for f in PLACEMENT_FIELDS:
            value = getattr(cs, f"_{f}", None)
            if _requires_grad(value):
                out.append((f"{prefix}.{f}", value))
        cs = getattr(cs, "reference_cs", None)
        prefix += ".reference_cs"
    return out


def placement_is_attached(cs) -> bool:
    """True when any field of ``cs`` or of its reference chain requires a gradient."""
    return bool(placement_tensors(cs))


# ---------------------------------------------------------------------------
# The attached transform
# ---------------------------------------------------------------------------


def _scalar_like(value, like):
    """``value`` as a 0-d tensor of ``like``'s dtype and device, keeping its graph."""
    import torch  # noqa: PLC0415

    if torch.is_tensor(value):
        return value.to(dtype=like.dtype, device=like.device).reshape(())
    return torch.tensor(float(np.asarray(value).reshape(())), dtype=like.dtype, device=like.device)


def _rotation_from_angles(rx, ry, rz):
    """``Rz @ Ry @ Rx`` from three 0-d tensors, differentiably (the coordinate system's convention)."""
    import torch  # noqa: PLC0415

    one = torch.ones_like(rx)
    zero = torch.zeros_like(rx)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    Rx = torch.stack(
        [torch.stack([one, zero, zero]), torch.stack([zero, cx, -sx]), torch.stack([zero, sx, cx])]
    )
    Ry = torch.stack(
        [torch.stack([cy, zero, sy]), torch.stack([zero, one, zero]), torch.stack([-sy, zero, cy])]
    )
    Rz = torch.stack(
        [torch.stack([cz, -sz, zero]), torch.stack([sz, cz, zero]), torch.stack([zero, zero, one])]
    )
    return Rz @ Ry @ Rx


def device_transform(cs, like):
    """The placement ``(translation, rotation)`` built on the device from the attached tensors.

    Mirrors :meth:`~optiland.coordinate_system.CoordinateSystem
    .get_effective_transform` operation for operation (``t_eff = t_ref +
    R_ref @ t``, ``R_eff = R_ref @ R``), in ``like``'s dtype and on its
    device, with every step differentiable. No host transfer: the angles and
    offsets are read as tensors, never as Python floats (a plain number in the
    chain is a constant and becomes a new constant tensor).

    Args:
        cs: The coordinate system.
        like: A tensor whose dtype and device the transform takes.

    Returns:
        ``(translation (3,), rotation (3, 3))`` as tensors.
    """
    import torch  # noqa: PLC0415

    t = torch.stack([_scalar_like(getattr(cs, f"_{f}"), like) for f in ("x", "y", "z")])
    R = _rotation_from_angles(*(_scalar_like(getattr(cs, f"_{f}"), like) for f in ("rx", "ry", "rz")))
    ref = getattr(cs, "reference_cs", None)
    if ref is None:
        return t, R
    t_ref, R_ref = device_transform(ref, like)
    return t_ref + R_ref @ t, R_ref @ R


def _tangent_only(x):
    """``x - sg(x)``: exactly zero in value, the derivative of ``x`` in the graph."""
    return x - x.detach()


def attach_placement(cs, translation, rotation):
    """Attach a placement's transform to the autograd graph, keeping its value.

    With nothing in ``cs`` (or its reference chain) requiring a gradient the
    host-built pair is returned as it was given -- the detached default and
    every existing number. Otherwise the transform is built again on the
    device (:func:`device_transform`) and joined to the host value so that the
    value is the host pair's bit for bit and the derivative is the device
    build's (the module docstring's formula).

    Args:
        cs: The owner's coordinate system.
        translation: The host-built translation, already an array of the
            active backend (what ``be.array(_get_transform(cs)[0])`` gave).
        rotation: The host-built rotation, likewise.

    Returns:
        ``(translation, rotation)`` of the active backend.
    """
    if be.get_backend() != "torch" or not placement_is_attached(cs):
        return translation, rotation
    t_dev, R_dev = device_transform(cs, rotation)
    return translation + _tangent_only(t_dev), rotation + _tangent_only(R_dev)


def attach_source_placement(rays: NSQRayBundle, source) -> NSQRayBundle:
    """Attach a source's placement to the rays it has just emitted.

    Sources sample on the host and place their rays with the detached
    float64 transform. A rigid placement moves every emitted point and
    direction by the same transform (Jacobian determinant one, so the birth
    weight is unchanged; chapter 09 section 9.7), so the attachment is done
    here, once per batch, on the device: with ``p_l = R0^T (p - t0)`` and
    ``d_l = R0^T d`` the local point and direction the host placed,

        p <- p + dR p_l + dt,   d <- d + dR d_l,

    where ``dt`` and ``dR`` are the device build's tangent-only parts (zero
    in value). The values are the ones the source emitted, to the bit; the
    derivative is that of ``t(theta) + R(theta) p_l``.

    Args:
        rays: The batch as the backend prepared it (tensors on the device).
        source: The source that emitted it.

    Returns:
        The same bundle, its position and direction attached when the
        source's placement carries a gradient; untouched otherwise.
    """
    cs = getattr(source, "cs", None)
    if cs is None or be.get_backend() != "torch" or not placement_is_attached(cs):
        return rays
    from optiland.nonsequential.components.base import _get_transform  # noqa: PLC0415

    t_host, R_host = _get_transform(cs)
    R0 = be.array(R_host)
    t0 = be.array(t_host)
    t_dev, R_dev = device_transform(cs, R0)
    dt = _tangent_only(t_dev)
    dR = _tangent_only(R_dev)
    p = be.stack([rays.x, rays.y, rays.z], axis=1)
    d = be.stack([rays.L, rays.M, rays.N], axis=1)
    p_local = (p - t0) @ R0
    d_local = d @ R0
    p_new = p + p_local @ dR.T + dt
    d_new = d + d_local @ dR.T
    rays.x, rays.y, rays.z = p_new[:, 0], p_new[:, 1], p_new[:, 2]
    rays.L, rays.M, rays.N = d_new[:, 0], d_new[:, 1], d_new[:, 2]
    return rays


# ---------------------------------------------------------------------------
# The register
# ---------------------------------------------------------------------------


def _classify(kind: str, path: str) -> tuple[str, str]:
    """The gradient class and stage of a non-placement parameter at ``path``."""
    leaf = path.rsplit(".", 1)[-1]
    if kind == "source" and leaf in _SOURCE_CONTRACT:
        return _SOURCE_CONTRACT[leaf]
    if kind == "detector" and "." not in path and leaf in _DETECTOR_CONTRACT:
        return _DETECTOR_CONTRACT[leaf]
    if leaf in _GEOMETRY_EXTENT and path.startswith("geometry."):
        return BOUNDARY_ONLY, "the finite-extent test (a boolean)"
    if leaf in CONTRACT:
        return CONTRACT[leaf]
    return UNCLASSIFIED, "not in the contract table"


def _walk(obj: Any, prefix: str, depth: int, seen: set[int]) -> Iterator[tuple[str, Any]]:
    """Every gradient-carrying tensor reachable from ``obj``'s public attributes."""
    if id(obj) in seen or depth > _MAX_DEPTH:
        return
    seen.add(id(obj))
    attrs = getattr(obj, "__dict__", None)
    if not attrs:
        return
    for key, value in list(attrs.items()):
        if key.startswith("_") or key in _SKIP_ATTRS:
            continue
        path = f"{prefix}{key}"
        if _requires_grad(value):
            yield path, value
        elif _is_engine_object(value):
            yield from _walk(value, path + ".", depth + 1, seen)


class ParameterRegister:
    """Every attached tensor of a scene, with its owner, name and gradient class.

    Built by :meth:`from_scene` when the trace uploads the scene; iterate it,
    :meth:`rows` for a table, :meth:`find` for one entry, :meth:`check` after
    the trace. Empty (falsy) when nothing in the scene requires a gradient.
    """

    def __init__(self, entries: Iterable[RegisteredParameter] = ()) -> None:
        self.entries: list[RegisteredParameter] = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[RegisteredParameter]:
        return iter(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def rows(self) -> list[dict[str, Any]]:
        """The register as a list of plain rows (no tensors)."""
        return [e.row() for e in self.entries]

    def find(self, owner: str, name: str) -> RegisteredParameter:
        """The entry of ``owner`` (or one of its co-owners) named ``name``.

        Raises:
            KeyError: If there is none.
        """
        for e in self.entries:
            if e.name == name and (e.owner == owner or owner in e.also_owned_by):
                return e
        raise KeyError(f"no registered parameter {owner}:{name}")

    @classmethod
    def from_scene(cls, scene) -> ParameterRegister:
        """Register every gradient-carrying tensor of ``scene``.

        Walks each surface, detector and source: its placement (the
        coordinate system and its reference chain) and its public attributes,
        following nested engine objects (geometry, materials, scatter models,
        coatings) a few levels deep. Private attributes (caches, memos,
        accumulators) are not parameters and are not walked. A tensor held by
        several owners is registered once, under the first, with the others
        listed.

        Args:
            scene: The scene about to be traced.

        Returns:
            The register (empty when nothing requires a gradient).
        """
        entries: list[RegisteredParameter] = []
        by_id: dict[int, RegisteredParameter] = {}

        def add(owner: str, kind: str, path: str, tensor, klass: str, stage: str) -> None:
            known = by_id.get(id(tensor))
            if known is not None:
                if owner != known.owner and owner not in known.also_owned_by:
                    known.also_owned_by.append(owner)
                return
            entry = RegisteredParameter(owner, path, tensor, kind, klass, stage)
            by_id[id(tensor)] = entry
            entries.append(entry)

        for owner, kind, obj in _scene_objects(scene):
            for path, tensor in placement_tensors(getattr(obj, "cs", None)):
                add(owner, kind, path, tensor, INTERIOR_BOUNDARY, _PLACEMENT_STAGE[kind])
            for path, tensor in _walk(obj, "", 0, set()):
                klass, stage = _classify(kind, path)
                add(owner, kind, path, tensor, klass, stage)
        return cls(entries)

    # -- after the trace ---------------------------------------------------

    def live(self, roots: Iterable[Any]) -> dict[int, bool]:
        """Which entries the autograd graph of ``roots`` reaches.

        A walk of the recorded graph (no backward pass, no gradient written):
        a leaf is live when its accumulator node is in the graph, a non-leaf
        when its own node is.

        Args:
            roots: Output tensors of the trace.

        Returns:
            ``{index into entries: live}``.
        """
        nodes: set[Any] = set()
        leaves: set[int] = set()
        stack = [r.grad_fn for r in roots if getattr(r, "grad_fn", None) is not None]
        while stack:
            node = stack.pop()
            if node is None or node in nodes:
                continue
            nodes.add(node)
            variable = getattr(node, "variable", None)
            if variable is not None:
                leaves.add(id(variable))
            for nxt, _ in getattr(node, "next_functions", ()):
                if nxt is not None and nxt not in nodes:
                    stack.append(nxt)
        out: dict[int, bool] = {}
        for i, e in enumerate(self.entries):
            fn = getattr(e.tensor, "grad_fn", None)
            out[i] = (fn in nodes) if fn is not None else (id(e.tensor) in leaves)
        return out

    def check(self, roots: Iterable[Any], reached: dict[str, int] | None = None, result=None) -> None:
        """Raise for every registered parameter no output of the trace depends on.

        Args:
            roots: The trace's output tensors.
            reached: Rays that reached each owner (surfaces by name), used to
                say why a parameter is dead; optional.
            result: The finished result, carried on the error.

        Raises:
            DeadParameterError: If any entry is dead, listing each with the
                stage its path ends at and the reason.
        """
        live = self.live(list(roots))
        dead: list[tuple[str, str, str, str]] = []
        for i, e in enumerate(self.entries):
            if live[i]:
                continue
            dead.append((e.owner, e.name, e.stage, self._reason(e, reached or {})))
        if dead:
            raise DeadParameterError(dead, result=result)

    @staticmethod
    def _reason(e: RegisteredParameter, reached: dict[str, int]) -> str:
        if e.gradient_class == BOUNDARY_ONLY:
            return (
                "detached by contract: it enters only an in-or-out test, whose "
                "derivative is a boundary term the pathwise gradient does not "
                "carry (chapter 09 section 9.3; boundary gradients are the "
                "research repository's issue 3)"
            )
        if e.gradient_class == DETACHED:
            return "detached by contract (chapter 09 R-09-2)"
        owners = [e.owner, *e.also_owned_by]
        counts = {o: reached[o] for o in owners if o in reached}
        if counts and not any(counts.values()):
            return (
                "in this scene it cannot influence the tallies: no ray reached "
                + ", ".join(owners)
            )
        return (
            "in this scene it cannot influence the tallies: no output of the "
            "trace depends on it"
        )


def _scene_objects(scene) -> Iterator[tuple[str, str, Any]]:
    """``(owner name, kind, object)`` for every surface, detector and source."""
    registry = getattr(getattr(scene, "component_registry", None), "_registry", None)
    if registry:
        for cname, compound in registry.items():
            for surf in compound.surfaces:
                yield (getattr(surf, "name", "") or cname), "surface", surf
    else:
        for i, surf in enumerate(scene.surfaces):
            yield (getattr(surf, "name", "") or f"surface {i}"), "surface", surf
    det_registry = getattr(getattr(scene, "detector_registry", None), "_registry", None)
    dets = det_registry.items() if det_registry else (
        (getattr(d, "name", "") or f"detector {i}", d) for i, d in enumerate(scene.detectors)
    )
    for name, det in dets:
        yield name, "detector", det
    src_registry = getattr(getattr(scene, "source_registry", None), "_registry", None)
    srcs = src_registry.items() if src_registry else (
        (getattr(s, "name", "") or f"source {i}", s) for i, s in enumerate(scene.sources)
    )
    for name, src in srcs:
        yield name, "source", src


def trace_outputs(scene, detector_results: dict[str, Any]) -> list[Any]:
    """The tensors a trace's caller can differentiate: every attached output.

    The detector results' tensors (the image buffer, the total flux), the
    detectors' own accumulators and every surface's ledger tensors.

    Args:
        scene: The traced scene.
        detector_results: ``SimulationResult.detectors``.

    Returns:
        The attached tensors among them.
    """
    out: list[Any] = []
    holders = [*detector_results.values(), *scene.detectors, *scene.surfaces]
    for holder in holders:
        for value in getattr(holder, "__dict__", {}).values():
            if getattr(value, "grad_fn", None) is not None:
                out.append(value)
            elif hasattr(value, "_dev") and getattr(value._dev, "grad_fn", None) is not None:
                out.append(value._dev)
    return out


def check_after_trace(register: ParameterRegister, scene, detector_results, hit_counts, result) -> None:
    """The trace's closing check (R-09-5); a no-op when autograd is off.

    Under ``torch.no_grad()`` nothing is recorded and nothing can be checked:
    a forward-only evaluation of a scene holding attached tensors (a finite
    difference, a replay check) is a request not to differentiate. Under the
    shared library's blanket differentiable mode (``be.grad_mode``) every
    number the scene was built from is a leaf that requires a gradient, not a
    parameter anyone asked for, so the register is kept on the result and no
    raise is made.

    Args:
        register: The trace's register.
        scene: The traced scene.
        detector_results: ``SimulationResult.detectors``.
        hit_counts: Rays that reached each surface, in ``scene.surfaces`` order.
        result: The finished result, carried on the error.
    """
    import torch  # noqa: PLC0415

    if not register or not torch.is_grad_enabled():
        return
    try:
        blanket = bool(be.grad_mode.requires_grad)
    except Exception:  # noqa: BLE001 - a backend with no grad mode cannot be in it
        blanket = False
    if blanket:
        return
    reached: dict[str, int] = {}
    names = [owner for owner, kind, _ in _scene_objects(scene) if kind == "surface"]
    for name, count in zip(names, hit_counts, strict=False):
        reached[name] = reached.get(name, 0) + int(count)
    register.check(trace_outputs(scene, detector_results), reached=reached, result=result)


def refuse_without_autograd(register: ParameterRegister) -> None:
    """R-09-3: a backend that cannot differentiate refuses the parameters.

    Raises:
        ParameterRefused: When the register is not empty and the active
            array backend has no autograd.
    """
    if register and be.get_backend() != "torch":
        names = ", ".join(f"{e.owner}:{e.name}" for e in register)
        raise ParameterRefused(
            f"the {be.get_backend()} backend has no autograd; these parameters "
            f"require a gradient and would be silently detached: {names}. Trace "
            "on the torch backend (docs/theory/09_differentiation.md R-09-3)."
        )
