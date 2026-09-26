"""Versioned JSON serialization for NSQ scenes.

Provides :func:`scene_to_dict` and :func:`scene_from_dict` which convert an
:class:`~optiland.nonsequential.scene.NSQScene` to and from a plain
JSON-serializable :class:`dict`.

Schema version
--------------
The top-level key ``"nsq_schema_version"`` is ``1``. NSQ has never been
officially released, so there is exactly one schema and no compatibility or
migration machinery for an earlier one: a file whose ``nsq_schema_version``
does not match the current loader is refused with a generic mismatch error
naming both versions. Since NSQ is still pre-release, the physics and the
schema can both change without notice; a scene built against an older
checkout should be rebuilt from its original construction code (or
converted again via
:func:`~optiland.nonsequential.convert.sequential_to_nonsequential`) against
the current API.

Tensor handling
---------------
All PyTorch tensors are detached and serialized as plain Python floats or
lists before writing.  ``requires_grad`` is **not** persisted.  A scene loaded
from JSON is plain-valued; users must re-wrap parameters in
``torch.tensor(..., requires_grad=True)`` to enable differentiation after
loading.

Coordinate systems
------------------
Only the local (x, y, z, rx, ry, rz) components are serialized.  Nested
``reference_cs`` chains are serialized recursively.

Materials
---------
String catalog names (e.g. ``'N-BK7'``) round-trip as strings.
:class:`~optiland.nonsequential.materials.nsq_material.NSQMaterial` instances
with an underlying optiland material round-trip via the catalog name stored on
the material object.  NSQMaterial vacuum (``optiland_material=None``) is
serialized as ``null``.

Not serialized
--------------
- :class:`~optiland.nonsequential.tracer.SimulationResult` / detector data
- Ray databases
- Mesh geometry file content (only the file path is stored)

Kramer Harrison, 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import os

    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.sources.base import Spectrum

NSQ_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float:
    """Convert a scalar (float, numpy scalar, or torch Tensor) to Python float.

    Args:
        value: Scalar value to convert.

    Returns:
        Plain Python float.
    """
    try:
        # torch.Tensor
        return float(value.detach().cpu().item())
    except AttributeError:
        return float(value)


def _to_list(value: Any) -> list:
    """Convert an array-like (numpy array or torch Tensor) to a Python list.

    Args:
        value: Array-like to convert.

    Returns:
        Plain Python list of floats.
    """
    try:
        # torch.Tensor
        return value.detach().cpu().tolist()
    except AttributeError:
        return np.asarray(value, dtype=float).tolist()


def _serialize_cs(cs: CoordinateSystem) -> dict:
    """Serialize a :class:`CoordinateSystem` to a JSON-safe dict.

    Recursively serializes any chained ``reference_cs``.

    Args:
        cs: Coordinate system to serialize.

    Returns:
        Dict with keys x, y, z, rx, ry, rz, and optionally reference_cs.
    """
    d: dict[str, Any] = {
        "x": _to_float(cs.x),
        "y": _to_float(cs.y),
        "z": _to_float(cs.z),
        "rx": _to_float(cs.rx),
        "ry": _to_float(cs.ry),
        "rz": _to_float(cs.rz),
        "reference_cs": _serialize_cs(cs.reference_cs) if cs.reference_cs else None,
    }
    return d


def _deserialize_cs(d: dict) -> CoordinateSystem:
    """Reconstruct a :class:`CoordinateSystem` from a serialized dict.

    Args:
        d: Dict previously produced by :func:`_serialize_cs`.

    Returns:
        Reconstructed :class:`CoordinateSystem`.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415

    ref = _deserialize_cs(d["reference_cs"]) if d.get("reference_cs") else None
    return CoordinateSystem(
        x=d.get("x", 0.0),
        y=d.get("y", 0.0),
        z=d.get("z", 0.0),
        rx=d.get("rx", 0.0),
        ry=d.get("ry", 0.0),
        rz=d.get("rz", 0.0),
        reference_cs=ref,
    )


def _serialize_spectrum(spectrum: Spectrum) -> dict:
    """Serialize a spectrum to a JSON-safe dict, through its spectrum kind.

    A line spectrum (the original :class:`Spectrum`) is written exactly as
    schema version 1 always wrote it -- ``wavelengths`` and ``weights`` and no
    ``kind`` key -- so every file written before other spectrum kinds existed
    reads back unchanged. Any other kind adds ``"kind": <name>``.

    Args:
        spectrum: A spectrum registered to a spectrum kind.

    Returns:
        JSON-safe dict.
    """
    from optiland.nonsequential import kinds  # noqa: PLC0415

    spec = kinds.SPECTRA.for_object(spectrum)
    body = spec.to_dict(spectrum)
    if spec.name == "lines":
        return body
    return {"kind": spec.name, **body}


def _deserialize_spectrum(d: dict) -> Spectrum:
    """Reconstruct a spectrum from a serialized dict (a missing ``kind`` is a
    line spectrum).

    Args:
        d: Dict previously produced by :func:`_serialize_spectrum`.

    Returns:
        Reconstructed spectrum.
    """
    from optiland.nonsequential import kinds  # noqa: PLC0415

    return kinds.SPECTRA.by_name(d.get("kind", "lines")).from_dict(d)


def _serialize_material(mat: str | NSQMaterial | None) -> Any:
    """Serialize a material reference to a JSON-safe value.

    - ``None`` or vacuum NSQMaterial -> ``null``
    - string catalog name -> that string
    - NSQMaterial with optiland_material -> ``{"type": "catalog", "name": ...}``

    Args:
        mat: Material to serialize; may be a catalog name string, an
            :class:`~optiland.nonsequential.materials.nsq_material.NSQMaterial`,
            or ``None``.

    Returns:
        JSON-serializable representation.

    Raises:
        ValueError: If the NSQMaterial cannot be round-tripped (no catalog name
            is available on the underlying material).
    """
    if mat is None:
        return None
    if isinstance(mat, str):
        return mat

    # NSQMaterial
    from optiland.nonsequential.materials.nsq_material import (  # noqa: PLC0415
        NSQMaterial,
    )

    if isinstance(mat, NSQMaterial):
        if mat.optiland_material is None:
            return None  # vacuum
        # Try to recover the catalog name from the underlying material
        underlying = mat.optiland_material
        glass_name = getattr(underlying, "name", None) or getattr(
            underlying, "_name", None
        )
        if glass_name is None:
            raise ValueError(
                f"Cannot serialize NSQMaterial: the underlying material "
                f"{underlying!r} does not expose a 'name' attribute. "
                "Only catalog-name materials can be round-tripped."
            )
        return {"type": "catalog", "name": glass_name}

    raise TypeError(f"Unrecognised material type: {type(mat).__name__}")


def _deserialize_material(d: Any) -> str | None:
    """Reconstruct a material from a serialized value.

    Args:
        d: Value produced by :func:`_serialize_material`.

    Returns:
        String catalog name (which ``add_lens`` etc. resolve at build time),
        or ``None`` for vacuum.
    """
    if d is None:
        return None
    if isinstance(d, str):
        return d
    if isinstance(d, dict) and d.get("type") == "catalog":
        return d["name"]  # scene builder resolves via NSQMaterial.from_glass
    raise ValueError(f"Cannot deserialize material: {d!r}")


# ---------------------------------------------------------------------------
# Component serialization
# ---------------------------------------------------------------------------


def _serialize_component(name: str, compound: Any) -> dict:
    """Serialize a named compound component to a JSON-safe dict.

    The component's kind (:data:`optiland.nonsequential.kinds.COMPONENTS`)
    writes its ``config`` block from the ``_config`` stored on every compound;
    this function adds ``type``, ``name`` and ``cs``.

    Args:
        name: Registry name of the component.
        compound: Compound component object.

    Returns:
        Dict describing the component type and configuration.

    Raises:
        TypeError: If the component type is not registered to a component
            kind.
    """
    from optiland.nonsequential import kinds  # noqa: PLC0415

    try:
        spec = kinds.COMPONENTS.for_object(compound)
    except TypeError:
        raise TypeError(
            f"Cannot serialize component '{name}' of type "
            f"'{type(compound).__name__}'. Registered component kinds: "
            f"{', '.join(kinds.COMPONENTS.names())}."
        ) from None
    return {
        "type": spec.name,
        "name": name,
        "cs": _serialize_cs(compound._cs),
        "config": spec.to_dict(compound),
    }


def _deserialize_component(d: dict, scene: NSQScene) -> None:
    """Reconstruct a compound component and add it to the scene.

    Args:
        d: Dict produced by :func:`_serialize_component`.
        scene: Target :class:`NSQScene` to populate.

    Raises:
        ValueError: If the component type is unknown.
    """
    from optiland.nonsequential import kinds  # noqa: PLC0415

    spec = kinds.COMPONENTS.by_name(d["type"])
    config = spec.from_dict(d["config"])
    kinds.COMPONENTS.check_gradients(spec, config)
    spec.build(scene, d["name"], _deserialize_cs(d["cs"]), config)


# ---------------------------------------------------------------------------
# Source serialization
# ---------------------------------------------------------------------------


def _serialize_source(name: str, source: Any) -> dict:
    """Serialize a named source to a JSON-safe dict.

    The shared fields (``type``, ``name``, ``cs``, ``spectrum``,
    ``total_flux``, ``medium``) are written here; the kind-specific ones by
    the source's kind (:data:`optiland.nonsequential.kinds.SOURCES`).

    Args:
        name: Registry name of the source.
        source: A source registered to a source kind.

    Returns:
        Dict describing the source type and parameters.

    Raises:
        TypeError: If the source type is not registered to a source kind.
    """
    from optiland.nonsequential import kinds  # noqa: PLC0415

    try:
        spec = kinds.SOURCES.for_object(source)
    except TypeError:
        raise TypeError(
            f"Cannot serialize source '{name}' of type '{type(source).__name__}'. "
            f"Registered source kinds: {', '.join(kinds.SOURCES.names())}."
        ) from None
    out = {
        "type": spec.name,
        "name": name,
        "cs": _serialize_cs(source.cs),
        "spectrum": _serialize_spectrum(source.spectrum),
        "total_flux": _to_float(source.total_flux),
        **spec.to_dict(source),
        "medium": _serialize_material(getattr(source, "medium", None)),
    }
    # The source's polarization (the research repository's issue 5), only
    # when one was set, so a scene without one serializes as it always did.
    polarization = getattr(source, "polarization", None)
    if polarization is not None:
        out["polarization"] = polarization.to_dict()
    return out


def _deserialize_source(d: dict, scene: NSQScene) -> None:
    """Reconstruct a source and add it to the scene.

    Args:
        d: Dict produced by :func:`_serialize_source`.
        scene: Target :class:`NSQScene` to populate.

    Raises:
        ValueError: If the source type is unknown.
    """
    from optiland.nonsequential import kinds  # noqa: PLC0415

    spec = kinds.SOURCES.by_name(d["type"])
    config = spec.from_dict(
        d,
        spectrum=_deserialize_spectrum(d["spectrum"]),
        total_flux=d["total_flux"],
        medium=_deserialize_material(d.get("medium")),
    )
    scene.add_source(d["name"], _deserialize_cs(d["cs"]), config)
    if d.get("polarization") is not None:
        from optiland.nonsequential.polarization import (  # noqa: PLC0415
            SourcePolarization,
        )

        scene.source_registry.get(d["name"]).polarization = (
            SourcePolarization.from_dict(d["polarization"])
        )


# ---------------------------------------------------------------------------
# Detector serialization
# ---------------------------------------------------------------------------


def _serialize_detector(name: str, detector: Any) -> dict:
    """Serialize a named detector to a JSON-safe dict.

    ``type``, ``name`` and ``cs`` are written here; the rest by the detector's
    kind (:data:`optiland.nonsequential.kinds.DETECTORS`). The lookup is by
    exact class, so the hemispherical collector (a subclass of the far-field
    detector) is never written as a flat far-field detector.

    Args:
        name: Registry name of the detector.
        detector: A detector registered to a detector kind.

    Returns:
        Dict describing the detector type and parameters.

    Raises:
        TypeError: If the detector type is not registered to a detector kind.
    """
    from optiland.nonsequential import kinds  # noqa: PLC0415

    try:
        spec = kinds.DETECTORS.for_object(detector)
    except TypeError:
        raise TypeError(
            f"Cannot serialize detector '{name}' of type "
            f"'{type(detector).__name__}'. Registered detector kinds: "
            f"{', '.join(kinds.DETECTORS.names())}."
        ) from None
    return {
        "type": spec.name,
        "name": name,
        "cs": _serialize_cs(detector.cs),
        **spec.to_dict(detector),
    }


def _deserialize_detector(d: dict, scene: NSQScene) -> None:
    """Reconstruct a detector and add it to the scene.

    Args:
        d: Dict produced by :func:`_serialize_detector`.
        scene: Target :class:`NSQScene` to populate.

    Raises:
        ValueError: If the detector type is unknown.
    """
    from optiland.nonsequential import kinds  # noqa: PLC0415

    spec = kinds.DETECTORS.by_name(d["type"])
    scene.add_detector(d["name"], _deserialize_cs(d["cs"]), spec.from_dict(d))


# ---------------------------------------------------------------------------
# Top-level scene serialization
# ---------------------------------------------------------------------------


def scene_to_dict(scene: NSQScene) -> dict:
    """Convert an :class:`NSQScene` to a JSON-serializable dict.

    The returned dict includes a top-level ``"nsq_schema_version"`` key.
    Simulation results and detector data are **not** included.

    Args:
        scene: The scene to serialize.

    Returns:
        JSON-serializable dict representing the scene structure.

    Raises:
        TypeError: If any component, source, or detector type is not supported.
        ValueError: If any material cannot be round-tripped.
    """
    components = []
    for name, compound in scene.component_registry._registry.items():
        components.append(_serialize_component(name, compound))

    sources = []
    for name, source in scene.source_registry._registry.items():
        sources.append(_serialize_source(name, source))

    detectors = []
    for name, detector in scene.detector_registry._registry.items():
        detectors.append(_serialize_detector(name, detector))

    return {
        "nsq_schema_version": NSQ_SCHEMA_VERSION,
        "components": components,
        "sources": sources,
        "detectors": detectors,
    }


def scene_from_dict(d: dict) -> NSQScene:
    """Reconstruct an :class:`NSQScene` from a serialized dict.

    Validates ``"nsq_schema_version"`` before loading.  A loaded scene is
    plain-valued; all parameters are plain Python floats, not tensors.

    Args:
        d: Dict previously produced by :func:`scene_to_dict` or read from a
            JSON file written by :meth:`NSQScene.to_json`.

    Returns:
        Reconstructed :class:`NSQScene`.

    Raises:
        ValueError: If ``"nsq_schema_version"`` is missing or does not match
            :data:`NSQ_SCHEMA_VERSION`. NSQ has never been officially
            released, so there is no compatibility mode or auto-migration
            for an older schema -- a mismatched file must be rebuilt against
            the current API.
        ValueError: If any component/source/detector type is unknown.
    """
    from optiland.nonsequential.scene import NSQScene  # noqa: PLC0415

    version = d.get("nsq_schema_version")
    if version is None:
        raise ValueError(
            "The JSON file is missing the required 'nsq_schema_version' key. "
            "This file may not be an Optiland NSQ scene file."
        )
    if version != NSQ_SCHEMA_VERSION:
        raise ValueError(
            f"NSQ schema version mismatch: the file uses version {version!r}, "
            f"but this version of Optiland only supports version "
            f"{NSQ_SCHEMA_VERSION}. "
            "Please update Optiland or re-export the scene."
        )

    scene = NSQScene()

    for comp_d in d.get("components", []):
        _deserialize_component(comp_d, scene)

    for src_d in d.get("sources", []):
        _deserialize_source(src_d, scene)

    for det_d in d.get("detectors", []):
        _deserialize_detector(det_d, scene)

    return scene


def scene_to_json(scene: NSQScene, path: str | os.PathLike) -> None:
    """Serialize an :class:`NSQScene` to a versioned JSON file.

    This is the low-level implementation called by
    :meth:`NSQScene.to_json`.

    Args:
        scene: The scene to serialize.
        path: Destination file path (created or overwritten).
    """
    import json  # noqa: PLC0415

    d = scene_to_dict(scene)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def scene_from_json(path: str | os.PathLike) -> NSQScene:
    """Load an :class:`NSQScene` from a versioned JSON file.

    This is the low-level implementation called by
    :meth:`NSQScene.from_json`.

    Args:
        path: Path to the JSON file previously written by
            :func:`scene_to_json` or :meth:`NSQScene.to_json`.

    Returns:
        Reconstructed :class:`NSQScene`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the schema version is missing or does not match.
    """
    import json  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"NSQ scene file not found: {path}. "
            "Check the path and ensure the file has not been moved or deleted."
        )
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return scene_from_dict(d)
