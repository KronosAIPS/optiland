"""Kind registries: the extension seam of the non-sequential engine.

A *kind* is one member of a family a scene is built from: a source kind
(``"point"``, ``"collimated"``, ...), a detector kind, a geometry kind, a
scatter (BSDF) kind, a compound-component kind, a spectrum kind. Every place
that used to branch on the Python class of an object -- the scene builder
(:mod:`optiland.nonsequential.scene`), JSON serialization
(:mod:`optiland.nonsequential.serialization`) and the lowering to the
data-only scene description (:mod:`optiland.nonsequential.ir.lower`) -- asks
the family's registry instead. A new kind is therefore one registration,
made inside the engine or from a separate package, with no edit to those
three places.

What a registration carries (:class:`KindSpec`)
------------------------------------------------
- ``name``: the kind's string key, as it appears in the scene JSON
  (``"type"``) and in the scene IR (``kind``).
- ``cls``: the live class. Lowering and serialization find the spec of a live
  object by its class: the exact class first, then -- only for specs
  registered with ``accept_subclasses=True`` -- the nearest registered
  ancestor. A subclass that is not registered and whose ancestor does not
  accept subclasses is refused by name rather than serialized as its parent,
  so a subclass carrying state of its own is never silently truncated.
- ``config_cls`` and ``build``: the config dataclass the scene accepts and
  the function that turns ``(cs, config)`` into the live object (sources,
  detectors), or ``(scene, name, cs, config)`` into a registered compound
  component (components).
- ``to_dict`` and ``from_dict``: the kind-specific part of the scene JSON and
  its inverse (``from_dict`` returns a config for the builder).
- ``lower``: the kind-specific ``params`` of the kind's IR record.
- The gradient rule, ``attached``: the config fields a gradient may flow
  through. A config field holding a tensor that requires a gradient and is
  not named in ``attached`` raises at build time instead of being silently
  detached (``docs/theory/09_differentiation.md`` R-09-5, the dead-parameter
  raise, in the research repository). ``attached="*"`` marks a built-in whose
  constructor already enforces the rule parameter by parameter
  (``as_param`` / ``as_detached_param``); a plug-in kind states its list.

Plug-ins
--------
A package outside the engine declares the entry-point group
``optiland.nonsequential`` in its own ``pyproject.toml``, pointing at a
zero-argument callable that calls the ``register_*`` functions below. The
group is loaded lazily, at most once per process, on the first lookup that
misses -- the same mechanism as :mod:`optiland.plugins` uses for surfaces,
materials and analyses::

    [project.entry-points."optiland.nonsequential"]
    my_kinds = "my_package.nsq_kinds:register"

The built-in kinds register themselves through the same functions, in
:mod:`optiland.nonsequential._builtin_kinds`, the first time any registry is
consulted.
"""

from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from optiland._suggest import options_hint
from optiland.nonsequential._utils import is_tensor

PLUGIN_GROUP = "optiland.nonsequential"

#: The families a kind can belong to.
FAMILIES = ("source", "detector", "geometry", "bsdf", "component", "spectrum")


@dataclass(frozen=True)
class KindSpec:
    """One registered kind of one family.

    Attributes:
        family: One of :data:`FAMILIES`.
        name: The kind's key in the scene JSON and the scene IR.
        cls: The live class the kind builds.
        config_cls: The config dataclass the scene builder accepts, or
            ``None`` for a family built outside the scene builder (geometry,
            BSDF).
        build: Builder: ``(cs, config) -> live object`` for sources and
            detectors; ``(scene, name, cs, config) -> None`` for compound
            components (it registers the component itself); ``(dict) ->
            live object`` for spectra.
        to_dict: ``live object -> dict``: the kind-specific fields of its
            scene JSON (the family's shared fields -- name, placement,
            spectrum, flux, medium -- are written by the serializer).
        from_dict: ``dict -> config`` for the builder (sources, detectors,
            components; the shared fields are passed as keyword arguments
            where the family has them), or ``dict -> live object`` for
            spectra.
        lower: ``live object -> params`` of the kind's IR record.
        attached: Config fields a gradient may flow through, or ``"*"``
            (the constructor enforces the rule itself).
        ir_kind: The ``kind`` written to the IR, when it differs from
            ``name`` (an infinite plane lowers as ``"plane"``).
        accept_subclasses: Whether an unregistered subclass of ``cls`` is
            handled by this spec.
        description: One line for listings and error messages.
    """

    family: str
    name: str
    cls: type
    config_cls: type | None = None
    build: Callable[..., Any] | None = None
    to_dict: Callable[[Any], dict] | None = None
    from_dict: Callable[..., Any] | None = None
    lower: Callable[[Any], dict] | None = None
    attached: tuple[str, ...] | str = ()
    ir_kind: str | None = None
    accept_subclasses: bool = False
    description: str = ""

    @property
    def lowered_kind(self) -> str:
        """The ``kind`` string this spec writes into the scene IR."""
        return self.ir_kind or self.name


class KindRegistry:
    """The registered kinds of one family, looked up by name, config or class.

    Args:
        family: One of :data:`FAMILIES`.
    """

    def __init__(self, family: str) -> None:
        if family not in FAMILIES:
            raise ValueError(f"Unknown kind family {family!r}; expected {FAMILIES}.")
        self.family = family
        self._by_name: dict[str, KindSpec] = {}
        self._by_cls: dict[type, KindSpec] = {}
        self._by_config: dict[type, KindSpec] = {}

    # -- registration ------------------------------------------------------

    def register(self, spec: KindSpec, *, overwrite: bool = False) -> KindSpec:
        """Add ``spec`` to this family.

        Args:
            spec: The kind to register; ``spec.family`` must be this family.
            overwrite: Replace an existing kind of the same name (and drop
                its class and config entries).

        Returns:
            ``spec``.

        Raises:
            ValueError: If the name, the class or the config class is already
                registered to another kind and ``overwrite`` is False, or the
                spec belongs to another family.
        """
        if spec.family != self.family:
            raise ValueError(
                f"A {spec.family!r} kind cannot be registered in the "
                f"{self.family!r} registry."
            )
        old = self._by_name.get(spec.name)
        if old is not None:
            if not overwrite:
                raise ValueError(
                    f"{self.family.capitalize()} kind {spec.name!r} is already "
                    "registered. Pass overwrite=True to replace it."
                )
            self._forget(old)
        for key, table, what in (
            (spec.cls, self._by_cls, "class"),
            (spec.config_cls, self._by_config, "config class"),
        ):
            if key is None:
                continue
            clash = table.get(key)
            if clash is not None and clash.name != spec.name and not overwrite:
                raise ValueError(
                    f"The {what} {key.__name__} is already registered as "
                    f"{self.family} kind {clash.name!r}."
                )
        self._by_name[spec.name] = spec
        self._by_cls[spec.cls] = spec
        if spec.config_cls is not None:
            self._by_config[spec.config_cls] = spec
        return spec

    def _forget(self, spec: KindSpec) -> None:
        self._by_name.pop(spec.name, None)
        if self._by_cls.get(spec.cls) is spec:
            del self._by_cls[spec.cls]
        if spec.config_cls is not None and self._by_config.get(spec.config_cls) is spec:
            del self._by_config[spec.config_cls]

    def unregister(self, name: str) -> None:
        """Remove kind ``name`` (tests and plug-in reloads use this).

        Raises:
            KeyError: If no such kind is registered.
        """
        _ensure_builtins()
        spec = self._by_name.get(name)
        if spec is None:
            raise KeyError(f"No {self.family} kind {name!r} is registered.")
        self._forget(spec)

    # -- lookup ------------------------------------------------------------

    def names(self) -> tuple[str, ...]:
        """The registered kind names, in registration order."""
        _ensure_builtins()
        return tuple(self._by_name)

    def by_name(self, name: str) -> KindSpec:
        """The spec registered under ``name``.

        Raises:
            ValueError: If no kind of that name is registered, after loading
                the plug-ins once; the message lists the registered names.
        """
        _ensure_builtins()
        spec = self._by_name.get(name)
        if spec is None:
            load_plugins()
            spec = self._by_name.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown {self.family} kind {name!r}."
                f"{options_hint(str(name), self._by_name)} A kind from a plug-in "
                f"package registers through the {PLUGIN_GROUP!r} entry-point "
                "group or an explicit register call before it is used."
            )
        return spec

    def for_config(self, config: Any) -> KindSpec:
        """The spec whose ``config_cls`` is the exact type of ``config``.

        Raises:
            TypeError: If the config type is not registered to this family.
        """
        return self.for_config_class(type(config))

    def for_config_class(self, config_cls: type) -> KindSpec:
        """The spec whose ``config_cls`` is exactly ``config_cls``.

        Raises:
            TypeError: If the config class is not registered to this family.
        """
        _ensure_builtins()
        spec = self._by_config.get(config_cls)
        if spec is None:
            load_plugins()
            spec = self._by_config.get(config_cls)
        if spec is None:
            known = ", ".join(
                s.config_cls.__name__ for s in self._by_name.values() if s.config_cls
            )
            raise TypeError(
                f"Unrecognised {self.family} config type: {config_cls.__name__}. "
                f"Registered {self.family} configs: {known}."
            )
        return spec

    def for_object(self, obj: Any) -> KindSpec:
        """The spec of a live object: its exact class, else the nearest
        registered ancestor whose spec accepts subclasses.

        Raises:
            TypeError: If no spec handles ``type(obj)``.
        """
        _ensure_builtins()
        spec = self._lookup_cls(type(obj))
        if spec is None:
            load_plugins()
            spec = self._lookup_cls(type(obj))
        if spec is None:
            raise TypeError(
                f"No {self.family} kind is registered for type "
                f"{type(obj).__name__}. Registered {self.family} kinds: "
                f"{', '.join(self._by_name) or 'none'}."
            )
        return spec

    def _lookup_cls(self, cls: type) -> KindSpec | None:
        spec = self._by_cls.get(cls)
        if spec is not None:
            return spec
        for ancestor in cls.__mro__[1:]:
            spec = self._by_cls.get(ancestor)
            if spec is not None:
                return spec if spec.accept_subclasses else None
        return None

    # -- the gradient rule ---------------------------------------------------

    def check_gradients(self, spec: KindSpec, config: Any) -> None:
        """Raise if a config field carries a gradient its kind does not declare.

        Args:
            spec: The kind ``config`` belongs to.
            config: A dataclass config instance.

        Raises:
            NotImplementedError: If a field holds a tensor with
                ``requires_grad`` and is not in ``spec.attached``.
        """
        if spec.attached == "*" or not is_dataclass(config):
            return
        for f in fields(config):
            value = getattr(config, f.name)
            if is_tensor(value) and value.requires_grad and f.name not in spec.attached:
                raise NotImplementedError(
                    f"{self.family.capitalize()} kind {spec.name!r}: the field "
                    f"{f.name!r} carries a gradient, but the kind does not "
                    "declare it attached, so the gradient would be silently "
                    f"dropped. Pass a plain float for {f.name!r}. Attached "
                    f"fields of this kind: {', '.join(spec.attached) or 'none'}."
                )

    def build(self, *args: Any) -> Any:
        """Build a live object from ``(..., config)`` through its kind.

        The config is always the last positional argument. The gradient rule
        is checked first.
        """
        config = args[-1]
        spec = self.for_config(config)
        self.check_gradients(spec, config)
        if spec.build is None:
            raise TypeError(f"{self.family} kind {spec.name!r} has no builder.")
        return spec.build(*args)


SOURCES = KindRegistry("source")
DETECTORS = KindRegistry("detector")
GEOMETRIES = KindRegistry("geometry")
BSDFS = KindRegistry("bsdf")
COMPONENTS = KindRegistry("component")
SPECTRA = KindRegistry("spectrum")

_REGISTRIES = {
    "source": SOURCES,
    "detector": DETECTORS,
    "geometry": GEOMETRIES,
    "bsdf": BSDFS,
    "component": COMPONENTS,
    "spectrum": SPECTRA,
}


def registry(family: str) -> KindRegistry:
    """The registry of ``family`` (one of :data:`FAMILIES`)."""
    try:
        return _REGISTRIES[family]
    except KeyError as err:
        raise ValueError(
            f"Unknown kind family {family!r}.{options_hint(str(family), FAMILIES)}"
        ) from err


def registered_kinds() -> dict[str, tuple[str, ...]]:
    """Every family's registered kind names, for listings and front ends."""
    return {family: reg.names() for family, reg in _REGISTRIES.items()}


def _register(family: str, name: str, cls: type, overwrite: bool, **kw: Any) -> KindSpec:
    return _REGISTRIES[family].register(
        KindSpec(family=family, name=name, cls=cls, **kw), overwrite=overwrite
    )


def register_source(
    name: str,
    cls: type,
    config_cls: type,
    build: Callable[..., Any],
    to_dict: Callable[[Any], dict],
    from_dict: Callable[..., Any],
    lower: Callable[[Any], dict],
    *,
    attached: tuple[str, ...] | str = (),
    overwrite: bool = False,
    description: str = "",
) -> KindSpec:
    """Register a source kind.

    ``build(cs, config)`` returns the live source. ``to_dict(source)`` returns
    the kind-specific JSON fields; the serializer adds ``type``, ``name``,
    ``cs``, ``spectrum``, ``total_flux`` and ``medium``.
    ``from_dict(d, spectrum=..., total_flux=..., medium=...)`` returns the
    config. ``lower(source)`` returns the kind-specific IR params; the
    lowering adds ``total_flux`` and ``spectrum``. The live source must
    implement ``generate(ray_id, rng)`` (see
    :class:`~optiland.nonsequential.sources.base.BaseNSQSource`).
    """
    return _register(
        "source", name, cls, overwrite, config_cls=config_cls, build=build,
        to_dict=to_dict, from_dict=from_dict, lower=lower, attached=attached,
        description=description,
    )


def register_detector(
    name: str,
    cls: type,
    config_cls: type,
    build: Callable[..., Any],
    to_dict: Callable[[Any], dict],
    from_dict: Callable[..., Any],
    lower: Callable[[Any], dict],
    *,
    attached: tuple[str, ...] | str = (),
    overwrite: bool = False,
    accept_subclasses: bool = False,
    description: str = "",
) -> KindSpec:
    """Register a detector kind.

    ``build(cs, config)`` returns the live detector (a
    :class:`~optiland.nonsequential.detectors.base.BaseDetector`).
    ``to_dict(detector)`` returns the kind-specific JSON fields (the
    serializer adds ``type``, ``name`` and ``cs``); ``from_dict(d)`` returns
    the config; ``lower(detector)`` the IR params (the lowering adds
    ``reflection_bins``).
    """
    return _register(
        "detector", name, cls, overwrite, config_cls=config_cls, build=build,
        to_dict=to_dict, from_dict=from_dict, lower=lower, attached=attached,
        accept_subclasses=accept_subclasses, description=description,
    )


def register_geometry(
    name: str,
    cls: type,
    lower: Callable[[Any], dict],
    *,
    ir_kind: str | None = None,
    accept_subclasses: bool = False,
    overwrite: bool = False,
    description: str = "",
) -> KindSpec:
    """Register a geometry kind: a
    :class:`~optiland.nonsequential.components.geometry.base.ComponentGeometry`
    subclass (its ``ray_intersect`` and ``bounding_box`` are what the loop
    calls) and its IR params."""
    return _register(
        "geometry", name, cls, overwrite, lower=lower, ir_kind=ir_kind,
        accept_subclasses=accept_subclasses, description=description,
    )


def register_bsdf(
    name: str,
    cls: type,
    lower: Callable[[Any], dict],
    *,
    accept_subclasses: bool = False,
    overwrite: bool = False,
    description: str = "",
) -> KindSpec:
    """Register a scatter kind: a
    :class:`~optiland.nonsequential.bsdf.base.BaseBSDF` subclass and its IR
    params."""
    return _register(
        "bsdf", name, cls, overwrite, lower=lower,
        accept_subclasses=accept_subclasses, description=description,
    )


def register_component(
    name: str,
    cls: type,
    config_cls: type,
    build: Callable[..., Any],
    to_dict: Callable[[Any], dict],
    from_dict: Callable[[dict], Any],
    *,
    attached: tuple[str, ...] | str = (),
    overwrite: bool = False,
    description: str = "",
) -> KindSpec:
    """Register a compound-component kind.

    ``build(scene, name, cs, config)`` adds the component to the scene;
    ``to_dict(compound)`` returns its ``config`` block; ``from_dict(cfg)``
    the config.
    """
    return _register(
        "component", name, cls, overwrite, config_cls=config_cls, build=build,
        to_dict=to_dict, from_dict=from_dict, attached=attached,
        description=description,
    )


def register_spectrum(
    name: str,
    cls: type,
    to_dict: Callable[[Any], dict],
    from_dict: Callable[[dict], Any],
    lower: Callable[[Any], dict],
    *,
    overwrite: bool = False,
    description: str = "",
) -> KindSpec:
    """Register a spectrum kind (the source's wavelength distribution).

    The live object must provide ``sample(ray_id, bounce, rng)`` and the
    ``wavelengths`` and ``weights`` arrays the photometric helpers read.
    """
    return _register(
        "spectrum", name, cls, overwrite, to_dict=to_dict, from_dict=from_dict,
        lower=lower, description=description,
    )


# ---------------------------------------------------------------------------
# Built-ins and plug-ins, both loaded lazily
# ---------------------------------------------------------------------------

_builtins_loaded = False
_plugins_loaded = False


def _ensure_builtins() -> None:
    """Register the engine's own kinds, once, on first use of any registry."""
    global _builtins_loaded  # noqa: PLW0603
    if _builtins_loaded:
        return
    _builtins_loaded = True
    from optiland.nonsequential import _builtin_kinds  # noqa: PLC0415

    _builtin_kinds.register_all()


def load_plugins() -> None:
    """Invoke every ``optiland.nonsequential`` entry point, once per process.

    A failing plug-in warns and is skipped; it never breaks the engine.
    """
    global _plugins_loaded  # noqa: PLW0603
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for entry_point in importlib.metadata.entry_points(group=PLUGIN_GROUP):
        try:
            entry_point.load()()
        except Exception as exc:  # noqa: BLE001 - a bad plug-in must not break the engine
            warnings.warn(
                f"Failed to load non-sequential plug-in {entry_point.name!r} "
                f"from group {PLUGIN_GROUP!r}: {exc}",
                UserWarning,
                stacklevel=2,
            )
