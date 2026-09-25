"""NSQ Material adapter -- thin differentiable wrapper over optiland.materials.

Evaluates refractive index as an attached computation node when the backend
is PyTorch, enabling gradients w.r.t. material dispersion parameters.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np

import optiland.backend as be
from optiland.nonsequential._compile import compiling, run_eagerly

if TYPE_CHECKING:
    import torch

    from optiland.materials import BaseMaterial
    from optiland.nonsequential.bsdf.base import BaseBSDF

# Wavelength input: Python float, numpy array, or torch Tensor
WavelengthInput = Union[float, np.ndarray, "torch.Tensor"]


def _grad_attached(value) -> bool:
    """True when ``value`` carries a live autograd graph.

    A plain attribute read (``requires_grad`` on a torch Tensor, absent on
    everything else) -- never a device-to-host transfer.
    """
    return bool(getattr(value, "requires_grad", False))


@dataclass
class NSQMaterial:
    """Thin differentiable adapter over optiland.materials.BaseMaterial.

    Evaluates ``n(wavelength_um)`` without any grad-severing casts
    (no ``float()``, no ``np.asarray()``) so the result stays in the
    autograd graph when using the Torch backend.

    ``n()``/``k()`` additionally memoize their result against the identity
    of the ``wavelength_um`` object most recently seen (one slot per
    property, not a general cache). ``optiland.materials.BaseMaterial``
    itself must re-inspect its live parameters and the wavelength array's
    content on every call -- issue #630 upstream, still fixed there --
    which on the Torch backend costs one host synchronization per call (the
    uniformity probe and the mutable-state fingerprint both read a tensor's
    value back to the host). That correctness fix is upstream's and stays
    untouched; the cost it added is paid here only once per *distinct*
    wavelength array object rather than once per bounce.

    This is sound because, within nsq's own usage, it is: a ray bundle's
    ``wavelength`` field is only ever replaced by slicing into a *new*
    array (``NSQRayBundle`` never writes into it in place -- masking,
    compaction and backend promotion all construct a fresh array), so the
    same wavelength array object recurs, unchanged, across every bounce
    that runs on an unchanged ray-state buffer (a "steady" bounce -- see
    ``tests/nonsequential/test_nsq_host_reads.py``), and a fresh trace
    always samples a brand-new wavelength array (``Spectrum.sample``
    allocates), so no cached result can outlive the material parameters
    that produced it -- nothing in the bounce loop mutates a material's own
    state mid-trace. A result that carries a live gradient
    (``requires_grad``) is never cached, matching ``BaseMaterial``'s own
    rule, so differentiable material-parameter optimization sees a fresh
    graph on every call exactly as it does without this cache.

    Attributes:
        optiland_material: Underlying material model. None means vacuum (n=1).
        bsdf: Optional surface scatter model.
    """

    optiland_material: BaseMaterial | None = None
    bsdf: BaseBSDF | None = None
    # Per-property, one-slot memo: (the wavelength object last seen, the
    # result computed for it). Excluded from dataclass equality/repr so
    # NSQMaterial's observable identity is unaffected by this optimization.
    _n_memo: tuple = field(default=(None, None), repr=False, compare=False)
    _k_memo: tuple = field(default=(None, None), repr=False, compare=False)

    @classmethod
    def from_glass(cls, name: str) -> NSQMaterial:
        """Resolve a glass catalog name to an NSQMaterial.

        Args:
            name: Glass catalog name (e.g. ``'N-BK7'``, ``'SF11'``).

        Returns:
            NSQMaterial wrapping the resolved BaseMaterial.

        Raises:
            ValueError: If the glass name is not found in the catalog.
        """
        from optiland.materials import Material  # noqa: PLC0415

        try:
            mat = Material(name)
        except Exception as exc:
            raise ValueError(
                f"Glass '{name}' not found in the Optiland material catalog."
            ) from exc
        return cls(optiland_material=mat)

    def reset_memo(self) -> None:
        """Clear the identity memo of both ``n()`` and ``k()``.

        Called once per material at the start of every trace (see
        ``ArrayBackend.trace``), so a wavelength array object reused across
        two separate traces -- the same ray bundle traced twice, with the
        material's own parameters changed in between -- cannot read a
        result computed under the old parameters. Within one trace nothing
        calls this: the memo stays valid for the trace's whole bounce loop,
        which is what makes it effective. A plain attribute write, no host
        read.
        """
        self._n_memo = (None, None)
        self._k_memo = (None, None)

    def n(self, wavelength_um: WavelengthInput) -> WavelengthInput:
        """Refractive index at the given wavelength(s).

        Differentiable: when ``wavelength_um`` is a torch Tensor with
        ``requires_grad=True``, the returned value carries attached gradients.
        No ``float()`` or ``np.asarray()`` casts are applied to the result.

        Args:
            wavelength_um: Wavelength(s) in micrometres [µm]. Accepts Python
                float, NumPy ndarray, or torch Tensor.

        Returns:
            Refractive index with the same array type as the input. Returns
            ``1.0`` (scalar) for vacuum when input is a scalar, or a
            ones-like array/tensor matching the input shape for array inputs.
        """
        if compiling():
            # Inside the compiled bounce step the property is evaluated as
            # ordinary Python and enters the compiled program as data: the
            # memo and the material library's caches key on the identity and
            # contents of their arguments, which, traced, would specialise
            # the program to one scene (_compile.py).
            return run_eagerly(self.n, wavelength_um)
        if self.optiland_material is None:
            # Vacuum: return ones matching the input type/device
            try:
                return be.ones_like(wavelength_um)
            except (TypeError, AttributeError):
                return 1.0
        memo_wavelength, memo_result = self._n_memo
        if wavelength_um is memo_wavelength:
            return memo_result
        # Pass wavelength directly -- preserves the grad graph
        result = self.optiland_material.n(wavelength_um)
        if not _grad_attached(result):
            self._n_memo = (wavelength_um, result)
        return result

    def k(self, wavelength_um: WavelengthInput) -> WavelengthInput:
        """Extinction coefficient at the given wavelength(s).

        Feeds Beer-Lambert bulk absorption: ``alpha = 4*pi*k/wavelength_um``
        [1/um], matching ``optiland.propagation.homogeneous
        .HomogeneousPropagation`` so NSQ and the sequential engine attenuate
        a glass path by the same amount.

        Args:
            wavelength_um: Wavelength(s) in micrometres [µm].

        Returns:
            Extinction coefficient (dimensionless). Returns ``0.0`` (scalar)
            for vacuum when input is a scalar, or a zeros-like array/tensor
            matching the input shape for array inputs.
        """
        if compiling():
            return run_eagerly(self.k, wavelength_um)  # as n(), above
        if self.optiland_material is None:
            # Vacuum: non-absorbing.
            try:
                return be.zeros_like(wavelength_um)
            except (TypeError, AttributeError):
                return 0.0
        memo_wavelength, memo_result = self._k_memo
        if wavelength_um is memo_wavelength:
            return memo_result
        result = self.optiland_material.k(wavelength_um)
        if not _grad_attached(result):
            self._k_memo = (wavelength_um, result)
        return result


# Module-level vacuum constant
VACUUM: NSQMaterial = NSQMaterial(optiland_material=None)


def medium_stack_id(material: NSQMaterial) -> int:
    """Canonical integer id for a medium, for the ray-level medium stack.

    Every vacuum-like material (``optiland_material is None``) maps to the
    same id ``0`` regardless of which ``NSQMaterial`` instance wraps it --
    two separately constructed vacuum wrappers are physically the same
    medium. Any other material is identified by Python object identity: two
    ``NSQMaterial`` instances that happen to wrap the same physical glass
    are only treated as the same medium if the scene reuses one instance
    for both (as :class:`~optiland.nonsequential.components.volume.Volume`
    and the ``Lens``/``Doublet`` builders do for a shared interior). This is
    a conservative default -- reusing distinct instances for the same
    physical glass on either side of a gap can produce a spurious
    ``medium_stack_underflows`` count, never a wrong flux/index result (the
    stack is a diagnostic cross-check; ``n1``/``n2`` are always resolved
    geometrically, never from the stack).

    Args:
        material: The medium to identify.

    Returns:
        A stable (for the process lifetime) non-negative integer id.
    """
    if material.optiland_material is None:
        return 0
    return id(material)


def medium_stack_id_value(material: NSQMaterial, like):
    """:func:`medium_stack_id`, in the form the bounce can use on ``like``'s device.

    Outside the compiled bounce step this is the Python int itself. Inside
    it, it is the same number as a 0-d int64 tensor on ``like``'s device,
    made once per material and device and returned as data: an object's
    identity read inside the compiled program would be a constant of it, and
    every new scene would compile the bounce again.

    Args:
        material: The medium to identify.
        like: A tensor on the device the medium stack lives on.

    Returns:
        ``medium_stack_id(material)``, as an int or a 0-d tensor.
    """
    if not compiling():
        return medium_stack_id(material)
    return run_eagerly(_medium_stack_id_tensor, material, like.device)


def _medium_stack_id_tensor(material: NSQMaterial, device):
    import torch  # noqa: PLC0415

    cache = material.__dict__.setdefault("_stack_id_tensors", {})
    key = str(device)
    value = cache.get(key)
    if value is None:
        value = torch.tensor(medium_stack_id(material), dtype=torch.int64, device=device)
        cache[key] = value
    return value
