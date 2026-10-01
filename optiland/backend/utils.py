"""
Utility functions for working with different backends.

To add support for a new backend, add a conversion function to the CONVERTERS
list.

Kramer Harrison, 2024
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    from optiland._types import ScalarOrArrayT


# Conversion functions for backends
def torch_to_numpy(obj: Tensor) -> NDArray:
    """Convert a torch Tensor to a NumPy array; raise TypeError otherwise."""
    if importlib.util.find_spec("torch"):
        import torch

        if isinstance(obj, torch.Tensor):
            if in_functorch_transform():
                return _numpy_inside_transform(obj)
            return obj.detach().cpu().numpy()
    raise TypeError


def in_functorch_transform() -> bool:
    """True while a ``torch.func`` transform (``jvp``, ``grad``, ``vmap``) is running."""
    import torch  # noqa: PLC0415

    functorch = getattr(torch._C, "_functorch", None)
    return functorch is not None and functorch.peek_interpreter_stack() is not None


def _numpy_inside_transform(obj: Tensor) -> NDArray:
    """A host copy of ``obj``'s value while a ``torch.func`` transform runs.

    Inside a transform every tensor refuses ``.numpy()`` (the transform's
    interpreter stack intercepts the data access), and an argument of the
    transform, or anything computed from it, is a wrapper with no storage.
    The value is read with the stack cleared, from the innermost wrapped
    tensor: the same bits a host read gives outside the transform. The
    derivative stays with the wrapper, which a host read drops in any case.
    """
    from torch._functorch import pyfunctorch  # noqa: PLC0415

    with pyfunctorch.temporarily_clear_interpreter_stack():
        return unwrap_functorch(obj).detach().cpu().numpy()


def is_functorch_wrapped(obj) -> bool:
    """True for a wrapper tensor of a ``torch.func`` transform (``jvp``, ``grad``, ``vmap``)."""
    import torch  # noqa: PLC0415

    functorch = getattr(torch._C, "_functorch", None)
    return (
        functorch is not None
        and isinstance(obj, torch.Tensor)
        and functorch.is_functorch_wrapped_tensor(obj)
    )


def unwrap_functorch(obj: Tensor) -> Tensor:
    """The plain tensor inside the wrappers of ``torch.func``'s transforms.

    Under ``torch.func.jvp`` (or ``grad``, ``vjp``) the arguments and
    everything computed from them are wrapper tensors with no storage of
    their own, so reading one's value on the host (``.numpy()``) fails. The
    value is the innermost wrapped tensor's; the derivative stays with the
    wrapper, which is what a host read drops in any case. A plain tensor is
    returned unchanged.

    Args:
        obj: A torch tensor, wrapped or not.

    Returns:
        The innermost plain tensor.
    """
    import torch  # noqa: PLC0415

    functorch = getattr(torch._C, "_functorch", None)
    if functorch is None:
        return obj
    while functorch.is_functorch_wrapped_tensor(obj):
        obj = functorch.get_unwrapped(obj)
    return obj


CONVERTERS = [torch_to_numpy]


def to_numpy(obj: ScalarOrArrayT) -> NDArray:
    """Converts input scalar or array to NumPy array, regardless of backend."""
    if isinstance(obj, np.ndarray):
        return obj

    elif isinstance(obj, int | float | np.number):
        return np.array(obj)

    # Handle lists: Iterate and convert elements individually
    elif isinstance(obj, list | tuple):
        # Recursively call to_numpy on each element to handle tensors correctly
        # This will use the CONVERTERS loop for tensor elements within the list
        # Then, construct a 1D numpy array from the processed scalar elements.
        processed_elements = []
        for item in obj:
            converted = to_numpy(
                item
            )  # Handles tensor detach, returns ndarray or scalar
            # Extract scalar value if it's a 0-dim or 1-element array
            if isinstance(converted, np.ndarray) and converted.size == 1:
                processed_elements.append(converted.item())
            # Handle if it was already converted to a Python/Numpy scalar
            elif isinstance(converted, int | float | np.number):
                processed_elements.append(converted)
            else:
                raise TypeError(
                    f"List element conversion resulted in non-scalar "
                    f"type: {type(converted)}"
                )
        return np.array(processed_elements, dtype=float)  # Ensure 1D float array

    for converter in CONVERTERS:
        try:
            return converter(obj)
        except TypeError:
            continue
    raise TypeError(f"Unsupported object type: {type(obj)}")


def is_torch_tensor(obj) -> bool:
    """Checks if an object is a PyTorch tensor.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is a PyTorch tensor, False otherwise.
    """
    if importlib.util.find_spec("torch"):
        import torch

        return isinstance(obj, torch.Tensor)
    return False
