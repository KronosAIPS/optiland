"""
PyTorch backend -- identity, capability flags, overrides, and precision.

Provides CapabilitiesMixin, used by TorchBackend (see
optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from numpy.typing import ArrayLike
    from torch import Tensor

    from optiland.backend.torch_backend.config import GradMode


#: Device types whose native complex square root is not exact. On Apple's
#: ``mps`` device (torch 2.14.0, and MLX 0.32.2 on the same GPU) the root of an
#: argument whose imaginary part is small beside its real part comes back with
#: that imaginary part wrong or zero: sqrt(5 + 0.0005j) returns 2.2360680 + 0j
#: where the root is 2.2360680 + 1.1180e-4j, and sqrt(5 + 0.005j) returns an
#: imaginary part 2.3 percent low. The real part of sqrt(-4 + 0.001j) is lost
#: the same way. Every other complex operation checked there (cos, sin, exp,
#: products, quotients, abs) is within a few float32 roundoffs of float64.
#: Measured on an Apple silicon GPU, 2026-09-24/25.
_INEXACT_COMPLEX_SQRT_DEVICES = frozenset({"mps"})


def csqrt_from_reals(z: Tensor) -> Tensor:
    """Principal complex square root formed from real operations.

    The half-angle form that never subtracts two nearly equal numbers: with
    ``r = |z| = hypot(a, b)`` and ``t = sqrt((r + |a|) / 2)``, the root is
    ``t + i b / (2t)`` for ``a >= 0`` and ``|b| / (2t) + i copysign(t, b)``
    for ``a < 0``. The part that is small beside the other is a quotient, not
    a difference, so it keeps its relative precision. For ``z = 0`` the root is
    ``0 + i b`` with ``b``'s own sign of zero, which is what the principal
    branch gives. A real non-negative argument returns exactly ``sqrt(a)``,
    and a real negative one exactly ``i sqrt(-a)`` with the sign of its zero
    imaginary part, as the native root does.

    Every step is an ordinary differentiable operation; the quotient's
    denominator is replaced by 1 where ``t`` is zero, so no branch that is
    not taken carries ``0/0`` into a reverse pass.

    Args:
        z: Complex tensor (complex64 or complex128), any device.

    Returns:
        The principal square root of ``z``, same dtype and device.
    """
    a = z.real
    b = z.imag
    r = torch.hypot(a, b)
    t = torch.sqrt((r + torch.abs(a)) * 0.5)
    positive = t > 0
    q = b / (2.0 * torch.where(positive, t, torch.ones_like(t)))
    right = a >= 0
    re = torch.where(right, t, torch.abs(q))
    im = torch.where(right, q, torch.copysign(t, b))
    return torch.complex(re, im)


class CapabilitiesMixin:
    """Identity, capability flags, overrides, and precision."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "torch"

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_gradients(self) -> bool:
        """Return True — PyTorch supports automatic differentiation."""
        return True

    @property
    def supports_gpu(self) -> bool:
        """Return True if CUDA is available."""
        return torch.cuda.is_available()

    @property
    def exact_complex_sqrt(self) -> bool:
        """False on a device whose native complex square root is not exact.

        The configured device's type against
        :data:`_INEXACT_COMPLEX_SQRT_DEVICES` (Apple's ``mps``); True on
        ``cpu`` and ``cuda``.
        """
        return str(self._config.get_device()) not in _INEXACT_COMPLEX_SQRT_DEVICES

    def csqrt(self, z: Any) -> Tensor:
        """Principal complex square root, exact on every device.

        The native ``torch.sqrt`` on a tensor whose device type computes it
        exactly, so ``cpu`` and ``cuda`` results are unchanged to the bit;
        :func:`csqrt_from_reals` on one that does not (``mps``). The rule is
        the tensor's own device type, the same set
        :attr:`exact_complex_sqrt` reads for the configured device, so a
        tensor that is on ``mps`` is protected whatever the configuration.

        Args:
            z: Complex input.

        Returns:
            Tensor: The principal square root of ``z``.
        """
        if (
            isinstance(z, torch.Tensor)
            and z.is_complex()
            and z.device.type in _INEXACT_COMPLEX_SQRT_DEVICES
        ):
            return csqrt_from_reals(z)
        return self.sqrt(z)

    # ------------------------------------------------------------------
    # Capability-gated overrides (torch has real implementations)
    # ------------------------------------------------------------------

    @property
    def grad_mode(self) -> GradMode:
        """Return the GradMode controller."""
        return self._config.grad_mode

    @property
    def autograd(self) -> Any:
        """Return the torch.autograd submodule."""
        return torch.autograd

    @contextlib.contextmanager
    def no_grad_unless_enabled(self) -> Generator[None, None, None]:
        """Suppress autograd bookkeeping unless gradients were requested.

        With ``grad_mode`` disabled no tensor requires grad, so no graph is
        built either way; running under ``torch.no_grad`` additionally skips
        the per-op autograd dispatch and version-counter work, which adds up
        over the hundreds of kernels a trace launches. With ``grad_mode``
        enabled this is a no-op so differentiable workflows are untouched.
        """
        if self._config.grad_mode.requires_grad:
            yield
        else:
            with torch.no_grad():
                yield

    @property
    def nn(self) -> Any:
        """Return the torch.nn submodule."""
        return torch.nn

    def set_device(self, device: str) -> None:
        """Set the compute device.

        Args:
            device: ``'cpu'``, ``'cuda'``, or ``'mps'`` (Apple GPU).
        """
        self._config.set_device(device)  # type: ignore[arg-type]

    def get_device(self) -> str:
        """Return the current compute device."""
        return self._config.get_device()

    def get_complex_precision(self) -> torch.dtype:
        """Return the complex dtype matching the current float precision.

        Returns:
            torch.dtype: ``torch.complex64`` or ``torch.complex128``.

        Raises:
            ValueError: If the current precision is unsupported.
        """
        prec = self._config.get_precision()
        if prec == torch.float32:
            return torch.complex64
        elif prec == torch.float64:
            return torch.complex128
        else:
            raise ValueError("Unsupported precision for complex dtype.")

    def tensor(self, data: Any, **kwargs: Any) -> Tensor:
        """Create a tensor from data with full kwargs support.

        Args:
            data: Input data (scalar, list, numpy array, etc.).
            **kwargs: Forwarded to ``torch.tensor`` (e.g. ``requires_grad``,
                ``dtype``, ``device``).

        Returns:
            Tensor: New tensor.
        """
        kwargs.setdefault("device", self._device())
        kwargs.setdefault("dtype", self._dtype())
        return torch.tensor(data, **kwargs)

    def copy_to(self, source: Tensor, destination: Tensor) -> None:
        """In-place copy from source to destination tensor.

        Safely handles tensors that require gradients.

        Args:
            source: Source tensor.
            destination: Destination tensor (modified in place).
        """
        if destination.requires_grad:
            destination.data.copy_(source)
        else:
            destination.copy_(source)

    def to_tensor(
        self,
        data: ArrayLike,
        device: str | torch.device | None = None,
    ) -> Tensor:
        """Convert data to a PyTorch tensor with the backend's precision.

        Args:
            data: The data to convert.
            device: Optional device override.

        Returns:
            Tensor: Converted tensor.
        """
        current_device = device or self._config.get_device()
        current_precision = self._config.get_precision()
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, device=current_device, dtype=current_precision)
        return data.to(device=current_device, dtype=current_precision)

    def get_bilinear_weights(
        self, coords: Tensor, bin_edges: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Compute differentiable bilinear interpolation weights.

        Args:
            coords: Ray coordinates tensor of shape (N, 2).
            bin_edges: Sequence of two edge tensors [x_edges, y_edges].

        Returns:
            tuple[Tensor, Tensor]: (all_indices, all_weights).
        """
        x_edges, y_edges = bin_edges
        x = coords[:, 0].contiguous()
        y = coords[:, 1].contiguous()

        valid_mask = (
            (x >= x_edges[0])
            & (x <= x_edges[-1])
            & (y >= y_edges[0])
            & (y <= y_edges[-1])
        )

        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        ix = torch.searchsorted(x_centers, x, right=True) - 1
        iy = torch.searchsorted(y_centers, y, right=True) - 1
        ix = torch.clamp(ix, 0, len(x_centers) - 2)
        iy = torch.clamp(iy, 0, len(y_centers) - 2)

        x0, x1 = x_centers[ix], x_centers[ix + 1]
        y0, y1 = y_centers[iy], y_centers[iy + 1]

        wx = (x - x0) / (x1 - x0 + 1e-9)
        wy = (y - y0) / (y1 - y0 + 1e-9)

        w00 = (1 - wx) * (1 - wy)
        w01 = (1 - wx) * wy
        w10 = wx * (1 - wy)
        w11 = wx * wy

        all_indices = torch.stack(
            [
                torch.stack([ix, iy], dim=1),
                torch.stack([ix, iy + 1], dim=1),
                torch.stack([ix + 1, iy], dim=1),
                torch.stack([ix + 1, iy + 1], dim=1),
            ],
            dim=1,
        )
        all_weights = torch.stack([w00, w01, w10, w11], dim=1)
        all_weights = all_weights * valid_mask.unsqueeze(1).to(all_weights.dtype)
        return all_indices, all_weights

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        """Set the floating-point precision.

        Args:
            precision: ``'float32'`` or ``'float64'``.
        """
        self._config.set_precision(precision)

    def get_precision(self) -> int:
        """Return the current precision as an integer (32 or 64)."""
        return 32 if self._config.get_precision() == torch.float32 else 64
