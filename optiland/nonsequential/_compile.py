"""What the engine does differently inside the torch backend's compiled bounce step.

``TorchBackend(compile_step=True)`` runs each bounce through ``torch.compile``
(``backends.torch_backend.compiled_bounce_body``). Inside that region Python
values the bounce reads become constants of the compiled program, and the
compiler guards on them: a value that differs from one trace to the next (a
seed, the identity of a scene object) makes every trace compile the bounce
again. The few places that read such values ask :func:`compiling` and, there
only, take a route whose Python state is the same for every trace -- device
data instead of a constant, or a call run eagerly outside the compiled program
(:func:`run_eagerly`). Outside a compiled region nothing here changes what the
engine does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import torch as _torch
except ImportError:  # pragma: no cover - numpy-only installs never compile
    _torch = None


def compiling() -> bool:
    """True inside a ``torch.compile`` region, False everywhere else."""
    return _torch is not None and bool(_torch.compiler.is_compiling())


def _call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    return fn(*args, **kwargs)


# The trampoline the compiler does not trace: a call through it is run as
# ordinary Python, and the compiled program resumes with its result.
_call_outside = _torch._dynamo.disable(_call, recursive=True) if _torch is not None else _call


def run_eagerly(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` outside the compiled program when inside one.

    For work whose Python state differs between traces or scenes (a
    material's property caches, keyed on the identity and contents of its
    arguments): traced, it would be specialised to one scene and compiled
    again for the next; run eagerly, its result enters the compiled program
    as data.

    Args:
        fn: The callable.
        *args: Its positional arguments.
        **kwargs: Its keyword arguments.

    Returns:
        ``fn(*args, **kwargs)``.
    """
    if compiling():
        return _call_outside(fn, *args, **kwargs)
    return fn(*args, **kwargs)
