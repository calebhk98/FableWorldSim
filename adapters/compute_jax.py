"""jax compute adapter: functional arrays on CPU/GPU/TPU.

JAX is functional and immutable: no in-place mutation, and ``jit``
constrains control flow and shapes.  Kernels must stay functional when
running here; ``supports_inplace_mutation`` is False so shared code can
branch, and kernels that cannot be expressed uniformly get a JAX-specific
implementation behind :mod:`ports.kernel`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ports.array_backend import ArrayBackend, ComputeBackendUnavailableError


class JaxBackend(ArrayBackend):
    """Array math via jax.numpy (functional, jit-compilable)."""

    def __init__(self) -> None:
        """Import jax lazily so the core never hard-depends on it."""
        try:
            self._jnp = import_module("jax.numpy")
            self._jax = import_module("jax")
        except ImportError as exc:
            msg = "compute backend 'jax' needs jax: pip install jax"
            raise ComputeBackendUnavailableError(msg) from exc

    @property
    def name(self) -> str:
        """Return the toggle name of this backend."""
        return "jax"

    @property
    def device(self) -> str:
        """Return 'gpu' when jax sees an accelerator, else 'cpu'."""
        platform = str(self._jax.default_backend())
        return "cpu" if platform == "cpu" else "gpu"

    @property
    def supports_inplace_mutation(self) -> bool:
        """JAX arrays are immutable."""
        return False

    def namespace(self) -> Any:
        """Return jax.numpy as the Array-API namespace."""
        return self._jnp


def create() -> ArrayBackend:
    """Create a :class:`JaxBackend`; registry entry point."""
    return JaxBackend()
