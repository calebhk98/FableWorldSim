"""jax compute adapter: functional arrays on CPU/GPU/TPU.

JAX is functional and immutable: no in-place mutation, and ``jit``
constrains control flow and shapes.  Kernels must stay functional when
running here; ``supports_inplace_mutation`` is False so shared code can
branch, and kernels that cannot be expressed uniformly get a JAX-specific
implementation behind :mod:`ports.kernel`.

Imported statically (typed via ``adapters/stubs/jax/``); the compute
registry converts an ImportError into a backend-unavailable error with
an install hint (``pip install 'fableworldsim[jax]'``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ports.array_backend import ArrayBackend


class JaxBackend(ArrayBackend):
    """Array math via jax.numpy (functional, jit-compilable)."""

    @property
    def name(self) -> str:
        """Return the toggle name of this backend."""
        return "jax"

    @property
    def device(self) -> str:
        """Return 'gpu' when jax sees an accelerator, else 'cpu'."""
        return "cpu" if jax.default_backend() == "cpu" else "gpu"

    @property
    def supports_inplace_mutation(self) -> bool:
        """Return False: JAX arrays are immutable."""
        return False

    def namespace(self) -> Any:
        """Return jax.numpy as the Array-API namespace."""
        return jnp


def create() -> ArrayBackend:
    """Create a :class:`JaxBackend`; registry entry point."""
    return JaxBackend()
