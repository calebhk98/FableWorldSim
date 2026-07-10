"""numpy compute adapter: the CPU default, available everywhere."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ports.array_backend import ArrayBackend, ComputeBackendUnavailableError


class NumpyBackend(ArrayBackend):
    """Array math on the CPU via numpy."""

    def __init__(self) -> None:
        """Import numpy lazily so the core never hard-depends on it."""
        try:
            self._np = import_module("numpy")
        except ImportError as exc:
            msg = "compute backend 'numpy' needs numpy: pip install numpy"
            raise ComputeBackendUnavailableError(msg) from exc

    @property
    def name(self) -> str:
        """Return the toggle name of this backend."""
        return "numpy"

    @property
    def device(self) -> str:
        """Return the device arrays live on."""
        return "cpu"

    @property
    def supports_inplace_mutation(self) -> bool:
        """Numpy arrays are mutable in place."""
        return True

    def namespace(self) -> Any:
        """Return the numpy module as the Array-API namespace."""
        return self._np


def create() -> ArrayBackend:
    """Create a :class:`NumpyBackend`; registry entry point."""
    return NumpyBackend()
