"""numpy compute adapter: the CPU default, available everywhere.

numpy is a base dependency, so this import cannot fail in a supported
install; the registry still converts ImportError uniformly for safety.
"""

from __future__ import annotations

from typing import Any

import numpy

from ports.array_backend import ArrayBackend


class NumpyBackend(ArrayBackend):
    """Array math on the CPU via numpy."""

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
        """Return True: numpy arrays are mutable in place."""
        return True

    def namespace(self) -> Any:
        """Return the numpy module as the Array-API namespace."""
        return numpy


def create() -> ArrayBackend:
    """Create a :class:`NumpyBackend`; registry entry point."""
    return NumpyBackend()
