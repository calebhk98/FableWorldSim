"""cupy compute adapter: numpy-compatible arrays on NVIDIA GPUs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ports.array_backend import ArrayBackend, ComputeBackendUnavailableError


class CupyBackend(ArrayBackend):
    """Array math on the GPU via cupy (tracks the numpy API closely)."""

    def __init__(self) -> None:
        """Import cupy lazily and verify a GPU is actually usable."""
        try:
            self._cp = import_module("cupy")
        except ImportError as exc:
            msg = "compute backend 'cupy' needs cupy: pip install cupy-cuda12x"
            raise ComputeBackendUnavailableError(msg) from exc
        try:
            self._cp.cuda.runtime.getDeviceCount()
        except Exception as exc:
            msg = "cupy is installed but no usable CUDA device was found"
            raise ComputeBackendUnavailableError(msg) from exc

    @property
    def name(self) -> str:
        """Return the toggle name of this backend."""
        return "cupy"

    @property
    def device(self) -> str:
        """Return the device arrays live on."""
        return "gpu"

    @property
    def supports_inplace_mutation(self) -> bool:
        """Cupy arrays are mutable in place, like numpy."""
        return True

    def namespace(self) -> Any:
        """Return the cupy module as the Array-API namespace."""
        return self._cp


def create() -> ArrayBackend:
    """Create a :class:`CupyBackend`; registry entry point."""
    return CupyBackend()
