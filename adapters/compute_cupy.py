"""cupy compute adapter: numpy-compatible arrays on NVIDIA GPUs.

Imported statically (typed via ``adapters/stubs/cupy.pyi``); the compute
registry converts an ImportError into a backend-unavailable error with
an install hint (``pip install 'fableworldsim[gpu]'``).
"""

from __future__ import annotations

from typing import Any

import cupy

from ports.array_backend import ArrayBackend, ComputeBackendUnavailableError


class CupyBackend(ArrayBackend):
    """Array math on the GPU via cupy (tracks the numpy API closely)."""

    def __init__(self) -> None:
        """Verify a CUDA device is actually usable, not just installed."""
        try:
            cupy.cuda.runtime.getDeviceCount()
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
        """Return True: cupy arrays are mutable in place, like numpy."""
        return True

    def namespace(self) -> Any:
        """Return the cupy module as the Array-API namespace."""
        return cupy


def create() -> ArrayBackend:
    """Create a :class:`CupyBackend`; registry entry point."""
    return CupyBackend()
