"""ArrayBackend port: per-cell field math via the Python Array API.

The goal is that the same domain code runs on numpy (CPU), cupy/jax
(GPU), or dask/ray (cluster), auto-detecting hardware and falling back
to CPU.  Reality check: the Array API standard narrows the gap but does
not make backends drop-in — JAX is functional and immutable (no in-place
mutation), so "same code, every backend" is a target enforced by the
shared backend-conformance test suite, not a freebie.  Hot kernels that
cannot be expressed uniformly get a per-backend adapter behind the
:mod:`ports.kernel` port instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class ComputeBackendUnavailableError(RuntimeError):
    """Raised when a known compute backend cannot run in this environment."""


class ArrayBackend(ABC):
    """An Array-API-compatible numerical backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the toggle name of this backend (e.g. ``"numpy"``)."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Return where arrays live: ``"cpu"`` or ``"gpu"``."""

    @property
    @abstractmethod
    def supports_inplace_mutation(self) -> bool:
        """Return whether arrays may be mutated in place.

        False for functional backends (JAX); kernels must branch on this
        or stay purely functional.
        """

    @abstractmethod
    def namespace(self) -> Any:
        """Return the Array-API namespace (``xp``) for this backend."""

    def asarray(self, values: Sequence[float]) -> Any:
        """Return ``values`` as this backend's array type."""
        return self.namespace().asarray(values)

    def to_list(self, array: Any) -> list[float]:
        """Return an array's contents as a plain list of floats."""
        return [float(value) for value in list(array)]

    @property
    def num_devices(self) -> int:
        """Return how many devices fields can be sharded across.

        One for single-device backends (numpy, cupy); the count of visible
        accelerators for a device-mesh backend (jax).  The auto-scaler's
        ``device_count`` recommendation is the *target*; this is what the
        backend can actually reach right now.
        """
        return 1

    def shard(self, array: Any, axis: int = 0) -> Any:
        """Return ``array`` partitioned across all devices along ``axis``.

        The default is a single-device identity — the array is returned
        unchanged.  Device-mesh backends override this to split a per-cell
        field across the machine's GPUs while keeping it an ordinary
        Array-API value, so callers never handle placement themselves
        ("the backend hides which device holds which shard").  A backend
        that cannot split evenly must degrade to a whole (unsharded)
        placement rather than raise — results stay correct, just local.
        """
        return array

    def shard_devices(self, array: Any) -> int:
        """Return how many devices hold a piece of ``array`` (1 if local).

        Lets orchestration and tests confirm a field was actually
        distributed instead of silently kept on one device.
        """
        return 1
