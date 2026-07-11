"""jax compute adapter: functional arrays on CPU/GPU/TPU.

JAX is functional and immutable: no in-place mutation, and ``jit``
constrains control flow and shapes.  Kernels must stay functional when
running here; ``supports_inplace_mutation`` is False so shared code can
branch, and kernels that cannot be expressed uniformly get a JAX-specific
implementation behind :mod:`ports.kernel`.

This is also the project's **multi-GPU path**: :meth:`JaxBackend.shard`
partitions a per-cell field across every visible device via a 1-D device
mesh (``jax.sharding``), so one GPU, two 3090s, or eight A100s run the
same code — jax places and moves shards, the caller just sees a normal
array.  The device count comes from the host probe; the auto-scaler
routes multi-GPU hosts here (see :func:`adapters.hardware.recommend`).

Imported statically (typed via ``adapters/stubs/jax/``); the compute
registry converts an ImportError into a backend-unavailable error with
an install hint (``pip install 'fableworldsim[jax]'``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ports.array_backend import ArrayBackend


class JaxBackend(ArrayBackend):
    """Array math via jax.numpy (functional, jit-compilable)."""

    # Mesh axis that a per-cell field is partitioned along.
    _CELL_AXIS = "cells"

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

    @property
    def num_devices(self) -> int:
        """Return the number of jax devices (GPUs/TPUs, or CPU shards)."""
        return int(jax.device_count())

    def shard(self, array: Any, axis: int = 0) -> Any:
        """Return ``array`` partitioned across every device along ``axis``.

        Builds a 1-D device mesh over ``jax.devices()`` and places the
        array with a :class:`~jax.sharding.NamedSharding` that splits the
        chosen axis.  Downstream ``jax.numpy`` ops on the result stay
        distributed automatically.  When there is a single device, or the
        axis length does not divide evenly across the mesh (an even split
        is required), the array is placed whole instead — correct, just
        not partitioned — so this never raises on an awkward grid size.
        """
        arr = jnp.asarray(array)
        devices = jax.devices()
        count = len(devices)
        if count <= 1 or arr.ndim == 0 or arr.shape[axis] % count != 0:
            return arr
        mesh = Mesh(numpy.asarray(devices), (self._CELL_AXIS,))
        spec: list[str | None] = [None] * arr.ndim
        spec[axis] = self._CELL_AXIS
        sharding = NamedSharding(mesh, PartitionSpec(*spec))
        return jax.device_put(arr, sharding)

    def shard_devices(self, array: Any) -> int:
        """Return how many devices hold a shard of ``array`` (1 if local)."""
        sharding = getattr(array, "sharding", None)
        if sharding is None:
            return 1
        try:
            return len(sharding.device_set)
        except (AttributeError, TypeError):
            return 1


def create() -> ArrayBackend:
    """Create a :class:`JaxBackend`; registry entry point."""
    return JaxBackend()
