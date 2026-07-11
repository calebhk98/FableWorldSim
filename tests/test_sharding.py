"""Tests for multi-device field sharding on the ArrayBackend port.

The default (numpy/cpu) path is single-device and verified in-process.
The jax device-mesh path is the project's multi-GPU story; there is no
GPU in CI, so real N-way distribution is proven on CPU by faking devices
with ``--xla_force_host_platform_device_count`` in a subprocess (jax
fixes the device count at first use, so it must be set before import).
Both jax tests skip cleanly when jax is not installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from adapters.compute_numpy import NumpyBackend

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FAKE_DEVICES = 8


def test_single_device_backend_shards_as_identity() -> None:
    """A single-device backend reports one device and shards in place."""
    backend = NumpyBackend()
    assert backend.num_devices == 1
    field = backend.asarray([1.0, 2.0, 3.0, 4.0])
    sharded = backend.shard(field)
    assert backend.shard_devices(sharded) == 1
    assert backend.to_list(sharded) == [1.0, 2.0, 3.0, 4.0]


def test_jax_backend_exposes_sharding_surface() -> None:
    """The jax backend implements the port even on a one-device host."""
    pytest.importorskip("jax")
    from adapters.compute_jax import JaxBackend

    backend = JaxBackend()
    assert backend.num_devices >= 1
    field = backend.asarray([1.0, 2.0, 3.0, 4.0])
    sharded = backend.shard(field)
    # Result stays a correct, usable array whatever the device count.
    assert backend.to_list(sharded) == [1.0, 2.0, 3.0, 4.0]
    assert backend.shard_devices(sharded) >= 1


# Runs inside a subprocess that sees _FAKE_DEVICES CPU devices.
_MULTI_DEVICE_SCRIPT = textwrap.dedent(
    """
    import jax.numpy as jnp
    from adapters.compute_jax import JaxBackend

    backend = JaxBackend()
    assert backend.num_devices == {devices}, backend.num_devices

    field = backend.asarray([float(i) for i in range({devices})])
    sharded = backend.shard(field, axis=0)

    # The field is genuinely spread across every device...
    assert backend.shard_devices(sharded) == {devices}, backend.shard_devices(sharded)
    # ...and distributed math still gives the right answer.
    total = float(jnp.sum(sharded * 2.0))
    expected = 2.0 * sum(range({devices}))
    assert total == expected, (total, expected)

    # An axis that does not divide evenly degrades to a whole placement
    # instead of raising.
    odd = backend.asarray([1.0, 2.0, 3.0])
    placed = backend.shard(odd, axis=0)
    assert backend.shard_devices(placed) == 1, backend.shard_devices(placed)

    print("SHARDING_OK")
    """
).format(devices=_FAKE_DEVICES)


def test_jax_shards_a_field_across_many_devices() -> None:
    """Proof on CPU: a field partitions across all faked devices."""
    pytest.importorskip("jax")
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={_FAKE_DEVICES}"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join((str(_REPO_ROOT), env.get("PYTHONPATH", "")))

    result = subprocess.run(
        [sys.executable, "-c", _MULTI_DEVICE_SCRIPT],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "SHARDING_OK" in result.stdout
