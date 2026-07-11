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


# Runs inside a subprocess that sees DEVICES CPU devices. Uses an H3-style
# cell count (2 + 120*7^r = 5882) that divides by none of the awkward
# device counts, proving capacity-padding — not luck — makes them shard.
_MULTI_DEVICE_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import jax.numpy as jnp
    from adapters.compute_jax import JaxBackend

    backend = JaxBackend()
    devices = {devices}
    assert backend.num_devices == devices, backend.num_devices

    length = 5882  # H3 resolution-2 cell count; % devices != 0 for 3/5/8/15/27
    values = backend.asarray([float(i % 7) for i in range(length)])
    areas = backend.asarray([1.0 + (i % 3) for i in range(length)])

    sv = backend.shard(values, axis=0)
    sa = backend.shard(areas, axis=0)

    # Genuinely spread across every device despite the indivisible size.
    assert backend.shard_devices(sv) == devices, backend.shard_devices(sv)

    # Distributed area-weighted mean matches the single-device reference:
    # zero-area pad cells drop out of the weighted sum with no masking.
    got = float(jnp.sum(sv * sa) / jnp.sum(sa))
    rv = np.array([float(i % 7) for i in range(length)])
    ra = np.array([1.0 + (i % 3) for i in range(length)])
    ref = float((rv * ra).sum() / ra.sum())
    assert abs(got - ref) < 1e-6, (got, ref, devices)

    # unshard drops the padding and round-trips the logical field exactly.
    back = backend.unshard(sv, length, axis=0)
    assert back.shape[0] == length, back.shape
    assert backend.to_list(back) == backend.to_list(values)

    print("SHARDING_OK")
    """
)


@pytest.mark.slow
@pytest.mark.parametrize("devices", [3, 5, 8, 15, 27])
def test_jax_shards_indivisible_field_across_any_device_count(devices: int) -> None:
    """Proof on CPU: an H3-sized field shards across 3/5/8/15/27 devices."""
    pytest.importorskip("jax")
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join((str(_REPO_ROOT), env.get("PYTHONPATH", "")))

    result = subprocess.run(
        [sys.executable, "-c", _MULTI_DEVICE_SCRIPT.format(devices=devices)],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "SHARDING_OK" in result.stdout
