"""Tests for backend-accelerated, shardable area-weighted reductions.

These prove the *integration seam*: a real sim reduction
(:func:`civ_population_total`) and the public area-weighted helpers run
the identical math through an ArrayBackend when one is supplied, matching
the pure-Python reference. The numpy path is exact and runs in-process;
the jax multi-device path is proven on CPU with faked devices (slow) and
skips where jax is absent.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from adapters.compute_numpy import NumpyBackend
from core.civilization.state import Civilization, civ_population_total
from core.grid.area_weighted import area_weighted_mean, area_weighted_total
from core.grid.field_reduce import area_weighted_mean_on, area_weighted_total_on
from tests.civ_helpers import FakeGrid

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]
_AREAS = [10.0, 20.0, 30.0, 40.0, 50.0]


def test_numpy_reducers_match_the_pure_reference_exactly() -> None:
    """On numpy (float64) the sharded path equals the pure-Python path."""
    backend = NumpyBackend()
    assert area_weighted_mean_on(backend, _VALUES, _AREAS) == area_weighted_mean(_VALUES, _AREAS)
    assert area_weighted_total_on(backend, _VALUES, _AREAS) == area_weighted_total(_VALUES, _AREAS)


def test_public_helpers_accept_a_backend() -> None:
    """The optional backend argument is the opt-in seam; results agree."""
    backend = NumpyBackend()
    assert area_weighted_mean(_VALUES, _AREAS, backend=backend) == area_weighted_mean(
        _VALUES, _AREAS
    )
    assert area_weighted_total(_VALUES, _AREAS, backend=backend) == area_weighted_total(
        _VALUES, _AREAS
    )


def test_civ_population_total_forwards_the_backend() -> None:
    """The real sim reduction routes through the backend when given one."""
    grid = FakeGrid()
    cells = list(grid.cells())[:5]
    civ = Civilization(
        civ_id="c1",
        species_id="s1",
        domain="surface",
        population_per_m2={cell: 0.5 + i for i, cell in enumerate(cells)},
        settlements=(),
    )
    reference = civ_population_total(grid, civ)
    on_backend = civ_population_total(grid, civ, backend=NumpyBackend())
    assert on_backend == pytest.approx(reference)


# Runs inside a subprocess that sees DEVICES CPU devices: a real sim
# reduction (area-weighted total) over an H3-sized field, sharded.
_SEAM_SCRIPT = textwrap.dedent(
    """
    from adapters.compute_jax import JaxBackend
    from core.grid.area_weighted import area_weighted_total

    backend = JaxBackend()
    devices = {devices}
    assert backend.num_devices == devices, backend.num_devices

    length = 5882  # H3 res-2 count; indivisible by 8 and 27
    densities = [0.1 * (i % 11) for i in range(length)]
    areas = [1_000.0 + (i % 5) for i in range(length)]

    # The field genuinely spreads across every device...
    sharded = backend.shard(backend.asarray(densities))
    assert backend.shard_devices(sharded) == devices, backend.shard_devices(sharded)

    # ...and the sharded reduction matches the pure-Python reference.
    reference = area_weighted_total(densities, areas)
    distributed = area_weighted_total(densities, areas, backend=backend)
    assert abs(distributed - reference) <= 1e-3 * abs(reference), (distributed, reference)

    print("SEAM_OK")
    """
)


@pytest.mark.slow
@pytest.mark.parametrize("devices", [8, 27])
def test_sim_reduction_runs_sharded_across_devices(devices: int) -> None:
    """A real sim reduction distributes across 8 and 27 faked devices."""
    pytest.importorskip("jax")
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join((str(_REPO_ROOT), env.get("PYTHONPATH", "")))

    result = subprocess.run(
        [sys.executable, "-c", _SEAM_SCRIPT.format(devices=devices)],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "SEAM_OK" in result.stdout
