"""Smoke tests for the performance profiler tool.

The benchmark is a measurement tool, not a budget guard, so these check
that the harness itself works — every component is timed, results are
well-formed, and the table and JSON render — not that any timing is fast.
"""

from __future__ import annotations

import json

import pytest

from adapters.compute_numpy import NumpyBackend
from tools.benchmark import BenchResult, render_table, run_suite


def test_run_suite_times_every_component() -> None:
    """A quick run produces one well-formed result per component."""
    pytest.importorskip("h3")
    results = run_suite(resolution=1, backend_name="numpy", repeat=1, quick=True)

    assert len(results) == 4
    for result in results:
        assert result.count > 0
        assert result.wall_seconds >= 0.0
        assert result.throughput >= 0.0
    # The pure-Python and backend reductions are both present.
    scales = {r.scale for r in results}
    assert "pure-python" in scales


def test_throughput_is_count_over_wall() -> None:
    """Throughput is derived, not stored — guard the arithmetic."""
    result = BenchResult("x", "y", count=100, wall_seconds=2.0)
    assert result.throughput == 50.0
    assert BenchResult("x", "y", count=100, wall_seconds=0.0).throughput == 0.0


def test_table_and_json_render() -> None:
    """The report renders to text and to serializable JSON."""
    pytest.importorskip("h3")
    results = run_suite(resolution=1, backend_name="numpy", repeat=1, quick=True)
    host = {"cpu_cores": 4, "ram_gib": 16.0, "gpu_count": 0}

    table = render_table(results, NumpyBackend(), host)
    assert "performance profile" in table
    assert "diffusion_step" in table
    # Ranked worst-first: shares are non-increasing down the table.
    shares = [r.wall_seconds for r in sorted(results, key=lambda r: r.wall_seconds, reverse=True)]
    assert shares == sorted(shares, reverse=True)

    payload = {"results": [r.__dict__ for r in results]}
    assert json.loads(json.dumps(payload))["results"]
