"""End-to-end batch world sweep: the concrete composition + its API command.

These build real worlds (grid + climate + biome + rivers + biology), so they
are marked slow; the pure engine is covered fast in test_world_sweep.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("h3")

from api.world_service import deepen_seed, evaluate_seed, report_to_dict, run_world_sweep
from core.sim.world_sweep import SweepReport

_RES = 1
_COUNT = 3
_TICKS = 2


@pytest.mark.slow
def test_evaluate_seed_scores_a_real_world() -> None:
    score, details = evaluate_seed(1, resolution=_RES)
    assert 0.0 <= score <= 1.0
    assert details["score"] == pytest.approx(score)
    assert 0.0 <= details["ocean_fraction"] <= 1.0


@pytest.mark.slow
def test_deepen_seed_routes_rivers_and_runs_biology_with_telemetry() -> None:
    result = deepen_seed(1, resolution=_RES, ticks=_TICKS)
    assert result["seed"] == 1
    assert result["biology_ticks"] == _TICKS
    assert result["channel_count"] >= 0
    telemetry = result["telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry["ticks"] == _TICKS
    assert telemetry["ticks_per_second"] > 0.0


@pytest.mark.slow
def test_run_world_sweep_ranks_and_deepens_the_winners() -> None:
    report = run_world_sweep(
        base_seed=1, count=_COUNT, keep_top_k=2, resolution=_RES, deep_ticks=_TICKS
    )
    assert isinstance(report, SweepReport)
    assert len(report.ranked) == _COUNT
    scores = [c.score for c in report.ranked]
    assert scores == sorted(scores, reverse=True)  # best first
    assert len(report.selected) == 2
    assert set(report.deepened) == {c.seed for c in report.selected}

    doc = report_to_dict(report)
    assert [c["seed"] for c in doc["ranked"]] == [c.seed for c in report.ranked]
    assert doc["selected"] == [c.seed for c in report.selected]


@pytest.mark.slow
def test_run_world_sweep_is_deterministic() -> None:
    first = run_world_sweep(base_seed=1, count=_COUNT, keep_top_k=1, resolution=_RES, deep_ticks=1)
    second = run_world_sweep(base_seed=1, count=_COUNT, keep_top_k=1, resolution=_RES, deep_ticks=1)
    assert [c.seed for c in first.ranked] == [c.seed for c in second.ranked]
    assert [c.score for c in first.ranked] == [c.score for c in second.ranked]


@pytest.mark.slow
def test_sweep_command_runs_over_the_api_and_surfaces_telemetry() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.settings import Settings

    app = create_app(Settings())
    client = TestClient(app)

    assert "run_world_sweep" in {e["name"] for e in client.get("/commands").json()}
    response = client.post(
        "/commands/run_world_sweep",
        json={
            "base_seed": 1,
            "count": _COUNT,
            "keep_top_k": 1,
            "resolution": _RES,
            "deep_ticks": 1,
        },
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert len(result["ranked"]) == _COUNT
    assert len(result["selected"]) == 1

    # The winner's deep-run telemetry is surfaced on /metrics.
    assert client.get("/metrics").json()["sim"]["ticks"] == 1


@pytest.mark.slow
def test_evaluate_seed_with_s2_backend() -> None:
    """Evaluate a seed using the s2 grid backend (skip if s2sphere unavailable)."""
    pytest.importorskip("s2sphere")
    score, details = evaluate_seed(1, resolution=_RES, grid_backend="s2")
    assert 0.0 <= score <= 1.0
    assert details["score"] == pytest.approx(score)
    assert 0.0 <= details["ocean_fraction"] <= 1.0


@pytest.mark.slow
def test_deepen_seed_with_s2_backend() -> None:
    """Deep-sim a seed using the s2 grid backend (skip if s2sphere unavailable)."""
    pytest.importorskip("s2sphere")
    result = deepen_seed(1, resolution=_RES, ticks=_TICKS, grid_backend="s2")
    assert result["seed"] == 1
    assert result["biology_ticks"] == _TICKS
    assert result["channel_count"] >= 0
    telemetry = result["telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry["ticks"] == _TICKS
    assert telemetry["ticks_per_second"] > 0.0
