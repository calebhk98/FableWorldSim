"""Sim-speed telemetry: wall-clock timing of orchestrator runs.

The orchestrator times each run against an injectable monotonic clock, so
these tests drive a deterministic fake clock and assert exact timing —
no flaky real-wall-clock thresholds. Telemetry is a side channel and must
never perturb world state or determinism.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from core.sim.orchestrator import Orchestrator, RunTelemetry

_TICK_SECONDS = 100.0


class _StepClock:
    """A fake monotonic clock that advances one unit per call."""

    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        now = self._t
        self._t += 1.0
        return now


@dataclass(frozen=True)
class _State:
    """A tiny world state recording who stepped."""

    log: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class _Recorder:
    """A process that appends its name to the state log."""

    name: str

    def step(self, state: _State, dt_s: float) -> _State:
        """Record this invocation and return the next state."""
        return replace(state, log=(*state.log, self.name))


def test_no_telemetry_before_a_run() -> None:
    orchestrator: Orchestrator[_State] = Orchestrator(tick_seconds=_TICK_SECONDS)
    assert orchestrator.telemetry is None


def test_single_process_run_records_exact_timing() -> None:
    """One process, N ticks, unit clock: timing is fully determined."""
    ticks = 4
    orchestrator: Orchestrator[_State] = Orchestrator(
        tick_seconds=_TICK_SECONDS, monotonic=_StepClock()
    )
    orchestrator.register(_Recorder("biology"))
    orchestrator.run(_State(), ticks=ticks)

    tel = orchestrator.telemetry
    assert tel is not None
    # Clock calls: start(0) + per tick before/after(2) + end => elapsed = 2N+1.
    assert tel.ticks == ticks
    assert tel.wall_seconds == pytest.approx(2 * ticks + 1)
    assert tel.ticks_per_second == pytest.approx(ticks / (2 * ticks + 1))
    assert tel.mean_seconds_per_tick == pytest.approx((2 * ticks + 1) / ticks)
    bio = tel.per_process["biology"]
    assert bio.calls == ticks
    assert bio.wall_seconds == pytest.approx(ticks)  # each step spans one unit


def test_per_process_call_counts_follow_cadence() -> None:
    """Fast process every tick, slow every N; per-process calls reflect it."""
    orchestrator: Orchestrator[_State] = Orchestrator(
        tick_seconds=_TICK_SECONDS, monotonic=_StepClock()
    )
    orchestrator.register(_Recorder("biology"), every_ticks=1)
    orchestrator.register(_Recorder("geology"), every_ticks=5)
    orchestrator.run(_State(), ticks=10)

    tel = orchestrator.telemetry
    assert tel is not None
    assert tel.per_process["biology"].calls == 10
    assert tel.per_process["geology"].calls == 2
    # Per-process wall time is a subset of the whole run.
    process_wall = sum(t.wall_seconds for t in tel.per_process.values())
    assert process_wall <= tel.wall_seconds


def test_telemetry_does_not_change_the_final_state() -> None:
    """Instrumentation is a side channel: state matches an untimed run."""
    a: Orchestrator[_State] = Orchestrator(tick_seconds=_TICK_SECONDS, monotonic=_StepClock())
    a.register(_Recorder("biology"))
    b: Orchestrator[_State] = Orchestrator(tick_seconds=_TICK_SECONDS)
    b.register(_Recorder("biology"))
    assert a.run(_State(), ticks=3) == b.run(_State(), ticks=3)


def test_to_metrics_is_json_shaped() -> None:
    orchestrator: Orchestrator[_State] = Orchestrator(monotonic=_StepClock())
    orchestrator.register(_Recorder("biology"))
    orchestrator.run(_State(), ticks=2)
    tel = orchestrator.telemetry
    assert tel is not None
    metrics = tel.to_metrics()
    assert metrics["ticks"] == 2
    assert "ticks_per_second" in metrics
    assert metrics["per_process"]["biology"]["calls"] == 2


def _telemetry_for(ticks: int) -> RunTelemetry:
    orchestrator: Orchestrator[_State] = Orchestrator(monotonic=_StepClock())
    orchestrator.register(_Recorder("biology"))
    orchestrator.run(_State(), ticks=ticks)
    assert orchestrator.telemetry is not None
    return orchestrator.telemetry


def test_app_state_records_and_broadcasts_telemetry() -> None:
    pytest.importorskip("fastapi")
    from api.settings import Settings
    from api.state import AppState, EventBus

    bus = EventBus()
    state = AppState(Settings(), bus)
    queue = bus.subscribe()
    assert state.sim_telemetry is None

    state.record_run_telemetry(_telemetry_for(3))

    assert state.sim_telemetry is not None
    assert state.sim_telemetry["ticks"] == 3
    event = queue.get_nowait()
    assert event.type == "run_telemetry"
    assert event.ticks == 3


def test_metrics_endpoint_and_command_expose_telemetry() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.settings import Settings

    app = create_app(Settings())
    client = TestClient(app)

    # The command is self-described and returns empty before any run.
    names = {entry["name"] for entry in client.get("/commands").json()}
    assert "get_run_telemetry" in names
    empty = client.post("/commands/get_run_telemetry", json={})
    assert empty.status_code == 200
    assert empty.json()["result"] == {}

    # /ws/schema documents the new streaming event.
    assert "run_telemetry" in client.get("/ws/schema").json()

    # After a run is recorded, both surfaces reflect it.
    app.state.sim.record_run_telemetry(_telemetry_for(2))
    assert client.get("/metrics").json()["sim"]["ticks"] == 2
    assert client.post("/commands/get_run_telemetry", json={}).json()["result"]["ticks"] == 2
