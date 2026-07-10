"""Tests for the multi-rate orchestrator (fast biology, slow geology)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from core.sim.orchestrator import DEFAULT_TICK_SECONDS, Orchestrator

_TICK_SECONDS = 100.0
_SLOW_CADENCE = 5
_TOTAL_TICKS = 10
_FAST_RUNS = 10
_SLOW_RUNS = 2


@dataclass(frozen=True)
class _State:
    """A tiny world state recording who stepped and with what dt."""

    log: tuple[tuple[str, float], ...] = field(default=())


@dataclass(frozen=True)
class _Recorder:
    """A process that appends (name, dt) to the state log."""

    name: str

    def step(self, state: _State, dt_s: float) -> _State:
        """Record this invocation and return the next state."""
        return replace(state, log=(*state.log, (self.name, dt_s)))


def test_processes_step_at_their_own_cadence() -> None:
    """Fast biology runs every tick; slow geology every N ticks."""
    orchestrator: Orchestrator[_State] = Orchestrator(tick_seconds=_TICK_SECONDS)
    orchestrator.register(_Recorder("biology"), every_ticks=1)
    orchestrator.register(_Recorder("geology"), every_ticks=_SLOW_CADENCE)
    final = orchestrator.run(_State(), ticks=_TOTAL_TICKS)
    names = [name for name, _ in final.log]
    assert names.count("biology") == _FAST_RUNS
    assert names.count("geology") == _SLOW_RUNS


def test_slow_processes_receive_their_full_elapsed_time() -> None:
    """A process running every N ticks sees dt = N * tick_seconds."""
    orchestrator: Orchestrator[_State] = Orchestrator(tick_seconds=_TICK_SECONDS)
    orchestrator.register(_Recorder("geology"), every_ticks=_SLOW_CADENCE)
    final = orchestrator.run(_State(), ticks=_TOTAL_TICKS)
    for _, dt_s in final.log:
        assert dt_s == pytest.approx(_SLOW_CADENCE * _TICK_SECONDS)


def test_processes_run_in_registration_order_within_a_tick() -> None:
    """Deterministic ordering: registration order within each tick."""
    orchestrator: Orchestrator[_State] = Orchestrator(tick_seconds=_TICK_SECONDS)
    orchestrator.register(_Recorder("first"))
    orchestrator.register(_Recorder("second"))
    final = orchestrator.run(_State(), ticks=1)
    assert [name for name, _ in final.log] == ["first", "second"]


def test_default_tick_is_about_a_year() -> None:
    """The default tick spans one simulated Julian year."""
    orchestrator: Orchestrator[_State] = Orchestrator()
    assert orchestrator.tick_seconds == pytest.approx(DEFAULT_TICK_SECONDS)
    assert DEFAULT_TICK_SECONDS == pytest.approx(365.25 * 86_400.0)


def test_validation_rejects_bad_cadence_and_ticks() -> None:
    """Zero cadences and negative tick counts raise."""
    orchestrator: Orchestrator[_State] = Orchestrator(tick_seconds=_TICK_SECONDS)
    with pytest.raises(ValueError, match="every_ticks"):
        orchestrator.register(_Recorder("bad"), every_ticks=0)
    with pytest.raises(ValueError, match="ticks"):
        orchestrator.run(_State(), ticks=-1)
    with pytest.raises(ValueError, match="tick_seconds"):
        Orchestrator(tick_seconds=0.0)
