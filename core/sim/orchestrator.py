"""Multi-rate simulation orchestrator.

Climate and tectonics evolve on much slower clocks than populations, so
the orchestrator steps registered processes at per-process cadences (fast
biology every tick, slow geology every N ticks) instead of one global
timestep.  A tick maps to a configurable span of real simulated time,
defaulting to about a year so a run covers the ~1-5k-year human-migration
horizon.

Processes are pure-looking: ``step`` receives the current world state and
the simulated seconds elapsed since that process last ran, and returns
the next state.  The state type is generic; layers define their own.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from core.sim.constants import SECONDS_PER_YEAR

DEFAULT_TICK_SECONDS = SECONDS_PER_YEAR
"""Default simulated time per tick: one year."""

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class ProcessTiming:
    """Wall-clock cost of one process across a run."""

    calls: int
    wall_seconds: float


@dataclass(frozen=True)
class RunTelemetry:
    """Wall-clock timing for one ``Orchestrator.run`` call.

    This is a side channel — it never enters world state, so it does not
    perturb determinism (a fixed seed + config still produces a bit-identical
    world; only the timing varies run to run). Injecting a monotonic clock
    lets tests assert exact telemetry against a deterministic fake clock.
    """

    ticks: int
    wall_seconds: float
    ticks_per_second: float
    mean_seconds_per_tick: float
    per_process: Mapping[str, ProcessTiming]

    def to_metrics(self) -> dict[str, object]:
        """Return a flat, JSON-able summary for the /metrics surface."""
        return {
            "ticks": self.ticks,
            "wall_seconds": self.wall_seconds,
            "ticks_per_second": self.ticks_per_second,
            "mean_seconds_per_tick": self.mean_seconds_per_tick,
            "per_process": {
                name: {"calls": t.calls, "wall_seconds": t.wall_seconds}
                for name, t in self.per_process.items()
            },
        }


def _build_telemetry(
    ticks: int,
    wall_seconds: float,
    calls: Mapping[str, int],
    wall: Mapping[str, float],
) -> RunTelemetry:
    """Reduce per-process accumulators into a :class:`RunTelemetry`."""
    ticks_per_second = ticks / wall_seconds if wall_seconds > 0.0 else 0.0
    mean = wall_seconds / ticks if ticks > 0 else 0.0
    per_process = {
        name: ProcessTiming(calls=calls[name], wall_seconds=wall[name]) for name in calls
    }
    return RunTelemetry(
        ticks=ticks,
        wall_seconds=wall_seconds,
        ticks_per_second=ticks_per_second,
        mean_seconds_per_tick=mean,
        per_process=per_process,
    )


class Process(Protocol[StateT]):
    """A simulation layer's stepping contract."""

    @property
    def name(self) -> str:
        """Return a human-readable process name (for logs/chronicle)."""
        ...

    def step(self, state: StateT, dt_s: float) -> StateT:
        """Advance ``state`` by ``dt_s`` simulated seconds; return the next."""
        ...


@dataclass(frozen=True)
class ScheduledProcess(Generic[StateT]):
    """A process bound to its cadence (run every ``every_ticks`` ticks)."""

    process: Process[StateT]
    every_ticks: int

    def __post_init__(self) -> None:
        """Validate the cadence."""
        if self.every_ticks < 1:
            msg = f"every_ticks must be >= 1, got {self.every_ticks}"
            raise ValueError(msg)


class Orchestrator(Generic[StateT]):
    """Steps registered processes at their own cadences."""

    def __init__(
        self,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        """Create an orchestrator with a fixed simulated-seconds-per-tick.

        ``monotonic`` is the wall-clock source used for run telemetry; it
        defaults to :func:`time.perf_counter` and is injectable so tests can
        supply a deterministic fake clock.
        """
        if tick_seconds <= 0:
            msg = f"tick_seconds must be > 0, got {tick_seconds}"
            raise ValueError(msg)
        self._tick_seconds = tick_seconds
        self._scheduled: list[ScheduledProcess[StateT]] = []
        self._monotonic = monotonic if monotonic is not None else time.perf_counter
        self._telemetry: RunTelemetry | None = None

    @property
    def tick_seconds(self) -> float:
        """Return the simulated seconds each tick represents."""
        return self._tick_seconds

    @property
    def telemetry(self) -> RunTelemetry | None:
        """Return wall-clock timing for the most recent run (None before any)."""
        return self._telemetry

    @property
    def scheduled(self) -> tuple[ScheduledProcess[StateT], ...]:
        """Return the registered processes in registration order."""
        return tuple(self._scheduled)

    def register(self, process: Process[StateT], every_ticks: int = 1) -> None:
        """Register a process to run every ``every_ticks`` ticks."""
        self._scheduled.append(ScheduledProcess(process, every_ticks))

    def run(self, state: StateT, ticks: int) -> StateT:
        """Advance the world by ``ticks`` ticks and return the final state.

        On each tick, every process whose cadence divides the tick number
        runs (in registration order) and receives the simulated seconds
        elapsed since its previous run.
        """
        if ticks < 0:
            msg = f"ticks must be >= 0, got {ticks}"
            raise ValueError(msg)
        calls: dict[str, int] = {}
        wall: dict[str, float] = {}
        started = self._monotonic()
        for tick in range(1, ticks + 1):
            state = self._step_due(state, tick, calls, wall)
        elapsed = self._monotonic() - started
        self._telemetry = _build_telemetry(ticks, elapsed, calls, wall)
        return state

    def _step_due(
        self,
        state: StateT,
        tick: int,
        calls: dict[str, int],
        wall: dict[str, float],
    ) -> StateT:
        """Run every process due at ``tick``, timing each, and return the next state."""
        for entry in self._scheduled:
            if tick % entry.every_ticks == 0:
                dt_s = entry.every_ticks * self._tick_seconds
                name = entry.process.name
                started = self._monotonic()
                state = entry.process.step(state, dt_s)
                wall[name] = wall.get(name, 0.0) + (self._monotonic() - started)
                calls[name] = calls.get(name, 0) + 1
        return state
