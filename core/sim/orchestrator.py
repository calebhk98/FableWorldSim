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

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from core.sim.constants import SECONDS_PER_YEAR

DEFAULT_TICK_SECONDS = SECONDS_PER_YEAR
"""Default simulated time per tick: one year."""

StateT = TypeVar("StateT")


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

    def __init__(self, tick_seconds: float = DEFAULT_TICK_SECONDS) -> None:
        """Create an orchestrator with a fixed simulated-seconds-per-tick."""
        if tick_seconds <= 0:
            msg = f"tick_seconds must be > 0, got {tick_seconds}"
            raise ValueError(msg)
        self._tick_seconds = tick_seconds
        self._scheduled: list[ScheduledProcess[StateT]] = []

    @property
    def tick_seconds(self) -> float:
        """Return the simulated seconds each tick represents."""
        return self._tick_seconds

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
        for tick in range(1, ticks + 1):
            state = self._step_due(state, tick)
        return state

    def _step_due(self, state: StateT, tick: int) -> StateT:
        """Run every process due at ``tick`` and return the next state."""
        for entry in self._scheduled:
            if tick % entry.every_ticks == 0:
                dt_s = entry.every_ticks * self._tick_seconds
                state = entry.process.step(state, dt_s)
        return state
