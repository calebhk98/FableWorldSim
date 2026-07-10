"""Simulation clock adapter: a deterministic tick/seconds counter."""

from __future__ import annotations

from ports.clock import Clock


class SimClock(Clock):
    """A clock advanced only by explicit simulation stepping."""

    def __init__(self) -> None:
        """Start at tick 0, zero seconds."""
        self._tick = 0
        self._sim_seconds = 0.0

    @property
    def tick(self) -> int:
        """Return the current tick number."""
        return self._tick

    @property
    def sim_seconds(self) -> float:
        """Return total simulated seconds elapsed."""
        return self._sim_seconds

    def advance(self, ticks: int, tick_seconds: float) -> None:
        """Advance by ``ticks`` ticks of ``tick_seconds`` each."""
        if ticks < 0:
            msg = f"ticks must be >= 0, got {ticks}"
            raise ValueError(msg)
        if tick_seconds <= 0:
            msg = f"tick_seconds must be > 0, got {tick_seconds}"
            raise ValueError(msg)
        self._tick += ticks
        self._sim_seconds += ticks * tick_seconds


def create() -> Clock:
    """Create a :class:`SimClock`; wiring entry point."""
    return SimClock()
