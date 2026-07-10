"""Clock port: deterministic simulation time.

Domain code never consults wall time; it asks this port, so runs are
reproducible and tests can drive time explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Clock(ABC):
    """A deterministic simulation clock."""

    @property
    @abstractmethod
    def tick(self) -> int:
        """Return the current tick number (0 before any stepping)."""

    @property
    @abstractmethod
    def sim_seconds(self) -> float:
        """Return total simulated seconds elapsed since world creation."""

    @abstractmethod
    def advance(self, ticks: int, tick_seconds: float) -> None:
        """Advance by ``ticks`` ticks of ``tick_seconds`` each."""
