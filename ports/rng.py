"""Rng port: seedable, forkable randomness for reproducible sims.

Every source of randomness in the domain flows through this port.  The
same world seed must replay the same history; independent subsystems
fork *named* streams so adding a random draw in one layer never shifts
the sequence seen by another.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

ItemT = TypeVar("ItemT")


class Rng(ABC):
    """A deterministic random stream."""

    @abstractmethod
    def fork(self, label: str) -> Rng:
        """Return an independent stream derived from this one by label.

        Deterministic: the same parent seed and label always yield the
        same child stream.
        """

    @abstractmethod
    def random(self) -> float:
        """Return a uniform float in [0, 1)."""

    @abstractmethod
    def uniform(self, low: float, high: float) -> float:
        """Return a uniform float in [low, high]."""

    @abstractmethod
    def randint(self, low: int, high: int) -> int:
        """Return a uniform integer in [low, high] inclusive."""

    def choice(self, items: Sequence[ItemT]) -> ItemT:
        """Return a uniformly chosen element of a non-empty sequence."""
        if not items:
            msg = "cannot choose from an empty sequence"
            raise ValueError(msg)
        return items[self.randint(0, len(items) - 1)]
