"""Seeded RNG adapter: deterministic, forkable streams on stdlib random.

Child streams are derived by hashing (parent seed, label), so the same
world seed always replays the same history and adding draws in one
subsystem never shifts another subsystem's sequence.
"""

from __future__ import annotations

import hashlib
import random

from ports.rng import Rng


def _derive_seed(seed: int, label: str) -> int:
    """Return a child seed from a parent seed and a stream label."""
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class SeededRng(Rng):
    """A deterministic stream backed by ``random.Random``."""

    def __init__(self, seed: int) -> None:
        """Create a stream from an integer seed."""
        self._seed = seed
        self._random = random.Random(seed)

    @property
    def seed(self) -> int:
        """Return the seed this stream was created from."""
        return self._seed

    def fork(self, label: str) -> Rng:
        """Return an independent child stream for ``label``."""
        return SeededRng(_derive_seed(self._seed, label))

    def random(self) -> float:
        """Return a uniform float in [0, 1)."""
        return self._random.random()

    def uniform(self, low: float, high: float) -> float:
        """Return a uniform float in [low, high]."""
        return self._random.uniform(low, high)

    def randint(self, low: int, high: int) -> int:
        """Return a uniform integer in [low, high] inclusive."""
        return self._random.randint(low, high)


def create(seed: int) -> Rng:
    """Create a :class:`SeededRng`; registry/wiring entry point."""
    return SeededRng(seed)
