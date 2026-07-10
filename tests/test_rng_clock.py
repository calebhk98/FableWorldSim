"""Determinism tests for the seeded RNG and simulation clock."""

from __future__ import annotations

import pytest

from adapters.clock_sim import SimClock
from adapters.rng_seeded import SeededRng

_SEED = 12345
_DRAWS = 10
_TICKS = 7
_TICK_SECONDS = 3_600.0


def test_same_seed_replays_the_same_history() -> None:
    """Two streams with one seed produce identical sequences."""
    a = SeededRng(_SEED)
    b = SeededRng(_SEED)
    assert [a.random() for _ in range(_DRAWS)] == [b.random() for _ in range(_DRAWS)]


def test_forked_streams_are_deterministic_and_independent() -> None:
    """Forks depend only on (seed, label); sibling draws don't interfere."""
    first = SeededRng(_SEED).fork("biology")
    again = SeededRng(_SEED).fork("biology")
    other = SeededRng(_SEED).fork("geology")
    sequence = [first.random() for _ in range(_DRAWS)]
    assert sequence == [again.random() for _ in range(_DRAWS)]
    assert sequence != [other.random() for _ in range(_DRAWS)]


def test_draw_helpers_respect_bounds() -> None:
    """uniform/randint/choice stay in range and choice rejects empties."""
    low, high = 2.0, 3.0
    die_min, die_max = 1, 6
    rng = SeededRng(_SEED)
    for _ in range(_DRAWS):
        assert low <= rng.uniform(low, high) <= high
        assert die_min <= rng.randint(die_min, die_max) <= die_max
    assert rng.choice(["a", "b", "c"]) in {"a", "b", "c"}
    with pytest.raises(ValueError, match="empty"):
        rng.choice([])


def test_clock_advances_deterministically() -> None:
    """Tick count and simulated seconds track advances exactly."""
    clock = SimClock()
    assert clock.tick == 0
    assert clock.sim_seconds == 0.0
    clock.advance(_TICKS, _TICK_SECONDS)
    assert clock.tick == _TICKS
    assert clock.sim_seconds == pytest.approx(_TICKS * _TICK_SECONDS)
    with pytest.raises(ValueError, match="ticks"):
        clock.advance(-1, _TICK_SECONDS)
    with pytest.raises(ValueError, match="tick_seconds"):
        clock.advance(1, 0.0)
