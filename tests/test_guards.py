"""Tests for degenerate-state guards (fail loudly, not silently)."""

from __future__ import annotations

import math

import pytest

from core.sim.guards import DegenerateStateError, assert_conserved, check_runaway


def test_conservation_within_tolerance_passes() -> None:
    assert_conserved(1000.0, 1000.0000001, quantity="energy")


def test_conservation_drift_raises_with_context() -> None:
    with pytest.raises(DegenerateStateError, match="energy"):
        assert_conserved(1000.0, 900.0, quantity="energy")
    with pytest.raises(DegenerateStateError, match="appeared from nothing"):
        assert_conserved(0.0, 5.0)
    with pytest.raises(DegenerateStateError):
        assert_conserved(1000.0, math.inf)


def test_runaway_growth_and_collapse_raise() -> None:
    check_runaway(100.0, 250.0, quantity="population")
    with pytest.raises(DegenerateStateError, match="exploded"):
        check_runaway(100.0, 5000.0, quantity="population")
    with pytest.raises(DegenerateStateError, match="collapsed"):
        check_runaway(100.0, 2.0, quantity="population")
    with pytest.raises(DegenerateStateError, match="non-finite"):
        check_runaway(100.0, math.nan)


def test_extinction_and_cold_start_are_not_runaway() -> None:
    check_runaway(100.0, 0.0)
    check_runaway(0.0, 50.0)
