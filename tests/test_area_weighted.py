"""Tests for area-weighted aggregation (the equal-surface rule)."""

from __future__ import annotations

import pytest

from core.grid.area_weighted import (
    area_fraction,
    area_weighted_mean,
    area_weighted_total,
)

_EXPECTED_MEAN = 2.0
_EXPECTED_TOTAL = 8.0
_EXPECTED_FRACTION = 0.25


def test_mean_weights_by_area() -> None:
    """Bigger cells count for more, per unit area — not per cell."""
    values = [1.0, 3.0]
    areas = [3.0, 1.0]
    assert area_weighted_mean(values, areas) == pytest.approx(1.5)


def test_mean_is_invariant_under_cell_splitting() -> None:
    """Splitting a cell into equal halves must not change the mean.

    This is the property that makes results invariant to the grid
    backend toggle and to cell-size variation within a backend.
    """
    coarse = area_weighted_mean([1.0, 3.0], [2.0, 2.0])
    fine = area_weighted_mean([1.0, 3.0, 3.0], [2.0, 1.0, 1.0])
    assert coarse == pytest.approx(_EXPECTED_MEAN)
    assert fine == pytest.approx(coarse)


def test_total_integrates_density_over_area() -> None:
    """A per-m^2 density integrates to density * area."""
    assert area_weighted_total([2.0, 1.0], [3.0, 2.0]) == pytest.approx(_EXPECTED_TOTAL)


def test_fraction_is_area_share_not_cell_share() -> None:
    """One flagged cell of four is not 25% unless areas say so."""
    flags = [True, False, False, False]
    areas = [1.0, 1.0, 1.0, 1.0]
    assert area_fraction(flags, areas) == pytest.approx(_EXPECTED_FRACTION)
    lopsided = area_fraction(flags, [3.0, 1.0, 1.0, 1.0])
    assert lopsided == pytest.approx(0.5)


def test_validation_rejects_mismatched_or_empty_cells() -> None:
    """Length mismatches, empty inputs, and bad areas raise."""
    with pytest.raises(ValueError, match="values"):
        area_weighted_mean([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="zero cells"):
        area_weighted_mean([], [])
    with pytest.raises(ValueError, match="area"):
        area_weighted_mean([1.0], [0.0])
