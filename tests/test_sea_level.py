"""Tests for the ocean-fraction-to-sea-level solver."""

from __future__ import annotations

import math

import pytest

from core.ocean.sea_level import ocean_fraction_at, solve_sea_level

_ELEVATIONS = [0.0, 10.0, 20.0, 30.0]
_EQUAL_AREAS = [1.0, 1.0, 1.0, 1.0]
_HALF = 0.5


def test_half_ocean_floods_the_lower_half() -> None:
    """A 50% target floods the two lowest of four equal cells."""
    level = solve_sea_level(_ELEVATIONS, _EQUAL_AREAS, _HALF)
    assert level == pytest.approx(10.0)
    assert ocean_fraction_at(_ELEVATIONS, _EQUAL_AREAS, level) == pytest.approx(_HALF)


def test_area_weighting_moves_the_threshold() -> None:
    """A big low cell can satisfy the target on its own."""
    areas = [3.0, 1.0, 1.0, 1.0]
    level = solve_sea_level(_ELEVATIONS, areas, _HALF)
    assert level == pytest.approx(0.0)
    assert ocean_fraction_at(_ELEVATIONS, areas, level) == pytest.approx(_HALF)


def test_bone_dry_world_has_no_sea() -> None:
    """A zero target yields a sea level below all terrain."""
    level = solve_sea_level(_ELEVATIONS, _EQUAL_AREAS, 0.0)
    assert level == -math.inf
    assert ocean_fraction_at(_ELEVATIONS, _EQUAL_AREAS, level) == 0.0


def test_water_world_floods_everything() -> None:
    """A full-ocean target floods the highest cell too."""
    level = solve_sea_level(_ELEVATIONS, _EQUAL_AREAS, 1.0)
    assert level == pytest.approx(30.0)
    assert ocean_fraction_at(_ELEVATIONS, _EQUAL_AREAS, level) == 1.0


def test_tied_elevations_flood_together() -> None:
    """Cells sharing an elevation cannot be split by the threshold."""
    elevations = [0.0, 5.0, 5.0, 30.0]
    level = solve_sea_level(elevations, _EQUAL_AREAS, _HALF)
    assert level == pytest.approx(5.0)
    achieved = ocean_fraction_at(elevations, _EQUAL_AREAS, level)
    assert achieved == pytest.approx(0.75)


def test_validation_rejects_bad_targets_and_shapes() -> None:
    """Out-of-range fractions and mismatched inputs raise."""
    with pytest.raises(ValueError, match="ocean fraction"):
        solve_sea_level(_ELEVATIONS, _EQUAL_AREAS, 1.5)
    with pytest.raises(ValueError, match="elevations"):
        solve_sea_level([0.0], _EQUAL_AREAS, _HALF)
    with pytest.raises(ValueError, match="at least one"):
        solve_sea_level([], [], _HALF)
