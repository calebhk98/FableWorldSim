"""Tests for explicit cross-backend reprojection (the world-move path)."""

from __future__ import annotations

import pytest

from adapters.grid_registry import create_grid
from core.grid.reproject import reproject_extensive, reproject_intensive

_RADIUS_M = 6_371_008.8
_UNIFORM_DENSITY = 2.5
_TOTAL_TOLERANCE = 0.05


def test_intensive_uniform_field_survives_reprojection() -> None:
    """A constant per-area field is exactly constant on the new grid."""
    pytest.importorskip("h3")
    pytest.importorskip("s2sphere")
    h3_grid = create_grid("h3", resolution=1, radius_m=_RADIUS_M)
    s2_grid = create_grid("s2", resolution=2, radius_m=_RADIUS_M)
    field = {cell: _UNIFORM_DENSITY for cell in h3_grid.cells()}
    moved = reproject_intensive(field, h3_grid, s2_grid)
    assert set(moved) == set(s2_grid.cells())
    assert all(value == _UNIFORM_DENSITY for value in moved.values())


def test_extensive_total_is_approximately_preserved() -> None:
    """Population-style totals survive an h3 -> s2 move to first order."""
    pytest.importorskip("h3")
    pytest.importorskip("s2sphere")
    h3_grid = create_grid("h3", resolution=1, radius_m=_RADIUS_M)
    s2_grid = create_grid("s2", resolution=2, radius_m=_RADIUS_M)
    amounts = {cell: _UNIFORM_DENSITY * h3_grid.area_m2(cell) for cell in h3_grid.cells()}
    moved = reproject_extensive(amounts, h3_grid, s2_grid)
    total_before = sum(amounts.values())
    total_after = sum(moved.values())
    assert abs(total_after - total_before) / total_before < _TOTAL_TOLERANCE
