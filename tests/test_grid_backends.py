"""Tests for the optional H3 and S2 grid adapters.

These skip cleanly when the optional dependency is not installed; the
port contract itself is covered by test_grid_port.py.
"""

from __future__ import annotations

import math

import pytest

from adapters.grid_registry import create_grid

_SPHERE_STERADIANS = 4.0 * math.pi
_TEST_RADIUS_M = 1_000_000.0
_MIN_HEX_NEIGHBORS = 5
_MAX_HEX_NEIGHBORS = 6
_S2_NEIGHBOR_COUNT = 4


def test_h3_covers_sphere_and_scales_with_radius() -> None:
    """H3 cell areas sum to 4*pi*r^2 and adjacency is 5 or 6 cells."""
    pytest.importorskip("h3")
    grid = create_grid("h3", resolution=0, radius_m=_TEST_RADIUS_M)
    cells = list(grid.cells())
    assert len(cells) == grid.cell_count
    total = sum(grid.area_m2(cell) for cell in cells)
    assert total == pytest.approx(_SPHERE_STERADIANS * _TEST_RADIUS_M**2, rel=1e-3)
    neighbor_counts = {len(grid.neighbors(cell)) for cell in cells}
    assert neighbor_counts <= {_MIN_HEX_NEIGHBORS, _MAX_HEX_NEIGHBORS}


def test_h3_point_lookup_roundtrip() -> None:
    """A cell's centroid falls back into that cell."""
    pytest.importorskip("h3")
    grid = create_grid("h3", resolution=1, radius_m=_TEST_RADIUS_M)
    cell = next(iter(grid.cells()))
    assert grid.cell_at(grid.centroid(cell)) == cell


def test_s2_covers_sphere_with_quad_adjacency() -> None:
    """S2 cell areas sum to 4*pi*r^2 and every cell has 4 edge neighbors."""
    pytest.importorskip("s2sphere")
    grid = create_grid("s2", resolution=2, radius_m=_TEST_RADIUS_M)
    cells = list(grid.cells())
    assert len(cells) == grid.cell_count
    total = sum(grid.area_m2(cell) for cell in cells)
    assert total == pytest.approx(_SPHERE_STERADIANS * _TEST_RADIUS_M**2, rel=1e-6)
    for cell in cells[:8]:
        assert len(grid.neighbors(cell)) == _S2_NEIGHBOR_COUNT


def test_s2_point_lookup_roundtrip() -> None:
    """A cell's centroid falls back into that cell."""
    pytest.importorskip("s2sphere")
    grid = create_grid("s2", resolution=3, radius_m=_TEST_RADIUS_M)
    cell = next(iter(grid.cells()))
    assert grid.cell_at(grid.centroid(cell)) == cell
