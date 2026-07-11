"""Tests for the optional H3, S2, and ISEA grid adapters.

These skip cleanly when the optional dependency is not installed (H3, S2)
or when the external DGGRID binary is not on PATH (ISEA); the port
contract itself is covered by test_grid_port.py.
"""

from __future__ import annotations

import math

import pytest

from adapters.grid_registry import create_grid
from ports.grid import GridBackendUnavailableError

_SPHERE_STERADIANS = 4.0 * math.pi
_TEST_RADIUS_M = 1_000_000.0
_MIN_HEX_NEIGHBORS = 5
_MAX_HEX_NEIGHBORS = 6
_S2_NEIGHBOR_COUNT = 4
_PENTAGONS_PER_RESOLUTION = 12


def _isea_grid(resolution: int, radius_m: float = _TEST_RADIUS_M):
    """Build an ISEA grid, skipping the test if DGGRID isn't installed."""
    try:
        return create_grid("isea", resolution=resolution, radius_m=radius_m)
    except GridBackendUnavailableError:
        pytest.skip("dggrid binary not available")


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


def test_isea_covers_sphere_with_hex_adjacency() -> None:
    """ISEA3H cell areas sum to 4*pi*r^2 and adjacency is 5 or 6 cells."""
    grid = _isea_grid(resolution=0)
    cells = list(grid.cells())
    assert len(cells) == grid.cell_count
    total = sum(grid.area_m2(cell) for cell in cells)
    assert total == pytest.approx(_SPHERE_STERADIANS * _TEST_RADIUS_M**2, rel=1e-6)
    neighbor_counts = {len(grid.neighbors(cell)) for cell in cells}
    assert neighbor_counts <= {_MIN_HEX_NEIGHBORS, _MAX_HEX_NEIGHBORS}


def test_isea_has_exactly_twelve_pentagons() -> None:
    """ISEA3H's icosahedral origin means exactly 12 cells have 5 neighbors."""
    grid = _isea_grid(resolution=1)
    degrees = [len(grid.neighbors(cell)) for cell in grid.cells()]
    assert degrees.count(_MIN_HEX_NEIGHBORS) == _PENTAGONS_PER_RESOLUTION


def test_isea_point_lookup_roundtrip() -> None:
    """A cell's centroid falls back into that cell."""
    grid = _isea_grid(resolution=1)
    cell = next(iter(grid.cells()))
    assert grid.cell_at(grid.centroid(cell)) == cell


def test_isea_cells_are_equal_area_within_the_pentagon_exception() -> None:
    """Every ISEA3H hex cell has the same area; pentagons are 5/6 of it.

    This is the entire point of the ISEA backend: it is a genuine
    equal-area DGGS (unlike H3/S2), modulo the topologically-necessary
    12 pentagons every icosahedral hex grid carries.
    """
    grid = _isea_grid(resolution=1)
    hex_areas = [
        grid.area_m2(cell)
        for cell in grid.cells()
        if len(grid.neighbors(cell)) == _MAX_HEX_NEIGHBORS
    ]
    pentagon_areas = [
        grid.area_m2(cell)
        for cell in grid.cells()
        if len(grid.neighbors(cell)) == _MIN_HEX_NEIGHBORS
    ]
    assert max(hex_areas) == pytest.approx(min(hex_areas), rel=1e-6)
    assert max(pentagon_areas) == pytest.approx(min(pentagon_areas), rel=1e-6)
    hex_area = sum(hex_areas) / len(hex_areas)
    pentagon_area = sum(pentagon_areas) / len(pentagon_areas)
    assert pentagon_area == pytest.approx(hex_area * 5.0 / 6.0, rel=1e-6)
