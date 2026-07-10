"""Cross-backend conservation: the single most important invariant.

These tests do not claim H3 or S2 are equal-area (they aren't).  They
verify the simulation code area-weights every aggregation using each
backend's own reported ``area_m2`` — so a non-equal-area backend still
gives unbiased, conserved global totals — and that flux math never
assumes a fixed neighbor degree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from adapters.grid_registry import create_grid
from core.grid.area_weighted import area_weighted_total
from core.grid.flux import diffusion_step

if TYPE_CHECKING:
    from ports.grid import Grid

_RADIUS_M = 6_371_008.8
_DIFFUSIVITY_M2_S = 1.0e7
_STEPS = 5
_HEX_DEGREES = {5, 6}
_PENTAGONS_PER_RESOLUTION = 12
_MAX_RELATIVE_DRIFT = 1e-9


def _checkerboard(grid: Grid) -> dict[str, float]:
    """Return an alternating intensive field over all cells."""
    return {cell: float(index % 2) for index, cell in enumerate(grid.cells())}


def _diffused_total_drift(grid: Grid) -> float:
    """Diffuse a checkerboard and return the relative drift in the total."""
    values = _checkerboard(grid)
    areas = {cell: grid.area_m2(cell) for cell in values}
    cells = list(values)
    before = area_weighted_total([values[c] for c in cells], [areas[c] for c in cells])
    dt_s = 0.05 * min(areas.values()) / _DIFFUSIVITY_M2_S
    for _ in range(_STEPS):
        values = diffusion_step(values, grid, _DIFFUSIVITY_M2_S, dt_s)
    after = area_weighted_total([values[c] for c in cells], [areas[c] for c in cells])
    return abs(after - before) / before


def test_conservation_on_h3_with_pentagons() -> None:
    """Diffusion conserves the global total on H3 despite 12 pentagons."""
    pytest.importorskip("h3")
    grid = create_grid("h3", resolution=0, radius_m=_RADIUS_M)
    assert _diffused_total_drift(grid) < _MAX_RELATIVE_DRIFT


def test_conservation_on_s2_quads() -> None:
    """Diffusion conserves the global total on S2's 4-neighbor quads."""
    pytest.importorskip("s2sphere")
    grid = create_grid("s2", resolution=1, radius_m=_RADIUS_M)
    assert _diffused_total_drift(grid) < _MAX_RELATIVE_DRIFT


def test_conservation_survives_resolution_change() -> None:
    """The same code conserves at a finer resolution too."""
    pytest.importorskip("h3")
    grid = create_grid("h3", resolution=1, radius_m=_RADIUS_M)
    assert _diffused_total_drift(grid) < _MAX_RELATIVE_DRIFT


def test_no_code_path_assumes_a_fixed_degree() -> None:
    """H3 grids really do mix 5- and 6-degree cells (12 pentagons)."""
    pytest.importorskip("h3")
    grid = create_grid("h3", resolution=1, radius_m=_RADIUS_M)
    degrees = [len(grid.neighbors(cell)) for cell in grid.cells()]
    assert set(degrees) == _HEX_DEGREES
    assert degrees.count(min(_HEX_DEGREES)) == _PENTAGONS_PER_RESOLUTION


def test_diffusion_relaxes_toward_uniform() -> None:
    """Diffusion smooths the field (variance strictly decreases)."""
    pytest.importorskip("h3")
    grid = create_grid("h3", resolution=0, radius_m=_RADIUS_M)
    values = _checkerboard(grid)
    areas = {cell: grid.area_m2(cell) for cell in values}
    dt_s = 0.05 * min(areas.values()) / _DIFFUSIVITY_M2_S
    spread_before = max(values.values()) - min(values.values())
    for _ in range(_STEPS):
        values = diffusion_step(values, grid, _DIFFUSIVITY_M2_S, dt_s)
    spread_after = max(values.values()) - min(values.values())
    assert spread_after < spread_before
