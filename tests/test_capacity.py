"""Capacity-budget regression guard: the coarse tier must stay cheap.

The doc names concrete targets per hardware tier; this pins the lowest
one so a regression that makes the coarse tier unusable on a 2010-class
no-GPU laptop fails CI. Budgets are generous ceilings (CI machines
vary), not performance goals.
"""

from __future__ import annotations

import time

import pytest

from adapters.grid_registry import create_grid
from core.grid.flux import diffusion_step

_COARSE_RESOLUTION = 2  # h3 res 2 = 5,882 cells (the ~10^4 coarse tier)
_MIN_COARSE_CELLS = 1_000
_MAX_COARSE_CELLS = 100_000
_BUDGET_SECONDS = 30.0
_DIFFUSIVITY = 1.0e7


def test_coarse_tier_diffusion_step_fits_the_laptop_budget() -> None:
    pytest.importorskip("h3")
    grid = create_grid("h3", resolution=_COARSE_RESOLUTION)
    assert _MIN_COARSE_CELLS <= grid.cell_count <= _MAX_COARSE_CELLS

    values = {cell: float(i % 2) for i, cell in enumerate(grid.cells())}
    min_area = min(grid.area_m2(cell) for cell in values)
    dt_s = 0.05 * min_area / _DIFFUSIVITY

    started = time.perf_counter()
    diffusion_step(values, grid, _DIFFUSIVITY, dt_s)
    elapsed = time.perf_counter() - started
    assert elapsed < _BUDGET_SECONDS, f"coarse-tier step took {elapsed:.1f}s"
