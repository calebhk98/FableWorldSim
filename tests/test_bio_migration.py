"""Migration: suitability-weighted, count-conserving local diffusion."""

from __future__ import annotations

import pytest

from adapters.rng_seeded import SeededRng
from core.biology.migration import DiffusionParams, Geometry, diffuse
from tests.civ_helpers import FakeGrid


def _line_geometry(grid: FakeGrid) -> Geometry:
    return Geometry(neighbors_of=grid.neighbors, area_of=grid.area_m2)


def test_diffusion_spreads_into_empty_suitable_neighbours() -> None:
    grid = FakeGrid(rows=1, cols=3)
    density = {"r0c0": 1.0, "r0c1": 0.0, "r0c2": 0.0}
    suitability = dict.fromkeys(density, 1.0)
    out = diffuse(density, _line_geometry(grid), suitability, DiffusionParams(0.2), SeededRng(1))
    assert out["r0c1"] > 0.0


def test_diffusion_conserves_head_count() -> None:
    grid = FakeGrid(rows=1, cols=3)
    density = {"r0c0": 1.0, "r0c1": 0.0, "r0c2": 0.0}
    suitability = dict.fromkeys(density, 1.0)
    out = diffuse(density, _line_geometry(grid), suitability, DiffusionParams(0.3), SeededRng(2))
    before = sum(value * grid.area_m2(cell) for cell, value in density.items())
    after = sum(value * grid.area_m2(cell) for cell, value in out.items())
    assert after == pytest.approx(before)


def test_no_suitable_neighbour_means_no_movement() -> None:
    grid = FakeGrid(rows=1, cols=3)
    density = {"r0c0": 1.0, "r0c1": 0.0, "r0c2": 0.0}
    suitability = {"r0c0": 1.0, "r0c1": 0.0, "r0c2": 0.0}
    out = diffuse(density, _line_geometry(grid), suitability, DiffusionParams(0.3), SeededRng(3))
    assert out == density
