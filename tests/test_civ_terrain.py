"""Terrain costs, cost-weighted reach, influence fields, and frontiers."""

from __future__ import annotations

from core.civilization.influence import assign_territory, contested_cells, influence_field
from core.civilization.terrain import (
    TerrainCostParams,
    cost_distances,
    subsurface_cost_field,
    surface_cost_field,
)
from tests.civ_helpers import FakeGrid, two_island_world

_PARAMS = TerrainCostParams(water_cost=40.0)
_RANGE_M = 600_000.0
_FLOOR = 1.0
_POWER = 50_000.0

_WEST_CAPITAL = "r4c1"
_EAST_CAPITAL = "r4c6"
_EAST_ISLAND = [f"r{r}c{c}" for r in range(8) for c in (5, 6, 7)]


def test_water_costs_dominate_land_costs_for_surface_civs() -> None:
    grid = FakeGrid()
    heights, mask = two_island_world(grid)
    costs = surface_cost_field(grid, heights, mask, _PARAMS)
    assert costs["r4c3"] == 40.0
    assert costs["r4c1"] < 2.0


def test_naval_multiplier_lowers_water_cost_but_never_below_base() -> None:
    grid = FakeGrid()
    heights, mask = two_island_world(grid)
    naval = surface_cost_field(grid, heights, mask, _PARAMS, water_multiplier=0.25)
    assert naval["r4c3"] == 10.0
    floor = surface_cost_field(grid, heights, mask, _PARAMS, water_multiplier=0.001)
    assert floor["r4c3"] == _PARAMS.base_cost


def test_subsurface_costs_ignore_relief_but_not_the_ocean() -> None:
    grid = FakeGrid()
    _heights, mask = two_island_world(grid)
    costs = subsurface_cost_field(grid, mask, _PARAMS)
    assert costs["r4c1"] == _PARAMS.rock_cost
    assert costs["r4c3"] == _PARAMS.water_cost


def test_cost_distances_grow_monotonically_from_the_source() -> None:
    grid = FakeGrid()
    heights, mask = two_island_world(grid)
    costs = surface_cost_field(grid, heights, mask, _PARAMS)
    distances = cost_distances(grid, costs, (_WEST_CAPITAL,))
    assert distances[_WEST_CAPITAL] == 0.0
    assert distances["r4c2"] < distances["r4c3"] < distances["r4c5"]


def test_influence_cannot_cross_the_ocean_without_naval_tech() -> None:
    grid = FakeGrid()
    heights, mask = two_island_world(grid)
    costs = surface_cost_field(grid, heights, mask, _PARAMS)
    field = influence_field(grid, costs, {_WEST_CAPITAL: _POWER}, _RANGE_M)
    east_max = max(field.get(cell, 0.0) for cell in _EAST_ISLAND)
    assert east_max < _FLOOR


def test_naval_tech_lets_influence_reach_the_far_island() -> None:
    grid = FakeGrid()
    heights, mask = two_island_world(grid)
    costs = surface_cost_field(grid, heights, mask, _PARAMS, water_multiplier=0.25)
    field = influence_field(grid, costs, {_WEST_CAPITAL: _POWER}, _RANGE_M)
    east_max = max(field.get(cell, 0.0) for cell in _EAST_ISLAND)
    assert east_max >= _FLOOR


def test_territory_goes_to_the_higher_influence_and_frontiers_are_detected() -> None:
    grid = FakeGrid()
    heights, mask = two_island_world(grid)
    costs = surface_cost_field(grid, heights, mask, _PARAMS)
    west = influence_field(grid, costs, {"r4c0": _POWER}, _RANGE_M)
    east = influence_field(grid, costs, {"r4c2": _POWER}, _RANGE_M)
    fields = {"w": west, "e": east}
    territory = assign_territory(fields, _FLOOR)
    assert territory["r4c0"] == "w"
    assert territory["r4c2"] == "e"
    contested = contested_cells(fields, territory, contest_ratio=0.5)
    assert "r4c1" in contested
