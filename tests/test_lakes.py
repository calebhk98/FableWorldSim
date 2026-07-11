"""Tests for the lake module (``core/hydrology/lakes.py``).

Uses a hand-built ridge/bowl grid (``FakeGrid``, 4-neighbour), mirroring
the hand-built valley in ``test_rivers.py``: a single pit sits behind a
low "notch" in its rim, with a monotonically descending corridor leading
from the notch out to an ocean edge. Whether that pit ends up dry, an
overflowing lake, or a terminal (endorheic) lake is controlled entirely by
the ``evaporation_mm_yr`` passed to :func:`build_lake_network`, so all
three scenarios share one geometry (built once and observed under three
different evaporation rates) rather than three hand-tuned grids.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from adapters.kernels_python import flow_accumulation
from core.hydrology.lake_rivers import lake_aware_receivers
from core.hydrology.lakes import (
    build_lake_network,
    fill_depressions,
)
from core.hydrology.rivers import NO_RECEIVER, steepest_descent_receivers
from core.hydrology.sea_mask import SeaMask
from ports.grid import CellId
from tests.civ_helpers import FakeGrid

_ROWS = 6
_COLS = 5

# The pit (r2c2) and a low shelf beside it (r2c3) form the lake footprint;
# the notch (r1c2) is the rim's lowest point; the remaining overrides carve
# a strictly descending corridor down column 0 to the ocean at row 5, well
# away from the basin's other (50 m) walls so it can never become a
# cheaper escape route than the notch.
_OVERRIDES = {
    "r2c2": 5.0,
    "r2c3": 8.0,
    "r1c2": 20.0,
    "r3c2": 50.0,
    "r2c1": 50.0,
    "r0c2": 15.0,
    "r0c1": 14.0,
    "r0c0": 13.0,
    "r1c0": 12.0,
    "r2c0": 11.0,
    "r3c0": 10.0,
    "r4c0": 9.0,
}
_DEFAULT_LAND_HEIGHT_M = 100.0
_OUTLET_CELL = "r1c2"
_PIT_CELL = "r2c2"
_SHELF_CELL = "r2c3"


@dataclass(frozen=True)
class _BowlWorld:
    """A hand-built grid with one pit basin behind a low rim notch."""

    grid: FakeGrid
    heights_m: Mapping[CellId, float]
    sea_mask: SeaMask


def _bowl_world() -> _BowlWorld:
    """Build the pit/notch/corridor grid shared by every test in this module."""
    grid = FakeGrid(rows=_ROWS, cols=_COLS, spacing_deg=2.0)
    heights: dict[CellId, float] = {}
    ocean: dict[CellId, bool] = {}
    for cell in grid.cells():
        row, _, _ = cell[1:].partition("c")
        is_ocean = int(row) == _ROWS - 1
        ocean[cell] = is_ocean
        heights[cell] = -10.0 if is_ocean else _DEFAULT_LAND_HEIGHT_M
    heights.update(_OVERRIDES)
    sea_mask = SeaMask(sea_level_m=0.0, ocean=ocean, intertidal=dict.fromkeys(ocean, False))
    return _BowlWorld(grid=grid, heights_m=heights, sea_mask=sea_mask)


def test_pit_is_a_true_sink_before_any_lake_is_considered() -> None:
    """Sanity check: the pit drains nowhere under plain steepest descent."""
    world = _bowl_world()
    cells, receivers, _, _ = steepest_descent_receivers(world.grid, world.heights_m, world.sea_mask)
    index_of = {cell: i for i, cell in enumerate(cells)}
    assert receivers[index_of[_PIT_CELL]] == NO_RECEIVER


def test_fill_depressions_finds_the_notch_as_the_pour_point() -> None:
    """The pit's pour-point elevation is the rim notch, not the 50 m walls."""
    world = _bowl_world()
    fill_level = fill_depressions(world.grid, world.heights_m, world.sea_mask)
    assert fill_level[_PIT_CELL] == world.heights_m[_OUTLET_CELL]
    assert fill_level[_PIT_CELL] < 50.0
    # A cell with a free path to the ocean (no depression) needs no fill.
    assert fill_level[_OUTLET_CELL] == world.heights_m[_OUTLET_CELL]


def test_net_positive_water_balance_forms_an_overflowing_lake() -> None:
    """Ample catchment inflow floods the basin to its rim and spills out."""
    world = _bowl_world()
    cells, receivers, _, _ = steepest_descent_receivers(world.grid, world.heights_m, world.sea_mask)
    network = build_lake_network(world.grid, world.heights_m, world.sea_mask, receivers, cells)

    assert len(network.lakes) == 1
    lake = network.lakes[0]

    # Plausible level: it fills all the way to the rim notch and no higher.
    assert lake.level_m == world.heights_m[_OUTLET_CELL]
    assert lake.spill_level_m == lake.level_m
    assert lake.cells == frozenset({_PIT_CELL, _SHELF_CELL})
    assert lake.outlet == _OUTLET_CELL
    assert lake.inflow_m3_s > lake.evaporation_m3_s > 0.0

    for cell in (_PIT_CELL, _SHELF_CELL):
        assert network.is_lake[cell]
        assert network.level_m[cell] == lake.level_m
        assert network.lake_id[cell] == 0
    assert not network.is_lake[_OUTLET_CELL]


def test_overflowing_lake_wires_into_downstream_flow() -> None:
    """A filled pit's outlet keeps the river network flowing past it.

    Re-deriving receivers over the lake-aware surface should let the
    basin's catchment (previously stranded at the pit) reach cells
    downstream of the rim notch, strictly increasing accumulated
    drainage area there relative to the plain (lake-free) routing.
    """
    world = _bowl_world()
    cells, receivers, areas_m2, elevations_m = steepest_descent_receivers(
        world.grid, world.heights_m, world.sea_mask
    )
    network = build_lake_network(world.grid, world.heights_m, world.sea_mask, receivers, cells)

    lake_cells, lake_receivers, lake_areas_m2, lake_elevations_m = lake_aware_receivers(
        world.grid, world.heights_m, world.sea_mask, network
    )
    index_of = {cell: i for i, cell in enumerate(cells)}
    lake_index_of = {cell: i for i, cell in enumerate(lake_cells)}

    # The pit no longer dead-ends; it (and the rest of the lake) drains
    # toward the notch, which drains onward down the corridor.
    assert lake_receivers[lake_index_of[_PIT_CELL]] != NO_RECEIVER
    assert lake_receivers[lake_index_of[_SHELF_CELL]] != NO_RECEIVER

    plain_drainage = flow_accumulation(receivers, areas_m2, elevations_m)
    routed_drainage = flow_accumulation(lake_receivers, lake_areas_m2, lake_elevations_m)
    downstream_cell = "r4c0"  # the last land cell before the ocean on the corridor
    assert (
        routed_drainage[lake_index_of[downstream_cell]] > plain_drainage[index_of[downstream_cell]]
    )


def test_evaporation_dominated_basin_stays_dry() -> None:
    """Evaporation that outpaces even the smallest puddle leaves no lake."""
    world = _bowl_world()
    cells, receivers, _, _ = steepest_descent_receivers(world.grid, world.heights_m, world.sea_mask)
    network = build_lake_network(
        world.grid,
        world.heights_m,
        world.sea_mask,
        receivers,
        cells,
        evaporation_mm_yr=1.0e9,
    )

    assert network.lakes == ()
    assert not network.is_lake[_PIT_CELL]
    assert network.level_m[_PIT_CELL] == 0.0
    assert network.lake_id[_PIT_CELL] is None

    # A dry basin must not perturb routing: the pit is still a plain sink.
    lake_cells, lake_receivers, _, _ = lake_aware_receivers(
        world.grid, world.heights_m, world.sea_mask, network
    )
    lake_index_of = {cell: i for i, cell in enumerate(lake_cells)}
    assert lake_receivers[lake_index_of[_PIT_CELL]] == NO_RECEIVER


def test_moderate_evaporation_yields_a_terminal_lake_below_the_pour_point() -> None:
    """Evaporation between the two extremes settles at a lower, closed level.

    Too much inflow to stay dry, but not enough to sustain the larger
    surface area needed to reach the rim: the basin should settle right
    at the point where adding the shelf cell's extra area would push
    evaporation above inflow, strictly below the rim notch, with no
    outlet (an endorheic lake, water in but none out).
    """
    world = _bowl_world()
    cells, receivers, _, _ = steepest_descent_receivers(world.grid, world.heights_m, world.sea_mask)
    network = build_lake_network(
        world.grid,
        world.heights_m,
        world.sea_mask,
        receivers,
        cells,
        evaporation_mm_yr=1500.0,
    )

    assert len(network.lakes) == 1
    lake = network.lakes[0]
    assert lake.outlet is None
    assert world.heights_m[_PIT_CELL] < lake.level_m < world.heights_m[_OUTLET_CELL]
    assert lake.cells == frozenset({_PIT_CELL})
    assert not network.is_lake[_SHELF_CELL]
    assert abs(lake.inflow_m3_s - lake.evaporation_m3_s) < lake.inflow_m3_s

    # Still a closed basin: no leak into downstream routing.
    lake_cells, lake_receivers, _, _ = lake_aware_receivers(
        world.grid, world.heights_m, world.sea_mask, network
    )
    lake_index_of = {cell: i for i, cell in enumerate(lake_cells)}
    assert lake_receivers[lake_index_of[_PIT_CELL]] == NO_RECEIVER
