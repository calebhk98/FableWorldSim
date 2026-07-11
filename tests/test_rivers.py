"""Tests for the river-network module (``core/hydrology/rivers.py``).

Real-world (H3) tests build a res-2 Earth world (5,882 cells; res-1's 842
cells are too coarse for any drainage basin to clear a meaningful channel
threshold, as observed while writing this suite) and share it across the
module via a fixture, mirroring the pattern in ``test_end_to_end.py`` and
``test_golden_master.py``. A small hand-built ridge/valley grid (FakeGrid,
4-neighbour) checks the routing logic directly without relying on
procedural terrain.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

pytest.importorskip("h3")

from adapters.grid_registry import create_grid
from adapters.kernels_python import flow_accumulation
from adapters.rng_seeded import SeededRng
from core.hydrology.rivers import (
    NO_RECEIVER,
    build_river_network,
    river_outlets,
    steepest_descent_receivers,
)
from core.hydrology.sea_mask import SeaMask, build_sea_mask
from core.sim.presets import earth
from core.topography.procedural import ProceduralTopography
from ports.grid import CellId, Grid
from tests.civ_helpers import FakeGrid

_SEED = 42
_RESOLUTION = 2


@dataclass(frozen=True)
class _EarthWorld:
    """A built known-seed Earth world: grid, heights, and sea mask."""

    grid: Grid
    heights_m: Mapping[CellId, float]
    sea_mask: SeaMask


def _build_earth_world() -> _EarthWorld:
    """Build the canonical known-seed Earth world used by this module."""
    planet = earth()
    grid = create_grid("h3", resolution=_RESOLUTION, radius_m=planet.radius_m)
    heights = ProceduralTopography(SeededRng(_SEED)).heights(grid)
    mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    return _EarthWorld(grid=grid, heights_m=heights, sea_mask=mask)


@pytest.fixture(scope="module")
def earth_world() -> _EarthWorld:
    """Build the Earth world once and share it across this module's tests."""
    return _build_earth_world()


def _mean_area_m2(grid: Grid) -> float:
    """Return the grid's mean per-cell area, used to scale test thresholds."""
    cells = list(grid.cells())
    return sum(grid.area_m2(c) for c in cells) / len(cells)


@pytest.mark.slow
def test_receivers_never_drain_uphill(earth_world: _EarthWorld) -> None:
    """Every routed cell's receiver sits at a strictly lower elevation."""
    grid, heights_m, sea_mask = earth_world.grid, earth_world.heights_m, earth_world.sea_mask
    _, receivers, _, elevations_m = steepest_descent_receivers(grid, heights_m, sea_mask)
    for index, receiver in enumerate(receivers):
        if receiver == NO_RECEIVER:
            continue
        assert elevations_m[receiver] < elevations_m[index]


@pytest.mark.slow
def test_every_receiver_chain_terminates_at_ocean_or_a_true_pit(earth_world: _EarthWorld) -> None:
    """Chains stop only at an ocean cell or a genuine local minimum."""
    grid, heights_m, sea_mask = earth_world.grid, earth_world.heights_m, earth_world.sea_mask
    cells, receivers, _, _ = steepest_descent_receivers(grid, heights_m, sea_mask)
    index_of = {cell: i for i, cell in enumerate(cells)}
    outlets = river_outlets(cells, receivers)

    assert outlets
    for cell, outlet in outlets.items():
        outlet_index = index_of[outlet]
        # Every chain must actually stop (reach a NO_RECEIVER cell).
        assert receivers[outlet_index] == NO_RECEIVER
        if sea_mask.is_land(outlet):
            # A land sink must be a genuine pit: no neighbor sits lower.
            lower = [n for n in grid.neighbors(outlet) if heights_m[n] < heights_m[outlet]]
            assert not lower, f"{cell} chain stopped at {outlet}, which is not a true pit"


@pytest.mark.slow
def test_most_channel_drainage_reaches_the_ocean(earth_world: _EarthWorld) -> None:
    """Most of a dense channel network's drainage area empties into the sea.

    A handful of the very largest trunk rivers can happen to end in
    inland pits on a coarse toy grid, so this uses a lower (denser)
    threshold than the module default to get a representative network
    rather than just the top few streams (observed on this seed: ~73% of
    channel cells and ~61% of channel drainage area reach the ocean).
    """
    grid, heights_m, sea_mask = earth_world.grid, earth_world.heights_m, earth_world.sea_mask
    threshold_m2 = 4.0 * _mean_area_m2(grid)
    network = build_river_network(
        grid, heights_m, sea_mask, flow_accumulation, channel_threshold_m2=threshold_m2
    )
    cells, receivers, _, _ = steepest_descent_receivers(grid, heights_m, sea_mask)
    outlets = river_outlets(cells, receivers)
    channel_cells = [c for c in cells if network.is_channel[c]]
    assert channel_cells

    reaches_ocean = [c for c in channel_cells if sea_mask.ocean[outlets[c]]]
    count_fraction = len(reaches_ocean) / len(channel_cells)
    total_area = sum(network.drainage_area_m2[c] for c in channel_cells)
    ocean_area = sum(network.drainage_area_m2[c] for c in reaches_ocean)
    assert count_fraction > 0.5
    assert ocean_area / total_area > 0.5


@pytest.mark.slow
def test_higher_threshold_yields_fewer_channels(earth_world: _EarthWorld) -> None:
    """Raising the channel threshold strictly shrinks the channel set."""
    grid, heights_m, sea_mask = earth_world.grid, earth_world.heights_m, earth_world.sea_mask
    mean_area_m2 = _mean_area_m2(grid)
    sparse = build_river_network(
        grid, heights_m, sea_mask, flow_accumulation, channel_threshold_m2=10.0 * mean_area_m2
    )
    dense = build_river_network(
        grid, heights_m, sea_mask, flow_accumulation, channel_threshold_m2=2.0 * mean_area_m2
    )
    sparse_count = sum(sparse.is_channel.values())
    dense_count = sum(dense.is_channel.values())
    assert sparse_count > 0
    assert dense_count > sparse_count


@pytest.mark.slow
def test_channel_width_and_depth_are_positive_and_monotone_in_drainage_area(
    earth_world: _EarthWorld,
) -> None:
    """Wider/deeper channels correspond to strictly larger drainage areas."""
    grid, heights_m, sea_mask = earth_world.grid, earth_world.heights_m, earth_world.sea_mask
    network = build_river_network(grid, heights_m, sea_mask, flow_accumulation)
    channel_cells = [c for c in network.is_channel if network.is_channel[c]]
    assert len(channel_cells) > 1

    ordered = sorted(channel_cells, key=lambda c: network.drainage_area_m2[c])
    widths_m = [network.width_m[c] for c in ordered]
    depths_m = [network.depth_m[c] for c in ordered]
    assert all(w > 0.0 for w in widths_m)
    assert all(d > 0.0 for d in depths_m)
    assert all(widths_m[i] <= widths_m[i + 1] for i in range(len(widths_m) - 1))
    assert all(depths_m[i] <= depths_m[i + 1] for i in range(len(depths_m) - 1))

    non_channel_cells = [c for c in network.is_channel if not network.is_channel[c]]
    assert non_channel_cells
    assert all(network.width_m[c] == 0.0 for c in non_channel_cells)
    assert all(network.depth_m[c] == 0.0 for c in non_channel_cells)


_VALLEY_ROWS = 8
_VALLEY_COLS = 7
_VALLEY_CENTER_COL = 3
_VALLEY_ROW_STEP_M = 40.0
_VALLEY_COL_STEP_M = 100.0


def _row_col(cell: CellId) -> tuple[int, int]:
    """Parse a FakeGrid ``r<row>c<col>`` id into its row and column."""
    row, _, col = cell[1:].partition("c")
    return int(row), int(col)


def _valley_world() -> tuple[FakeGrid, dict[CellId, float], SeaMask]:
    """Build a hand-shaped ridge/valley grid draining south to a sea row.

    Elevation falls fastest laterally toward the center column (a "V"
    cross-section funneling water into the valley) and falls south along
    every column (draining the whole grid toward the ocean row at the
    bottom) more slowly than the lateral fall, so land cells route toward
    the valley center before flowing south down it.
    """
    grid = FakeGrid(rows=_VALLEY_ROWS, cols=_VALLEY_COLS, spacing_deg=2.0)
    heights: dict[CellId, float] = {}
    ocean: dict[CellId, bool] = {}
    for cell in grid.cells():
        row, col = _row_col(cell)
        heights[cell] = (
            (_VALLEY_ROWS - 1 - row) * _VALLEY_ROW_STEP_M
            + abs(col - _VALLEY_CENTER_COL) * _VALLEY_COL_STEP_M
            + 10.0
        )
        ocean[cell] = row == _VALLEY_ROWS - 1
    intertidal = dict.fromkeys(grid.cells(), False)
    sea_mask = SeaMask(sea_level_m=0.0, ocean=ocean, intertidal=intertidal)
    return grid, heights, sea_mask


def test_hand_built_valley_channelizes_the_valley_not_the_ridges() -> None:
    """A carved valley drains into a channel; the flanking ridges do not."""
    grid, heights_m, sea_mask = _valley_world()
    threshold_m2 = 2.5 * grid.area_m2("r0c0")
    network = build_river_network(
        grid, heights_m, sea_mask, flow_accumulation, channel_threshold_m2=threshold_m2
    )

    land_rows = range(_VALLEY_ROWS - 1)
    valley_cells = [f"r{r}c{_VALLEY_CENTER_COL}" for r in land_rows]
    ridge_cells = [f"r{r}c{c}" for r in land_rows for c in (0, _VALLEY_COLS - 1)]

    assert all(network.is_channel[c] for c in valley_cells)
    assert not any(network.is_channel[c] for c in ridge_cells)

    # The valley channel widens/deepens downstream as tributaries join.
    widths_m = [network.width_m[c] for c in valley_cells]
    assert all(widths_m[i] < widths_m[i + 1] for i in range(len(widths_m) - 1))
