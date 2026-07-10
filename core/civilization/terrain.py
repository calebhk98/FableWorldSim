"""Terrain movement cost and cost-weighted distances over the grid.

One distance-plus-terrain-cost model underlies both influence (borders) and
supply projection (how much strength a civ can bring to a far cell), so reach
is bounded the same way everywhere.  Water is near-impassable for land civs
until naval tech lowers its cost multiplier; mountains get more expensive
with relief.  Subsurface civs ignore relief but pay a uniform rock cost.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.grid.geodesy import great_circle_distance_m

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid


@dataclass(frozen=True)
class TerrainCostParams:
    """Tunable knobs for the movement-cost model."""

    base_cost: float = 1.0
    water_cost: float = 40.0
    rock_cost: float = 3.0
    mountain_scale_m: float = 1_500.0
    mountain_cost: float = 2.0

    def __post_init__(self) -> None:
        """Reject non-positive scales that would break the cost model."""
        if self.base_cost <= 0 or self.water_cost <= 0 or self.rock_cost <= 0:
            msg = "terrain costs must be > 0"
            raise ValueError(msg)
        if self.mountain_scale_m <= 0 or self.mountain_cost < 0:
            msg = "mountain scale must be > 0 and mountain cost >= 0"
            raise ValueError(msg)


def surface_cost_field(
    grid: Grid,
    heights_m: Mapping[CellId, float],
    sea_mask: SeaMask,
    params: TerrainCostParams,
    water_multiplier: float = 1.0,
) -> dict[CellId, float]:
    """Return per-cell movement cost for a surface civilization.

    ``water_multiplier`` comes from tech effects: naval tech shrinks it, and
    flight shrinks it further, which is the entire water/air-crossing
    mechanism in M1.
    """
    costs: dict[CellId, float] = {}
    for cell in grid.cells():
        if sea_mask.ocean[cell]:
            costs[cell] = max(params.base_cost, params.water_cost * water_multiplier)
        else:
            relief = max(0.0, heights_m[cell] - sea_mask.sea_level_m) / params.mountain_scale_m
            costs[cell] = params.base_cost + params.mountain_cost * relief
    return costs


def subsurface_cost_field(
    grid: Grid,
    sea_mask: SeaMask,
    params: TerrainCostParams,
) -> dict[CellId, float]:
    """Return per-cell movement cost for a subsurface civilization.

    Tunnelling ignores surface relief (mountains are home, not obstacles)
    but pays a uniform rock cost; the water column above ocean cells still
    blocks expansion in M1.
    """
    return {
        cell: params.water_cost if sea_mask.ocean[cell] else params.rock_cost
        for cell in grid.cells()
    }


def cost_distances(
    grid: Grid,
    costs: Mapping[CellId, float],
    sources: Iterable[CellId],
    max_cost_m: float = math.inf,
) -> dict[CellId, float]:
    """Return cost-weighted distances (meters x cost) from the nearest source.

    Runs Dijkstra over the cell graph; each edge weighs the great-circle
    distance between centroids by the mean terrain cost of its endpoints.
    Cells beyond ``max_cost_m`` are omitted (treat missing as unreachable).
    """
    distances: dict[CellId, float] = {}
    heap: list[tuple[float, CellId]] = [(0.0, source) for source in sources]
    heapq.heapify(heap)
    while heap:
        found, cell = heapq.heappop(heap)
        if cell in distances or found > max_cost_m:
            continue
        distances[cell] = found
        origin = grid.centroid(cell)
        for neighbor in grid.neighbors(cell):
            if neighbor in distances:
                continue
            step_m = great_circle_distance_m(origin, grid.centroid(neighbor), grid.radius_m)
            weight = step_m * 0.5 * (costs[cell] + costs[neighbor])
            heapq.heappush(heap, (found + weight, neighbor))
    return distances
