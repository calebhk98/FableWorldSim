"""Influence fields, territory assignment, and frontier detection.

Borders are a field, not a polygon: every settlement emits influence
proportional to its power, decaying exponentially with cost-weighted
distance (the same distance-plus-terrain-cost model supply projection
uses).  A cell belongs to the civ with the highest influence there, and the
frontier is wherever a rival's influence is comparable to the holder's.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.civilization.terrain import cost_distances

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports.grid import CellId, Grid

_CUTOFF_RANGES = 12.0


def decay(distance_cost_m: float, range_m: float) -> float:
    """Return the exponential falloff factor at a cost-weighted distance."""
    return math.exp(-distance_cost_m / range_m)


def influence_field(
    grid: Grid,
    costs: Mapping[CellId, float],
    emitters: Mapping[CellId, float],
    range_m: float,
) -> dict[CellId, float]:
    """Sum every emitter's decayed influence over the grid.

    ``emitters`` maps a settlement's cell to its power.  Each emitter gets
    its own Dijkstra pass, cut off where its contribution becomes
    negligible, so cost stays proportional to actual reach.
    """
    total: dict[CellId, float] = {}
    cutoff = range_m * _CUTOFF_RANGES
    for source, power in emitters.items():
        for cell, distance in cost_distances(grid, costs, (source,), cutoff).items():
            total[cell] = total.get(cell, 0.0) + power * decay(distance, range_m)
    return total


def assign_territory(
    fields: Mapping[str, Mapping[CellId, float]],
    floor: float,
) -> dict[CellId, str]:
    """Assign each cell to the civ with the highest influence above the floor."""
    best: dict[CellId, tuple[float, str]] = {}
    for civ_id, field in fields.items():
        for cell, value in field.items():
            if value >= floor and (cell not in best or value > best[cell][0]):
                best[cell] = (value, civ_id)
    return {cell: civ_id for cell, (_, civ_id) in best.items()}


def contested_cells(
    fields: Mapping[str, Mapping[CellId, float]],
    territory: Mapping[CellId, str],
    contest_ratio: float,
) -> dict[CellId, str]:
    """Return frontier cells mapped to the strongest challenger.

    A cell is contested when the best rival influence is at least
    ``contest_ratio`` of the holder's — the design's "frontiers are where
    two civs' influence is comparable".
    """
    challengers: dict[CellId, str] = {}
    for cell, holder in territory.items():
        holder_value = fields[holder].get(cell, 0.0)
        best_id: str | None = None
        best_value = 0.0
        for civ_id, field in fields.items():
            value = field.get(cell, 0.0)
            if civ_id != holder and value > best_value:
                best_id, best_value = civ_id, value
        if best_id is not None and holder_value > 0 and best_value / holder_value >= contest_ratio:
            challengers[cell] = best_id
    return challengers
