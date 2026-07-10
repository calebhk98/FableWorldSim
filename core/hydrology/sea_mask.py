"""Hydrology part A: sea level, land/ocean mask, and the intertidal band.

Runs *before* climate (M1.2): ocean current routing and surface albedo
both need the land mask, while nothing here needs climate.  Rivers and
lakes (part B) run after climate because they consume precipitation.

The intertidal band — a distinct habitat, not just "coast" — derives
from the planet's tidal range: cells whose elevation sits within half
the tidal range of sea level flood and drain with the tide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.ocean.sea_level import solve_sea_level

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports.grid import CellId, Grid


@dataclass(frozen=True)
class SeaMask:
    """Solved sea level and the per-cell water classification."""

    sea_level_m: float
    ocean: Mapping[CellId, bool]
    intertidal: Mapping[CellId, bool]

    def is_land(self, cell: CellId) -> bool:
        """Return whether a cell is dry land (above sea level)."""
        return not self.ocean[cell]


def build_sea_mask(
    grid: Grid,
    heights_m: Mapping[CellId, float],
    ocean_fraction: float,
    tidal_range_m: float = 0.0,
) -> SeaMask:
    """Reconcile the target water fraction with the height field.

    Solves the area-weighted sea-level threshold, classifies every cell
    as ocean (elevation <= sea level) or land, and marks the intertidal
    band around the shoreline when the planet has tides.
    """
    cells = list(heights_m)
    areas = [grid.area_m2(cell) for cell in cells]
    sea_level = solve_sea_level([heights_m[c] for c in cells], areas, ocean_fraction)
    ocean = {cell: heights_m[cell] <= sea_level for cell in cells}
    half_range = tidal_range_m / 2.0
    intertidal = {
        cell: half_range > 0.0 and abs(heights_m[cell] - sea_level) <= half_range for cell in cells
    }
    return SeaMask(sea_level_m=sea_level, ocean=ocean, intertidal=intertidal)
