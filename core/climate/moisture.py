"""Moisture: evaporation, downwind transport, orographic rain and shadow.

Evaporation over ice-free water supplies humidity; winds carry it
cell-to-cell over the neighbor graph; rain falls from convergence and
from **orographic lift** (ascending toward higher terrain rains on the
windward slope and leaves the lee side in shadow).  Tall ranges deplete
crossing moisture almost entirely — the Himalaya/Tibet barrier effect —
because shadowing is emergent from depletion, not scripted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.climate.temperature import FREEZING_POINT_K
from core.grid.vectors import unit_direction

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.grid.vectors import Vector2
    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid

_EVAPORATION_FULL_K = 303.0
_EVAPORATION_ZERO_K = 243.0
_LAND_EVAPORATION_FACTOR = 0.15
_BASE_RAINOUT_FRACTION = 0.25
_OROGRAPHIC_FULL_UPLIFT_M = 1_500.0
_MAX_RAINOUT_FRACTION = 0.95
_TRANSPORT_HOPS = 5
_MM_PER_HUMIDITY_UNIT = 1_200.0
_CALM_WIND_M_S = 1e-9


def evaporation_field(
    temps: Mapping[CellId, float],
    sea_mask: SeaMask,
    ice: Mapping[CellId, bool],
) -> dict[CellId, float]:
    """Return per-cell humidity sourced by evaporation (dimensionless).

    Warm open water evaporates fully; frozen or cold surfaces barely at
    all; land contributes a small soil/vegetation term.
    """
    humidity = {}
    for cell, temp in temps.items():
        warmth = (temp - _EVAPORATION_ZERO_K) / (_EVAPORATION_FULL_K - _EVAPORATION_ZERO_K)
        strength = min(1.0, max(0.0, warmth)) ** 2
        if ice[cell]:
            strength *= 0.05
        elif not sea_mask.ocean[cell]:
            strength *= _LAND_EVAPORATION_FACTOR
        humidity[cell] = strength
    return humidity


def downwind_neighbor(
    grid: Grid,
    cell: CellId,
    wind: Vector2,
) -> CellId | None:
    """Return the neighbor best aligned with the wind, or None if calm."""
    if math.hypot(wind[0], wind[1]) < _CALM_WIND_M_S:
        return None
    origin = grid.centroid(cell)
    best: tuple[float, CellId] | None = None
    for neighbor in grid.neighbors(cell):
        direction = unit_direction(origin, grid.centroid(neighbor), grid.radius_m)
        alignment = direction[0] * wind[0] + direction[1] * wind[1]
        if best is None or alignment > best[0]:
            best = (alignment, neighbor)
    if best is None or best[0] <= 0.0:
        return None
    return best[1]


def _rainout_fraction(origin_height: float, target_height: float, origin_is_ocean: bool) -> float:
    """Return the fraction of moving moisture that rains this hop."""
    uplift = max(0.0, target_height - origin_height)
    orographic = 0.0 if origin_is_ocean else uplift / _OROGRAPHIC_FULL_UPLIFT_M
    return min(_MAX_RAINOUT_FRACTION, _BASE_RAINOUT_FRACTION + orographic)


@dataclass(frozen=True)
class MoistureInputs:
    """Everything the moisture cycle consumes for one season."""

    temps: Mapping[CellId, float]
    winds: Mapping[CellId, Vector2]
    heights_m: Mapping[CellId, float]
    sea_mask: SeaMask
    ice: Mapping[CellId, bool]


def precipitation_field(
    grid: Grid,
    inputs: MoistureInputs,
    hops: int = _TRANSPORT_HOPS,
) -> dict[CellId, float]:
    """Return per-cell precipitation in mm/year for one season.

    Humidity advects downwind for a few hops; each hop rains out a base
    convergence fraction plus an orographic term where the parcel is
    forced upslope, so windward slopes are wet and lee sides dry.
    Whatever survives the journey falls as drizzle where it ends up.
    """
    humidity = evaporation_field(inputs.temps, inputs.sea_mask, inputs.ice)
    rain = dict.fromkeys(humidity, 0.0)
    for _ in range(hops):
        moved = dict.fromkeys(humidity, 0.0)
        for cell, load in humidity.items():
            if load <= 0.0:
                continue
            target = downwind_neighbor(grid, cell, inputs.winds[cell])
            if target is None:
                rain[cell] += load
                continue
            fraction = _rainout_fraction(
                inputs.heights_m[cell],
                inputs.heights_m[target],
                inputs.sea_mask.ocean[cell],
            )
            rain[cell] += load * fraction
            moved[target] += load * (1.0 - fraction)
        humidity = moved
    for cell, leftover in humidity.items():
        rain[cell] += leftover
    return {
        cell: amount * _MM_PER_HUMIDITY_UNIT * _cold_air_factor(inputs.temps[cell])
        for cell, amount in rain.items()
    }


def _cold_air_factor(temp_k: float) -> float:
    """Return how much water cold air can actually deliver (0.05..1)."""
    span = 40.0
    factor = (temp_k - (FREEZING_POINT_K - span)) / span
    return min(1.0, max(0.05, factor))
