"""Sea-level solver: reconcile a target ocean fraction with a height field.

The planet's water budget is a first-class dial ("90% ocean" through
bone-dry).  Rather than picking a sea level and hoping, we solve for the
elevation threshold whose flooded area matches the target fraction —
area-weighted, so the answer is invariant to the grid backend.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _check_cells(elevations_m: Sequence[float], areas_m2: Sequence[float]) -> None:
    """Validate that elevations and areas describe the same cells."""
    if len(elevations_m) != len(areas_m2):
        msg = f"got {len(elevations_m)} elevations but {len(areas_m2)} areas"
        raise ValueError(msg)
    if not elevations_m:
        msg = "need at least one cell"
        raise ValueError(msg)
    if any(area <= 0 for area in areas_m2):
        msg = "every cell area must be > 0"
        raise ValueError(msg)


def solve_sea_level(
    elevations_m: Sequence[float],
    areas_m2: Sequence[float],
    ocean_fraction: float,
) -> float:
    """Return the sea level matching a target ocean fraction.

    A cell is ocean when its elevation is <= the returned sea level.
    Returns ``-inf`` for a bone-dry world (fraction 0).  Cells sharing
    an elevation flood together, so with a discrete height field the
    achieved fraction is the smallest reachable value >= the target.
    """
    _check_cells(elevations_m, areas_m2)
    if not 0.0 <= ocean_fraction <= 1.0:
        msg = f"ocean fraction must be in [0, 1], got {ocean_fraction}"
        raise ValueError(msg)
    if ocean_fraction == 0.0:
        return -math.inf

    area_at_elevation: dict[float, float] = {}
    for elevation, area in zip(elevations_m, areas_m2, strict=True):
        area_at_elevation[elevation] = area_at_elevation.get(elevation, 0.0) + area

    total_area = sum(areas_m2)
    levels = sorted(area_at_elevation)
    flooded = 0.0
    for elevation in levels:
        flooded += area_at_elevation[elevation]
        if flooded / total_area >= ocean_fraction:
            return elevation
    return levels[-1]


def ocean_fraction_at(
    elevations_m: Sequence[float],
    areas_m2: Sequence[float],
    sea_level_m: float,
) -> float:
    """Return the area fraction flooded at a given sea level."""
    _check_cells(elevations_m, areas_m2)
    total_area = sum(areas_m2)
    flooded = sum(
        area
        for elevation, area in zip(elevations_m, areas_m2, strict=True)
        if elevation <= sea_level_m
    )
    return flooded / total_area
