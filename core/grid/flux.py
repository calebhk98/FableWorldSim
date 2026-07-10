"""Edge-weighted flux exchange between cells: the no-fixed-degree rule.

Flows between cells (heat, moisture, migration) are fluxes across the
shared edge, weighted by ``edge_length_m`` and divided by ``area_m2`` —
never per-cell constants.  The code iterates ``neighbors(cell)`` so it
is correct at H3/ISEA3H pentagons (5 edges), hexes (6), and S2 quads (4)
alike; hardcoding a degree would silently break the backend toggle.

Because each edge's flux enters both cells with opposite signs, the
area-weighted global total is conserved exactly (up to float error) on
any backend — the cross-backend conservation test pins this down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.grid.geodesy import great_circle_distance_m

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports.grid import CellId, Grid


def diffusion_step(
    values: Mapping[CellId, float],
    grid: Grid,
    diffusivity_m2_s: float,
    dt_s: float,
) -> dict[CellId, float]:
    """Return ``values`` after one explicit diffusion step.

    ``values`` is an intensive field (per-area or per-volume quantity)
    covering every cell.  The flux across each shared edge is
    ``D * (v_neighbor - v_cell) / centroid_distance * edge_length``; a
    cell's change is the edge-flux sum times ``dt`` over its own area.
    Callers must pick ``dt_s`` small enough for explicit stability.
    """
    if diffusivity_m2_s < 0:
        msg = f"diffusivity must be >= 0, got {diffusivity_m2_s}"
        raise ValueError(msg)
    if dt_s <= 0:
        msg = f"dt_s must be > 0, got {dt_s}"
        raise ValueError(msg)
    next_values: dict[CellId, float] = {}
    for cell, value in values.items():
        exchange = 0.0
        for neighbor in grid.neighbors(cell):
            distance = great_circle_distance_m(
                grid.centroid(cell), grid.centroid(neighbor), grid.radius_m
            )
            gradient = (values[neighbor] - value) / distance
            exchange += diffusivity_m2_s * gradient * grid.edge_length_m(cell, neighbor)
        next_values[cell] = value + exchange * dt_s / grid.area_m2(cell)
    return next_values
