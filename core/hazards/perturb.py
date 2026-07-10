"""Disturbances: two triggers, one field-perturbation mechanism.

Scripted events (a volcano's ash cloud, a meteor strike, a climate
nudge) and emergent hazards (wildfire, later plate volcanism) both
route through the same perturbation code — the only difference is *who
fires it*.  M1 ships this mechanism as test scaffolding plus the
scripted vocabulary; emergent wildfire lands with the biology layer
(it needs a fuel field).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.grid.geodesy import great_circle_distance_m

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports.grid import CellId, Grid


@dataclass(frozen=True)
class Disturbance:
    """One perturbation of world fields, whoever triggered it.

    ``field_deltas`` maps field name -> per-cell additive delta.
    ``trigger`` records the source ("scripted:<command>" or
    "emergent:<hazard>") for the chronicle.
    """

    kind: str
    trigger: str
    field_deltas: Mapping[str, Mapping[CellId, float]] = field(default_factory=dict)


def apply_disturbance(
    fields: Mapping[str, Mapping[CellId, float]],
    disturbance: Disturbance,
) -> dict[str, dict[CellId, float]]:
    """Return fields with the disturbance's deltas applied.

    Unknown field names raise (a disturbance may never invent state);
    untouched fields pass through unchanged.
    """
    unknown = set(disturbance.field_deltas) - set(fields)
    if unknown:
        msg = f"disturbance {disturbance.kind!r} targets unknown fields: {sorted(unknown)}"
        raise KeyError(msg)
    result = {name: dict(values) for name, values in fields.items()}
    for name, deltas in disturbance.field_deltas.items():
        target = result[name]
        for cell, delta in deltas.items():
            if cell not in target:
                msg = f"disturbance {disturbance.kind!r} targets unknown cell {cell!r}"
                raise KeyError(msg)
            target[cell] += delta
    return result


def radial_deltas(
    grid: Grid,
    center: CellId,
    radius_m: float,
    peak_delta: float,
) -> dict[CellId, float]:
    """Return Gaussian-falloff deltas around a center cell.

    The shared footprint builder for area effects (ash clouds, impact
    craters, epidemics): full ``peak_delta`` at the center falling to
    ~1% at ``radius_m``, walked over the neighbor graph so it works on
    any backend.
    """
    if radius_m <= 0:
        msg = f"radius_m must be > 0, got {radius_m}"
        raise ValueError(msg)
    origin = grid.centroid(center)
    deltas: dict[CellId, float] = {}
    frontier = [center]
    seen = {center}
    while frontier:
        cell = frontier.pop()
        distance = great_circle_distance_m(origin, grid.centroid(cell), grid.radius_m)
        if distance > radius_m:
            continue
        falloff = math.exp(-((3.0 * distance / radius_m) ** 2) / 2.0)
        deltas[cell] = peak_delta * falloff
        for neighbor in grid.neighbors(cell):
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return deltas


def ash_cloud(
    grid: Grid,
    center: CellId,
    radius_m: float,
    cooling_k: float,
) -> Disturbance:
    """Return a scripted volcanic-ash-cloud disturbance (regional cooling)."""
    if cooling_k <= 0:
        msg = f"cooling_k must be > 0, got {cooling_k}"
        raise ValueError(msg)
    return Disturbance(
        kind="ash_cloud",
        trigger="scripted:ash_cloud",
        field_deltas={"temperature_k": radial_deltas(grid, center, radius_m, -cooling_k)},
    )
