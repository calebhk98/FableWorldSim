"""Frontier contests: relative projected strength with stochasticity.

A civ's raw contest strength is population it can commit x a tech multiplier
x a resource (arms) multiplier — a stat built up over time, not a free
constant.  What it can bring to a *given* cell is that raw strength scaled by
a projection factor decaying with cost-weighted distance from the nearest
supplying settlement, so Rome cannot conquer China: its projected capacity
there is ~0.  Contested cells flip stochastically by relative projected
strength, never by coin flip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from core.civilization.resources import ResourceDefinition
    from core.civilization.tech import TechEffects
    from ports.grid import CellId
    from ports.rng import Rng

_MAX_ARMS_BONUS = 4.0


@dataclass(frozen=True)
class CivStrength:
    """Precomputed contest inputs for one civilization."""

    civ_id: str
    raw_strength: float
    supply_distance_m: Mapping[CellId, float]
    supply_range_m: float

    def projected(self, cell: CellId) -> float:
        """Return the strength the civ can actually bring to a cell."""
        distance = self.supply_distance_m.get(cell, math.inf)
        return self.raw_strength * math.exp(-distance / self.supply_range_m)


def arms_multiplier(
    stockpiles: Mapping[str, float],
    resources: Sequence[ResourceDefinition],
    arms_scale: float,
) -> float:
    """Return the resource multiplier drawn from what the civ has invested."""
    invested = sum(
        stockpiles.get(resource.resource_id, 0.0) * resource.arms_value for resource in resources
    )
    return 1.0 + min(_MAX_ARMS_BONUS, invested * arms_scale)


def raw_strength(
    population_total: float,
    commit_fraction: float,
    effects: TechEffects,
    arms: float,
) -> float:
    """Return committed population x tech multiplier x resource multiplier."""
    return population_total * commit_fraction * effects.strength * arms


def resolve_contests(
    rng: Rng,
    contested: Mapping[CellId, str],
    territory: Mapping[CellId, str],
    strengths: Mapping[str, CivStrength],
) -> tuple[dict[CellId, str], tuple[tuple[CellId, str, str], ...]]:
    """Resolve every contested cell and return (territory, flips).

    Each flip records ``(cell, loser, winner)``.  Cells are visited in
    sorted order so outcomes are deterministic for a given RNG stream.
    """
    resolved = dict(territory)
    flips: list[tuple[CellId, str, str]] = []
    for cell in sorted(contested):
        holder = territory[cell]
        challenger = contested[cell]
        attack = strengths[challenger].projected(cell)
        defence = strengths[holder].projected(cell)
        total = attack + defence
        if total <= 0.0:
            continue
        if rng.random() < attack / total:
            resolved[cell] = challenger
            flips.append((cell, holder, challenger))
    return resolved, tuple(flips)
