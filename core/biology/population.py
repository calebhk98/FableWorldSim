"""Population update: feeding-gated logistic growth toward carrying capacity.

Densities glide toward a per-cell carrying capacity ``K = crowding cap x
suitability`` — so even with infinite food a large-bodied species stays
sparse, and a hostile cell caps everything.  Two update maps toggle the
predator-prey regime: the **smoothed** default (a monotone Beverton-Holt
logistic) glides to K without overshoot, while the **oscillatory** toggle
(the overcompensating Ricker map) produces the lag-driven boom/bust of the
snowshoe-hare/lynx cycle.  How well a consumer fed scales its growth, so
starvation turns growth negative and drives decline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.biology.demography import intrinsic_growth_rate

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.biology.organism import Organism

_STARVATION_PIVOT = 0.5
"""Fed fraction at which growth is neutral; below it a consumer declines."""


@dataclass(frozen=True)
class GrowthParams:
    """Timing knobs for one growth step."""

    dt_years: float
    no_habitat_decay_per_year: float


def env_capacity(
    organism: Organism,
    suitability: Mapping[str, float],
) -> dict[str, float]:
    """Return per-location carrying capacity: crowding cap scaled by suitability."""
    return {loc: organism.crowding_cap_per_m2 * score for loc, score in suitability.items()}


def logistic_step(value: float, capacity: float, growth_rt: float, oscillatory: bool) -> float:
    """Return the next population value under the chosen growth map.

    ``growth_rt`` is the effective per-step growth exponent (rate x dt,
    already scaled by feeding).  ``capacity`` must be positive.
    """
    if oscillatory:
        return value * math.exp(growth_rt * (1.0 - value / capacity))
    factor = math.exp(growth_rt)
    denominator = capacity + value * (factor - 1.0)
    if denominator <= 0.0:
        return 0.0
    return capacity * value * factor / denominator


def _effective_rt(
    organism: Organism,
    base_rate: float,
    fed_fraction: float,
    dt_years: float,
) -> float:
    """Return the feeding-scaled growth exponent for one step.

    Autotrophs ignore feeding (the environment already gates them); a
    consumer fed below the pivot gets a negative rate — starvation.
    """
    if organism.is_autotroph:
        return base_rate * dt_years
    feeding_scale = (fed_fraction - _STARVATION_PIVOT) / _STARVATION_PIVOT
    return base_rate * feeding_scale * dt_years


def grow(
    field: Mapping[str, float],
    capacity: Mapping[str, float],
    organism: Organism,
    fed_fraction: Mapping[str, float],
    params: GrowthParams,
) -> dict[str, float]:
    """Grow one species' density field one step across every occupied cell."""
    base_rate = intrinsic_growth_rate(organism)
    grown: dict[str, float] = {}
    for loc, value in field.items():
        if value <= 0.0:
            grown[loc] = 0.0
            continue
        cap = capacity.get(loc, 0.0)
        if cap <= 0.0:
            grown[loc] = value * max(0.0, 1.0 - params.no_habitat_decay_per_year * params.dt_years)
            continue
        rt = _effective_rt(organism, base_rate, fed_fraction.get(loc, 1.0), params.dt_years)
        grown[loc] = max(0.0, logistic_step(value, cap, rt, organism.oscillatory))
    return grown
