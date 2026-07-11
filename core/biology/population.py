"""Population update: feeding-gated logistic growth toward carrying capacity.

Densities glide toward a per-cell carrying capacity ``K = min(crowding cap x
suitability, food-limited cap)`` — so even with infinite food a large-bodied
species stays sparse, a hostile cell caps everything, *and* a consumer with
no reachable prey or plant biomass has nowhere to sustain a population at
all, however good the habitat otherwise looks.  Two update maps toggle the
predator-prey regime: the **smoothed** default (a monotone Beverton-Holt
logistic) glides to K without overshoot, while the **oscillatory** toggle
(the overcompensating Ricker map) produces the lag-driven boom/bust of the
snowshoe-hare/lynx cycle.

Food gates growth twice, deliberately, at two different timescales rather
than double-counting one signal: :func:`env_capacity`'s food term is a
**stock** question ("how large a population can this reachable biomass
support at equilibrium?"), independent of how many consumers are currently
present, while :func:`_effective_rt`'s feeding scale is a **flow** question
("did *this* population's actual demand get met *this* tick?").  The two
share the same intake/offtake constants so they agree at equilibrium — a
population sitting at its food-limited K is, by construction, drawing
close to the sustainable offtake share and so is fed near the starvation
pivot — instead of one silently amplifying the other.
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


@dataclass(frozen=True)
class FoodCapacityParams:
    """Tuning for the food-limited term of carrying capacity.

    Mirrors :class:`core.biology.foodweb.FeedingParams` deliberately: using
    the same per-capita intake and sustainable-offtake constants for both
    "how well is this population fed right now" and "how large a
    population can this food support" keeps the two mechanisms consistent
    with each other instead of one silently double-penalizing scarce food.
    """

    intake_kg_per_kg_body_year: float = 8.0
    max_offtake_fraction: float = 0.5


def env_capacity(
    organism: Organism,
    suitability: Mapping[str, float],
    food_biomass_per_m2: Mapping[str, float] | None = None,
    food_params: FoodCapacityParams | None = None,
) -> dict[str, float]:
    """Return per-location ``K = min(crowding cap x suitability, food cap)``.

    The habitat term (crowding cap scaled by suitability) is unchanged from
    before; the new food term converts diet-reachable biomass
    (:func:`core.biology.foodweb.reachable_food_biomass`) into the
    population density a sustainable offtake share of it can feed, so a
    consumer's ceiling can never exceed what its food supply can carry even
    when crowding and habitat quality would allow more.  Autotrophs have no
    diet and so no food term — their own suitability already gates them.
    ``food_biomass_per_m2`` and ``food_params`` default to "no known food",
    which only matters for heterotrophs a caller chooses not to supply
    food for; real callers (see :mod:`core.biology.domain`) always pass both.
    """
    habitat_cap = {loc: organism.crowding_cap_per_m2 * score for loc, score in suitability.items()}
    if organism.is_autotroph:
        return habitat_cap
    food_biomass = food_biomass_per_m2 if food_biomass_per_m2 is not None else {}
    params = food_params if food_params is not None else FoodCapacityParams()
    demand_per_head = organism.body_mass_kg * params.intake_kg_per_kg_body_year
    if demand_per_head <= 0.0:
        return habitat_cap
    capacity: dict[str, float] = {}
    for loc, cap in habitat_cap.items():
        sustainable_supply = params.max_offtake_fraction * food_biomass.get(loc, 0.0)
        capacity[loc] = min(cap, sustainable_supply / demand_per_head)
    return capacity


def logistic_step(value: float, capacity: float, growth_rt: float, oscillatory: bool) -> float:
    """Return the next population value under the chosen growth map.

    ``growth_rt`` is the effective per-step growth exponent (rate x dt,
    already scaled by feeding).  ``capacity`` must be positive.

    The Ricker overcompensation term assumes ``growth_rt`` is the rate a
    population *would* grow at below capacity; its sign is what makes
    ``1 - value / capacity`` correctly flip growth into decline once
    ``value`` clears ``capacity``.  With a food-limited capacity
    (:func:`env_capacity`) that ceiling can now itself collapse toward zero
    in a single tick (its prey crashed), leaving a population that was
    fine last tick enormously above a near-zero capacity *while already
    starving* (``growth_rt < 0``).  Feeding a already-negative rate into
    that same term flips its sign again, turning a starving, over-capacity
    population's overcompensation term into runaway growth instead of
    collapse — and can overflow ``exp`` outright.  A starving population
    has nothing left to overcompensate toward: plain per-tick decay at
    ``growth_rt`` is what the "no habitat" branch below already does for
    an analogous degenerate case, so mirror it here instead of trusting
    the sign-sensitive crowding term outside its valid range.
    """
    if oscillatory:
        if growth_rt < 0.0 and value > capacity:
            return value * math.exp(growth_rt)
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
