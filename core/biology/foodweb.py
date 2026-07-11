"""The food web: specific diets, competition, and predation offtake.

Diet is a set of **specific** food edges with preference weights — a horse
grazes grass, not bamboo — so a species can starve amid food it cannot
eat.  When several consumers draw on one food pool the pool is split
**proportionally to demand times competitive ability** (population x body
size x preference), which yields **competitive exclusion** for free: a
much stronger competitor takes the lion's share and starves the weaker one
out of that cell.  What each consumer eats is removed from its food — so
clear-cutting a forest starves the deer, and predators crash their prey —
and how well it fed scales its growth next.

The web is geometry-free: :func:`feed_location` resolves one cell (or one
subsurface node) from the populations present there.  Emit the whole web
as a graph (:func:`food_web_graph`) for the client's food-chain chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.biology.organism import Organism


@dataclass(frozen=True)
class FeedingParams:
    """Tuning for one feeding pass (bundled so callers pass one value)."""

    intake_kg_per_kg_body_year: float = 8.0
    max_offtake_fraction: float = 0.5
    dt_years: float = 1.0


@dataclass(frozen=True)
class FeedingResult:
    """Outcome of feeding one location.

    ``fed_fraction`` maps each consumer to how much of its food demand was
    met (0..1); ``offtake`` maps each eaten species to the population units
    removed from it (biomass kg/m2 for plants, count/m2 for animals).
    """

    fed_fraction: Mapping[str, float]
    offtake: Mapping[str, float]


def _biomass(value: float, organism: Organism) -> float:
    """Return a species' standing biomass (kg/m2) from its population value."""
    if organism.is_autotroph:
        return value
    return value * organism.body_mass_kg


def _need(value: float, organism: Organism, params: FeedingParams) -> float:
    """Return a consumer's total food demand this step, in kg/m2."""
    intake_per_individual = organism.body_mass_kg * params.intake_kg_per_kg_body_year
    return intake_per_individual * value * params.dt_years


def feed_location(
    populations: Mapping[str, float],
    organisms: Mapping[str, Organism],
    params: FeedingParams,
) -> FeedingResult:
    """Resolve feeding, competition, and offtake at one location."""
    present = {sid: value for sid, value in populations.items() if value > 0.0}
    consumers = [sid for sid in present if organisms[sid].diet]
    demand = _Demand(
        consumers=consumers,
        need={sid: _need(present[sid], organisms[sid], params) for sid in consumers},
        organisms=organisms,
        present=present,
    )
    grants = {sid: 0.0 for sid in consumers}
    offtake: dict[str, float] = {}
    for food_id in _eaten_foods(consumers, organisms, present):
        taken = _split_food_pool(food_id, demand, params, grants)
        if taken > 0.0:
            offtake[food_id] = taken
    fed = {
        sid: min(1.0, grants[sid] / demand.need[sid]) if demand.need[sid] > 0 else 1.0
        for sid in consumers
    }
    return FeedingResult(fed_fraction=fed, offtake=offtake)


def _eaten_foods(
    consumers: list[str],
    organisms: Mapping[str, Organism],
    present: Mapping[str, float],
) -> set[str]:
    """Return every food id at least one present consumer can eat here."""
    return {food_id for sid in consumers for food_id in organisms[sid].diet if food_id in present}


@dataclass(frozen=True)
class _Demand:
    """One location's consumers and everything their allocation needs."""

    consumers: list[str]
    need: Mapping[str, float]
    organisms: Mapping[str, Organism]
    present: Mapping[str, float]


def _split_food_pool(
    food_id: str,
    demand: _Demand,
    params: FeedingParams,
    grants: dict[str, float],
) -> float:
    """Divide one food pool by competitive weight; return the total eaten."""
    organisms, present = demand.organisms, demand.present
    weights = {
        sid: organisms[sid].diet[food_id] * present[sid] * organisms[sid].body_mass_kg
        for sid in demand.consumers
        if food_id in organisms[sid].diet
    }
    total_weight = sum(weights.values())
    supply = params.max_offtake_fraction * _biomass(present[food_id], organisms[food_id])
    if total_weight <= 0.0 or supply <= 0.0:
        return 0.0
    taken = 0.0
    for sid, weight in weights.items():
        want = organisms[sid].diet[food_id] * demand.need[sid]
        granted = min(weight / total_weight * supply, want)
        grants[sid] += granted
        taken += granted
    return taken


def reachable_food_biomass(
    organism: Organism,
    populations: Mapping[str, Mapping[str, float]],
    organisms: Mapping[str, Organism],
) -> dict[str, float]:
    """Return each location's diet-weighted biomass reachable by a consumer.

    Sums, per location, ``preference x standing biomass`` over every food
    species in ``organism``'s diet that is present there — the same
    preference weighting :func:`feed_location` uses to split a shared food
    pool, now feeding the food-limited term of carrying capacity
    (:func:`core.biology.population.env_capacity`) instead of one tick's
    offtake.  An autotroph's diet is always empty, so this is always empty
    for it too — its own suitability alone gates its capacity.
    """
    reachable: dict[str, float] = {}
    for food_id, preference in organism.diet.items():
        food_field = populations.get(food_id)
        if not food_field:
            continue
        food_org = organisms[food_id]
        for loc, value in food_field.items():
            if value <= 0.0:
                continue
            reachable[loc] = reachable.get(loc, 0.0) + preference * _biomass(value, food_org)
    return reachable


def apply_offtake(
    value: float,
    organism: Organism,
    removed_biomass_kg_m2: float,
) -> float:
    """Return a prey population reduced by the biomass eaten from it."""
    if organism.is_autotroph:
        return max(0.0, value - removed_biomass_kg_m2)
    return max(0.0, value - removed_biomass_kg_m2 / organism.body_mass_kg)


def food_web_graph(organisms: Mapping[str, Organism]) -> dict[str, list[dict[str, object]]]:
    """Emit the food web as a ``GraphLayout``-ready node/edge graph.

    Nodes carry the trophic role (autotroph vs consumer); edges point from
    consumer to each food with the preference weight, so the client can lay
    out a food-chain chart with the same elkjs pipeline the tech DAG uses.
    """
    nodes: list[dict[str, object]] = [
        {
            "id": sid,
            "name_key": org.name_key,
            "trophic": "autotroph" if org.is_autotroph else "consumer",
        }
        for sid, org in sorted(organisms.items())
    ]
    edges: list[dict[str, object]] = [
        {"source": sid, "target": food_id, "weight": weight}
        for sid, org in sorted(organisms.items())
        for food_id, weight in sorted(org.diet.items())
    ]
    return {"nodes": nodes, "edges": edges}
