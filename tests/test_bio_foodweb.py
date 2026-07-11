"""Food web: specific diets, competitive exclusion, and predation offtake."""

from __future__ import annotations

from core.biology.foodweb import FeedingParams, apply_offtake, feed_location, food_web_graph
from tests.bio_helpers import make_organism


def _orgs(*organisms: object) -> dict[str, object]:
    return {org.species_id: org for org in organisms}  # type: ignore[attr-defined]


def test_competitive_exclusion_favours_the_specialist() -> None:
    grass = make_organism("grass")
    specialist = make_organism("specialist", diet={"grass": 1.0})
    generalist = make_organism("generalist", diet={"grass": 0.3})
    result = feed_location(
        {"grass": 0.5, "specialist": 0.01, "generalist": 0.01},
        _orgs(grass, specialist, generalist),  # type: ignore[arg-type]
        FeedingParams(),
    )
    assert result.fed_fraction["specialist"] > result.fed_fraction["generalist"]


def test_offtake_removes_eaten_biomass() -> None:
    grass = make_organism("grass")
    herbivore = make_organism("herbivore", diet={"grass": 1.0})
    result = feed_location(
        {"grass": 2.0, "herbivore": 0.05},
        _orgs(grass, herbivore),
        FeedingParams(),  # type: ignore[arg-type]
    )
    assert result.offtake["grass"] > 0.0
    assert apply_offtake(2.0, grass, result.offtake["grass"]) < 2.0


def test_clear_cutting_starves_the_dependent_herbivore() -> None:
    grass = make_organism("grass")
    herbivore = make_organism("herbivore", diet={"grass": 1.0}, body_mass_kg=50.0)
    orgs = _orgs(grass, herbivore)
    lush = feed_location({"grass": 5.0, "herbivore": 0.01}, orgs, FeedingParams())  # type: ignore[arg-type]
    cleared = feed_location({"grass": 0.05, "herbivore": 0.01}, orgs, FeedingParams())  # type: ignore[arg-type]
    assert cleared.fed_fraction["herbivore"] < lush.fed_fraction["herbivore"]


def test_food_web_graph_carries_trophic_nodes_and_diet_edges() -> None:
    grass = make_organism("grass")
    herbivore = make_organism("herbivore", diet={"grass": 1.0})
    graph = food_web_graph(_orgs(grass, herbivore))  # type: ignore[arg-type]
    trophic = {node["id"]: node["trophic"] for node in graph["nodes"]}
    assert trophic == {"grass": "autotroph", "herbivore": "consumer"}
    assert {"source": "herbivore", "target": "grass", "weight": 1.0} in graph["edges"]
