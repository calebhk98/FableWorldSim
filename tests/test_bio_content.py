"""The shipped biology roster loads, stays data-driven, and runs on a planet."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.content_toml import TomlContentRegistry
from core.biology.foodweb import food_web_graph
from core.biology.medium import AERIAL, AQUATIC, SUBTERRANEAN, TERRESTRIAL
from core.biology.organism import Organism, load_organisms

_REPO_CONTENT = Path(__file__).resolve().parents[1] / "content"
_MIN_SAPIENT = 2
_EXPECTED = frozenset(
    {"grass", "oak", "seaweed", "lion", "sparrow", "dragon", "human", "dwarf", "cave_moss"}
)


def _organisms() -> tuple[Organism, ...]:
    return load_organisms(TomlContentRegistry([("base", _REPO_CONTENT)]))


def test_roster_exercises_every_habitat_axis() -> None:
    organisms = _organisms()
    media = {org.medium for org in organisms}
    assert {TERRESTRIAL, AQUATIC, AERIAL, SUBTERRANEAN} <= media
    assert any(org.is_autotroph for org in organisms), "needs plants"
    assert any(not org.is_autotroph for org in organisms), "needs animals"
    assert sum(org.sapient for org in organisms) >= _MIN_SAPIENT


def test_expected_species_are_present() -> None:
    ids = {org.species_id for org in _organisms()}
    assert _EXPECTED <= ids


def test_every_diet_edge_points_at_a_real_species() -> None:
    organisms = _organisms()
    ids = {org.species_id for org in organisms}
    for org in organisms:
        assert set(org.diet) <= ids, f"{org.species_id} eats an unknown species"


def test_name_keys_follow_the_locale_convention() -> None:
    for org in _organisms():
        assert org.name_key == f"species-{org.species_id}"


def test_food_web_graph_matches_the_roster() -> None:
    organisms = {org.species_id: org for org in _organisms()}
    graph = food_web_graph(organisms)
    node_ids = {node["id"] for node in graph["nodes"]}
    assert node_ids == set(organisms)
    assert graph["edges"], "a roster with predators should form real food-chain edges"
    for edge in graph["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids


def test_biology_runs_on_a_real_earth_climate() -> None:
    pytest.importorskip("h3")
    from adapters.grid_registry import create_grid
    from adapters.rng_seeded import SeededRng
    from core.biology.biome import biome_field, load_biomes
    from core.biology.context import LayersBelow, build_biology_context
    from core.biology.engine import step_biology
    from core.biology.seed import seed_biosphere
    from core.biology.state import plant_biomass_field
    from core.climate.model import simulate_climate
    from core.hydrology.sea_mask import build_sea_mask
    from core.sim.constants import SECONDS_PER_YEAR
    from core.sim.presets import earth
    from core.topography.procedural import ProceduralTopography

    registry = TomlContentRegistry([("base", _REPO_CONTENT)])
    planet = earth()
    grid = create_grid("h3", resolution=1, radius_m=planet.radius_m)
    heights = dict(ProceduralTopography(SeededRng(42)).heights(grid))
    mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    climate = simulate_climate(grid, planet, heights, mask, season_count=2)
    biomes = load_biomes(registry)
    organisms = {org.species_id: org for org in load_organisms(registry)}

    below = LayersBelow(grid, climate, mask, heights, biome_field(climate, mask, biomes))
    ctx = build_biology_context(below, organisms)
    state = seed_biosphere(ctx)
    assert any(state.surface_populations.get("grass", {}).values()), "grass should find a home"

    result = step_biology(state, ctx, SeededRng(0), SECONDS_PER_YEAR)
    assert result.tick == 1
    biomass = plant_biomass_field(result, list(organisms.values()))
    assert any(value > 0.0 for value in biomass.values()), "vegetation should persist a step"
