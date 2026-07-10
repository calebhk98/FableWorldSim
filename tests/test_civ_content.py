"""The shipped civilization content loads, validates, and stays data-driven."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.content_toml import TomlContentRegistry
from core.civilization.resources import MODES, load_resources
from core.civilization.species import (
    SUBSURFACE_DOMAIN,
    load_species,
    sapient_species,
    species_name_keys,
)
from core.civilization.tech import (
    TechDefinition,
    load_techs,
    tech_dag,
    validate_tech_dag,
)

_REPO_CONTENT = Path(__file__).resolve().parents[1] / "content"

_MIN_SAPIENT_RACES = 2


def _registry() -> TomlContentRegistry:
    return TomlContentRegistry([("base", _REPO_CONTENT)])


def test_shipped_species_include_multiple_sapient_races_and_a_subsurface_one() -> None:
    species = load_species(_registry())
    sapient = sapient_species(species)
    assert len(sapient) >= _MIN_SAPIENT_RACES
    domains = {spec.domain for spec in sapient}
    assert SUBSURFACE_DOMAIN in domains


def test_species_name_keys_follow_locale_convention() -> None:
    species = load_species(_registry())
    keys = species_name_keys(species)
    for species_id, key in keys.items():
        assert key == f"species-{species_id}"


def test_shipped_tech_dag_loads_and_validates() -> None:
    techs = load_techs(_registry())
    assert techs
    ids = {tech.tech_id for tech in techs}
    for tech in techs:
        assert set(tech.prerequisites) <= ids


def test_tech_dag_export_matches_node_ids() -> None:
    techs = load_techs(_registry())
    dag = tech_dag(techs)
    node_ids = {str(node["id"]) for node in dag["nodes"]}
    assert node_ids == {tech.tech_id for tech in techs}
    for edge in dag["edges"]:
        assert str(edge["source"]) in node_ids
        assert str(edge["target"]) in node_ids
    assert dag["edges"], "shipped techs should form a real DAG, not a flat list"


def test_some_shipped_tech_is_resource_gated_by_a_real_resource() -> None:
    techs = load_techs(_registry())
    resources = load_resources(_registry())
    resource_ids = {res.resource_id for res in resources}
    gated = [tech for tech in techs if tech.requires_resources]
    assert gated, "the design requires environmentally gated techs"
    for tech in gated:
        assert set(tech.requires_resources) <= resource_ids


def test_shipped_resources_cover_all_three_modes() -> None:
    resources = load_resources(_registry())
    assert {res.mode for res in resources} == set(MODES)
    assert any(res.feeds_population for res in resources)
    assert any(res.arms_value > 0 for res in resources)
    assert any(res.deposit_stock > 0 for res in resources)


def test_tech_cycle_is_rejected() -> None:
    loop = (
        TechDefinition(tech_id="a", cost=1.0, prerequisites=("b",)),
        TechDefinition(tech_id="b", cost=1.0, prerequisites=("a",)),
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_tech_dag(loop)


def test_unknown_prerequisite_is_rejected() -> None:
    orphan = (TechDefinition(tech_id="a", cost=1.0, prerequisites=("ghost",)),)
    with pytest.raises(ValueError, match="unknown tech"):
        validate_tech_dag(orphan)
