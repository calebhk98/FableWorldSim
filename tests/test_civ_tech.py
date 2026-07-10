"""Tech effects compose, gating works, and research choice follows propensity."""

from __future__ import annotations

from adapters.rng_seeded import SeededRng
from core.civilization.tech import (
    TechDefinition,
    TechEffects,
    available_techs,
    choose_research,
    combined_effects,
)

_A = TechDefinition(tech_id="alpha", cost=5.0, effects=TechEffects(water_cost=0.25))
_B = TechDefinition(
    tech_id="beta",
    cost=10.0,
    prerequisites=("alpha",),
    effects=TechEffects(water_cost=0.4, strength=2.0),
)
_GATED = TechDefinition(tech_id="gated", cost=5.0, requires_resources=("ore",))
_TECHS = (_A, _B, _GATED)


def test_combined_effects_multiply_across_unlocked_techs() -> None:
    effects = combined_effects(frozenset({"alpha", "beta"}), _TECHS)
    assert effects.water_cost == 0.25 * 0.4
    assert effects.strength == 2.0
    assert effects.research == 1.0


def test_available_techs_respect_prerequisites() -> None:
    none_unlocked = available_techs(_TECHS, frozenset(), frozenset())
    assert [tech.tech_id for tech in none_unlocked] == ["alpha"]
    with_alpha = available_techs(_TECHS, frozenset({"alpha"}), frozenset())
    assert {tech.tech_id for tech in with_alpha} == {"beta"}


def test_available_techs_respect_resource_gating() -> None:
    without_ore = available_techs(_TECHS, frozenset(), frozenset())
    assert all(tech.tech_id != "gated" for tech in without_ore)
    with_ore = available_techs(_TECHS, frozenset(), frozenset({"ore"}))
    assert any(tech.tech_id == "gated" for tech in with_ore)


def test_choose_research_is_deterministic_for_a_seed() -> None:
    candidates = available_techs(_TECHS, frozenset(), frozenset({"ore"}))
    first = choose_research(SeededRng(3), candidates, {})
    second = choose_research(SeededRng(3), candidates, {})
    assert first is not None and second is not None
    assert first.tech_id == second.tech_id


def test_choose_research_follows_heavy_propensity() -> None:
    candidates = available_techs(_TECHS, frozenset(), frozenset({"ore"}))
    weights = {"gated": 10_000.0, "alpha": 0.0001}
    picks = {
        choose_research(SeededRng(seed), candidates, weights).tech_id  # type: ignore[union-attr]
        for seed in range(10)
    }
    assert picks == {"gated"}


def test_choose_research_returns_none_without_candidates() -> None:
    assert choose_research(SeededRng(1), (), {}) is None
