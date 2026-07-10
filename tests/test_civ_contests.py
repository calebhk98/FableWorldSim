"""Frontier contests weigh projected strength; reach is supply-bounded."""

from __future__ import annotations

import math

from adapters.rng_seeded import SeededRng
from core.civilization.contests import (
    CivStrength,
    arms_multiplier,
    raw_strength,
    resolve_contests,
)
from core.civilization.resources import MINE_MODE, ResourceDefinition
from core.civilization.tech import IDENTITY_EFFECTS, TechEffects


def _strength(civ_id: str, raw: float, distance: float) -> CivStrength:
    return CivStrength(
        civ_id=civ_id,
        raw_strength=raw,
        supply_distance_m={"x": distance},
        supply_range_m=1_000_000.0,
    )


def test_projected_strength_decays_with_supply_distance() -> None:
    near = _strength("a", 100.0, 0.0).projected("x")
    far = _strength("a", 100.0, 2_000_000.0).projected("x")
    assert near == 100.0
    assert far == 100.0 * math.exp(-2.0)
    unreachable = CivStrength("a", 100.0, {}, 1_000_000.0).projected("x")
    assert unreachable == 0.0


def test_overwhelming_challenger_takes_the_cell() -> None:
    strengths = {
        "holder": _strength("holder", 1e-9, 0.0),
        "raider": _strength("raider", 1e9, 0.0),
    }
    resolved, flips = resolve_contests(SeededRng(0), {"x": "raider"}, {"x": "holder"}, strengths)
    assert resolved["x"] == "raider"
    assert flips == (("x", "holder", "raider"),)


def test_unsupplied_challenger_cannot_take_anything() -> None:
    strengths = {
        "holder": _strength("holder", 10.0, 0.0),
        "rome": CivStrength("rome", 1e9, {}, 1_000_000.0),
    }
    for seed in range(20):
        resolved, flips = resolve_contests(
            SeededRng(seed), {"x": "rome"}, {"x": "holder"}, strengths
        )
        assert resolved["x"] == "holder", "projection ~0 means no conquest across the world"
        assert not flips


def test_even_contests_are_stochastic_not_a_fixed_rule() -> None:
    strengths = {
        "a": _strength("a", 100.0, 0.0),
        "b": _strength("b", 100.0, 0.0),
    }
    outcomes = set()
    for seed in range(20):
        resolved, _ = resolve_contests(SeededRng(seed), {"x": "b"}, {"x": "a"}, strengths)
        outcomes.add(resolved["x"])
    assert outcomes == {"a", "b"}, "comparable strength must sometimes flip, sometimes hold"


def test_raw_strength_is_built_from_population_tech_and_resources() -> None:
    ore = ResourceDefinition(resource_id="ore", mode=MINE_MODE, arms_value=1.0)
    armed = arms_multiplier({"ore": 100.0}, (ore,), arms_scale=0.01)
    unarmed = arms_multiplier({}, (ore,), arms_scale=0.01)
    assert armed == 2.0
    assert unarmed == 1.0
    strong = raw_strength(10_000.0, 0.1, TechEffects(strength=1.5), armed)
    weak = raw_strength(10_000.0, 0.1, IDENTITY_EFFECTS, unarmed)
    assert strong == 10_000.0 * 0.1 * 1.5 * 2.0
    assert weak == 1_000.0
