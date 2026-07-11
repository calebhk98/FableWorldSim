"""Population dynamics: smoothed glide vs oscillatory boom/bust, starvation."""

from __future__ import annotations

from itertools import pairwise

import pytest

from core.biology.population import (
    FoodCapacityParams,
    GrowthParams,
    env_capacity,
    grow,
    logistic_step,
)
from tests.bio_helpers import make_organism


def test_smoothed_growth_glides_monotonically_without_overshoot() -> None:
    value, capacity = 0.1, 1.0
    series = [value]
    for _ in range(25):
        value = logistic_step(value, capacity, 0.5, oscillatory=False)
        series.append(value)
    assert all(later >= earlier for earlier, later in pairwise(series))
    assert max(series) <= capacity + 1e-9
    assert series[-1] == pytest.approx(capacity, abs=0.05)


def test_oscillatory_growth_overshoots_capacity() -> None:
    value, capacity = 0.5, 1.0
    peak = 0.0
    for _ in range(10):
        value = logistic_step(value, capacity, 2.5, oscillatory=True)
        peak = max(peak, value)
    assert peak > capacity


def test_starvation_makes_a_consumer_shrink() -> None:
    herbivore = make_organism("herbivore", diet={"grass": 1.0})
    grown = grow({"c": 0.5}, {"c": 1.0}, herbivore, {"c": 0.0}, GrowthParams(1.0, 0.5))
    assert grown["c"] < 0.5


def test_crowding_cap_bounds_capacity_even_at_full_suitability() -> None:
    organism = make_organism("giant", crowding_cap_per_m2=0.001)
    assert env_capacity(organism, {"c": 1.0}) == {"c": 0.001}


def test_reduced_reachable_food_lowers_capacity_not_just_growth_rate() -> None:
    carnivore = make_organism(
        "carnivore", diet={"prey": 1.0}, body_mass_kg=50.0, crowding_cap_per_m2=10.0
    )
    params = FoodCapacityParams(intake_kg_per_kg_body_year=8.0, max_offtake_fraction=0.5)
    plentiful = env_capacity(carnivore, {"c": 1.0}, {"c": 1000.0}, params)
    scarce = env_capacity(carnivore, {"c": 1.0}, {"c": 10.0}, params)
    assert scarce["c"] < plentiful["c"]
    # Food, not crowding or suitability, is the binding constraint here.
    assert scarce["c"] < carnivore.crowding_cap_per_m2


def test_no_reachable_food_zeroes_a_consumers_capacity() -> None:
    carnivore = make_organism("carnivore", diet={"prey": 1.0}, crowding_cap_per_m2=10.0)
    capacity = env_capacity(carnivore, {"c": 1.0}, {}, FoodCapacityParams())
    assert capacity["c"] == 0.0


def test_autotrophs_ignore_the_food_term_entirely() -> None:
    plant = make_organism("plant", crowding_cap_per_m2=2.0)
    capacity = env_capacity(plant, {"c": 1.0}, {}, FoodCapacityParams())
    assert capacity["c"] == 2.0
