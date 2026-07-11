"""Extinction and minimum-viable-population viability gates."""

from __future__ import annotations

from core.biology.extinction import enforce_viability, total_individuals
from tests.bio_helpers import make_organism


def _unit_area(_node: str) -> float:
    return 1.0


def test_total_individuals_is_area_weighted() -> None:
    assert total_individuals({"a": 2.0, "b": 3.0}, _unit_area) == 5.0


def test_sexual_species_below_mvp_goes_extinct() -> None:
    deer = make_organism("deer", reproduction="sexual", min_viable_population=2)
    culled, alive = enforce_viability({"c": 1.5}, deer, _unit_area, 1e-15)
    assert not alive
    assert all(value == 0.0 for value in culled.values())


def test_asexual_species_rebounds_from_a_single_survivor() -> None:
    grass = make_organism("grass", reproduction="asexual", min_viable_population=1)
    culled, alive = enforce_viability({"c": 0.5}, grass, _unit_area, 1e-15)
    assert alive
    assert culled["c"] == 0.5


def test_numerical_dust_is_culled_to_zero() -> None:
    organism = make_organism("x", reproduction="asexual", min_viable_population=1)
    culled, alive = enforce_viability({"c": 1e-20}, organism, _unit_area, 1e-15)
    assert culled["c"] == 0.0
    assert not alive
