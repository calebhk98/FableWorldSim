"""Growth rate derived from life-history traits (cohort-lite)."""

from __future__ import annotations

from core.biology.demography import intrinsic_growth_rate
from tests.bio_helpers import make_organism


def test_fast_maturing_species_outbreeds_a_slow_one() -> None:
    grass = make_organism("grass", maturity_years=0.3, litter_size=50.0, lifespan_years=2.0)
    dragon = make_organism("dragon", maturity_years=40.0, litter_size=1.2, lifespan_years=500.0)
    assert intrinsic_growth_rate(grass) > intrinsic_growth_rate(dragon)


def test_a_species_that_cannot_replace_itself_has_nonpositive_rate() -> None:
    barren = make_organism("barren", litter_size=1.0, maturity_years=5.0, lifespan_years=10.0)
    assert intrinsic_growth_rate(barren) <= 0.0


def test_defaults_reduce_to_the_pre_gestation_formula() -> None:
    """gestation=0, fertility_window=inf must reproduce the old rate exactly."""
    baseline = make_organism("baseline", maturity_years=2.0, litter_size=3.0, lifespan_years=8.0)
    explicit = make_organism(
        "explicit",
        maturity_years=2.0,
        litter_size=3.0,
        lifespan_years=8.0,
        gestation_years=0.0,
        fertility_window_years=float("inf"),
    )
    assert intrinsic_growth_rate(baseline) == intrinsic_growth_rate(explicit)


def test_longer_gestation_lengthens_the_birth_interval_and_lowers_r() -> None:
    quick = make_organism(
        "quick", maturity_years=2.0, litter_size=3.0, lifespan_years=8.0, gestation_years=0.0
    )
    slow_gestation = make_organism(
        "slow-gestation",
        maturity_years=2.0,
        litter_size=3.0,
        lifespan_years=8.0,
        gestation_years=3.0,
    )
    assert intrinsic_growth_rate(slow_gestation) < intrinsic_growth_rate(quick)


def test_a_narrower_fertility_window_lowers_r() -> None:
    long_fertile = make_organism(
        "long-fertile",
        maturity_years=2.0,
        litter_size=3.0,
        lifespan_years=20.0,
        fertility_window_years=18.0,
    )
    short_fertile = make_organism(
        "short-fertile",
        maturity_years=2.0,
        litter_size=3.0,
        lifespan_years=20.0,
        fertility_window_years=2.0,
    )
    assert intrinsic_growth_rate(short_fertile) < intrinsic_growth_rate(long_fertile)


def test_two_species_differing_only_in_gestation_and_fertility_window_get_different_r() -> None:
    common = {"maturity_years": 3.0, "litter_size": 4.0, "lifespan_years": 15.0}
    species_a = make_organism("a", gestation_years=0.2, fertility_window_years=10.0, **common)
    species_b = make_organism("b", gestation_years=1.5, fertility_window_years=3.0, **common)
    assert intrinsic_growth_rate(species_a) != intrinsic_growth_rate(species_b)
