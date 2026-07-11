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
