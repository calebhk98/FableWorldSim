"""Growth rate from simple life-history traits (cohort-lite, minimal M1).

A species carries a lifespan, a maturity age, a litter size, a gestation
length, and a fertility window; the intrinsic rate of increase *r* falls
out of them, so a fast-maturing, many-offspring species (grass, rabbits)
outbreeds a slow, single-offspring one (dragons) without either being
tuned by hand.  Full age-structured (Leslie-matrix) dynamics are an
option/roadmap; M1 collapses the life history to one Euler-Lotka-style
rate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.biology.organism import Organism

_MIN_LITTER = 1.0e-3
"""Floor on litter size so the birth-rate logarithm stays defined."""

_MIN_YEARS = 1.0e-3
"""Floor on any interval-of-years denominator so it stays defined."""


def intrinsic_growth_rate(organism: Organism) -> float:
    """Return the per-year intrinsic growth rate *r* for an organism.

    Births per year approximate ``ln(litter_size) / birth_interval``, the
    Euler-Lotka short form with generation time taken as the birth
    interval — maturity age plus gestation, since a longer pregnancy
    lengthens the gap between litters just as a longer maturity age
    delays the first one.  The result is scaled by ``fertile_fraction``,
    the share of adult life (lifespan minus maturity) that falls inside
    the species' fertility window: a species that stops reproducing long
    before it dies contributes fewer lifetime births per year of life than
    one fertile for its whole adulthood.  Deaths approximate
    ``1 / lifespan``.  A species whose litter cannot outrun its mortality
    gets a non-positive rate and cannot sustain a population on its own —
    exactly the intended failure mode.

    The defaults (``gestation_years=0``, ``fertility_window_years=inf``)
    reduce this exactly to the pre-gestation formula — ``birth_interval``
    is just ``maturity_years`` and ``fertile_fraction`` is 1 — so existing
    species are unaffected unless their content opts into the new fields.
    """
    birth_interval = max(_MIN_YEARS, organism.maturity_years + organism.gestation_years)
    births = math.log(max(_MIN_LITTER, organism.litter_size)) / birth_interval
    adult_lifespan = max(_MIN_YEARS, organism.lifespan_years - organism.maturity_years)
    fertile_fraction = min(1.0, organism.fertility_window_years / adult_lifespan)
    deaths = 1.0 / organism.lifespan_years
    return births * fertile_fraction - deaths
