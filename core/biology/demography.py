"""Growth rate from simple life-history traits (cohort-lite, minimal M1).

A species carries a lifespan, a maturity age, and a litter size; the
intrinsic rate of increase *r* falls out of them, so a fast-maturing,
many-offspring species (grass, rabbits) outbreeds a slow, single-offspring
one (dragons) without either being tuned by hand.  Full age-structured
(Leslie-matrix) dynamics are an option/roadmap; M1 collapses the life
history to one Euler-Lotka-style rate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.biology.organism import Organism

_MIN_LITTER = 1.0e-3
"""Floor on litter size so the birth-rate logarithm stays defined."""


def intrinsic_growth_rate(organism: Organism) -> float:
    """Return the per-year intrinsic growth rate *r* for an organism.

    Births per year approximate ``ln(litter_size) / generation_time`` with
    generation time taken as the maturity age (the Euler-Lotka short form);
    deaths approximate ``1 / lifespan``.  A species whose litter cannot
    outrun its mortality gets a non-positive rate and cannot sustain a
    population on its own — exactly the intended failure mode.
    """
    births = math.log(max(_MIN_LITTER, organism.litter_size)) / organism.maturity_years
    deaths = 1.0 / organism.lifespan_years
    return births - deaths
