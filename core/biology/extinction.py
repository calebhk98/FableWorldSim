"""Extinction and minimum-viable-population viability gates.

A population can dwindle to local disappearance and, globally, to
extinction — recorded as a first-class chronicle event.  A minimum-viable
gate reflects breeding biology: a **sexual** species needs enough breeding
stock left to recover, so a remnant below its threshold is a dead lineage,
whereas an **asexual** species (most plants here) can rebound from a single
survivor.  Vanishingly small per-cell densities are also culled to zero so
numerical dust never masquerades as a surviving population.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.biology.organism import SEXUAL

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from core.biology.organism import Organism


def total_individuals(field: Mapping[str, float], area_of: Callable[[str], float]) -> float:
    """Return the area-weighted total (head count, or total biomass for plants)."""
    return sum(value * area_of(loc) for loc, value in field.items())


def enforce_viability(
    field: Mapping[str, float],
    organism: Organism,
    area_of: Callable[[str], float],
    epsilon_per_m2: float,
) -> tuple[dict[str, float], bool]:
    """Cull dust, apply the viability gate, and report whether the species lives.

    Returns the cleaned field and whether the organism is still present
    anywhere.  A sexual species whose surviving total falls below its
    minimum viable population is wiped out (a lineage that cannot recover).
    """
    culled = {loc: (value if value > epsilon_per_m2 else 0.0) for loc, value in field.items()}
    total = total_individuals(culled, area_of)
    if total <= 0.0:
        return culled, False
    if organism.reproduction == SEXUAL and total < organism.min_viable_population:
        return dict.fromkeys(culled, 0.0), False
    return culled, True
