"""Named settlements: the discrete-entity overlay above the density fields.

Borders, population, and tech are continuous fields; settlements are the
thin overlay of named objects that sits on top (rulers, treaties, and wars
join them on the roadmap).  Each settlement anchors influence emission and
supply lines, carries a generated toponym, and is tiered capital/city/town.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ports.grid import CellId, Grid
    from ports.rng import Rng

CAPITAL_TIER = "capital"
CITY_TIER = "city"
TOWN_TIER = "town"
TIERS = (CAPITAL_TIER, CITY_TIER, TOWN_TIER)


@dataclass(frozen=True)
class Settlement:
    """One named settlement belonging to a civilization."""

    settlement_id: str
    name: str
    civ_id: str
    cell: CellId
    domain: str
    tier: str
    founded_tick: int = 0

    def __post_init__(self) -> None:
        """Reject unknown tiers and empty names."""
        if self.tier not in TIERS:
            msg = f"settlement {self.settlement_id!r} has unknown tier {self.tier!r}"
            raise ValueError(msg)
        if not self.name:
            msg = f"settlement {self.settlement_id!r} must have a name"
            raise ValueError(msg)


def generate_toponym(
    rng: Rng,
    prefixes: Sequence[str],
    suffixes: Sequence[str],
) -> str:
    """Compose a settlement name from the species' toponym syllables."""
    if not prefixes or not suffixes:
        return f"Site {rng.randint(100, 999)}"
    return f"{rng.choice(list(prefixes))}{rng.choice(list(suffixes))}"


def local_population(
    grid: Grid,
    population_per_m2: Mapping[CellId, float],
    cell: CellId,
) -> float:
    """Return the population count living on a cell and its direct neighbors."""
    cells = (cell, *grid.neighbors(cell))
    return sum(population_per_m2.get(here, 0.0) * grid.area_m2(here) for here in cells)


def settlement_emitters(
    grid: Grid,
    population_per_m2: Mapping[CellId, float],
    settlements: Sequence[Settlement],
) -> dict[CellId, float]:
    """Map each settlement cell to its influence power.

    Power is the population the settlement actually commands (its cell plus
    direct neighbors), so influence — and therefore borders — grows with the
    people a civ builds up over time, not a free constant.
    """
    emitters: dict[CellId, float] = {}
    for settlement in settlements:
        power = local_population(grid, population_per_m2, settlement.cell)
        emitters[settlement.cell] = emitters.get(settlement.cell, 0.0) + power
    return emitters


def tier_for(local_count: float, city_threshold: float) -> str:
    """Return the tier a non-capital settlement deserves at a population count."""
    return CITY_TIER if local_count >= city_threshold else TOWN_TIER
