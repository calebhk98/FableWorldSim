"""Civilization state: continuous fields plus the named-entity overlay.

The substrate is field-shaped, consistent with the rest of the sim:
population density, influence, and territory live as per-cell mappings.
On top sits the discrete overlay of named settlements.  Territory is kept
per domain, so a mountain-surface race and an under-mountain race can hold
the same column without contesting — they only compete within a domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.grid.area_weighted import area_weighted_total

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.civilization.settlements import Settlement
    from ports.array_backend import ArrayBackend
    from ports.grid import CellId, Grid


@dataclass(frozen=True)
class Civilization:
    """One civilization: fields, entities, tech, and stockpiles."""

    civ_id: str
    species_id: str
    domain: str
    population_per_m2: Mapping[CellId, float]
    settlements: tuple[Settlement, ...]
    unlocked_techs: frozenset[str] = frozenset()
    research_points: float = 0.0
    current_research: str | None = None
    stockpiles: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject settlements that belong to another civ."""
        for settlement in self.settlements:
            if settlement.civ_id != self.civ_id:
                msg = (
                    f"settlement {settlement.settlement_id!r} belongs to "
                    f"{settlement.civ_id!r}, not {self.civ_id!r}"
                )
                raise ValueError(msg)


@dataclass(frozen=True)
class WorldCivState:
    """The whole civilization layer threaded through the orchestrator."""

    civs: tuple[Civilization, ...]
    plant_biomass_kg_m2: Mapping[CellId, float]
    mine_stocks: Mapping[str, Mapping[CellId, float]]
    territory: Mapping[str, Mapping[CellId, str]]
    tick: int = 0


def civ_by_id(state: WorldCivState, civ_id: str) -> Civilization:
    """Return the civilization with the given id."""
    for civ in state.civs:
        if civ.civ_id == civ_id:
            return civ
    msg = f"unknown civilization {civ_id!r}"
    raise KeyError(msg)


def civ_population_total(
    grid: Grid,
    civ: Civilization,
    backend: ArrayBackend | None = None,
) -> float:
    """Return the civ's total head count, area-weighted from the density field.

    Passing a sharded ``backend`` runs this planet-wide reduction across
    every device; ``None`` uses the pure-Python path.  This is the seam
    that lets the same call scale from a laptop to a multi-GPU host.
    """
    cells = list(civ.population_per_m2)
    if not cells:
        return 0.0
    return area_weighted_total(
        [civ.population_per_m2[cell] for cell in cells],
        [grid.area_m2(cell) for cell in cells],
        backend=backend,
    )


def territory_cells(state: WorldCivState, civ: Civilization) -> tuple[CellId, ...]:
    """Return the cells the civ holds in its own domain."""
    owned = state.territory.get(civ.domain)
    if owned is None:
        return ()
    return tuple(cell for cell, owner in owned.items() if owner == civ.civ_id)
