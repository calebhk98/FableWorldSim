"""Biology state: per-species density fields on surface and in the volume.

The substrate is field-shaped, consistent with the rest of the sim: each
species is a per-location density mapping.  Surface species (plants,
animals, birds) live on the surface grid keyed by cell; subterranean
species (cave flora, dwarves) live on the ``SubsurfaceGrid`` keyed by node.
Plant fields are standing biomass in kg/m2; animal fields are count density
in individuals/m2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from core.biology.organism import Organism
    from ports.grid import CellId


@dataclass(frozen=True)
class WorldBiologyState:
    """Every species' density field, threaded through the orchestrator."""

    surface_populations: Mapping[str, Mapping[str, float]]
    subsurface_populations: Mapping[str, Mapping[str, float]]
    tick: int = 0

    def field_for(self, species_id: str) -> Mapping[str, float]:
        """Return a species' density field (surface or subsurface)."""
        if species_id in self.surface_populations:
            return self.surface_populations[species_id]
        return self.subsurface_populations.get(species_id, {})


def population_total(field: Mapping[str, float], area_of: Callable[[str], float]) -> float:
    """Return the area-weighted total of a density field."""
    return sum(value * area_of(loc) for loc, value in field.items())


def plant_biomass_field(
    state: WorldBiologyState,
    organisms: Sequence[Organism],
) -> dict[CellId, float]:
    """Sum surface autotroph biomass per cell — the field civilization grazes.

    This is the coupling seam to layer 6: forestry and farmland draw down
    the very same standing plant biomass the food web depends on, so
    clear-cutting a forest starves the herbivores that grazed it. See
    :func:`apply_biomass_drawdown` for the write side of that seam.
    """
    plants = [org.species_id for org in organisms if org.is_autotroph]
    totals: dict[CellId, float] = {}
    for species_id in plants:
        for cell, biomass in state.surface_populations.get(species_id, {}).items():
            totals[cell] = totals.get(cell, 0.0) + biomass
    return totals


def apply_biomass_drawdown(
    state: WorldBiologyState,
    organisms: Sequence[Organism],
    removed_kg_m2: Mapping[CellId, float],
) -> WorldBiologyState:
    """Remove civilization's harvest from the autotroph fields it came from.

    ``removed_kg_m2`` is a per-cell total (e.g. what :mod:`core.civilization`
    extraction just cut), split across the cell's autotroph species in
    proportion to each one's share of the standing biomass there — a
    mixed forest of two tree species loses both species pro rata, not
    whichever happens to be first.  This is the write half of the same
    seam :func:`plant_biomass_field` reads: after this call, the very
    populations the food web feeds on are down by exactly what was cut.
    """
    plants = [org for org in organisms if org.is_autotroph]
    if not plants:
        return state
    totals = plant_biomass_field(state, plants)
    surface = {sid: dict(field) for sid, field in state.surface_populations.items()}
    for cell, removed in removed_kg_m2.items():
        total = totals.get(cell, 0.0)
        if removed <= 0.0 or total <= 0.0:
            continue
        for org in plants:
            field = surface.get(org.species_id)
            if not field or cell not in field:
                continue
            share = field[cell] / total
            field[cell] = max(0.0, field[cell] - removed * share)
    return replace(state, surface_populations=surface)
