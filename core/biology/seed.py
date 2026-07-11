"""Seed the initial biosphere: every species placed in its suitable habitat.

At world start each species is dropped at a modest fraction of its crowding
cap into the cells (or nodes) whose suitability clears a threshold, so the
first tick already has plants on fertile ground, herbivores among them, and
predators where prey can live.  Dynamics take over from there — starvation,
competition, migration, and extinction reshape the map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.biology.medium import SUBTERRANEAN
from core.biology.state import WorldBiologyState
from core.biology.suitability import (
    SurfaceEnvironment,
    subsurface_suitability,
    surface_suitability,
)

if TYPE_CHECKING:
    from core.biology.context import BiologyContext
    from core.biology.organism import Organism

_SEED_SUITABILITY_THRESHOLD = 0.3
"""Only cells at least this suitable receive a founding population."""

_DEFAULT_SEED_FRACTION = 0.05
"""Founding density as a fraction of the species' crowding cap."""


def _seed_field(
    organism: Organism,
    suitability: dict[str, float],
    fraction: float,
) -> dict[str, float]:
    """Return a founding density field over the sufficiently suitable locations."""
    return {
        loc: organism.crowding_cap_per_m2 * fraction
        for loc, score in suitability.items()
        if score >= _SEED_SUITABILITY_THRESHOLD
    }


def _seed_surface(ctx: BiologyContext, organism: Organism, fraction: float) -> dict[str, float]:
    """Seed one surface species across its suitable cells."""
    environment = SurfaceEnvironment(
        axis_fields=ctx.axis_fields,
        biome_field=ctx.biome_field,
        sea_mask=ctx.sea_mask,
        base_preference=ctx.params.base_biome_preference,
    )
    suitability = surface_suitability(organism, list(ctx.grid.cells()), environment)
    return _seed_field(organism, suitability, fraction)


def _seed_subsurface(ctx: BiologyContext, organism: Organism, fraction: float) -> dict[str, float]:
    """Seed one subterranean species across its suitable volume nodes."""
    if ctx.subsurface is None:
        return {}
    suitability = subsurface_suitability(
        organism,
        list(ctx.subsurface.nodes()),
        ctx.subsurface_temperature,
        ctx.subsurface_diggability,
    )
    return _seed_field(organism, suitability, fraction)


def seed_biosphere(
    ctx: BiologyContext,
    fraction: float = _DEFAULT_SEED_FRACTION,
) -> WorldBiologyState:
    """Build the initial :class:`WorldBiologyState` from habitat suitability."""
    surface: dict[str, dict[str, float]] = {}
    subsurface: dict[str, dict[str, float]] = {}
    for species_id, organism in ctx.organisms.items():
        if organism.medium == SUBTERRANEAN:
            subsurface[species_id] = _seed_subsurface(ctx, organism, fraction)
        else:
            surface[species_id] = _seed_surface(ctx, organism, fraction)
    return WorldBiologyState(surface_populations=surface, subsurface_populations=subsurface, tick=0)
