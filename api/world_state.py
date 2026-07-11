"""Persisted, steppable world: the live world the API holds and advances.

Unlike :func:`api.world_service.build_world` (an ephemeral ``FastWorld`` its
caller immediately discards) or ``deepen_seed`` (which runs a whole
simulation and returns only a summary), this module builds a world **once**
and keeps it -- terrain, climate, and a seeded biology + civilization
state -- so the API can advance it tick by tick and answer per-cell queries
against the *same* world across many requests. See ``AppState.world`` in
:mod:`api.state`, and the ``create_world``/``get_world``/``step_world``/
``query_field``/``grid_geometry``/``export_recipe`` commands in
:mod:`api.world_commands` that wrap the functions here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from api.world_service import (
    FastWorld,
    assemble_coupled_run,
    build_world,
    resolve_planet,
)
from core.civilization.state import civ_population_total
from core.sim.coupling import WorldState
from core.sim.orchestrator import Orchestrator
from core.sim.presets import earth
from core.sim.recipe import WorldRecipe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.sim.planet_config import PlanetConfig
    from ports.grid import CellId

_DEFAULT_RESOLUTION = 0
"""Default create_world resolution: 122 h3 cells -- cheap enough to build
and to seed biology/civilization for inline on an API request."""

FIELD_NAMES = ("height", "elevation", "temperature", "precipitation")
"""Field names :func:`query_field` understands (elevation aliases height)."""

_FIELD_ALIASES: dict[str, str] = {"elevation": "height"}


@dataclass
class PersistedWorld:
    """The one live world the server holds: terrain plus a stepping state.

    Mutable in place -- ``tick`` and ``state`` advance under the server's
    single-writer lock (see ``AppState.write_lock``) -- while every other
    field is fixed at creation and doubles as the reproducibility recipe
    (see :func:`export_recipe`).
    """

    seed: int
    preset_name: str | None
    planet_config: Mapping[str, Any] | None
    resolution: int
    season_count: int
    grid_backend: str
    planet: PlanetConfig
    fast: FastWorld
    state: WorldState
    orchestrator: Orchestrator[WorldState]
    tick: int = 0


def create_persisted_world(  # noqa: PLR0913 - one keyword param per create_world knob
    seed: int,
    *,
    preset_name: str | None = "earth",
    planet_config: Mapping[str, Any] | None = None,
    resolution: int = _DEFAULT_RESOLUTION,
    season_count: int = 2,
    grid_backend: str = "h3",
) -> PersistedWorld:
    """Build a world and seed a biology + civilization state ready to step.

    Mirrors ``api.world_service.deepen_seed``'s composition (terrain ->
    climate -> biome -> biology/civ context -> seeded state, coupled
    through :mod:`core.sim.coupling`) but *keeps* the result instead of
    discarding it after one summarized run. Unlike ``deepen_seed`` this
    skips river/lake routing: neither the biology nor civilization context
    reads the river network (only climate, terrain, and biome), so
    omitting it keeps creation cheap enough to run inline on a request --
    a deliberate scope cut, not an oversight.
    """
    planet = resolve_planet(preset_name, planet_config)
    resolved_planet = planet if planet is not None else earth()
    fast = build_world(
        seed,
        resolution=resolution,
        season_count=season_count,
        planet=planet,
        grid_backend=grid_backend,
    )
    state, _coupled_ctx, orchestrator = assemble_coupled_run(fast, seed)
    return PersistedWorld(
        seed=seed,
        preset_name=preset_name,
        planet_config=planet_config,
        resolution=resolution,
        season_count=season_count,
        grid_backend=grid_backend,
        planet=resolved_planet,
        fast=fast,
        state=state,
        orchestrator=orchestrator,
    )


def world_summary(world: PersistedWorld | None) -> dict[str, object]:
    """Return a JSON-able summary of the current world (or none created yet)."""
    if world is None:
        return {"exists": False}
    grid = world.fast.grid
    return {
        "exists": True,
        "tick": world.tick,
        "seed": world.seed,
        "planet_name": world.planet.name,
        "grid_backend": grid.backend_name,
        "grid_resolution": grid.resolution,
        "cell_count": grid.cell_count,
        "fields_available": list(FIELD_NAMES),
    }


def step_persisted_world(world: PersistedWorld, ticks: int) -> dict[str, object]:
    """Advance ``world`` by ``ticks`` orchestrator ticks, in place.

    Returns cheap post-step stats (new tick, per-species survivor counts,
    civilization populations, and the run's wall-clock telemetry) -- enough
    for a client to confirm the step actually moved something without
    re-fetching the whole per-cell field state.
    """
    world.state = world.orchestrator.run(world.state, ticks=ticks)
    world.tick += ticks
    telemetry = world.orchestrator.telemetry
    populations = {
        species_id: sum(field_values.values())
        for species_id, field_values in world.state.biology.surface_populations.items()
    }
    civ_population = {
        civ.civ_id: civ_population_total(world.fast.grid, civ) for civ in world.state.civ.civs
    }
    return {
        "tick": world.tick,
        "ticks_advanced": ticks,
        "surviving_species": sum(1 for total in populations.values() if total > 0.0),
        "populations": populations,
        "civ_population": civ_population,
        "telemetry": telemetry.to_metrics() if telemetry is not None else {},
    }


def query_field(world: PersistedWorld, field_name: str) -> dict[CellId, float]:
    """Return one field's value at every cell: ``cell_id -> value``.

    Supports height/elevation (terrain) and temperature/precipitation
    (annual climate summaries) -- the layers a web client's data-layer
    picker needs first. Unknown names raise ``KeyError`` (-> 404 over the
    command surface, see ``api.app._execute_command``).
    """
    name = _FIELD_ALIASES.get(field_name, field_name)
    if name == "height":
        return dict(world.fast.heights_m)
    if name == "temperature":
        return dict(world.fast.climate.annual_mean_temperature_k)
    if name == "precipitation":
        return dict(world.fast.climate.annual_precipitation_mm_yr)
    known = ", ".join(FIELD_NAMES)
    raise KeyError(f"unknown field {field_name!r}; known: {known}")


def grid_geometry(world: PersistedWorld) -> dict[CellId, dict[str, float]]:
    """Return each cell's centroid lat/lng -- what a globe needs to place cells.

    The ``Grid`` port exposes no cell-boundary-vertex accessor (only
    centroid/area/neighbors/edge_length), so this is centroid-only; a
    boundary-polygon endpoint would need that port extended first. Noted
    as deferred LOD/geometry fidelity, not implemented here.
    """
    grid = world.fast.grid
    geometry: dict[CellId, dict[str, float]] = {}
    for cell in grid.cells():
        point = grid.centroid(cell)
        geometry[cell] = {"lat": point.lat_deg, "lng": point.lon_deg}
    return geometry


def export_recipe(world: PersistedWorld) -> dict[str, object]:
    """Return the reproducibility recipe for the current world.

    Seed + grid backend/resolution + season count + the resolved planet
    config are everything :func:`create_persisted_world` needs to
    regenerate this world's terrain/climate bit-for-bit (biology/civ RNG
    forks from the same seed too); see :class:`core.sim.recipe.WorldRecipe`.
    """
    planet_config = (
        dict(world.planet_config)
        if world.planet_config is not None
        else dataclasses.asdict(world.planet)
    )
    recipe = WorldRecipe(
        world_name=world.planet.name,
        seed=world.seed,
        grid_backend=world.grid_backend,
        grid_resolution=world.resolution,
        settings={
            "preset_name": world.preset_name,
            "season_count": world.season_count,
            "planet_config": planet_config,
        },
    )
    return dataclasses.asdict(recipe)
