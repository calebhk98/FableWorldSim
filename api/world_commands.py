"""World lifecycle commands: create/get/step a persisted world, plus reads.

Covers per-cell geometry, per-cell fields, and the reproducibility recipe.
Split out of :mod:`api.commands` (which calls
:func:`register_world_commands`) to keep both files under the line-length
cap. The persisted world itself lives on ``AppState.world``; every handler
here just validates params and delegates to :mod:`api.world_state`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from api.world_state import (
    create_persisted_world,
    export_recipe,
    grid_geometry,
    query_field,
    step_persisted_world,
    world_summary,
)
from ports.grid import GridBackendUnavailableError

if TYPE_CHECKING:
    from api.commands import Command, CommandRegistry
    from api.state import AppState
    from api.world_state import PersistedWorld


class _EmptyParams(BaseModel):
    """Empty params for the read commands below that take no arguments."""


class CreateWorldParams(BaseModel):
    """Arguments for create_world: seed + preset/custom planet + fidelity."""

    seed: int = Field(description="Random seed for terrain generation.")
    preset_name: str | None = Field(
        default="earth", description="Name of a scenario preset (earth, mars, venus, luna, etc)."
    )
    planet_config: dict[str, Any] | None = Field(
        default=None,
        description="Full PlanetConfig dict; if provided, overrides preset_name.",
    )
    resolution: int = Field(
        default=0, ge=0, le=4, description="Grid resolution (0 = 122 cells, coarsest/fastest)."
    )
    season_count: int = Field(default=2, ge=1, le=4, description="Number of seasons to simulate.")
    grid_backend: str = Field(default="h3", description="Grid backend: h3 (default), s2, or isea.")


class StepWorldParams(BaseModel):
    """Arguments for step_world."""

    ticks: int = Field(default=1, ge=1, le=1000, description="Orchestrator ticks to advance by.")


class QueryFieldParams(BaseModel):
    """Arguments for query_field."""

    field: str = Field(description="Field name: height, elevation, temperature, or precipitation.")


def _require_world(state: AppState) -> PersistedWorld:
    """Return the live world or raise KeyError (-> 404) if none exists yet."""
    if state.world is None:
        msg = "no world created yet; call create_world first"
        raise KeyError(msg)
    return state.world


def _create_world(state: AppState, params: BaseModel) -> object:
    """Build a world from a seed/preset and store it as the live world."""
    if not isinstance(params, CreateWorldParams):
        msg = "create_world invoked with the wrong params model"
        raise TypeError(msg)
    try:
        state.world = create_persisted_world(
            params.seed,
            preset_name=params.preset_name,
            planet_config=params.planet_config,
            resolution=params.resolution,
            season_count=params.season_count,
            grid_backend=params.grid_backend,
        )
    except GridBackendUnavailableError as exc:
        raise ValueError(f"grid backend unavailable: {exc}") from exc
    return world_summary(state.world)


def _get_world(state: AppState, params: BaseModel) -> object:
    """Return a summary of the current world (or ``exists: false``)."""
    return world_summary(state.world)


def _step_world(state: AppState, params: BaseModel) -> object:
    """Advance the current world by N orchestrator ticks."""
    if not isinstance(params, StepWorldParams):
        msg = "step_world invoked with the wrong params model"
        raise TypeError(msg)
    return step_persisted_world(_require_world(state), params.ticks)


def _query_field(state: AppState, params: BaseModel) -> object:
    """Return cell_id -> value for one field of the current world."""
    if not isinstance(params, QueryFieldParams):
        msg = "query_field invoked with the wrong params model"
        raise TypeError(msg)
    return query_field(_require_world(state), params.field)


def _grid_geometry(state: AppState, params: BaseModel) -> object:
    """Return cell_id -> {lat, lng} centroid for the current world's grid."""
    return grid_geometry(_require_world(state))


def _export_recipe(state: AppState, params: BaseModel) -> object:
    """Return the reproducibility recipe for the current world."""
    return export_recipe(_require_world(state))


def register_world_commands(registry: CommandRegistry, command_cls: type[Command]) -> None:
    """Register create/get/step/query_field/grid_geometry/export_recipe.

    Takes the ``Command`` class as a parameter (rather than importing it)
    so this module never imports :mod:`api.commands` at module scope --
    that module imports this one to call this function, and a top-level
    back-import would be circular.
    """
    registry.register(
        command_cls(
            "create_world",
            "Build a world from a seed and a preset or custom PlanetConfig, seed its "
            "biology + civilization state, and store it as the live, steppable world "
            "(replacing any previous one). Returns a summary (tick=0, cell count, ...).",
            CreateWorldParams,
            _create_world,
            mutates=True,
        )
    )
    registry.register(
        command_cls(
            "get_world",
            "Return a summary of the current live world: whether one exists, its "
            "tick, cell count, and which query_field field names are available.",
            _EmptyParams,
            _get_world,
        )
    )
    registry.register(
        command_cls(
            "step_world",
            "Advance the live world by N orchestrator ticks (the coupled biology + "
            "civilization step). Returns the new tick plus cheap per-species/civ stats.",
            StepWorldParams,
            _step_world,
            mutates=True,
        )
    )
    registry.register(
        command_cls(
            "query_field",
            "Return cell_id -> value for one per-cell field of the live world "
            "(height/elevation, temperature, or precipitation).",
            QueryFieldParams,
            _query_field,
        )
    )
    registry.register(
        command_cls(
            "grid_geometry",
            "Return cell_id -> {lat, lng} centroid for every cell of the live "
            "world's grid -- what a client needs to place cells on a globe.",
            _EmptyParams,
            _grid_geometry,
        )
    )
    registry.register(
        command_cls(
            "export_recipe",
            "Return the reproducibility recipe (seed, grid backend/resolution, "
            "resolved planet config) for the live world.",
            _EmptyParams,
            _export_recipe,
        )
    )
