"""Scenario/preset commands — build a world from a named preset or config.

Split out of :mod:`api.commands` (which registers these) to keep each
module under the file-length limit. The handlers here expose the preset
library in :mod:`core.sim.presets` over the ``/commands`` surface so a
client can enumerate scenarios and create Earth/Mars/Venus/tidally-locked
worlds through the API alone, instead of the core hardcoding Earth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from api.world_service import build_world, get_all_presets, resolve_planet
from ports.grid import GridBackendUnavailableError

if TYPE_CHECKING:
    from api.state import AppState


class ListScenariosParams(BaseModel):
    """Arguments for list_scenarios (empty)."""


class WorldBuildParams(BaseModel):
    """Shared world-building args: seed + preset/custom planet + fidelity.

    The common base for the ``build_world`` and ``create_world`` commands so
    the two never drift; subclasses tweak only defaults (e.g. resolution).
    Either specify ``preset_name`` for a built-in scenario, or provide a full
    ``planet_config`` dict matching the PlanetConfig schema.
    """

    seed: int = Field(description="Random seed for terrain generation.")
    preset_name: str | None = Field(
        default="earth", description="Name of a scenario preset (earth, mars, venus, luna, etc)."
    )
    planet_config: dict[str, Any] | None = Field(
        default=None,
        description="Full PlanetConfig dict; if provided, overrides preset_name.",
    )
    resolution: int = Field(
        default=1, ge=0, le=4, description="Grid resolution (coarser = faster)."
    )
    season_count: int = Field(default=2, ge=1, le=4, description="Number of seasons to simulate.")
    grid_backend: str = Field(default="h3", description="Grid backend: h3 (default), s2, or isea.")


class BuildWorldParams(WorldBuildParams):
    """Arguments for the build_world command (see :class:`WorldBuildParams`)."""


def scenario_summaries() -> list[dict[str, object]]:
    """Return each preset as a name/description/planet-name summary dict.

    The single source the ``/scenarios`` REST route and the
    ``list_scenarios`` command both serve, so the two surfaces can never
    drift apart.
    """
    return [
        {"name": name, "description": description, "planet_name": config.name}
        for name, (config, description) in get_all_presets().items()
    ]


def list_scenarios(state: AppState, params: BaseModel) -> object:
    """Return all available scenario presets with names and descriptions."""
    return {"scenarios": scenario_summaries()}


def build_world_command(state: AppState, params: BaseModel) -> object:
    """Build a world from a seed and a preset or custom PlanetConfig.

    Returns a summary of the generated world's properties.
    """
    if not isinstance(params, BuildWorldParams):
        msg = "build_world invoked with the wrong params model"
        raise TypeError(msg)

    planet = resolve_planet(params.preset_name, params.planet_config)

    try:
        world = build_world(
            params.seed,
            resolution=params.resolution,
            season_count=params.season_count,
            planet=planet,
            grid_backend=params.grid_backend,
        )
    except GridBackendUnavailableError as exc:
        raise ValueError(f"grid backend unavailable: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid grid backend: {exc}") from exc

    planet_name = planet.name if planet is not None else "Unknown"

    return {
        "seed": params.seed,
        "preset": params.preset_name,
        "planet_name": planet_name,
        "grid_cell_count": len(list(world.grid.cells())),
        "ocean_fraction": sum(1 for on in world.sea_mask.ocean.values() if on)
        / len(list(world.grid.cells())),
    }
