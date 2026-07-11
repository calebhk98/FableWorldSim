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

from api.world_service import build_world, get_all_presets
from core.sim.planet_config import PlanetConfig

if TYPE_CHECKING:
    from api.state import AppState


class ListScenariosParams(BaseModel):
    """Arguments for list_scenarios (empty)."""


class BuildWorldParams(BaseModel):
    """Arguments for build_world command.

    Either specify preset_name for a built-in scenario, or provide a full
    planet_config dict matching PlanetConfig schema.
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

    planet = None
    if params.planet_config is not None:
        try:
            planet = PlanetConfig(**params.planet_config)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid planet_config: {exc}") from exc
    elif params.preset_name:
        presets = get_all_presets()
        if params.preset_name not in presets:
            known = ", ".join(presets.keys())
            raise KeyError(f"unknown preset {params.preset_name!r}; known: {known}")
        planet, _description = presets[params.preset_name]

    world = build_world(
        params.seed,
        resolution=params.resolution,
        season_count=params.season_count,
        planet=planet,
    )

    planet_name = planet.name if planet is not None else "Unknown"

    return {
        "seed": params.seed,
        "preset": params.preset_name,
        "planet_name": planet_name,
        "grid_cell_count": len(list(world.grid.cells())),
        "ocean_fraction": sum(1 for on in world.sea_mask.ocean.values() if on)
        / len(list(world.grid.cells())),
    }
