"""The season-resolved climate model: one code path for any planet.

Per orbit phase: insolation -> temperature with ice-albedo feedback ->
ocean thermal buffering -> winds -> ocean currents advecting heat back
into coastal air -> precipitation.  Seasons matter for a world sim, so
M1 resolves the seasonal cycle (monsoons emerge from land-sea heat
contrast); day-to-day weather (storms, fronts) needs a time-dependent
GCM and is roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.climate.insolation import insolation_field
from core.climate.moisture import MoistureInputs, precipitation_field
from core.climate.temperature import (
    FREEZING_POINT_K,
    solve_temperature_with_ice,
)
from core.climate.wind import wind_field
from core.grid.flux import diffusion_step
from core.ocean.surface import SurfaceOcean
from ports.ocean import OceanForcing

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.grid.vectors import Vector2
    from core.hydrology.sea_mask import SeaMask
    from core.sim.planet_config import PlanetConfig
    from ports.grid import CellId, Grid

_OCEAN_SEASONAL_DAMPING = 0.25
_SST_AIR_BLEND = 0.25
_COASTAL_SPREAD_ITERATIONS = 3
_COASTAL_DIFFUSIVITY_M2_S = 2.0e6
_WIND_STRESS_PER_SPEED_PA = 0.02
_AIR_SEA_COUPLING_W_M2_K = 20.0


@dataclass(frozen=True)
class SeasonClimate:
    """One season's fields."""

    orbit_phase: float
    temperature_k: Mapping[CellId, float]
    precipitation_mm_yr: Mapping[CellId, float]
    ice: Mapping[CellId, bool]
    wind: Mapping[CellId, Vector2]


@dataclass(frozen=True)
class ClimateState:
    """The full seasonal climate plus annual summaries."""

    seasons: tuple[SeasonClimate, ...]
    annual_mean_temperature_k: Mapping[CellId, float]
    annual_precipitation_mm_yr: Mapping[CellId, float]
    permanent_ice: Mapping[CellId, bool]
    sea_surface_temp_k: Mapping[CellId, float]


def _seasonal_raw_temperatures(
    grid: Grid,
    planet: PlanetConfig,
    sea_mask: SeaMask,
    phases: list[float],
) -> list[dict[CellId, float]]:
    """Return unbuffered per-phase temperatures (with ice feedback)."""
    fields = []
    for phase in phases:
        insolation = insolation_field(grid, planet, phase)
        temps, _ice = solve_temperature_with_ice(grid, planet, insolation, sea_mask)
        fields.append(temps)
    return fields


def _buffer_ocean_seasons(
    raw: dict[CellId, float],
    annual: Mapping[CellId, float],
    sea_mask: SeaMask,
) -> dict[CellId, float]:
    """Damp seasonal swings over water (ocean heat capacity)."""
    return {
        cell: (
            annual[cell] + _OCEAN_SEASONAL_DAMPING * (temp - annual[cell])
            if sea_mask.ocean[cell]
            else temp
        )
        for cell, temp in raw.items()
    }


@dataclass(frozen=True)
class _WorldContext:
    """The per-run invariants every season step needs."""

    grid: Grid
    planet: PlanetConfig
    sea_mask: SeaMask
    ocean: SurfaceOcean
    season_dt_s: float


def _couple_ocean(
    context: _WorldContext,
    temps: dict[CellId, float],
    winds: Mapping[CellId, Vector2],
) -> dict[CellId, float]:
    """Step the ocean under this season's winds and blend heat back.

    Currents move SST; ocean-cell air blends toward the moved SST, and
    a few diffusion passes carry the anomaly onto downwind coasts —
    the Gulf Stream keeping high-latitude shores warmer than their
    latitude deserves.
    """
    grid = context.grid
    ocean = context.ocean
    sst = ocean.surface().sea_surface_temp_k
    forcing = OceanForcing(
        wind_stress_east_pa={c: winds[c][0] * _WIND_STRESS_PER_SPEED_PA for c in sst},
        wind_stress_north_pa={c: winds[c][1] * _WIND_STRESS_PER_SPEED_PA for c in sst},
        surface_heat_flux_w_m2={c: _AIR_SEA_COUPLING_W_M2_K * (temps[c] - sst[c]) for c in sst},
        ocean_mask={c: context.sea_mask.ocean[c] for c in temps},
    )
    ocean.step(forcing, context.season_dt_s)
    moved = ocean.surface().sea_surface_temp_k
    coupled = {
        cell: (
            (1.0 - _SST_AIR_BLEND) * temp + _SST_AIR_BLEND * moved[cell] if cell in moved else temp
        )
        for cell, temp in temps.items()
    }
    min_area = min(grid.area_m2(cell) for cell in coupled)
    dt_diffuse = 0.2 * min_area / _COASTAL_DIFFUSIVITY_M2_S
    for _ in range(_COASTAL_SPREAD_ITERATIONS):
        coupled = diffusion_step(coupled, grid, _COASTAL_DIFFUSIVITY_M2_S, dt_diffuse)
    return coupled


def simulate_climate(
    grid: Grid,
    planet: PlanetConfig,
    heights_m: Mapping[CellId, float],
    sea_mask: SeaMask,
    season_count: int = 4,
) -> ClimateState:
    """Run the seasonal climate and return all fields.

    ``season_count`` is a fidelity knob (4 = quarterly, 12 = monthly).
    """
    if season_count < 1:
        msg = f"season_count must be >= 1, got {season_count}"
        raise ValueError(msg)
    phases = [i / season_count for i in range(season_count)]
    raw_fields = _seasonal_raw_temperatures(grid, planet, sea_mask, phases)
    cells = list(raw_fields[0])
    annual = {cell: sum(field[cell] for field in raw_fields) / len(raw_fields) for cell in cells}
    context = _WorldContext(
        grid=grid,
        planet=planet,
        sea_mask=sea_mask,
        ocean=SurfaceOcean(grid, planet, sea_mask, annual),
        season_dt_s=planet.orbital_period_s / season_count,
    )

    seasons = []
    for phase, raw in zip(phases, raw_fields, strict=True):
        temps = _buffer_ocean_seasons(raw, annual, sea_mask)
        winds = wind_field(grid, planet, temps)
        temps = _couple_ocean(context, temps, winds)
        ice = {cell: temps[cell] < FREEZING_POINT_K for cell in temps}
        precip = precipitation_field(
            grid,
            MoistureInputs(
                temps=temps, winds=winds, heights_m=heights_m, sea_mask=sea_mask, ice=ice
            ),
        )
        seasons.append(
            SeasonClimate(
                orbit_phase=phase,
                temperature_k=temps,
                precipitation_mm_yr=precip,
                ice=ice,
                wind=winds,
            )
        )

    return ClimateState(
        seasons=tuple(seasons),
        annual_mean_temperature_k={
            cell: sum(s.temperature_k[cell] for s in seasons) / len(seasons) for cell in cells
        },
        annual_precipitation_mm_yr={
            cell: sum(s.precipitation_mm_yr[cell] for s in seasons) / len(seasons) for cell in cells
        },
        permanent_ice={cell: all(s.ice[cell] for s in seasons) for cell in cells},
        sea_surface_temp_k=dict(context.ocean.surface().sea_surface_temp_k),
    )
