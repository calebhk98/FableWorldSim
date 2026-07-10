"""Temperature: per-cell energy balance with transport and ice-albedo.

Each cell radiates what it absorbs (Stefan-Boltzmann) plus the
atmosphere's greenhouse offset; lateral heat transport is edge-weighted
diffusion whose strength scales with atmospheric pressure (thick CO2 ->
even Venus-like temps; thin/none -> huge contrasts like Mars/Moon).
Snow/ice cover feeds back through albedo: more ice -> more reflection ->
cooler -> more ice, iterated to a stable cover — this is a large part of
why poles are cold and climate has tipping behavior.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.grid.flux import diffusion_step
from core.sim.constants import EARTH_SURFACE_PRESSURE_PA

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.hydrology.sea_mask import SeaMask
    from core.sim.planet_config import PlanetConfig
    from ports.grid import CellId, Grid

STEFAN_BOLTZMANN = 5.670_374e-8
FREEZING_POINT_K = 273.15
OCEAN_ALBEDO = 0.06
LAND_ALBEDO = 0.25
ICE_ALBEDO = 0.60
_TRANSPORT_DIFFUSIVITY_M2_S = 2.0e6
_MAX_TRANSPORT_ITERATIONS = 60
_ICE_FEEDBACK_ITERATIONS = 3
_COSMIC_BACKGROUND_K = 3.0


def pressure_ratio(planet: PlanetConfig) -> float:
    """Return surface pressure relative to Earth's."""
    return planet.atmosphere.surface_pressure_pa / EARTH_SURFACE_PRESSURE_PA


def transport_iterations(planet: PlanetConfig) -> int:
    """Return how many diffusion passes the atmosphere earns.

    Scales with sqrt(pressure): 0 for airless (no transport at all),
    ~20 for Earth, capped for Venus-thick.
    """
    ratio = pressure_ratio(planet)
    return min(_MAX_TRANSPORT_ITERATIONS, round(20.0 * math.sqrt(ratio)))


def cell_albedo(cell: CellId, sea_mask: SeaMask, ice: Mapping[CellId, bool]) -> float:
    """Return surface albedo: ice > ocean/land distinction."""
    if ice[cell]:
        return ICE_ALBEDO
    return OCEAN_ALBEDO if sea_mask.ocean[cell] else LAND_ALBEDO


def radiative_temperature_k(
    insolation_wm2: float,
    albedo: float,
    greenhouse_offset_k: float,
) -> float:
    """Return the local no-transport equilibrium temperature."""
    absorbed = insolation_wm2 * (1.0 - albedo)
    base = float((absorbed / STEFAN_BOLTZMANN) ** 0.25)
    return max(_COSMIC_BACKGROUND_K, base + greenhouse_offset_k)


def _transport(temps: dict[CellId, float], grid: Grid, iterations: int) -> dict[CellId, float]:
    """Run pressure-scaled lateral heat diffusion (conserves the mean)."""
    if iterations == 0:
        return temps
    min_area = min(grid.area_m2(cell) for cell in temps)
    dt_s = 0.2 * min_area / _TRANSPORT_DIFFUSIVITY_M2_S
    for _ in range(iterations):
        temps = diffusion_step(temps, grid, _TRANSPORT_DIFFUSIVITY_M2_S, dt_s)
    return temps


def temperature_field(
    grid: Grid,
    planet: PlanetConfig,
    insolation: Mapping[CellId, float],
    sea_mask: SeaMask,
    ice: Mapping[CellId, bool],
) -> dict[CellId, float]:
    """Return per-cell temperature for a fixed ice cover."""
    greenhouse = planet.atmosphere.greenhouse_offset_k
    temps = {
        cell: radiative_temperature_k(
            insolation[cell], cell_albedo(cell, sea_mask, ice), greenhouse
        )
        for cell in insolation
    }
    return _transport(temps, grid, transport_iterations(planet))


def solve_temperature_with_ice(
    grid: Grid,
    planet: PlanetConfig,
    insolation: Mapping[CellId, float],
    sea_mask: SeaMask,
) -> tuple[dict[CellId, float], dict[CellId, bool]]:
    """Iterate the ice-albedo feedback to a stable (T, ice) pair.

    Starts ice-free, freezes cells below 0 degC, recomputes with the
    higher albedo, and repeats a few rounds — enough for the feedback
    to lock in polar caps without hunting for exact fixed points.
    """
    ice = {cell: False for cell in insolation}
    temps = temperature_field(grid, planet, insolation, sea_mask, ice)
    for _ in range(_ICE_FEEDBACK_ITERATIONS):
        ice = {cell: temps[cell] < FREEZING_POINT_K for cell in temps}
        temps = temperature_field(grid, planet, insolation, sea_mask, ice)
    return temps, ice
