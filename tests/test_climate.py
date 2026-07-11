"""Qualitative climate verification: emergence checked, not scripted.

Bands, seasonal cycles, ice caps, rain shadows, and regime differences
between planets must *fall out of the physics*; these tests assert the
qualitative structure the design doc names for each regime.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

pytest.importorskip("h3")

from adapters.grid_registry import create_grid
from adapters.rng_seeded import SeededRng
from core.climate.insolation import insolation_field
from core.climate.model import ClimateState, simulate_climate
from core.climate.temperature import solve_temperature_with_ice
from core.climate.wind import zonal_band_wind
from core.hydrology.sea_mask import SeaMask, build_sea_mask
from core.sim.planet_config import PlanetConfig
from core.sim.presets import earth, mars, tidally_locked_ocean, venus
from core.topography.procedural import ProceduralTopography
from ports.grid import CellId, Grid

_EarthBundle = tuple[Grid, SeaMask, "ClimateState"]

_SEED = 42
_RES = 1  # 842 cells: coarse but structured


def _world(planet: PlanetConfig) -> tuple[Grid, dict[CellId, float], SeaMask]:
    grid = create_grid("h3", resolution=_RES, radius_m=planet.radius_m)
    heights = dict(ProceduralTopography(SeededRng(_SEED)).heights(grid))
    mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    return grid, heights, mask


@pytest.fixture(scope="module")
def earth_climate() -> _EarthBundle:
    """Simulate the Earth preset once for all Earth assertions."""
    planet = earth()
    grid, heights, mask = _world(planet)
    return grid, mask, simulate_climate(grid, planet, heights, mask)


def test_earth_equator_warmer_than_poles(earth_climate: _EarthBundle) -> None:
    grid, _mask, state = earth_climate
    tropics = []
    polar = []
    for cell, temp in state.annual_mean_temperature_k.items():
        lat = abs(grid.centroid(cell).lat_deg)
        if lat < 15.0:
            tropics.append(temp)
        elif lat > 70.0:
            polar.append(temp)
    contrast = sum(tropics) / len(tropics) - sum(polar) / len(polar)
    assert contrast > 15.0


def test_earth_has_polar_ice_but_not_a_snowball(earth_climate: _EarthBundle) -> None:
    grid, _mask, state = earth_climate
    icy = [c for c, frozen in state.permanent_ice.items() if frozen]
    assert icy, "expected some permanent ice"
    assert len(icy) < len(state.permanent_ice) // 2, "half the planet froze"
    mean_ice_lat = sum(abs(grid.centroid(c).lat_deg) for c in icy) / len(icy)
    assert mean_ice_lat > 45.0


def test_earth_ocean_buffers_seasons_more_than_land(earth_climate: _EarthBundle) -> None:
    _grid, mask, state = earth_climate

    def swing(cell: str) -> float:
        temps = [s.temperature_k[cell] for s in state.seasons]
        return max(temps) - min(temps)

    ocean_swings = [swing(c) for c, wet in mask.ocean.items() if wet]
    land_swings = [swing(c) for c, wet in mask.ocean.items() if not wet]
    assert sum(ocean_swings) / len(ocean_swings) < sum(land_swings) / len(land_swings)


def test_earth_land_seasonality_reverses_between_hemispheres(earth_climate: _EarthBundle) -> None:
    grid, mask, state = earth_climate
    north_phase = _warmest_phase(grid, state, mask, hemisphere=+1)
    south_phase = _warmest_phase(grid, state, mask, hemisphere=-1)
    assert north_phase != south_phase


def _warmest_phase(grid: Grid, state: ClimateState, mask: SeaMask, hemisphere: int) -> float:
    """Return the orbit phase whose mid-latitude land is warmest."""
    best: tuple[float, float] = (-math.inf, 0.0)
    for season in state.seasons:
        temps = [
            season.temperature_k[c]
            for c in season.temperature_k
            if not mask.ocean[c] and 20.0 < hemisphere * grid.centroid(c).lat_deg < 60.0
        ]
        if temps:
            best = max(best, (sum(temps) / len(temps), season.orbit_phase))
    return best[1]


def test_earth_fields_are_finite_and_plausible(earth_climate: _EarthBundle) -> None:
    _grid, _mask, state = earth_climate
    temps = state.annual_mean_temperature_k.values()
    assert all(math.isfinite(t) and 150.0 < t < 350.0 for t in temps)
    rains = state.annual_precipitation_mm_yr.values()
    assert all(math.isfinite(r) and r >= 0.0 for r in rains)
    assert max(rains) > 0.0


def test_tidally_locked_world_has_an_eye_not_bands() -> None:
    planet = tidally_locked_ocean()
    grid, heights, mask = _world(planet)
    state = simulate_climate(grid, planet, heights, mask, season_count=2)
    day, night = [], []
    for cell, temp in state.annual_mean_temperature_k.items():
        lon = abs(grid.centroid(cell).lon_deg)
        if lon < 40.0:
            day.append(temp)
        elif lon > 140.0:
            night.append(temp)
    assert sum(day) / len(day) - sum(night) / len(night) > 30.0


def test_thin_air_swings_harder_than_thick_air() -> None:
    grid = create_grid("h3", resolution=0)
    heights = ProceduralTopography(SeededRng(_SEED)).heights(grid)

    def spread(planet: PlanetConfig) -> float:
        mask = build_sea_mask(grid, heights, planet.ocean_fraction)
        state = simulate_climate(grid, planet, heights, mask, season_count=2)
        temps = state.annual_mean_temperature_k.values()
        return max(temps) - min(temps)

    assert spread(mars()) > spread(venus())


def test_retrograde_spin_reverses_the_bands() -> None:
    prograde = earth()
    retrograde = venus()
    lat = 45.0
    east_prograde = zonal_band_wind(lat, prograde)[0]
    east_retrograde = zonal_band_wind(lat, retrograde)[0]
    assert east_prograde * east_retrograde < 0.0


def test_slow_rotation_kills_the_bands() -> None:
    assert zonal_band_wind(45.0, tidally_locked_ocean()) == (0.0, 0.0)


def test_orographic_shadow_on_real_grid() -> None:
    """A meridional ridge on a westerly-wind world: wet west, dry east."""
    from core.climate.moisture import MoistureInputs, precipitation_field

    planet = earth()
    grid = create_grid("h3", resolution=1, radius_m=planet.radius_m)
    heights = {}
    for cell in grid.cells():
        lon = grid.centroid(cell).lon_deg
        heights[cell] = 3_000.0 if 0.0 <= lon <= 12.0 else 0.0
    mask = build_sea_mask(grid, heights, ocean_fraction=0.0)
    temps = dict.fromkeys(heights, 295.0)
    ice = dict.fromkeys(heights, False)
    winds = dict.fromkeys(heights, (7.0, 0.0))  # pure westerlies

    rain = precipitation_field(
        grid,
        MoistureInputs(temps=temps, winds=winds, heights_m=heights, sea_mask=mask, ice=ice),
    )

    def band_mean(lon_min: float, lon_max: float) -> float:
        values = [
            rain[c]
            for c in rain
            if lon_min <= grid.centroid(c).lon_deg <= lon_max
            and abs(grid.centroid(c).lat_deg) < 45.0
        ]
        return sum(values) / len(values)

    windward = band_mean(-14.0, 0.0)
    lee = band_mean(12.0, 26.0)
    assert windward > lee * 1.5


def test_earth_has_the_three_zonal_wind_bands() -> None:
    """Tropics, mid-latitudes, and poles must each drive a different band."""
    planet = earth()
    tropical_east = zonal_band_wind(15.0, planet)[0]
    mid_east = zonal_band_wind(45.0, planet)[0]
    polar_east = zonal_band_wind(75.0, planet)[0]
    assert tropical_east < 0.0, "expected Hadley-cell trade easterlies"
    assert mid_east > 0.0, "expected Ferrel-cell westerlies"
    assert polar_east < 0.0, "expected polar easterlies"


@pytest.mark.slow
def test_atmosphere_thickness_alone_sets_the_day_night_swing() -> None:
    """Same world, same everything, only surface pressure differs."""
    grid = create_grid("h3", resolution=0)
    heights = ProceduralTopography(SeededRng(_SEED)).heights(grid)
    base = earth()

    def with_pressure(factor: float, name: str) -> PlanetConfig:
        atmosphere = dataclasses.replace(
            base.atmosphere, surface_pressure_pa=base.atmosphere.surface_pressure_pa * factor
        )
        return dataclasses.replace(base, name=name, atmosphere=atmosphere)

    thin = with_pressure(0.1, "thin-air Earth")
    thick = with_pressure(5.0, "thick-air Earth")

    def spread(planet: PlanetConfig) -> float:
        mask = build_sea_mask(grid, heights, planet.ocean_fraction)
        state = simulate_climate(grid, planet, heights, mask, season_count=2)
        temps = state.annual_mean_temperature_k.values()
        return max(temps) - min(temps)

    assert spread(thin) > spread(thick)


@pytest.mark.slow
def test_ice_albedo_feedback_amplifies_cooling() -> None:
    """Cooling the same insolation field grows ice AND drops mean temperature together."""
    planet = earth()
    grid = create_grid("h3", resolution=0)
    heights = ProceduralTopography(SeededRng(_SEED)).heights(grid)
    mask = build_sea_mask(grid, heights, planet.ocean_fraction)
    base_insolation = insolation_field(grid, planet, orbit_phase=0.25)
    warm_insolation = {cell: value * 1.15 for cell, value in base_insolation.items()}
    cold_insolation = {cell: value * 0.85 for cell, value in base_insolation.items()}

    warm_temps, warm_ice = solve_temperature_with_ice(grid, planet, warm_insolation, mask)
    cold_temps, cold_ice = solve_temperature_with_ice(grid, planet, cold_insolation, mask)

    def ice_fraction(ice: dict[CellId, bool]) -> float:
        return sum(1 for frozen in ice.values() if frozen) / len(ice)

    def mean_temp(temps: dict[CellId, float]) -> float:
        return sum(temps.values()) / len(temps)

    assert ice_fraction(cold_ice) > ice_fraction(warm_ice)
    assert mean_temp(cold_temps) < mean_temp(warm_temps)
