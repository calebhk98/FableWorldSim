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
from core.climate.temperature import solve_temperature_with_ice, temperature_field
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


def test_convergence_zone_rains_more_than_divergence_zone() -> None:
    """A real wind-divergence field, not a flat per-hop fraction, drives rain.

    An all-ocean, flat, uniform-temperature world isolates the
    convergence term from evaporation/orographic effects: winds blowing
    toward the same meridian converge there (piling moisture up); the
    mirrored pair blowing away from it diverges. The old model couldn't
    tell these apart -- its "convergence" was one constant fraction
    regardless of wind (issue #20).
    """
    from core.climate.moisture import MoistureInputs, precipitation_field

    planet = earth()
    grid = create_grid("h3", resolution=1, radius_m=planet.radius_m)
    heights = dict.fromkeys(grid.cells(), 0.0)
    mask = build_sea_mask(grid, heights, ocean_fraction=1.0)
    temps = dict.fromkeys(heights, 295.0)
    ice = dict.fromkeys(heights, False)

    def rain_for(inward: bool) -> dict[CellId, float]:
        sign = -1.0 if inward else 1.0
        winds = {
            cell: (sign * (-1.0 if grid.centroid(cell).lon_deg < 0.0 else 1.0) * 7.0, 0.0)
            for cell in grid.cells()
        }
        inputs = MoistureInputs(temps=temps, winds=winds, heights_m=heights, sea_mask=mask, ice=ice)
        return precipitation_field(grid, inputs)

    def band_mean(rain: dict[CellId, float]) -> float:
        values = [
            rain[c]
            for c in rain
            if -12.0 <= grid.centroid(c).lon_deg <= 12.0 and abs(grid.centroid(c).lat_deg) < 45.0
        ]
        return sum(values) / len(values)

    assert band_mean(rain_for(inward=True)) > band_mean(rain_for(inward=False)) * 5.0


def test_lake_surface_evaporates_like_open_water() -> None:
    """A lake cell should contribute humidity like the ocean, not like bare
    land, so a lake shows up as a moisture source for downwind precipitation
    (the climate side of issue #12's lake wiring)."""
    from core.climate.moisture import evaporation_field

    cells = ["land", "lake"]
    temps = dict.fromkeys(cells, 295.0)
    ice = dict.fromkeys(cells, False)
    mask = SeaMask(
        sea_level_m=0.0, ocean=dict.fromkeys(cells, False), intertidal=dict.fromkeys(cells, False)
    )

    bare = evaporation_field(temps, mask, ice)
    assert bare["land"] == bare["lake"], "identical cells evaporate alike with no lake_mask"

    watered = evaporation_field(temps, mask, ice, lake_mask={"lake": True})
    assert watered["land"] == bare["land"], "a non-lake cell is unaffected"
    assert watered["lake"] > bare["lake"], "a lake cell should evaporate more than bare land"


def test_precipitation_field_gets_extra_moisture_from_a_lake() -> None:
    """A lake upwind of a dry stretch should measurably wet it more than an
    otherwise identical, lake-free run of the same terrain."""
    from core.climate.moisture import MoistureInputs, precipitation_field

    planet = earth()
    grid = create_grid("h3", resolution=1, radius_m=planet.radius_m)
    heights = dict.fromkeys(grid.cells(), 0.0)
    mask = build_sea_mask(grid, heights, ocean_fraction=0.0)
    temps = dict.fromkeys(heights, 295.0)
    ice = dict.fromkeys(heights, False)
    winds = dict.fromkeys(heights, (7.0, 0.0))  # pure westerlies
    lake_mask = {next(iter(heights)): True}

    def total_rain(lakes: dict[CellId, bool] | None) -> float:
        rain = precipitation_field(
            grid,
            MoistureInputs(
                temps=temps,
                winds=winds,
                heights_m=heights,
                sea_mask=mask,
                ice=ice,
                lake_mask=lakes if lakes is not None else {},
            ),
        )
        return sum(rain.values())

    assert total_rain(lake_mask) > total_rain(None)


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


def _mean_no_ice_temp(atmosphere_overrides: dict[str, object]) -> float:
    """Return an Earth-based world's mean temperature for one atmosphere tweak.

    Ice is pinned off so the (separately emergent, separately tested)
    ice-albedo feedback can't add its own nonlinearity on top of the one
    atmosphere.* field under test.
    """
    base = earth()
    grid = create_grid("h3", resolution=0, radius_m=base.radius_m)
    heights = ProceduralTopography(SeededRng(_SEED)).heights(grid)
    mask = build_sea_mask(grid, heights, base.ocean_fraction)
    ice = dict.fromkeys(grid.cells(), False)
    atmosphere = dataclasses.replace(base.atmosphere, **atmosphere_overrides)
    planet = dataclasses.replace(base, atmosphere=atmosphere)
    insolation = insolation_field(grid, planet, orbit_phase=0.0)
    temps = temperature_field(grid, planet, insolation, mask, ice)
    return sum(temps.values()) / len(temps)


def test_greenhouse_warming_increases_monotonically_with_co2() -> None:
    """More CO2 at fixed pressure must raise mean temperature.

    Only ``composition`` changes between runs -- no per-preset
    ``greenhouse_offset_k`` literal is involved anywhere in this loop
    (issue #19): the warming has to fall out of
    ``Atmosphere.greenhouse_offset_k`` deriving more optical depth from
    more absorber, not from a hand-tuned number.
    """
    fractions = [0.0, 0.001, 0.01, 0.1, 0.5, 0.95]
    means = [_mean_no_ice_temp({"composition": {"N2": 1.0 - f, "CO2": f}}) for f in fractions]
    assert means == sorted(means), means
    assert means[-1] > means[0] + 5.0, "expected a meaningful CO2-driven warming spread"


def test_greenhouse_warming_increases_monotonically_with_pressure() -> None:
    """More surface pressure at fixed composition must raise mean temperature.

    Companion to the CO2 test above (issue #19): the same CO2-bearing
    mix under a thicker column has more absorber in its path (higher
    partial pressure), so a thicker atmosphere warms more -- Venus-thick
    versus Mars-thin falls out of this, not a literal per-planet.
    """
    pressures = [0.0, 1_000.0, 10_000.0, 101_325.0, 1_000_000.0]
    means = [_mean_no_ice_temp({"surface_pressure_pa": p}) for p in pressures]
    assert means == sorted(means), means
    assert means[-1] > means[0] + 5.0, "expected a meaningful pressure-driven warming spread"


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
