"""Qualitative ocean emergence: Stommel western-boundary currents and monsoons.

Companion to ``tests/test_climate.py``: these assert structure that must
*fall out of the physics* rather than being scripted, per the
verification checklist -- a western-boundary current (Gulf
Stream/Kuroshio analogue) intensifying on the poleward, land-backed
margin of an ocean basin, and a seasonal monsoon wind reversal driven
by land-sea thermal contrast.
"""

from __future__ import annotations

import pytest

pytest.importorskip("h3")

from adapters.grid_registry import create_grid
from core.climate.model import SeasonClimate, simulate_climate
from core.climate.wind import zonal_band_wind
from core.grid.vectors import unit_direction
from core.hydrology.sea_mask import SeaMask, build_sea_mask
from core.ocean.surface import SurfaceOcean
from core.sim.presets import earth
from ports.grid import CellId, Grid
from ports.ocean import OceanForcing

_RES = 1  # 842 cells: coarse but structured, matches tests/test_climate.py
# Bulk aerodynamic drag scale (Pa per m/s of wind), same order of magnitude as
# core.climate.model's air-sea coupling constant.
_STRESS_PER_WIND_SPEED_PA = 0.02


def _has_land_to_east(grid: Grid, mask: SeaMask, cell: CellId) -> bool:
    """Mirror ``SurfaceOcean._has_land_to_west`` but look eastward.

    Used to pick the *opposite* margin of the basin -- open ocean cells
    whose nearest land is behind them (to the east) rather than ahead of
    them (to the west) -- as the non-intensified control group.
    """
    origin = grid.centroid(cell)
    best: tuple[float, CellId] | None = None
    for neighbor in grid.neighbors(cell):
        direction = unit_direction(origin, grid.centroid(neighbor), grid.radius_m)
        eastness = direction[0]
        if best is None or eastness > best[0]:
            best = (eastness, neighbor)
    return best is not None and best[0] > 0.0 and mask.is_land(best[1])


@pytest.mark.slow
def test_western_boundary_current_warms_the_poleward_coast() -> None:
    """A land margin on the WEST of a basin should intensify poleward flow.

    Unit-level test on ``SurfaceOcean`` directly (a full simulate_climate
    pass at this resolution is too noisy for a robust signal): a thin
    north-south continent splits an otherwise all-ocean sphere. Ocean
    cells immediately east of that continent have land to their west
    (the Stommel western-boundary condition); ocean cells immediately
    west of the continent -- same latitudes, same distance from the
    coast -- have land only to their east (the non-intensified control
    margin).

    The forcing is the *realistic* zonal wind pattern (trade easterlies
    giving way to westerlies near 30 deg, the same ``zonal_band_wind``
    the climate model uses) rather than a scripted poleward push -- no
    north-south wind component is imposed at all. That shear is exactly
    the wind-stress curl that should drive an equatorward Sverdrup
    interior in the subtropics; mass conservation then has to close that
    transport somewhere, and it must close through the western margin,
    not the eastern one, purely because of the land mask. The result
    should be the real Gulf-Stream signature: the western margin
    develops a much stronger poleward (north) current than the eastern
    margin, emergent from the physics rather than a scripted factor, and
    carries anomalously warm water into its own poleward reach while the
    eastern margin does not.
    """
    planet = earth()
    grid = create_grid("h3", resolution=_RES, radius_m=planet.radius_m)

    # A thin north-south continent near the prime meridian; ocean covers
    # the rest of the sphere as one wraparound basin.
    land = {cell: -5.0 <= grid.centroid(cell).lon_deg <= 5.0 for cell in grid.cells()}
    mask = SeaMask(
        sea_level_m=0.0,
        ocean={cell: not is_land for cell, is_land in land.items()},
        intertidal=dict.fromkeys(land, False),
    )

    # Realistic equator-warm / pole-cold SST profile.
    initial_sst = {
        cell: 300.0 - 0.6 * abs(grid.centroid(cell).lat_deg)
        for cell in grid.cells()
        if mask.ocean[cell]
    }
    ocean = SurfaceOcean(grid, planet, mask, initial_sst)

    west_margin = [cell for cell in initial_sst if ocean._has_land_to_west(cell)]
    east_margin = [
        cell
        for cell in initial_sst
        if _has_land_to_east(grid, mask, cell) and not ocean._has_land_to_west(cell)
    ]
    assert west_margin, "expected ocean cells with land to their west"
    assert east_margin, "expected ocean cells with land to their east"

    # Zonal-only wind stress (east/west, no imposed north component) --
    # the only asymmetry between the two margins is the land mask, and
    # any poleward current has to come from Ekman + Sverdrup + the
    # boundary-layer closure, not from a hand-fed north stress.
    forcing = OceanForcing(
        wind_stress_east_pa={
            cell: _STRESS_PER_WIND_SPEED_PA
            * zonal_band_wind(grid.centroid(cell).lat_deg, planet)[0]
            for cell in initial_sst
        },
        wind_stress_north_pa=dict.fromkeys(initial_sst, 0.0),
        surface_heat_flux_w_m2=dict.fromkeys(initial_sst, 0.0),
        ocean_mask=dict.fromkeys(initial_sst, True),
    )
    dt_s = planet.orbital_period_s / 4.0
    for _ in range(3):
        ocean.step(forcing, dt_s)
    surface = ocean.surface()

    def mean_north_current(cells: list[CellId]) -> float:
        return sum(surface.current_north_m_s[c] for c in cells) / len(cells)

    # The subtropical-gyre latitudes, where the trades give way to the
    # westerlies (near 30 deg) -- the same band the heat check below
    # uses.
    def gyre_band(cells: list[CellId]) -> list[CellId]:
        return [c for c in cells if 20.0 <= grid.centroid(c).lat_deg <= 65.0]

    west_current = mean_north_current(gyre_band(west_margin))
    east_current = mean_north_current(gyre_band(east_margin))
    assert west_current > 0.0, f"expected a poleward western boundary current, got {west_current}"
    assert west_current > 2.0 * abs(east_current), (
        f"expected emergent western-boundary intensification, got west={west_current} "
        f"east={east_current}"
    )

    # The intensified current should also carry anomalously warm water
    # poleward on the western margin, much more so than the eastern
    # margin (which only carries the weak, unintensified interior flow).
    heat = ocean.heat_transport_to_atmosphere_w_m2()

    def poleward_band_mean_heat(cells: list[CellId]) -> float:
        band = [heat[c] for c in cells if 20.0 <= grid.centroid(c).lat_deg <= 65.0]
        return sum(band) / len(band)

    west_heat = poleward_band_mean_heat(west_margin)
    east_heat = poleward_band_mean_heat(east_margin)
    assert west_heat > 20.0, (
        f"expected a strong warm anomaly on the western margin, got {west_heat}"
    )
    assert east_heat < west_heat / 3.0, (
        f"expected the western margin far warmer than the control margin, got west={west_heat} "
        f"east={east_heat}"
    )


def test_western_boundary_layer_responds_to_wind_forcing() -> None:
    """Western-boundary current strength must respond to wind, not be hardcoded.

    Regression guard against a constant multiplier (e.g., hardcoded north *= 3.0)
    creeping back into the western-boundary layer closure. The WBC magnitude
    must be an *emergent* property of mass conservation and area ratio, not a
    fixed factor: doubling the wind stress should roughly double the Sverdrup
    interior transport and thus the required boundary-layer closure current,
    and zero wind should produce near-zero boundary current regardless of the
    basin geometry.
    """
    planet = earth()
    grid = create_grid("h3", resolution=_RES, radius_m=planet.radius_m)

    # A thin north-south continent: same setup as
    # test_western_boundary_current_warms_the_poleward_coast.
    land = {cell: -5.0 <= grid.centroid(cell).lon_deg <= 5.0 for cell in grid.cells()}
    mask = SeaMask(
        sea_level_m=0.0,
        ocean={cell: not is_land for cell, is_land in land.items()},
        intertidal=dict.fromkeys(land, False),
    )
    initial_sst = {
        cell: 300.0 - 0.6 * abs(grid.centroid(cell).lat_deg)
        for cell in grid.cells()
        if mask.ocean[cell]
    }

    # Cells on the western margin (land to the west) in the subtropical band.
    ocean = SurfaceOcean(grid, planet, mask, initial_sst.copy())
    west_margin = [
        cell
        for cell in initial_sst
        if ocean._has_land_to_west(cell) and 20.0 <= grid.centroid(cell).lat_deg <= 65.0
    ]
    assert west_margin, "expected western-margin cells in subtropical band"

    def run_scenario(wind_stress_scale: float) -> float:
        """Run ocean with scaled wind stress; return mean WBC north speed."""
        ocean = SurfaceOcean(grid, planet, mask, initial_sst.copy())
        base_east_wind = {
            cell: _STRESS_PER_WIND_SPEED_PA
            * zonal_band_wind(grid.centroid(cell).lat_deg, planet)[0]
            for cell in initial_sst
        }
        scaled_wind = {cell: stress * wind_stress_scale for cell, stress in base_east_wind.items()}
        forcing = OceanForcing(
            wind_stress_east_pa=scaled_wind,
            wind_stress_north_pa=dict.fromkeys(initial_sst, 0.0),
            surface_heat_flux_w_m2=dict.fromkeys(initial_sst, 0.0),
            ocean_mask=dict.fromkeys(initial_sst, True),
        )
        dt_s = planet.orbital_period_s / 4.0
        for _ in range(3):
            ocean.step(forcing, dt_s)
        surface = ocean.surface()
        return sum(abs(surface.current_north_m_s[c]) for c in west_margin) / len(west_margin)

    normal_wbc = run_scenario(1.0)
    doubled_wbc = run_scenario(2.0)
    zero_wbc = run_scenario(0.0)

    # Doubled wind should increase the boundary current (at least 1.1x, to guard
    # against a hardcoded constant that ignores forcing). Allow up to 3x to account
    # for nonlinear speed clamping and model discretization noise.
    assert doubled_wbc > 1.1 * normal_wbc, (
        f"doubled wind should increase WBC, got normal={normal_wbc:.4f}, doubled={doubled_wbc:.4f}"
    )
    assert doubled_wbc < 3.0 * normal_wbc, (
        f"WBC scaling should stay sub-linear due to speed clamping, got normal={normal_wbc:.4f}, "
        f"doubled={doubled_wbc:.4f}"
    )
    # Zero wind should produce negligible boundary current (< 5% of normal).
    assert zero_wbc < 0.05 * normal_wbc, (
        f"zero wind should produce negligible WBC, got normal={normal_wbc:.4f}, zero={zero_wbc:.4f}"
    )


def test_seasonal_monsoon_reverses_the_coastal_wind() -> None:
    """A continent's coast should flip from onshore to offshore wind.

    Land fills the eastern hemisphere, ocean the western hemisphere.
    Along the west-facing coast (land immediately east of the ocean),
    "onshore" is a positive east-wind component (blowing from ocean to
    land) and "offshore" is negative. In the hemisphere's own summer
    (its warmest season) the heated continent should pull maritime air
    onshore; in its own winter (coldest season) the cooled continent
    should push air offshore -- the thermal-low / thermal-high reversal
    that drives real monsoons.
    """
    planet = earth()
    grid = create_grid("h3", resolution=_RES, radius_m=planet.radius_m)

    heights = {
        cell: 500.0 if 0.0 <= grid.centroid(cell).lon_deg <= 180.0 else -500.0
        for cell in grid.cells()
    }
    # ocean_fraction is set just under the true area split (~0.497) so the
    # binary height field resolves cleanly to the western hemisphere,
    # rather than the solver flooding the land plateau too.
    mask = build_sea_mask(grid, heights, ocean_fraction=0.45)
    assert mask.sea_level_m == -500.0, "expected the ocean band alone to be flooded"

    state = simulate_climate(grid, planet, heights, mask, season_count=4)

    coastal = [
        cell
        for cell in grid.cells()
        if mask.is_land(cell)
        and 0.0 <= grid.centroid(cell).lon_deg <= 20.0
        and 10.0 <= grid.centroid(cell).lat_deg <= 30.0
    ]
    assert coastal, "expected a coastal land band near the west-facing shore"

    def band_mean_temp(season: SeasonClimate) -> float:
        return sum(season.temperature_k[c] for c in coastal) / len(coastal)

    def band_mean_onshore(season: SeasonClimate) -> float:
        return sum(season.wind[c][0] for c in coastal) / len(coastal)

    summer = max(state.seasons, key=band_mean_temp)
    winter = min(state.seasons, key=band_mean_temp)
    assert summer is not winter

    summer_onshore = band_mean_onshore(summer)
    winter_onshore = band_mean_onshore(winter)
    assert summer_onshore > 0.0, f"expected onshore flow in summer, got {summer_onshore}"
    assert winter_onshore < 0.0, f"expected offshore flow in winter, got {winter_onshore}"

    # The reversal should hold cell-by-cell, not just on average.
    for cell in coastal:
        assert summer.wind[cell][0] > 0.0, f"{cell} not onshore in summer"
        assert winter.wind[cell][0] < 0.0, f"{cell} not offshore in winter"
