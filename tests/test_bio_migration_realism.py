"""Migration-realism additions: oxygen altitude, seasonal extremes, long-range pull.

Three independent behaviours, each documented in the module it lives in:

* :data:`core.biology.suitability.OXYGEN_AXIS` — altitude-derived partial O2
  pressure, opt-in via ``[traits.oxygen_kpa]``.
* :data:`core.biology.suitability.COLDEST_SEASON_AXIS` — a cell's coldest
  seasonal mean, distinct from the annual mean, opt-in via
  ``[traits.coldest_month_temperature_k]``.
* :func:`core.biology.migration.seasonal_pull` — the bounded multi-hop
  seasonal redistribution a ``seasonal_migration=True`` organism gets on
  top of local :func:`core.biology.migration.diffuse`.
"""

from __future__ import annotations

import pytest

from adapters.rng_seeded import SeededRng
from core.biology.bands import EnvBand
from core.biology.context import BiologyContext, BiologyParams, LayersBelow, build_biology_context
from core.biology.engine import step_biology
from core.biology.state import WorldBiologyState
from core.biology.suitability import (
    COLDEST_SEASON_AXIS,
    OXYGEN_AXIS,
    TEMPERATURE_AXIS,
    TEMPERATURE_RANGE_AXIS,
    SurfaceEnvironment,
    surface_suitability,
)
from core.climate.model import ClimateState, SeasonClimate
from core.sim.constants import SECONDS_PER_YEAR
from tests.bio_helpers import all_land_mask, make_organism
from tests.civ_helpers import FakeGrid

_CALM_WIND = (0.0, 0.0)


def _season(cells: list[str], temperature_k: dict[str, float], phase: float) -> SeasonClimate:
    """Build one bare-bones season over ``cells`` (precip/ice/wind are inert)."""
    return SeasonClimate(
        orbit_phase=phase,
        temperature_k=temperature_k,
        precipitation_mm_yr=dict.fromkeys(cells, 800.0),
        ice=dict.fromkeys(cells, False),
        wind=dict.fromkeys(cells, _CALM_WIND),
    )


# ---------------------------------------------------------------------------
# 1) Oxygen axis
# ---------------------------------------------------------------------------


def test_oxygen_axis_excludes_thin_air_but_not_sea_level() -> None:
    grid = FakeGrid(rows=1, cols=2)
    mask = all_land_mask(grid)
    sea_level, peak = list(grid.cells())
    heights = {sea_level: 0.0, peak: 6000.0}
    # Same temperature everywhere so it is not the limiting factor.
    temps = dict.fromkeys(heights, 290.0)
    climate = ClimateState(
        seasons=(_season(list(heights), temps, 0.0),),
        annual_mean_temperature_k=temps,
        annual_precipitation_mm_yr=dict.fromkeys(heights, 800.0),
        permanent_ice=dict.fromkeys(heights, False),
        sea_surface_temp_k={},
    )
    below = LayersBelow(grid, climate, mask, heights, dict.fromkeys(heights, ""))
    lowlander = make_organism(
        "lowlander",
        traits={
            OXYGEN_AXIS: EnvBand(
                comfort_min=18.0, comfort_max=21.5, tolerance_min=14.0, tolerance_max=25.0
            )
        },
    )
    ctx = build_biology_context(below, {"lowlander": lowlander})
    environment = SurfaceEnvironment(
        axis_fields=ctx.axis_fields, biome_field=ctx.biome_field, sea_mask=ctx.sea_mask
    )
    suitability = surface_suitability(lowlander, list(grid.cells()), environment)
    assert suitability[sea_level] == pytest.approx(1.0)
    assert suitability.get(peak, 0.0) == 0.0


def test_build_biology_context_populates_oxygen_from_the_barometric_formula() -> None:
    grid = FakeGrid(rows=1, cols=1)
    mask = all_land_mask(grid)
    cell = next(iter(grid.cells()))
    heights = {cell: 8000.0}  # exactly one scale height up.
    temps = {cell: 290.0}
    climate = ClimateState(
        seasons=(_season(list(heights), temps, 0.0),),
        annual_mean_temperature_k=temps,
        annual_precipitation_mm_yr=dict.fromkeys(heights, 800.0),
        permanent_ice=dict.fromkeys(heights, False),
        sea_surface_temp_k={},
    )
    below = LayersBelow(grid, climate, mask, heights, dict.fromkeys(heights, ""))
    ctx = build_biology_context(below, {})
    # 21 kPa * exp(-1) — one scale height thins the air to ~1/e of sea level.
    assert ctx.axis_fields[OXYGEN_AXIS][cell] == pytest.approx(21.0 * 0.36787944117, rel=1e-6)


# ---------------------------------------------------------------------------
# 2) Seasonal-extreme axis
# ---------------------------------------------------------------------------


def test_seasonal_extreme_excludes_cold_intolerant_species_from_the_high_swing_cell() -> None:
    grid = FakeGrid(rows=1, cols=2)
    mask = all_land_mask(grid)
    stable, swingy = list(grid.cells())
    cells = [stable, swingy]
    # Both cells share an annual mean of 290 K; only swingy has a hard winter.
    per_season_temps = [
        {stable: 290.0, swingy: 260.0},
        {stable: 290.0, swingy: 290.0},
        {stable: 290.0, swingy: 320.0},
        {stable: 290.0, swingy: 290.0},
    ]
    seasons = tuple(
        _season(cells, temps, phase=i / len(per_season_temps))
        for i, temps in enumerate(per_season_temps)
    )
    annual_mean = {c: sum(s.temperature_k[c] for s in seasons) / len(seasons) for c in cells}
    assert annual_mean[stable] == pytest.approx(annual_mean[swingy])
    climate = ClimateState(
        seasons=seasons,
        annual_mean_temperature_k=annual_mean,
        annual_precipitation_mm_yr=dict.fromkeys(cells, 800.0),
        permanent_ice=dict.fromkeys(cells, False),
        sea_surface_temp_k={},
    )
    below = LayersBelow(grid, climate, mask, dict.fromkeys(cells, 100.0), dict.fromkeys(cells, ""))
    cold_intolerant = make_organism(
        "cold_intolerant",
        traits={
            COLDEST_SEASON_AXIS: EnvBand(
                comfort_min=275.0, comfort_max=310.0, tolerance_min=265.0, tolerance_max=320.0
            )
        },
    )
    ctx = build_biology_context(below, {"cold_intolerant": cold_intolerant})
    environment = SurfaceEnvironment(
        axis_fields=ctx.axis_fields, biome_field=ctx.biome_field, sea_mask=ctx.sea_mask
    )
    suitability = surface_suitability(cold_intolerant, cells, environment)
    assert suitability[stable] == pytest.approx(1.0)
    assert suitability.get(swingy, 0.0) == 0.0


def test_build_biology_context_degrades_gracefully_with_a_single_season() -> None:
    grid = FakeGrid(rows=1, cols=1)
    mask = all_land_mask(grid)
    cell = next(iter(grid.cells()))
    temps = {cell: 290.0}
    climate = ClimateState(
        seasons=(_season([cell], temps, 0.0),),
        annual_mean_temperature_k=temps,
        annual_precipitation_mm_yr={cell: 800.0},
        permanent_ice={cell: False},
        sea_surface_temp_k={},
    )
    below = LayersBelow(grid, climate, mask, {cell: 0.0}, {cell: ""})
    ctx = build_biology_context(below, {})
    # With one season, "coldest" is just that season, and the range is zero.
    assert ctx.axis_fields[COLDEST_SEASON_AXIS][cell] == pytest.approx(290.0)
    assert ctx.axis_fields[TEMPERATURE_RANGE_AXIS][cell] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3) Long-range seasonal pull
# ---------------------------------------------------------------------------


_LineContext = tuple[dict[str, dict[str, float]], dict[str, str], dict[str, float]]


def _line_context(cells: list[str]) -> _LineContext:
    """Return (axis_fields, biome_field, dryness) for a 10-cell suitability ramp.

    Temperature climbs steadily from cell 0 (out of tolerance, suitability
    0) to cell 9 (deep in comfort, suitability 1), so both organisms face
    an identical gradient pulling them from a bad starting cell toward a
    good distant one.
    """
    temps = {cell: 260.0 + _col(cell) * 4.0 for cell in cells}
    axis_fields = {TEMPERATURE_AXIS: temps}
    biome_field = dict.fromkeys(cells, "")
    dryness = dict.fromkeys(cells, 0.0)
    return axis_fields, biome_field, dryness


def _col(cell: str) -> int:
    return int(cell.split("c")[1])


def _center_of_mass(field: dict[str, float]) -> float | None:
    """Return the population-weighted mean column, or None if the field is empty."""
    total = sum(field.values())
    if total <= 0.0:
        return None
    return sum(_col(cell) * value for cell, value in field.items()) / total


@pytest.mark.slow
def test_seasonal_migration_outruns_local_diffusion_under_the_same_gradient() -> None:
    """A migratory species' population-center-of-mass moves farther, faster.

    Observed with seed 42 over a 1x10 line, both species starting entirely
    in the coldest (suitability-0) cell, under an identical warming
    gradient (see :func:`_line_context`)::

        tick  resident_com  migrant_com
        0     0.10          0.97
        1     0.41          2.19
        2     0.90          3.17
        3     1.50          4.06

    The gap opens immediately and only widens, so a 3-tick run with a
    generous margin is a robust, non-flaky assertion of the effect.
    """
    grid = FakeGrid(rows=1, cols=10)
    cells = list(grid.cells())
    mask = all_land_mask(grid)
    axis_fields, biome_field, dryness = _line_context(cells)
    band = EnvBand(comfort_min=290.0, comfort_max=305.0, tolerance_min=260.0, tolerance_max=330.0)
    resident = make_organism(
        "resident",
        crowding_cap_per_m2=1.0,
        reproduction="asexual",
        min_viable_population=1,
        traits={TEMPERATURE_AXIS: band},
        seasonal_migration=False,
    )
    migrant = make_organism(
        "migrant",
        crowding_cap_per_m2=1.0,
        reproduction="asexual",
        min_viable_population=1,
        traits={TEMPERATURE_AXIS: band},
        seasonal_migration=True,
    )
    ctx = BiologyContext(
        grid=grid,
        sea_mask=mask,
        organisms={"resident": resident, "migrant": migrant},
        axis_fields=axis_fields,
        biome_field=biome_field,
        dryness_by_cell=dryness,
        land_cells=tuple(cells),
        params=BiologyParams(),
    )
    start_cell = cells[0]
    state = WorldBiologyState(
        surface_populations={"resident": {start_cell: 1.0}, "migrant": {start_cell: 1.0}},
        subsurface_populations={},
    )
    rng = SeededRng(42)
    for _ in range(3):
        state = step_biology(state, ctx, rng, SECONDS_PER_YEAR)
    resident_com = _center_of_mass(state.surface_populations["resident"])
    migrant_com = _center_of_mass(state.surface_populations["migrant"])
    assert resident_com is not None
    assert migrant_com is not None
    assert migrant_com - resident_com > 1.5
