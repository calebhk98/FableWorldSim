"""End-to-end smoke test: headless core of the checklist's "End-to-end smoke".

Scope: this file exercises the same headless simulation outputs the full
checklist item describes (load Earth topography, run across seasons, and
confirm the emergent outputs are plausible) but stays entirely inside the
Python simulation core. It does **not** attempt `docker compose up` or a
web-client render: `clients/web` and `clients/cli` are still README stubs
with nothing to build or serve. Docker/API/web-client rendering is a
roadmap wrapper that will eventually display these same headless outputs
(temperature fields, river networks, population densities) over HTTP/websui
— once it exists, it can be smoke-tested on top of, not instead of, what
this file already covers.

Three emergent properties are checked directly against the domain layer,
each built the same way the rest of the suite builds it (see
``tests/test_golden_master.py::_summarize`` for the Earth-world pattern and
``tests/test_bio_migration.py`` for suitability-gradient diffusion):

* seasonal temperature bands (equator warmer than the poles, with real
  season-to-season variation);
* rivers routed by steepest descent draining to the sea, using the same
  ``flow_accumulation`` kernel the hydrology kernel-conformance tests use;
* a population diffusing up a suitability gradient (center of mass moves
  toward the warmer, more-suitable end of a line of cells).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pytest

pytest.importorskip("h3")

from adapters.grid_registry import create_grid
from adapters.kernels_python import NO_RECEIVER, flow_accumulation
from adapters.rng_seeded import SeededRng
from core.biology.bands import EnvBand
from core.biology.context import BiologyContext, BiologyParams
from core.biology.engine import step_biology
from core.biology.state import WorldBiologyState
from core.biology.suitability import TEMPERATURE_AXIS
from core.climate.model import ClimateState
from core.climate.model import simulate_climate as run_seasonal_climate
from core.hydrology.sea_mask import SeaMask, build_sea_mask
from core.sim.constants import SECONDS_PER_YEAR
from core.sim.presets import earth
from core.topography.procedural import ProceduralTopography
from ports.grid import CellId, Grid
from tests.bio_helpers import all_land_mask, make_organism
from tests.civ_helpers import FakeGrid

_SEED = 42
_SEASON_COUNT = 4
_EQUATOR_MAX_ABS_LAT_DEG = 15.0
_POLAR_MIN_ABS_LAT_DEG = 60.0


@dataclass(frozen=True)
class _EarthWorld:
    """A built known-seed Earth world: grid, heights, sea mask, climate."""

    grid: Grid
    heights_m: Mapping[CellId, float]
    sea_mask: SeaMask
    climate: ClimateState


def _build_earth_world() -> _EarthWorld:
    """Build the canonical known-seed Earth world (mirrors the golden master)."""
    planet = earth()
    grid = create_grid("h3", resolution=1, radius_m=planet.radius_m)
    heights = ProceduralTopography(SeededRng(_SEED)).heights(grid)
    mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    climate = run_seasonal_climate(grid, planet, heights, mask, season_count=_SEASON_COUNT)
    return _EarthWorld(grid=grid, heights_m=heights, sea_mask=mask, climate=climate)


@pytest.fixture(scope="module")
def earth_world() -> _EarthWorld:
    """Build the Earth world once and share it across this module's tests."""
    return _build_earth_world()


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of an iterable of floats."""
    values = list(values)
    return sum(values) / len(values)


@pytest.mark.slow
def test_earth_climate_shows_seasonal_temperature_bands(earth_world: _EarthWorld) -> None:
    """The equator stays far warmer than the poles, and seasons are not flat."""
    grid, climate = earth_world.grid, earth_world.climate
    cells = list(climate.annual_mean_temperature_k)
    equator = [c for c in cells if abs(grid.centroid(c).lat_deg) <= _EQUATOR_MAX_ABS_LAT_DEG]
    polar = [c for c in cells if abs(grid.centroid(c).lat_deg) >= _POLAR_MIN_ABS_LAT_DEG]
    assert equator
    assert polar

    equator_mean_k = _mean(climate.annual_mean_temperature_k[c] for c in equator)
    polar_mean_k = _mean(climate.annual_mean_temperature_k[c] for c in polar)
    # Observed gap on this seed is ~83 K; require a wide, robust margin.
    assert equator_mean_k - polar_mean_k > 40.0

    # Seasonality: the planet-wide mean is not identical across seasons.
    season_means_k = [_mean(season.temperature_k.values()) for season in climate.seasons]
    assert max(season_means_k) - min(season_means_k) > 5.0

    # Equator-over-pole holds within every individual season too, not just
    # in the annual average.
    for season in climate.seasons:
        season_equator_mean_k = _mean(season.temperature_k[c] for c in equator)
        season_polar_mean_k = _mean(season.temperature_k[c] for c in polar)
        assert season_equator_mean_k > season_polar_mean_k


def _steepest_descent_receiver(
    cell: CellId,
    grid: Grid,
    heights_m: Mapping[CellId, float],
    index_of: Mapping[CellId, int],
) -> int:
    """Return the index of ``cell``'s steepest-descent neighbor, or NO_RECEIVER."""
    lower = [n for n in grid.neighbors(cell) if heights_m[n] < heights_m[cell]]
    if not lower:
        return NO_RECEIVER
    return index_of[min(lower, key=lambda n: heights_m[n])]


def _build_receivers(
    cells: list[CellId],
    grid: Grid,
    heights_m: Mapping[CellId, float],
    sea_mask: SeaMask,
) -> list[int]:
    """Route every land cell downhill; ocean cells are terminal sinks.

    Water that reaches the ocean has arrived at the sea — an ocean cell
    does not need to keep draining within the ocean itself.
    """
    index_of = {c: i for i, c in enumerate(cells)}
    receivers = []
    for cell in cells:
        if sea_mask.ocean[cell]:
            receivers.append(NO_RECEIVER)
        else:
            receivers.append(_steepest_descent_receiver(cell, grid, heights_m, index_of))
    return receivers


def _terminal_index(start: int, receivers: list[int]) -> int:
    """Walk the receiver chain from ``start`` to where it stops draining."""
    seen: set[int] = set()
    current = start
    while receivers[current] != NO_RECEIVER and current not in seen:
        seen.add(current)
        current = receivers[current]
    return current


@pytest.mark.slow
def test_rivers_flow_downhill_to_the_sea(earth_world: _EarthWorld) -> None:
    """Steepest-descent routing drains land downhill and mostly reaches the sea."""
    grid, heights_m, sea_mask = earth_world.grid, earth_world.heights_m, earth_world.sea_mask
    cells = list(heights_m)
    areas_m2 = [grid.area_m2(c) for c in cells]
    elevations_m = [heights_m[c] for c in cells]
    receivers = _build_receivers(cells, grid, heights_m, sea_mask)

    # Every cell that has a receiver drains to a not-higher neighbor.
    for index, receiver in enumerate(receivers):
        if receiver != NO_RECEIVER:
            assert elevations_m[receiver] <= elevations_m[index]

    accumulated_m2 = flow_accumulation(receivers, areas_m2, elevations_m)
    land_indices = [i for i, c in enumerate(cells) if not sea_mask.ocean[c]]
    assert land_indices

    terminates_at_ocean = sum(
        1 for i in land_indices if sea_mask.ocean[cells[_terminal_index(i, receivers)]]
    )
    # Most land drains to the ocean; the rest pools in endorheic basins
    # (real geography has these too — the Caspian, the Great Basin).
    assert terminates_at_ocean / len(land_indices) > 0.5

    river_mouths = [
        i
        for i in land_indices
        if receivers[i] != NO_RECEIVER and sea_mask.ocean[cells[receivers[i]]]
    ]
    assert river_mouths
    average_area_m2 = _mean(areas_m2)
    biggest_mouth = max(river_mouths, key=lambda i: accumulated_m2[i])
    # A real river outlet drains a whole upstream network, not just itself.
    assert accumulated_m2[biggest_mouth] > 5.0 * average_area_m2


def _column_index(cell: CellId) -> int:
    """Return a FakeGrid cell's column index from its ``r<row>c<col>`` id."""
    return int(cell.split("c")[1])


def _center_of_mass(field: Mapping[CellId, float]) -> float:
    """Return the density-weighted mean column index of a population field."""
    total = sum(field.values())
    return sum(value * _column_index(cell) for cell, value in field.items()) / total


def _build_gradient_context(grid: FakeGrid) -> tuple[BiologyContext, list[CellId]]:
    """Build a biology context over a temperature gradient favoring high columns."""
    cells = list(grid.cells())
    temperature_k = {cell: 290.0 + _column_index(cell) * 4.0 for cell in cells}
    organism = make_organism(
        "critter",
        crowding_cap_per_m2=5.0,
        reproduction="asexual",
        traits={
            TEMPERATURE_AXIS: EnvBand(
                comfort_min=308.0, comfort_max=312.0, tolerance_min=286.0, tolerance_max=316.0
            )
        },
    )
    ctx = BiologyContext(
        grid=grid,
        sea_mask=all_land_mask(grid),
        organisms={organism.species_id: organism},
        axis_fields={TEMPERATURE_AXIS: temperature_k},
        biome_field={},
        dryness_by_cell={},
        land_cells=tuple(cells),
        params=BiologyParams(migration_per_year=0.3, migration_jitter=0.2),
    )
    return ctx, cells


@pytest.mark.slow
def test_population_migrates_toward_higher_suitability() -> None:
    """A population seeded at the cold end diffuses toward the warm end."""
    grid = FakeGrid(rows=1, cols=6)
    ctx, cells = _build_gradient_context(grid)
    state = WorldBiologyState(
        surface_populations={
            "critter": {cells[0]: 5.0, **dict.fromkeys(cells[1:], 0.0)},
        },
        subsurface_populations={},
    )
    before_com = _center_of_mass(state.surface_populations["critter"])
    assert before_com == pytest.approx(0.0)

    rng = SeededRng(7)
    for _tick in range(15):
        state = step_biology(state, ctx, rng, SECONDS_PER_YEAR)

    after_field = state.surface_populations["critter"]
    after_com = _center_of_mass(after_field)
    # Observed center of mass settles near column 3.3 (of 0..5, warm end is
    # 5); require it moved well past the midpoint, with margin to spare.
    assert after_com > 2.0

    cold_half_total = sum(after_field[c] for c in cells[:3])
    warm_half_total = sum(after_field[c] for c in cells[3:])
    assert warm_half_total > cold_half_total
