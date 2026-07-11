"""The biology step end to end: coupling, fire, extinction, and the volume graph."""

from __future__ import annotations

from dataclasses import replace

from adapters.rng_seeded import SeededRng
from core.biology.context import BiologyParams
from core.biology.engine import BiologyProcess, _surface_suitability, step_biology
from core.biology.state import WorldBiologyState, plant_biomass_field
from core.biology.wildfire import FireParams
from core.chronicle.log import Chronicle
from core.sim.constants import SECONDS_PER_YEAR
from core.sim.orchestrator import Orchestrator
from tests.bio_helpers import FakeSubsurface, UniformEnv, make_organism, uniform_context
from tests.civ_helpers import FakeGrid

_YEAR = SECONDS_PER_YEAR


def _grass_and_herbivore() -> tuple[object, object]:
    grass = make_organism("grass", crowding_cap_per_m2=5.0, reproduction="asexual")
    herbivore = make_organism(
        "herbivore", diet={"grass": 1.0}, body_mass_kg=50.0, crowding_cap_per_m2=1.0e-3
    )
    return grass, herbivore


def _total(field: dict[str, float]) -> float:
    return sum(field.values())


def test_engine_suitability_pass_honors_the_context_lake_mask() -> None:
    """The suitability pass must thread ``ctx.lake_mask`` through to the
    medium gate, not just ``ctx.sea_mask`` -- otherwise a hydrology-built
    lake network never actually reaches a running biology step."""
    grid = FakeGrid(rows=2, cols=2)
    fish = make_organism("fish", medium="aquatic")
    ctx = uniform_context(grid, [fish])
    ctx = replace(ctx, lake_mask={"r0c0": True})

    fields = _surface_suitability(ctx, ["fish"])

    assert fields["fish"] == {"r0c0": 1.0}


def test_process_runs_under_the_orchestrator_and_advances_the_tick() -> None:
    grass, herbivore = _grass_and_herbivore()
    grid = FakeGrid(rows=3, cols=3)
    ctx = uniform_context(grid, [grass, herbivore])  # type: ignore[list-item]
    land = list(grid.cells())
    state = WorldBiologyState(
        surface_populations={
            "grass": dict.fromkeys(land, 2.5),
            "herbivore": dict.fromkeys(land, 5.0e-4),
        },
        subsurface_populations={},
    )
    orchestrator: Orchestrator[WorldBiologyState] = Orchestrator(tick_seconds=_YEAR)
    orchestrator.register(BiologyProcess(ctx, SeededRng(3)))
    result = orchestrator.run(state, ticks=4)
    assert result.tick == 4


def test_same_seed_replays_identically() -> None:
    grass, herbivore = _grass_and_herbivore()
    grid = FakeGrid(rows=3, cols=3)
    ctx = uniform_context(grid, [grass, herbivore])  # type: ignore[list-item]
    land = list(grid.cells())
    state = WorldBiologyState(
        surface_populations={
            "grass": dict.fromkeys(land, 2.5),
            "herbivore": dict.fromkeys(land, 5.0e-4),
        },
        subsurface_populations={},
    )
    first = step_biology(state, ctx, SeededRng(11), _YEAR)
    second = step_biology(state, ctx, SeededRng(11), _YEAR)
    assert first.surface_populations == second.surface_populations


def test_clear_cutting_the_plants_starves_the_herbivore() -> None:
    grass, herbivore = _grass_and_herbivore()
    grid = FakeGrid(rows=2, cols=2)
    ctx = uniform_context(grid, [grass, herbivore])  # type: ignore[list-item]
    land = list(grid.cells())
    herd = dict.fromkeys(land, 5.0e-4)
    healthy = WorldBiologyState(
        surface_populations={"grass": dict.fromkeys(land, 2.5), "herbivore": dict(herd)},
        subsurface_populations={},
    )
    cleared = replace(
        healthy,
        surface_populations={"grass": dict.fromkeys(land, 0.01), "herbivore": dict(herd)},
    )
    grown = step_biology(healthy, ctx, SeededRng(5), _YEAR)
    starved = step_biology(cleared, ctx, SeededRng(5), _YEAR)
    assert _total(starved.surface_populations["herbivore"]) < _total(
        grown.surface_populations["herbivore"]
    )


def test_wildfire_burns_vegetation_and_is_chronicled() -> None:
    grass = make_organism("grass", crowding_cap_per_m2=5.0, reproduction="asexual")
    grid = FakeGrid(rows=3, cols=3)
    params = BiologyParams(
        fire=FireParams(
            base_rate_per_year=100.0,
            fuel_threshold_kg_m2=0.1,
            dryness_threshold=0.0,
            radius_m=5.0e6,
            burn_fraction=0.9,
        )
    )
    env = UniformEnv(precipitation_mm_yr=100.0)
    ctx = uniform_context(grid, [grass], env=env, params=params)  # type: ignore[list-item]
    land = list(grid.cells())
    state = WorldBiologyState(
        surface_populations={"grass": dict.fromkeys(land, 5.0)}, subsurface_populations={}
    )
    chronicle = Chronicle()
    burned = step_biology(state, ctx, SeededRng(1), _YEAR, chronicle)
    assert _total(burned.surface_populations["grass"]) < _total(state.surface_populations["grass"])
    assert any(event.kind == "wildfire" for event in chronicle.events())


def test_a_species_below_viability_goes_extinct_with_a_chronicle_event() -> None:
    grass = make_organism("grass", crowding_cap_per_m2=5.0, reproduction="asexual")
    doomed = make_organism(
        "doomed", diet={"grass": 1.0}, crowding_cap_per_m2=1.0e-3, min_viable_population=10**18
    )
    grid = FakeGrid(rows=2, cols=2)
    ctx = uniform_context(grid, [grass, doomed])  # type: ignore[list-item]
    land = list(grid.cells())
    state = WorldBiologyState(
        surface_populations={
            "grass": dict.fromkeys(land, 2.5),
            "doomed": dict.fromkeys(land, 5.0e-4),
        },
        subsurface_populations={},
    )
    chronicle = Chronicle()
    result = step_biology(state, ctx, SeededRng(2), _YEAR, chronicle)
    assert _total(result.surface_populations["doomed"]) == 0.0
    assert any(event.kind == "extinction" for event in chronicle.events())


def test_subterranean_species_migrate_through_the_volume_graph_not_the_surface() -> None:
    grid = FakeGrid(rows=1, cols=4)
    tunnels = {"r0c0": ("r0c3",), "r0c3": ("r0c0",)}
    subsurface = FakeSubsurface(grid, tunnels)
    moss = make_organism(
        "moss", medium="subterranean", crowding_cap_per_m2=1.0, reproduction="asexual"
    )
    ctx = uniform_context(grid, [moss], subsurface=subsurface)  # type: ignore[list-item]
    state = WorldBiologyState(
        surface_populations={}, subsurface_populations={"moss": {"r0c0@0": 0.5}}
    )
    result = step_biology(state, ctx, SeededRng(9), _YEAR)
    moss_after = result.subsurface_populations["moss"]
    # r0c3 is NOT surface-adjacent to r0c0, only tunnel-connected; r0c1 is surface-adjacent.
    assert moss_after.get("r0c3@0", 0.0) > 0.0
    assert moss_after.get("r0c1@0", 0.0) == 0.0


def test_plant_biomass_field_couples_to_the_civilization_layer() -> None:
    grass, herbivore = _grass_and_herbivore()
    land = ["r0c0", "r0c1"]
    state = WorldBiologyState(
        surface_populations={
            "grass": dict.fromkeys(land, 3.0),
            "herbivore": dict.fromkeys(land, 1.0e-4),
        },
        subsurface_populations={},
    )
    field = plant_biomass_field(state, [grass, herbivore])  # type: ignore[list-item]
    # Only the autotroph contributes standing biomass to what civ forestry grazes.
    assert field == {"r0c0": 3.0, "r0c1": 3.0}
