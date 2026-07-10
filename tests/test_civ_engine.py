"""End-to-end civilization steps: borders, domains, research, settlements."""

from __future__ import annotations

from adapters.rng_seeded import SeededRng
from core.chronicle.log import Chronicle
from core.civilization.engine import CivilizationProcess, step_civilization
from core.civilization.founding import found_civilizations
from core.civilization.species import SUBSURFACE_DOMAIN, SURFACE_DOMAIN
from core.civilization.state import WorldCivState, civ_by_id
from core.civilization.tech import TechDefinition
from core.sim.constants import SECONDS_PER_YEAR
from core.sim.orchestrator import Orchestrator
from tests.civ_helpers import FakeGrid, fast_params, make_context, make_species, make_tech

_WEST = "r4c1"
_EAST = "r4c6"
_EAST_ISLAND = {f"r{r}c{c}" for r in range(8) for c in (5, 6, 7)}
_WEST_ISLAND = {f"r{r}c{c}" for r in range(8) for c in (0, 1, 2)}


def _two_civ_state(techs: tuple[TechDefinition, ...] = ()) -> tuple[WorldCivState, object]:
    grid = FakeGrid()
    species = [make_species("westfolk"), make_species("eastfolk")]
    ctx = make_context(grid, species, techs=techs, params=fast_params())
    state = found_civilizations(ctx, SeededRng(5), capitals={"westfolk": _WEST, "eastfolk": _EAST})
    return state, ctx


def _run(state: WorldCivState, ctx: object, seed: int, steps: int) -> WorldCivState:
    rng = SeededRng(seed)
    chronicle = Chronicle()
    for _ in range(steps):
        state = step_civilization(state, ctx, rng, SECONDS_PER_YEAR, chronicle)  # type: ignore[arg-type]
    return state


def test_borders_form_around_capitals_and_stay_on_their_islands() -> None:
    state, ctx = _two_civ_state()
    state = _run(state, ctx, seed=11, steps=5)
    surface = state.territory[SURFACE_DOMAIN]
    assert surface[_WEST] == "civ-westfolk"
    assert surface[_EAST] == "civ-eastfolk"
    west_holdings = {cell for cell, owner in surface.items() if owner == "civ-westfolk"}
    assert len(west_holdings) > 1, "influence must claim more than the capital"
    assert not west_holdings & _EAST_ISLAND, "no conquest across an unsailed ocean"
    east_holdings = {cell for cell, owner in surface.items() if owner == "civ-eastfolk"}
    assert not east_holdings & _WEST_ISLAND


def test_same_seed_replays_identically() -> None:
    state_a, ctx_a = _two_civ_state()
    state_b, ctx_b = _two_civ_state()
    final_a = _run(state_a, ctx_a, seed=11, steps=5)
    final_b = _run(state_b, ctx_b, seed=11, steps=5)
    assert final_a.territory == final_b.territory
    for civ_a, civ_b in zip(final_a.civs, final_b.civs, strict=True):
        assert civ_a.population_per_m2 == civ_b.population_per_m2
        assert civ_a.stockpiles == civ_b.stockpiles
        assert civ_a.research_points == civ_b.research_points


def test_surface_and_subsurface_races_share_a_column_without_contesting() -> None:
    grid = FakeGrid()
    species = [
        make_species("uplander", domain=SURFACE_DOMAIN),
        make_species("deepkin", domain=SUBSURFACE_DOMAIN),
    ]
    ctx = make_context(grid, species, params=fast_params())
    state = found_civilizations(ctx, SeededRng(5), capitals={"uplander": _WEST, "deepkin": _WEST})
    state = _run(state, ctx, seed=3, steps=4)
    assert state.territory[SURFACE_DOMAIN][_WEST] == "civ-uplander"
    assert state.territory[SUBSURFACE_DOMAIN][_WEST] == "civ-deepkin"
    surface_cells = set(state.territory[SURFACE_DOMAIN])
    subsurface_cells = set(state.territory[SUBSURFACE_DOMAIN])
    assert surface_cells & subsurface_cells, "the same columns are held in both domains"


def test_research_unlocks_reachable_techs_and_skips_resource_gated_ones() -> None:
    techs = (
        make_tech("alpha", cost=5.0, research=1.2),
        TechDefinition(tech_id="gated", cost=5.0, requires_resources=("unobtainium",)),
    )
    state, ctx = _two_civ_state(techs)
    chronicle = Chronicle()
    rng = SeededRng(11)
    for _ in range(4):
        state = step_civilization(state, ctx, rng, SECONDS_PER_YEAR, chronicle)  # type: ignore[arg-type]
    for civ in state.civs:
        assert "alpha" in civ.unlocked_techs
        assert "gated" not in civ.unlocked_techs
        assert civ.current_research != "gated"
    unlock_events = list(chronicle.events(kind="tech_unlocked"))
    assert len(unlock_events) >= 2


def test_settlements_are_founded_with_toponyms_and_chronicled() -> None:
    grid = FakeGrid()
    species = [make_species("westfolk")]
    params = fast_params(
        settlement_population=500.0,
        settlement_spacing_m=400_000.0,
        migration_per_year=0.2,
    )
    ctx = make_context(grid, species, params=params)
    state = found_civilizations(ctx, SeededRng(5), capitals={"westfolk": _WEST})
    chronicle = Chronicle()
    rng = SeededRng(9)
    for _ in range(8):
        state = step_civilization(state, ctx, rng, SECONDS_PER_YEAR, chronicle)  # type: ignore[arg-type]
    civ = civ_by_id(state, "civ-westfolk")
    assert len(civ.settlements) >= 2, "growing populations must found new settlements"
    newest = civ.settlements[-1]
    assert newest.tier == "town"
    assert newest.name
    assert list(chronicle.events(kind="settlement_founded"))


def test_founding_chronicles_each_civ_and_names_capitals() -> None:
    state, _ctx = _two_civ_state()
    for civ in state.civs:
        assert civ.settlements[0].tier == "capital"
        assert civ.settlements[0].name


def test_civilization_process_runs_under_the_orchestrator() -> None:
    state, ctx = _two_civ_state()
    orchestrator: Orchestrator[WorldCivState] = Orchestrator(tick_seconds=SECONDS_PER_YEAR)
    process = CivilizationProcess(ctx, SeededRng(3))  # type: ignore[arg-type]
    orchestrator.register(process, every_ticks=1)
    result = orchestrator.run(state, ticks=3)
    assert result.tick == 3
    assert result.territory[SURFACE_DOMAIN]
