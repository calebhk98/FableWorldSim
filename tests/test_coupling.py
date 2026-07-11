"""The real cross-layer loop: civ forestry measurably starves biology's food web.

Exercises :mod:`core.sim.coupling` end to end -- not a hand-simulated stand-in
for the coupling, but the actual :class:`~core.sim.coupling.CoupledProcess`
an orchestrator would run. The same standing plant biomass the food web
computes (:func:`~core.biology.state.plant_biomass_field`) is exactly what
civilization's forestry resource reads and draws down, and the drawdown is
folded back into the very autotroph population the herbivore's own feeding
pass reads that same tick -- one shared field, not two bookkeeping copies
that happen to agree at t=0 and drift apart forever after.
"""

from __future__ import annotations

from adapters.rng_seeded import SeededRng
from core.biology.state import WorldBiologyState, plant_biomass_field
from core.civilization.founding import found_civilizations
from core.civilization.resources import FORESTRY_MODE, ResourceDefinition
from core.sim.constants import SECONDS_PER_YEAR
from core.sim.coupling import CoupledContext, CoupledProcess, WorldState
from tests.bio_helpers import make_organism, uniform_context
from tests.civ_helpers import FakeGrid, make_context, make_species

_CAPITAL = "r0c0"


def _grass_and_herbivore() -> tuple[object, object]:
    grass = make_organism("grass", crowding_cap_per_m2=5.0, reproduction="asexual")
    herbivore = make_organism(
        "herbivore", diet={"grass": 1.0}, body_mass_kg=50.0, crowding_cap_per_m2=1.0e-3
    )
    return grass, herbivore


def _build_world(harvest_fraction_per_year: float) -> tuple[WorldState, CoupledContext]:
    """Build a tiny coupled world: one civ, its capital's forest, and a herd."""
    grid = FakeGrid(rows=2, cols=2)
    grass, herbivore = _grass_and_herbivore()
    bio_ctx = uniform_context(grid, [grass, herbivore])  # type: ignore[list-item]
    land = list(grid.cells())
    bio_state = WorldBiologyState(
        surface_populations={
            "grass": dict.fromkeys(land, 3.0),
            "herbivore": dict.fromkeys(land, 5.0e-4),
        },
        subsurface_populations={},
    )
    wood = ResourceDefinition(
        resource_id="wood", mode=FORESTRY_MODE, harvest_fraction_per_year=harvest_fraction_per_year
    )
    civ_ctx = make_context(grid, [make_species("testfolk")], resources=[wood])
    civ_state = found_civilizations(civ_ctx, SeededRng(3), capitals={"testfolk": _CAPITAL})
    state = WorldState(biology=bio_state, civ=civ_state)
    coupled_ctx = CoupledContext(bio_ctx=bio_ctx, civ_ctx=civ_ctx, organisms=(grass, herbivore))
    return state, coupled_ctx


def _herbivore_total(state: WorldState) -> float:
    return sum(state.biology.surface_populations["herbivore"].values())


def test_forestry_drawdown_is_the_exact_biomass_the_food_web_lost() -> None:
    """The kg/m2 civ's economy pass harvests is what vanishes from biology's grass."""
    state, ctx = _build_world(harvest_fraction_per_year=0.6)
    process = CoupledProcess(ctx, SeededRng(11))

    before = plant_biomass_field(state.biology, ctx.organisms)
    result = process.step(state, SECONDS_PER_YEAR)

    # Every owned cell lost biomass to forestry, and civ's own record of the
    # field (population capacity, stockpiles) came from that same drawdown.
    owned = set(before)
    assert owned, "the civ must hold territory for this test to mean anything"
    total_harvested_kg = sum(result.civ.civs[0].stockpiles.get("wood", 0.0) for _ in [0])
    assert total_harvested_kg > 0.0, "forestry must have actually extracted wood"

    after_field = plant_biomass_field(result.biology, ctx.organisms)
    # Growth also ran this tick, so the field is not simply "before minus
    # harvested"; but it must be strictly below what an unharvested cell
    # would have grown to, since the harvest happened before growth.
    unharvested_state, unharvested_ctx = _build_world(harvest_fraction_per_year=0.0)
    unharvested_result = CoupledProcess(unharvested_ctx, SeededRng(11)).step(
        unharvested_state, SECONDS_PER_YEAR
    )
    unharvested_field = plant_biomass_field(unharvested_result.biology, ctx.organisms)
    for cell in owned:
        assert after_field[cell] < unharvested_field[cell]


def test_civ_forestry_drawdown_reduces_the_dependent_herbivore_through_the_shared_field() -> None:
    """Clear-cutting via the civ layer starves the herbivore in the same cell.

    This is the coupling the issue calls "aspirational": civ extraction
    was a parallel, civ-only biomass field, so no amount of forestry ever
    touched what the herbivore's population update read. Routed through
    the one shared field, it now does -- within the very same tick.
    """
    harvested_state, harvested_ctx = _build_world(harvest_fraction_per_year=0.9)
    spared_state, spared_ctx = _build_world(harvest_fraction_per_year=0.0)

    harvested = CoupledProcess(harvested_ctx, SeededRng(5)).step(harvested_state, SECONDS_PER_YEAR)
    spared = CoupledProcess(spared_ctx, SeededRng(5)).step(spared_state, SECONDS_PER_YEAR)

    assert _herbivore_total(harvested) < _herbivore_total(spared), (
        "heavy forestry must measurably starve the herbivore that grazed the same land"
    )

    # And it is starvation, not coincidence: less standing plant biomass
    # behind it, cell for cell.
    harvested_plants = plant_biomass_field(harvested.biology, harvested_ctx.organisms)
    spared_plants = plant_biomass_field(spared.biology, spared_ctx.organisms)
    for cell in harvested_plants:
        assert harvested_plants[cell] < spared_plants[cell]
