"""Mines run out, forests can be over-cut, farmland renews, people conserve."""

from __future__ import annotations

from core.civilization.context import CivParams
from core.civilization.economy import (
    CivEconomy,
    ExtractionRun,
    grow_population,
    regrow_biomass,
)
from core.civilization.resources import (
    FARMLAND_MODE,
    FORESTRY_MODE,
    MINE_MODE,
    ResourceDefinition,
)
from core.civilization.settlements import CAPITAL_TIER, Settlement
from core.civilization.state import Civilization
from core.civilization.tech import IDENTITY_EFFECTS
from tests.civ_helpers import FakeGrid, make_context, make_species

_CAPITAL = "r4c1"
_OWNED = ("r4c0", "r4c1", "r4c2")


def _civ(grid: FakeGrid, stockpiles: dict[str, float] | None = None) -> Civilization:
    density = 50_000.0 / grid.area_m2(_CAPITAL)
    capital = Settlement(
        settlement_id="civ-test-s0",
        name="Teston",
        civ_id="civ-test",
        cell=_CAPITAL,
        domain="surface",
        tier=CAPITAL_TIER,
    )
    return Civilization(
        civ_id="civ-test",
        species_id="testfolk",
        domain="surface",
        population_per_m2={_CAPITAL: density},
        settlements=(capital,),
        stockpiles=stockpiles or {},
    )


def _run(
    grid: FakeGrid,
    resource: ResourceDefinition,
    biomass: dict[str, float],
    mine_stocks: dict[str, dict[str, float]],
) -> ExtractionRun:
    species = make_species("testfolk")
    ctx = make_context(grid, [species], resources=[resource])
    econ = CivEconomy(
        civ=_civ(grid),
        species=species,
        effects=IDENTITY_EFFECTS,
        owned_cells=_OWNED,
    )
    return ExtractionRun(ctx, biomass, mine_stocks, econ, dt_years=1.0)


def test_mines_deplete_and_then_yield_nothing() -> None:
    grid = FakeGrid()
    ore = ResourceDefinition(resource_id="ore", mode=MINE_MODE, deposit_stock=10.0)
    stocks = {"ore": {_CAPITAL: 10.0}}
    first = _run(grid, ore, {}, stocks).run()
    assert first.stockpiles["ore"] == 10.0
    assert stocks["ore"][_CAPITAL] == 0.0
    second = _run(grid, ore, {}, stocks).run()
    assert "ore" not in second.stockpiles


def test_forestry_can_outrun_regrowth_and_crash_the_shared_biomass() -> None:
    grid = FakeGrid()
    wood = ResourceDefinition(resource_id="wood", mode=FORESTRY_MODE, harvest_fraction_per_year=0.5)
    capacity = 5.0
    biomass = dict.fromkeys(_OWNED, capacity)
    for _ in range(3):
        _run(grid, wood, biomass, {}).run()
        biomass = regrow_biomass(
            biomass, dict.fromkeys(_OWNED, capacity), rate_per_year=0.08, dt_years=1.0
        )
    assert all(biomass[cell] < 0.4 * capacity for cell in _OWNED), (
        "clear-cutting must measurably strip the biomass the food web grazes"
    )


def test_farmland_renews_while_the_underlying_biology_holds() -> None:
    grid = FakeGrid()
    crop = ResourceDefinition(
        resource_id="crop",
        mode=FARMLAND_MODE,
        harvest_fraction_per_year=0.05,
        feeds_population=True,
    )
    capacity = 5.0
    biomass = dict.fromkeys(_OWNED, capacity)
    fed = 0.0
    for _ in range(5):
        fed = _run(grid, crop, biomass, {}).run().fed_fraction
        biomass = regrow_biomass(
            biomass, dict.fromkeys(_OWNED, capacity), rate_per_year=0.08, dt_years=1.0
        )
    assert all(biomass[cell] > 0.9 * capacity for cell in _OWNED)
    assert fed == 1.0


def test_regrowth_is_logistic_and_capped_at_capacity() -> None:
    biomass = {"a": 2.0, "b": 5.0, "c": 0.0}
    capacity = {"a": 5.0, "b": 5.0, "c": 5.0}
    grown = regrow_biomass(biomass, capacity, rate_per_year=0.5, dt_years=1.0)
    assert 2.0 < grown["a"] <= 5.0
    assert grown["b"] == 5.0
    assert grown["c"] == 0.0, "extinct biomass cannot regrow from nothing"


def test_population_migration_conserves_head_count() -> None:
    grid = FakeGrid()
    species = make_species("testfolk", growth_rate_per_year=0.0)
    ctx = make_context(grid, [species], params=CivParams(migration_per_year=0.2))
    civ = _civ(grid)
    econ = CivEconomy(civ=civ, species=species, effects=IDENTITY_EFFECTS, owned_cells=_OWNED)
    biomass = dict.fromkeys(_OWNED, 5.0)
    before = sum(d * grid.area_m2(c) for c, d in civ.population_per_m2.items())
    grown = grow_population(ctx, biomass, econ, fed_fraction=1.0, dt_years=1.0)
    after = sum(d * grid.area_m2(c) for c, d in grown.items())
    assert abs(after - before) / before < 1e-9
