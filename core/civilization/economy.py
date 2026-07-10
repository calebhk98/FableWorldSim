"""Population growth, extraction, and the biomass feedback loop.

Forestry and farmland draw on the *same* plant-biomass field the food web
uses, so clear-cutting a forest measurably starves whatever grazed it —
resource use and ecology are one coupled system, not parallel bookkeeping.
Mines deplete a finite stock; extraction feeds population, settlements, and
research even before a market exists (trade is roadmap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.civilization.resources import FORESTRY_MODE, MINE_MODE
from core.civilization.state import civ_population_total, territory_cells

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from core.biology.biome import BiomeDefinition
    from core.civilization.context import CivContext
    from core.civilization.resources import ResourceDefinition
    from core.civilization.species import SpeciesDefinition
    from core.civilization.state import Civilization, WorldCivState
    from core.civilization.tech import TechEffects
    from ports.grid import CellId


@dataclass(frozen=True)
class CivEconomy:
    """Working view of one civ for the economy pass."""

    civ: Civilization
    species: SpeciesDefinition
    effects: TechEffects
    owned_cells: tuple[CellId, ...]


@dataclass(frozen=True)
class ExtractionResult:
    """New stockpiles plus how well the population was fed."""

    stockpiles: dict[str, float]
    fed_fraction: float


def biomass_capacity_from_biomes(
    biome_ids: Mapping[CellId, str],
    biomes: Sequence[BiomeDefinition],
    peak_kg_m2: float,
) -> dict[CellId, float]:
    """Derive a biomass-capacity field from per-cell biome assignments."""
    density = {biome.biome_id: biome.vegetation_density for biome in biomes}
    return {cell: peak_kg_m2 * density[biome_id] for cell, biome_id in biome_ids.items()}


def regrow_biomass(
    biomass: Mapping[CellId, float],
    capacity: Mapping[CellId, float],
    rate_per_year: float,
    dt_years: float,
) -> dict[CellId, float]:
    """Regrow plant biomass logistically toward each cell's capacity."""
    grown: dict[CellId, float] = {}
    for cell, value in biomass.items():
        cap = capacity.get(cell, 0.0)
        if cap <= 0.0:
            grown[cell] = 0.0
            continue
        step = rate_per_year * value * (1.0 - value / cap) * dt_years
        grown[cell] = min(cap, max(0.0, value + step))
    return grown


def accessible_resource_ids(
    state: WorldCivState,
    civ: Civilization,
    resources: Sequence[ResourceDefinition],
) -> frozenset[str]:
    """Return the resources the civ can reach inside its own territory.

    This is the environmental gating for tech: a civ with no coal or iron
    deposits in reach simply never sees an industrial revolution.
    """
    owned = set(territory_cells(state, civ))
    accessible: set[str] = set()
    for resource in resources:
        if resource.mode == MINE_MODE:
            stocks = state.mine_stocks.get(resource.resource_id, {})
            if any(stock > 0 and cell in owned for cell, stock in stocks.items()):
                accessible.add(resource.resource_id)
        elif any(state.plant_biomass_kg_m2.get(cell, 0.0) > 0 for cell in owned):
            accessible.add(resource.resource_id)
    return frozenset(accessible)


@dataclass
class ExtractionRun:
    """One civ's extraction over the step's working field copies.

    The ``biomass`` and ``mine_stocks`` dicts are this step's working
    copies, shared across civs so extraction is naturally competitive; the
    run mutates them in place and returns the civ's new stockpiles.
    """

    ctx: CivContext
    biomass: dict[CellId, float]
    mine_stocks: dict[str, dict[CellId, float]]
    econ: CivEconomy
    dt_years: float

    def run(self) -> ExtractionResult:
        """Extract every resource, then feed the population from stockpiles."""
        stockpiles = dict(self.econ.civ.stockpiles)
        for resource in self.ctx.resources:
            gained = self._yield_for(resource)
            if gained > 0.0:
                stockpiles[resource.resource_id] = (
                    stockpiles.get(resource.resource_id, 0.0) + gained
                )
        fed = self._consume_food(stockpiles)
        return ExtractionResult(stockpiles=stockpiles, fed_fraction=fed)

    def _workers(self) -> float:
        """Return the head count available for extraction work."""
        total = civ_population_total(self.ctx.grid, self.econ.civ)
        return total * self.ctx.params.workforce_fraction

    def _yield_for(self, resource: ResourceDefinition) -> float:
        """Dispatch extraction by resource mode."""
        if resource.mode == MINE_MODE:
            return self._mine(resource)
        if resource.mode == FORESTRY_MODE:
            return self._harvest(resource, self.econ.effects.extraction, 1.0)
        return self._harvest(
            resource,
            self.econ.effects.farm_yield,
            self.ctx.params.farm_drawdown_fraction,
        )

    def _mine(self, resource: ResourceDefinition) -> float:
        """Deplete finite deposits inside the civ's territory."""
        stocks = self.mine_stocks.get(resource.resource_id)
        if stocks is None:
            return 0.0
        sites = [cell for cell in self.econ.owned_cells if stocks.get(cell, 0.0) > 0.0]
        if not sites:
            return 0.0
        rate = (
            self.ctx.params.mine_rate_per_worker_year
            * self.econ.effects.extraction
            * self.econ.species.mine_affinity
        )
        want_per_site = self._workers() * rate * self.dt_years / len(sites)
        total = 0.0
        for cell in sites:
            taken = min(stocks[cell], want_per_site)
            stocks[cell] -= taken
            total += taken
        return total

    def _harvest(
        self,
        resource: ResourceDefinition,
        multiplier: float,
        drawdown_fraction: float,
    ) -> float:
        """Harvest plant biomass; forestry removes all of it, farmland a buffer.

        Forestry (``drawdown_fraction`` = 1) cuts standing biomass outright
        and can outrun regrowth; farmland only draws down a fraction of what
        it harvests, renewing while the underlying biology holds.
        """
        fraction = resource.harvest_fraction_per_year * multiplier * self.dt_years
        fraction = min(0.95, fraction)
        total_kg = 0.0
        for cell in self.econ.owned_cells:
            standing = self.biomass.get(cell, 0.0)
            if standing <= 0.0:
                continue
            harvested = standing * fraction
            self.biomass[cell] = standing - harvested * drawdown_fraction
            total_kg += harvested * self.ctx.grid.area_m2(cell)
        return total_kg / self.ctx.params.stockpile_unit_kg

    def _consume_food(self, stockpiles: dict[str, float]) -> float:
        """Eat food-bearing stockpiles; return the fraction of demand met."""
        params = self.ctx.params
        food = sum(
            stockpiles.get(res.resource_id, 0.0) * params.stockpile_unit_kg
            for res in self.ctx.resources
            if res.feeds_population
        )
        need = (
            civ_population_total(self.ctx.grid, self.econ.civ)
            * params.food_per_person_year
            * self.dt_years
            / params.food_value_per_kg
        )
        if need <= 0.0:
            return 1.0
        fed = min(1.0, food / need)
        remaining_share = 1.0 - fed * need / food if food > 0 else 0.0
        for res in self.ctx.resources:
            if res.feeds_population and res.resource_id in stockpiles:
                stockpiles[res.resource_id] *= remaining_share
        return fed


def grow_population(
    ctx: CivContext,
    biomass: Mapping[CellId, float],
    econ: CivEconomy,
    fed_fraction: float,
    dt_years: float,
) -> dict[CellId, float]:
    """Grow the density field logistically inside held territory, then migrate.

    Capacity is proportional to local plant biomass scaled by farm tech, so
    the land — and what extraction left of it — bounds the people.
    """
    params = ctx.params
    owned = set(econ.owned_cells)
    rate = econ.species.growth_rate_per_year * (0.25 + 0.75 * fed_fraction)
    grown: dict[CellId, float] = {}
    for cell, density in econ.civ.population_per_m2.items():
        if cell not in owned:
            grown[cell] = density
            continue
        cap = biomass.get(cell, 0.0) * params.capacity_per_biomass * econ.effects.farm_yield
        if cap <= 0.0:
            grown[cell] = density * max(0.0, 1.0 - rate * dt_years)
            continue
        grown[cell] = max(0.0, density + rate * density * (1.0 - density / cap) * dt_years)
    return _migrate(ctx, grown, owned, params.migration_per_year * dt_years)


def _migrate(
    ctx: CivContext,
    density: dict[CellId, float],
    owned: set[CellId],
    move_fraction: float,
) -> dict[CellId, float]:
    """Spread a count-conserving share of each cell's people to owned neighbors."""
    move_fraction = min(0.5, move_fraction)
    if move_fraction <= 0.0:
        return density
    result = dict(density)
    for cell, value in density.items():
        if value <= 0.0 or cell not in owned:
            continue
        targets = [nb for nb in ctx.grid.neighbors(cell) if nb in owned]
        if not targets:
            continue
        moving_count = value * ctx.grid.area_m2(cell) * move_fraction
        result[cell] -= value * move_fraction
        share = moving_count / len(targets)
        for target in targets:
            result[target] = result.get(target, 0.0) + share / ctx.grid.area_m2(target)
    return result
