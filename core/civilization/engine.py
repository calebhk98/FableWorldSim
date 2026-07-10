"""The civilization step: territory, economy, research, settlements.

Composes the layer's passes in a fixed order each orchestrator tick:
influence fields decide territory, frontier contests resolve stochastically
by projected strength, extraction feeds stockpiles (and back into biomass),
populations grow toward what the land supports, and research advances the
tech DAG.  Wraps it all in a ``Process``-shaped class for the orchestrator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from core.civilization.contests import CivStrength, arms_multiplier, raw_strength, resolve_contests
from core.civilization.economy import CivEconomy, ExtractionRun, grow_population, regrow_biomass
from core.civilization.economy import accessible_resource_ids as _accessible
from core.civilization.founding import update_settlements
from core.civilization.influence import assign_territory, contested_cells, influence_field
from core.civilization.settlements import settlement_emitters
from core.civilization.species import DOMAINS, SUBSURFACE_DOMAIN
from core.civilization.state import WorldCivState, civ_population_total, territory_cells
from core.civilization.tech import available_techs, choose_research, combined_effects
from core.civilization.terrain import cost_distances, subsurface_cost_field, surface_cost_field
from core.sim.constants import SECONDS_PER_YEAR

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.chronicle.log import Chronicle
    from core.civilization.context import CivContext
    from core.civilization.state import Civilization
    from core.civilization.tech import TechEffects
    from ports.grid import CellId
    from ports.rng import Rng

PROCESS_NAME = "civilization"


@dataclass
class _StepScope:
    """Everything one step shares across its passes."""

    ctx: CivContext
    rng: Rng
    chronicle: Chronicle | None
    dt_years: float
    effects: dict[str, TechEffects]


def _cost_field(ctx: CivContext, civ: Civilization, effects: TechEffects) -> dict[CellId, float]:
    """Return the civ's movement-cost field, shaped by domain and tech."""
    if civ.domain == SUBSURFACE_DOMAIN:
        return subsurface_cost_field(ctx.grid, ctx.sea_mask, ctx.params.terrain)
    return surface_cost_field(
        ctx.grid, ctx.heights_m, ctx.sea_mask, ctx.params.terrain, effects.water_cost
    )


def _territory_pass(state: WorldCivState, scope: _StepScope) -> WorldCivState:
    """Recompute influence, assign territory per domain, resolve contests."""
    ctx = scope.ctx
    fields: dict[str, dict[CellId, float]] = {}
    strengths: dict[str, CivStrength] = {}
    for civ in state.civs:
        effects = scope.effects[civ.civ_id]
        costs = _cost_field(ctx, civ, effects)
        emitters = settlement_emitters(ctx.grid, civ.population_per_m2, civ.settlements)
        fields[civ.civ_id] = influence_field(
            ctx.grid, costs, emitters, ctx.params.influence_range_m
        )
        arms = arms_multiplier(civ.stockpiles, ctx.resources, ctx.params.arms_scale)
        commit = ctx.species[civ.species_id].commit_fraction
        strengths[civ.civ_id] = CivStrength(
            civ_id=civ.civ_id,
            raw_strength=raw_strength(civ_population_total(ctx.grid, civ), commit, effects, arms),
            supply_distance_m=cost_distances(
                ctx.grid, costs, tuple(s.cell for s in civ.settlements)
            ),
            supply_range_m=ctx.params.supply_range_m * effects.supply_range,
        )
    territory = {
        domain: _domain_territory(state, scope, fields, strengths, domain) for domain in DOMAINS
    }
    return replace(state, territory=territory)


def _domain_territory(
    state: WorldCivState,
    scope: _StepScope,
    fields: Mapping[str, Mapping[CellId, float]],
    strengths: Mapping[str, CivStrength],
    domain: str,
) -> dict[CellId, str]:
    """Assign and contest one domain's territory; civs never cross domains."""
    domain_fields = {civ.civ_id: fields[civ.civ_id] for civ in state.civs if civ.domain == domain}
    if not domain_fields:
        return {}
    territory = assign_territory(domain_fields, scope.ctx.params.influence_floor)
    contested = contested_cells(domain_fields, territory, scope.ctx.params.contest_ratio)
    resolved, flips = resolve_contests(scope.rng, contested, territory, strengths)
    if scope.chronicle is not None:
        for cell, loser, winner in flips:
            scope.chronicle.append(
                tick=state.tick,
                kind="border_shift",
                subject=winner,
                payload={"cell": cell, "taken_from": loser, "domain": domain},
            )
    return resolved


def _economy_pass(state: WorldCivState, scope: _StepScope) -> WorldCivState:
    """Extract resources, feed and grow every civ, regrow shared biomass."""
    ctx = scope.ctx
    biomass = regrow_biomass(
        state.plant_biomass_kg_m2,
        ctx.biomass_capacity_kg_m2,
        ctx.params.biomass_regrowth_per_year,
        scope.dt_years,
    )
    mine_stocks = {rid: dict(stocks) for rid, stocks in state.mine_stocks.items()}
    civs = []
    for civ in state.civs:
        econ = CivEconomy(
            civ=civ,
            species=ctx.species[civ.species_id],
            effects=scope.effects[civ.civ_id],
            owned_cells=territory_cells(state, civ),
        )
        result = ExtractionRun(ctx, biomass, mine_stocks, econ, scope.dt_years).run()
        population = grow_population(ctx, biomass, econ, result.fed_fraction, scope.dt_years)
        civs.append(replace(civ, population_per_m2=population, stockpiles=result.stockpiles))
    return replace(state, civs=tuple(civs), plant_biomass_kg_m2=biomass, mine_stocks=mine_stocks)


def _research_pass(state: WorldCivState, scope: _StepScope) -> WorldCivState:
    """Accumulate research and unlock techs through the prerequisite DAG."""
    civs = []
    for civ in state.civs:
        civs.append(_advance_research(state, scope, civ))
    return replace(state, civs=tuple(civs))


def _advance_research(
    state: WorldCivState,
    scope: _StepScope,
    civ: Civilization,
) -> Civilization:
    """Advance one civ's research: pick a target, add points, maybe unlock."""
    ctx = scope.ctx
    spec = ctx.species[civ.species_id]
    accessible = _accessible(state, civ, ctx.resources)
    candidates = available_techs(ctx.techs, civ.unlocked_techs, accessible)
    candidate_ids = {tech.tech_id for tech in candidates}
    current = civ.current_research if civ.current_research in candidate_ids else None
    if current is None:
        chosen = choose_research(scope.rng, candidates, spec.propensity)
        current = chosen.tech_id if chosen else None
    points = civ.research_points + _research_gain(state, scope, civ)
    unlocked = civ.unlocked_techs
    if current is not None:
        cost = next(tech.cost for tech in ctx.techs if tech.tech_id == current)
        if points >= cost:
            points -= cost
            unlocked = unlocked | {current}
            if scope.chronicle is not None:
                scope.chronicle.append(
                    tick=state.tick,
                    kind="tech_unlocked",
                    subject=civ.civ_id,
                    payload={"tech": current},
                )
            current = None
    return replace(civ, research_points=points, current_research=current, unlocked_techs=unlocked)


def _research_gain(state: WorldCivState, scope: _StepScope, civ: Civilization) -> float:
    """Return research points earned this step.

    Population x resource surplus x settlement density, scaled by the
    species' aptitude and unlocked research multipliers — progress is
    driven by what the civ has actually built, never free.
    """
    ctx = scope.ctx
    population = civ_population_total(ctx.grid, civ)
    if population <= 0.0:
        return 0.0
    surplus = 1.0 + min(1.0, sum(civ.stockpiles.values()) / max(1.0, population / 1_000.0))
    settlement_factor = 1.0 + 0.1 * len(civ.settlements)
    aptitude = ctx.species[civ.species_id].research_aptitude
    research_mult = scope.effects[civ.civ_id].research
    return (
        ctx.params.research_rate
        * math.sqrt(population)
        * surplus
        * settlement_factor
        * aptitude
        * research_mult
        * scope.dt_years
    )


def step_civilization(
    state: WorldCivState,
    ctx: CivContext,
    rng: Rng,
    dt_s: float,
    chronicle: Chronicle | None = None,
) -> WorldCivState:
    """Advance the whole civilization layer by one step of ``dt_s`` seconds."""
    scope = _StepScope(
        ctx=ctx,
        rng=rng,
        chronicle=chronicle,
        dt_years=dt_s / SECONDS_PER_YEAR,
        effects={civ.civ_id: combined_effects(civ.unlocked_techs, ctx.techs) for civ in state.civs},
    )
    state = _territory_pass(state, scope)
    state = _economy_pass(state, scope)
    state = _research_pass(state, scope)
    state = update_settlements(state, ctx, scope.rng, chronicle)
    return replace(state, tick=state.tick + 1)


class CivilizationProcess:
    """Orchestrator process wrapper binding context, RNG, and chronicle."""

    def __init__(self, ctx: CivContext, rng: Rng, chronicle: Chronicle | None = None) -> None:
        """Fork a dedicated RNG stream so other layers stay unperturbed."""
        self._ctx = ctx
        self._rng = rng.fork(PROCESS_NAME)
        self._chronicle = chronicle

    @property
    def name(self) -> str:
        """Return the process name shown by the orchestrator."""
        return PROCESS_NAME

    def step(self, state: WorldCivState, dt_s: float) -> WorldCivState:
        """Advance the civilization layer by one orchestrator step."""
        return step_civilization(state, self._ctx, self._rng, dt_s, self._chronicle)
