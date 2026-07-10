"""Seeding civilizations and founding or upgrading settlements.

Every sapient species founds one civilization at world start (>= 2 races
ship, so inter-civ contact runs from tick one).  Capitals prefer fertile
land for surface races and high ground for subsurface races; later
settlements appear where a civ's people concentrate far from any existing
settlement, each with a generated toponym.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from core.civilization.resources import generate_mine_deposits
from core.civilization.settlements import (
    CAPITAL_TIER,
    TOWN_TIER,
    Settlement,
    generate_toponym,
    local_population,
    tier_for,
)
from core.civilization.species import SUBSURFACE_DOMAIN, sapient_species
from core.civilization.state import Civilization, WorldCivState, territory_cells
from core.grid.geodesy import great_circle_distance_m

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.chronicle.log import Chronicle
    from core.civilization.context import CivContext
    from core.civilization.species import SpeciesDefinition
    from ports.grid import CellId
    from ports.rng import Rng

_CAPITAL_SEPARATION_RANGES = 3.0


def _site_score(ctx: CivContext, spec: SpeciesDefinition, cell: CellId) -> float:
    """Score a capital candidate: fertility above ground, relief below it."""
    if spec.domain == SUBSURFACE_DOMAIN:
        return ctx.heights_m[cell]
    return ctx.biomass_capacity_kg_m2.get(cell, 0.0)


def _capital_site(
    ctx: CivContext,
    spec: SpeciesDefinition,
    taken: list[CellId],
) -> CellId:
    """Pick the best-scoring land cell far enough from earlier capitals."""
    min_separation = ctx.params.influence_range_m * _CAPITAL_SEPARATION_RANGES
    candidates = sorted(
        (cell for cell in ctx.grid.cells() if ctx.sea_mask.is_land(cell)),
        key=lambda cell: (-_site_score(ctx, spec, cell), cell),
    )
    for cell in candidates:
        centroid = ctx.grid.centroid(cell)
        far_enough = all(
            great_circle_distance_m(centroid, ctx.grid.centroid(other), ctx.grid.radius_m)
            >= min_separation
            for other in taken
        )
        if far_enough:
            return cell
    return candidates[0]


def found_civilizations(
    ctx: CivContext,
    rng: Rng,
    chronicle: Chronicle | None = None,
    capitals: Mapping[str, CellId] | None = None,
) -> WorldCivState:
    """Create the initial world state: one civilization per sapient species.

    ``capitals`` optionally pins species to explicit capital cells (used by
    tests and scenarios); unpinned species pick sites automatically.
    """
    species = sorted(sapient_species(tuple(ctx.species.values())), key=lambda s: s.species_id)
    taken: list[CellId] = []
    civs: list[Civilization] = []
    names = rng.fork("toponyms")
    for spec in species:
        site = capitals.get(spec.species_id) if capitals else None
        if site is None:
            site = _capital_site(ctx, spec, taken)
        taken.append(site)
        civ_id = f"civ-{spec.species_id}"
        capital = Settlement(
            settlement_id=f"{civ_id}-s0",
            name=generate_toponym(names, spec.toponym_prefixes, spec.toponym_suffixes),
            civ_id=civ_id,
            cell=site,
            domain=spec.domain,
            tier=CAPITAL_TIER,
        )
        density = ctx.params.initial_population / ctx.grid.area_m2(site)
        civs.append(
            Civilization(
                civ_id=civ_id,
                species_id=spec.species_id,
                domain=spec.domain,
                population_per_m2={site: density},
                settlements=(capital,),
            )
        )
        if chronicle is not None:
            chronicle.append(
                tick=0,
                kind="civ_founded",
                subject=civ_id,
                payload={"species": spec.species_id, "capital": capital.name, "cell": site},
            )
    return WorldCivState(
        civs=tuple(civs),
        plant_biomass_kg_m2=dict(ctx.biomass_capacity_kg_m2),
        mine_stocks=generate_mine_deposits(ctx.grid, rng, ctx.resources, ctx.sea_mask),
        territory={},
    )


def update_settlements(
    state: WorldCivState,
    ctx: CivContext,
    rng: Rng,
    chronicle: Chronicle | None = None,
) -> WorldCivState:
    """Found at most one new settlement per civ and upgrade existing tiers."""
    new_civs = []
    for civ in state.civs:
        settlements = _upgraded_tiers(ctx, civ)
        site = _founding_site(state, ctx, civ)
        if site is not None:
            spec = ctx.species[civ.species_id]
            settlement = Settlement(
                settlement_id=f"{civ.civ_id}-s{len(settlements)}",
                name=generate_toponym(rng, spec.toponym_prefixes, spec.toponym_suffixes),
                civ_id=civ.civ_id,
                cell=site,
                domain=civ.domain,
                tier=TOWN_TIER,
                founded_tick=state.tick,
            )
            settlements = (*settlements, settlement)
            if chronicle is not None:
                chronicle.append(
                    tick=state.tick,
                    kind="settlement_founded",
                    subject=settlement.settlement_id,
                    payload={"civ": civ.civ_id, "name": settlement.name, "cell": site},
                )
        new_civs.append(replace(civ, settlements=settlements))
    return replace(state, civs=tuple(new_civs))


def _upgraded_tiers(ctx: CivContext, civ: Civilization) -> tuple[Settlement, ...]:
    """Promote towns to cities where enough people have gathered."""
    upgraded = []
    for settlement in civ.settlements:
        if settlement.tier == CAPITAL_TIER:
            upgraded.append(settlement)
            continue
        count = local_population(ctx.grid, civ.population_per_m2, settlement.cell)
        upgraded.append(replace(settlement, tier=tier_for(count, ctx.params.city_population)))
    return tuple(upgraded)


def _founding_site(
    state: WorldCivState,
    ctx: CivContext,
    civ: Civilization,
) -> CellId | None:
    """Return the best owned cell that supports a new settlement, if any."""
    params = ctx.params
    existing = [ctx.grid.centroid(s.cell) for s in civ.settlements]
    best: tuple[float, CellId] | None = None
    for cell in territory_cells(state, civ):
        count = local_population(ctx.grid, civ.population_per_m2, cell)
        if count < params.settlement_population:
            continue
        centroid = ctx.grid.centroid(cell)
        spacing_ok = all(
            great_circle_distance_m(centroid, other, ctx.grid.radius_m)
            >= params.settlement_spacing_m
            for other in existing
        )
        if spacing_ok and (best is None or count > best[0]):
            best = (count, cell)
    return best[1] if best else None
