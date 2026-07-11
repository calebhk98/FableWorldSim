"""The biology step: suitability, feeding, growth, migration, fire, extinction.

Composes the layer's passes in a fixed order each orchestrator tick.  A
:class:`~core.biology.domain.DomainRunner` drives the surface shell and,
when present, the subsurface volume through the same feed-grow-migrate
pipeline; between a surface run and its viability gate, dry vegetation may
catch fire.  Wrapped in a ``Process``-shaped class for the multi-rate
orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.biology.domain import DomainRunner
from core.biology.medium import SUBTERRANEAN, is_surface_medium
from core.biology.state import WorldBiologyState
from core.biology.suitability import (
    SurfaceEnvironment,
    subsurface_suitability,
    surface_suitability,
)
from core.biology.wildfire import IgnitionField, burn, ignite
from core.sim.constants import SECONDS_PER_YEAR
from core.sim.process import BoundProcess

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from core.biology.context import BiologyContext
    from core.biology.organism import Organism
    from core.chronicle.log import Chronicle
    from ports.rng import Rng

PROCESS_NAME = "biology"

_Fields = dict[str, dict[str, float]]


@dataclass(frozen=True)
class _StepScope:
    """Everything one biology step shares across its passes."""

    ctx: BiologyContext
    rng: Rng
    chronicle: Chronicle | None
    dt_years: float
    tick: int


def _partition(organisms: Mapping[str, Organism]) -> tuple[list[str], list[str]]:
    """Split species ids into the surface set and the subterranean set."""
    surface = sorted(sid for sid, org in organisms.items() if is_surface_medium(org.medium))
    subsurface = sorted(sid for sid, org in organisms.items() if org.medium == SUBTERRANEAN)
    return surface, subsurface


def _surface_suitability(ctx: BiologyContext, species_ids: Sequence[str]) -> _Fields:
    """Grade every surface species over the whole grid."""
    environment = SurfaceEnvironment(
        axis_fields=ctx.axis_fields,
        biome_field=ctx.biome_field,
        sea_mask=ctx.sea_mask,
        base_preference=ctx.params.base_biome_preference,
        lake_mask=ctx.lake_mask,
    )
    cells = list(ctx.grid.cells())
    return {sid: surface_suitability(ctx.organisms[sid], cells, environment) for sid in species_ids}


def _subsurface_suitability(ctx: BiologyContext, species_ids: Sequence[str]) -> _Fields:
    """Grade every subterranean species over the volume graph."""
    if ctx.subsurface is None:
        return {sid: {} for sid in species_ids}
    nodes = list(ctx.subsurface.nodes())
    return {
        sid: subsurface_suitability(
            ctx.organisms[sid], nodes, ctx.subsurface_temperature, ctx.subsurface_diggability
        )
        for sid in species_ids
    }


def _fuel_by_cell(plant_fields: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Sum standing plant biomass per cell — the wildfire fuel load."""
    fuel: dict[str, float] = {}
    for field in plant_fields.values():
        for cell, biomass in field.items():
            fuel[cell] = fuel.get(cell, 0.0) + biomass
    return fuel


def _apply_wildfire(scope: _StepScope, surface_pops: _Fields) -> _Fields:
    """Ignite and burn surface vegetation, then record the fires."""
    ctx = scope.ctx
    plant_fields = {
        sid: field for sid, field in surface_pops.items() if ctx.organisms[sid].is_autotroph
    }
    if not plant_fields:
        return surface_pops
    field = IgnitionField(
        fuel_by_cell=_fuel_by_cell(plant_fields),
        dryness_by_cell=ctx.dryness_by_cell,
        land_cells=ctx.land_cells,
    )
    ignited = ignite(field, ctx.params.fire, scope.dt_years, scope.rng.fork(f"fire:{scope.tick}"))
    if not ignited:
        return surface_pops
    burned = burn(plant_fields, ctx.grid, ignited, ctx.params.fire)
    _log_fires(scope, ignited)
    return {**surface_pops, **burned}


def _log_fires(scope: _StepScope, ignited: Sequence[str]) -> None:
    """Append a chronicle event per ignited cell."""
    if scope.chronicle is None:
        return
    for cell in ignited:
        scope.chronicle.append(
            tick=scope.tick, kind="wildfire", subject=cell, payload={"fires": len(ignited)}
        )


def _subsurface_area_of(ctx: BiologyContext) -> Callable[[str], float]:
    """Return a node-area function using the surface column footprint."""
    subsurface = ctx.subsurface
    if subsurface is None:
        msg = "subsurface area requested without a subsurface grid"
        raise ValueError(msg)

    def area_of(node: str) -> float:
        return ctx.grid.area_m2(subsurface.surface_cell(node))

    return area_of


def _make_runner(
    scope: _StepScope,
    domain: str,
    geometry: tuple[Callable[[str], Sequence[str]], Callable[[str], float]],
) -> DomainRunner:
    """Build a :class:`DomainRunner` bound to one domain's geometry."""
    neighbors_of, area_of = geometry
    return DomainRunner(
        organisms=scope.ctx.organisms,
        neighbors_of=neighbors_of,
        area_of=area_of,
        params=scope.ctx.params,
        rng=scope.rng,
        dt_years=scope.dt_years,
        tick=scope.tick,
        domain=domain,
        chronicle=scope.chronicle,
    )


def _step_surface(
    scope: _StepScope,
    state: WorldBiologyState,
    species_ids: Sequence[str],
) -> _Fields:
    """Run surface populations: feed, grow, migrate, burn, then viability."""
    runner = _make_runner(scope, "surface", (scope.ctx.grid.neighbors, scope.ctx.grid.area_m2))
    suitability = _surface_suitability(scope.ctx, species_ids)
    stepped = runner.step(species_ids, state.surface_populations, suitability)
    stepped = _apply_wildfire(scope, stepped)
    return runner.finalize(species_ids, stepped, state.surface_populations)


def _step_subsurface(
    scope: _StepScope,
    state: WorldBiologyState,
    species_ids: Sequence[str],
) -> _Fields:
    """Run subterranean populations through the volume graph, if one exists."""
    if scope.ctx.subsurface is None or not species_ids:
        return {sid: dict(state.subsurface_populations.get(sid, {})) for sid in species_ids}
    geometry = (scope.ctx.subsurface.neighbors, _subsurface_area_of(scope.ctx))
    runner = _make_runner(scope, "subsurface", geometry)
    suitability = _subsurface_suitability(scope.ctx, species_ids)
    stepped = runner.step(species_ids, state.subsurface_populations, suitability)
    return runner.finalize(species_ids, stepped, state.subsurface_populations)


def step_biology(
    state: WorldBiologyState,
    ctx: BiologyContext,
    rng: Rng,
    dt_s: float,
    chronicle: Chronicle | None = None,
) -> WorldBiologyState:
    """Advance the whole biology layer by one step of ``dt_s`` seconds."""
    scope = _StepScope(
        ctx=ctx,
        rng=rng,
        chronicle=chronicle,
        dt_years=dt_s / SECONDS_PER_YEAR,
        tick=state.tick,
    )
    surface_ids, subsurface_ids = _partition(ctx.organisms)
    return WorldBiologyState(
        surface_populations=_step_surface(scope, state, surface_ids),
        subsurface_populations=_step_subsurface(scope, state, subsurface_ids),
        tick=state.tick + 1,
    )


class BiologyProcess(BoundProcess["WorldBiologyState", "BiologyContext"]):
    """Orchestrator process for the biology layer (see :class:`BoundProcess`)."""

    def __init__(self, ctx: BiologyContext, rng: Rng, chronicle: Chronicle | None = None) -> None:
        """Bind the biology step to its context and a forked RNG stream."""
        super().__init__(PROCESS_NAME, ctx, rng, chronicle, step_biology)
