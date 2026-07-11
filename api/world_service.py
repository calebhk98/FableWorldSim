"""Concrete world-sweep service: the composition root that wires the stack.

The pure sweep engine (:mod:`core.sim.world_sweep`) sequences generate ->
score -> select -> deepen over injected callables; this module supplies the
real callables, assembling grid + topography + climate + biome + rivers +
biology from concrete adapters. It is the "generate three dozen quick worlds,
keep the best, deep-sim the winners" workflow, runnable unattended and
exposed over the API (see the ``run_world_sweep`` command).

Fast preview: coarse grid, two seasons, biome classification -> habitability
score. Deep run (winners only): four seasons, a routed river network, and a
seeded biology + civilization run -- coupled through one shared plant-biomass
field -- whose wall-clock telemetry is captured.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from adapters.content_toml import TomlContentRegistry
from adapters.grid_registry import create_grid
from adapters.kernels_python import flow_accumulation
from adapters.rng_seeded import SeededRng
from core.biology.biome import biome_field, load_biomes
from core.biology.context import LayersBelow, build_biology_context
from core.biology.organism import load_organisms
from core.biology.seed import seed_biosphere
from core.civilization.context import CivContext
from core.civilization.economy import biomass_capacity_from_biomes
from core.civilization.founding import found_civilizations
from core.civilization.resources import load_resources
from core.civilization.species import load_species
from core.civilization.state import civ_population_total
from core.civilization.tech import load_techs
from core.climate.model import simulate_climate
from core.hydrology.lakes import build_lake_network
from core.hydrology.rivers import build_river_network, steepest_descent_receivers
from core.hydrology.sea_mask import build_sea_mask
from core.sim.coupling import CoupledContext, CoupledProcess, WorldState
from core.sim.orchestrator import Orchestrator
from core.sim.planet_config import PlanetConfig
from core.sim.presets import (
    earth,
    fantasy_default,
    high_tilt,
    luna,
    mars,
    tidally_locked_ocean,
    venus,
)
from core.sim.world_sweep import SweepReport, habitability_score, run_sweep
from core.topography.procedural import ProceduralTopography

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.biology.biome import BiomeDefinition
    from core.biology.organism import Organism
    from core.civilization.resources import ResourceDefinition
    from core.civilization.species import SpeciesDefinition
    from core.civilization.tech import TechDefinition
    from core.climate.model import ClimateState
    from core.hydrology.lakes import LakeNetwork
    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid

_REPO_CONTENT = Path(__file__).resolve().parents[1] / "content"
_FAST_SEASONS = 2
_DEEP_SEASONS = 4
_CHANNEL_THRESHOLD_CELL_MULTIPLE = 4.0
_DEFAULT_RESOLUTION = 1
_DEFAULT_DEEP_TICKS = 5
PEAK_BIOMASS_CAPACITY_KG_M2 = 5.0
"""Peak standing biomass (fully vegetated biome) civ capital-siting scores by.

Public (not module-private) because :mod:`api.world_state` needs the same
constant to build a civilization context for the persisted, steppable
world -- one number both call sites agree on.
"""


@dataclass(frozen=True)
class FastWorld:
    """The layers a fast preview produces, bundled for scoring and deepening."""

    grid: Grid
    heights_m: Mapping[CellId, float]
    sea_mask: SeaMask
    climate: ClimateState
    biome_field: Mapping[CellId, str]


@functools.lru_cache(maxsize=1)
def base_content() -> tuple[tuple[Organism, ...], tuple[BiomeDefinition, ...]]:
    """Load and cache the base content's organisms and biomes.

    Public: shared by this module's own deep-sim/sweep path and by
    :mod:`api.world_state`'s persisted-world builder, so both load the
    same cached registry instead of duplicating the TOML read.
    """
    registry = TomlContentRegistry([("base", _REPO_CONTENT)])
    return load_organisms(registry), load_biomes(registry)


def get_all_presets() -> dict[str, tuple[PlanetConfig, str]]:
    """Return all available scenario presets as name -> (config, description)."""
    return {
        "earth": (
            earth(),
            "Earth baseline: 1 AU from the Sun, day-night cycle, one moon.",
        ),
        "mars": (
            mars(),
            "Mars: thin CO2 atmosphere, bone-dry, two tiny moons.",
        ),
        "venus": (
            venus(),
            "Venus: retrograde spin, crushing greenhouse effect.",
        ),
        "luna": (
            luna(),
            "The Moon: airless, tidally locked, no day-night cycle.",
        ),
        "tidally_locked_ocean": (
            tidally_locked_ocean(),
            "Eyeball planet: tidally locked ocean world around a dim star.",
        ),
        "high_tilt": (
            high_tilt(),
            "Seasonal extremes: Uranus-grade axial tilt.",
        ),
        "fantasy_default": (
            fantasy_default(),
            "Fantasia: gentle Earth-plus for storytelling.",
        ),
    }


@functools.lru_cache(maxsize=1)
def civ_content() -> tuple[
    tuple[SpeciesDefinition, ...], tuple[ResourceDefinition, ...], tuple[TechDefinition, ...]
]:
    """Load and cache the civilization layer's species, resources, and techs.

    Species content lives in the same ``content/species`` directory
    ``base_content`` reads for biology organisms (each file doubles as an
    ``Organism`` and, for the sapient ones, a ``SpeciesDefinition``), so
    both loaders point at the same registry and the same species ids line
    up across layers for free. Public for the same reason as
    ``base_content``: :mod:`api.world_state` builds a civ context too.
    """
    registry = TomlContentRegistry([("base", _REPO_CONTENT)])
    return load_species(registry), load_resources(registry), load_techs(registry)


def resolve_planet(
    preset_name: str | None, planet_config: Mapping[str, object] | None
) -> PlanetConfig | None:
    """Resolve a :class:`PlanetConfig` from a custom dict or a named preset.

    ``planet_config`` wins when both are given (mirrors ``build_world``'s
    own precedence). Returns ``None`` when neither is given, meaning
    "use the caller's own default" (``build_world`` defaults to Earth).
    Raises ``ValueError`` for a malformed ``planet_config`` and
    ``KeyError`` for an unknown ``preset_name`` -- both already handled as
    422/404 by the command dispatcher (see ``api.app._execute_command``).
    """
    if planet_config is not None:
        try:
            return PlanetConfig.from_dict(planet_config)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid planet_config: {exc}") from exc
    if preset_name:
        presets = get_all_presets()
        if preset_name not in presets:
            known = ", ".join(presets.keys())
            raise KeyError(f"unknown preset {preset_name!r}; known: {known}")
        planet, _description = presets[preset_name]
        return planet
    return None


def build_world(
    seed: int,
    *,
    resolution: int,
    season_count: int,
    planet: PlanetConfig | None = None,
    grid_backend: str = "h3",
) -> FastWorld:
    """Generate one world from a seed at the given grid + seasonal fidelity.

    If planet is None, defaults to Earth.
    The grid_backend defaults to "h3" for backward compatibility.
    """
    _organisms, biomes = base_content()
    if planet is None:
        planet = earth()
    grid = create_grid(grid_backend, resolution=resolution, radius_m=planet.radius_m)
    heights = dict(ProceduralTopography(SeededRng(seed)).heights(grid))
    sea_mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    climate = simulate_climate(grid, planet, heights, sea_mask, season_count=season_count)
    field = biome_field(climate, sea_mask, biomes)
    return FastWorld(
        grid=grid, heights_m=heights, sea_mask=sea_mask, climate=climate, biome_field=field
    )


def evaluate_seed(
    seed: int,
    *,
    resolution: int = _DEFAULT_RESOLUTION,
    planet: PlanetConfig | None = None,
    grid_backend: str = "h3",
) -> tuple[float, dict[str, float]]:
    """Fast-score a seed by habitability (the cheap preview)."""
    world = build_world(
        seed,
        resolution=resolution,
        season_count=_FAST_SEASONS,
        planet=planet,
        grid_backend=grid_backend,
    )
    return habitability_score(
        world.grid, world.biome_field, world.climate.annual_mean_temperature_k, world.sea_mask
    )


def _mean_cell_area(grid: Grid) -> float:
    """Return the mean cell area over the grid (for a river threshold)."""
    cells = list(grid.cells())
    return sum(grid.area_m2(cell) for cell in cells) / len(cells)


def assemble_coupled_run(
    fast: FastWorld, seed: int, *, lakes: LakeNetwork | None = None
) -> tuple[WorldState, CoupledContext, Orchestrator[WorldState]]:
    """Seed biology + civilization on one world and register the coupled process.

    The shared spine of both the batch deep-run (:func:`deepen_seed`) and the
    persisted interactive world (:mod:`api.world_state`): build both layers'
    contexts from the same content, seed the biosphere and found civs off one
    forked RNG, and register :class:`~core.sim.coupling.CoupledProcess` on a
    fresh orchestrator. Callers either run the orchestrator to completion or
    hold it to step interactively. ``lakes`` (when routed) reaches biology as
    aquatic habitat via ``LayersBelow``.
    """
    organisms, biomes = base_content()
    species, resources, techs = civ_content()
    below = LayersBelow(
        fast.grid, fast.climate, fast.sea_mask, fast.heights_m, fast.biome_field, lakes
    )
    bio_ctx = build_biology_context(below, {org.species_id: org for org in organisms})
    civ_ctx = CivContext(
        grid=fast.grid,
        heights_m=fast.heights_m,
        sea_mask=fast.sea_mask,
        species={spec.species_id: spec for spec in species},
        techs=techs,
        resources=resources,
        biomass_capacity_kg_m2=biomass_capacity_from_biomes(
            fast.biome_field, biomes, PEAK_BIOMASS_CAPACITY_KG_M2
        ),
    )
    rng = SeededRng(seed)
    state = WorldState(
        biology=seed_biosphere(bio_ctx),
        civ=found_civilizations(civ_ctx, rng.fork("founding")),
    )
    coupled_ctx = CoupledContext(bio_ctx=bio_ctx, civ_ctx=civ_ctx, organisms=organisms)
    orchestrator: Orchestrator[WorldState] = Orchestrator()
    orchestrator.register(CoupledProcess(coupled_ctx, rng))
    return state, coupled_ctx, orchestrator


def deepen_seed(
    seed: int,
    *,
    resolution: int = _DEFAULT_RESOLUTION,
    ticks: int = _DEFAULT_DEEP_TICKS,
    planet: PlanetConfig | None = None,
    grid_backend: str = "h3",
) -> dict[str, object]:
    """Run the expensive simulation on one winner and summarize it.

    Rebuilds the world at higher seasonal fidelity, routes a river network,
    fills its depressions into a lake network (feeding aquatic habitat
    suitability), and steps a seeded biology + civilization run under the
    orchestrator —
    the two layers coupled through one shared plant-biomass field (see
    :mod:`core.sim.coupling`), so a civ's forestry and farmland measurably
    draw down what the food web itself depends on. Captures wall-clock
    telemetry and returns a JSON-able summary.
    """
    world = build_world(
        seed,
        resolution=resolution,
        season_count=_DEEP_SEASONS,
        planet=planet,
        grid_backend=grid_backend,
    )
    rivers = build_river_network(
        world.grid,
        world.heights_m,
        world.sea_mask,
        flow_accumulation,
        precipitation_mm_yr=world.climate.annual_precipitation_mm_yr,
        channel_threshold_m2=_CHANNEL_THRESHOLD_CELL_MULTIPLE * _mean_cell_area(world.grid),
    )
    # Lakes fill the depressions steepest-descent routing left as pits,
    # using the same flow-direction graph rivers just routed and the
    # climate's own precipitation for catchment inflow (build_lake_network
    # runs after climate/rivers, consuming their output -- see the module
    # docstring's sequencing note).
    lake_cells, lake_receivers, _, _ = steepest_descent_receivers(
        world.grid, world.heights_m, world.sea_mask
    )
    lakes = build_lake_network(
        world.grid,
        world.heights_m,
        world.sea_mask,
        lake_receivers,
        lake_cells,
        precipitation_mm_yr=world.climate.annual_precipitation_mm_yr,
    )
    state, _coupled_ctx, orchestrator = assemble_coupled_run(world, seed, lakes=lakes)
    state = orchestrator.run(state, ticks=ticks)
    telemetry = orchestrator.telemetry

    channel_count = sum(1 for on in rivers.is_channel.values() if on)
    lake_count = len(lakes.lakes)
    populations = {
        species_id: sum(field.values())
        for species_id, field in state.biology.surface_populations.items()
    }
    civ_population = {civ.civ_id: civ_population_total(world.grid, civ) for civ in state.civ.civs}
    return {
        "seed": seed,
        "channel_count": channel_count,
        "lake_count": lake_count,
        "biology_ticks": ticks,
        "surviving_species": sum(1 for total in populations.values() if total > 0.0),
        "populations": populations,
        "civ_population": civ_population,
        "telemetry": telemetry.to_metrics() if telemetry is not None else {},
    }


def _seeds(base_seed: int, count: int) -> list[int]:
    """Return ``count`` consecutive seeds starting at ``base_seed``."""
    return [base_seed + offset for offset in range(count)]


def run_world_sweep(  # noqa: PLR0913 - one keyword param per sweep knob
    *,
    base_seed: int = 1,
    count: int = 6,
    keep_top_k: int = 2,
    resolution: int = _DEFAULT_RESOLUTION,
    deep_ticks: int = _DEFAULT_DEEP_TICKS,
    planet: PlanetConfig | None = None,
    grid_backend: str = "h3",
) -> SweepReport:
    """Sweep ``count`` seeds, keep the best ``keep_top_k``, and deep-sim them."""

    def evaluate(seed: int) -> tuple[float, dict[str, float]]:
        """Fast-score one seed at the sweep's resolution."""
        return evaluate_seed(seed, resolution=resolution, planet=planet, grid_backend=grid_backend)

    def deepen(seed: int) -> dict[str, object]:
        """Deep-sim one winning seed at the sweep's fidelity."""
        return deepen_seed(
            seed,
            resolution=resolution,
            ticks=deep_ticks,
            planet=planet,
            grid_backend=grid_backend,
        )

    return run_sweep(_seeds(base_seed, count), evaluate, keep_top_k, deepen=deepen)


def report_to_dict(report: SweepReport) -> dict[str, object]:
    """Convert a :class:`SweepReport` to a JSON-able document."""
    return {
        "ranked": [
            {"seed": c.seed, "score": c.score, "details": dict(c.details)} for c in report.ranked
        ],
        "selected": [c.seed for c in report.selected],
        "deepened": {str(seed): result for seed, result in report.deepened.items()},
    }
