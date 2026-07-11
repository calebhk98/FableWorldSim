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
from core.hydrology.rivers import build_river_network
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
    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid

_REPO_CONTENT = Path(__file__).resolve().parents[1] / "content"
_FAST_SEASONS = 2
_DEEP_SEASONS = 4
_CHANNEL_THRESHOLD_CELL_MULTIPLE = 4.0
_DEFAULT_RESOLUTION = 1
_DEFAULT_DEEP_TICKS = 5
_PEAK_BIOMASS_CAPACITY_KG_M2 = 5.0
"""Peak standing biomass (fully vegetated biome) civ capital-siting scores by."""


@dataclass(frozen=True)
class FastWorld:
    """The layers a fast preview produces, bundled for scoring and deepening."""

    grid: Grid
    heights_m: Mapping[CellId, float]
    sea_mask: SeaMask
    climate: ClimateState
    biome_field: Mapping[CellId, str]


@functools.lru_cache(maxsize=1)
def _content() -> tuple[tuple[Organism, ...], tuple[BiomeDefinition, ...]]:
    """Load and cache the base content's organisms and biomes."""
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
def _civ_content() -> tuple[
    tuple[SpeciesDefinition, ...], tuple[ResourceDefinition, ...], tuple[TechDefinition, ...]
]:
    """Load and cache the civilization layer's species, resources, and techs.

    Species content lives in the same ``content/species`` directory
    ``_content`` reads for biology organisms (each file doubles as an
    ``Organism`` and, for the sapient ones, a ``SpeciesDefinition``), so
    both loaders point at the same registry and the same species ids line
    up across layers for free.
    """
    registry = TomlContentRegistry([("base", _REPO_CONTENT)])
    return load_species(registry), load_resources(registry), load_techs(registry)


def build_world(
    seed: int,
    *,
    resolution: int,
    season_count: int,
    planet: PlanetConfig | None = None,
) -> FastWorld:
    """Generate one world from a seed at the given grid + seasonal fidelity.

    If planet is None, defaults to Earth.
    """
    _organisms, biomes = _content()
    if planet is None:
        planet = earth()
    grid = create_grid("h3", resolution=resolution, radius_m=planet.radius_m)
    heights = dict(ProceduralTopography(SeededRng(seed)).heights(grid))
    sea_mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    climate = simulate_climate(grid, planet, heights, sea_mask, season_count=season_count)
    field = biome_field(climate, sea_mask, biomes)
    return FastWorld(
        grid=grid, heights_m=heights, sea_mask=sea_mask, climate=climate, biome_field=field
    )


def evaluate_seed(
    seed: int, *, resolution: int = _DEFAULT_RESOLUTION, planet: PlanetConfig | None = None
) -> tuple[float, dict[str, float]]:
    """Fast-score a seed by habitability (the cheap preview)."""
    world = build_world(seed, resolution=resolution, season_count=_FAST_SEASONS, planet=planet)
    return habitability_score(
        world.grid, world.biome_field, world.climate.annual_mean_temperature_k, world.sea_mask
    )


def _mean_cell_area(grid: Grid) -> float:
    """Return the mean cell area over the grid (for a river threshold)."""
    cells = list(grid.cells())
    return sum(grid.area_m2(cell) for cell in cells) / len(cells)


def deepen_seed(
    seed: int,
    *,
    resolution: int = _DEFAULT_RESOLUTION,
    ticks: int = _DEFAULT_DEEP_TICKS,
    planet: PlanetConfig | None = None,
) -> dict[str, object]:
    """Run the expensive simulation on one winner and summarize it.

    Rebuilds the world at higher seasonal fidelity, routes a river network,
    and steps a seeded biology + civilization run under the orchestrator —
    the two layers coupled through one shared plant-biomass field (see
    :mod:`core.sim.coupling`), so a civ's forestry and farmland measurably
    draw down what the food web itself depends on. Captures wall-clock
    telemetry and returns a JSON-able summary.
    """
    world = build_world(seed, resolution=resolution, season_count=_DEEP_SEASONS, planet=planet)
    rivers = build_river_network(
        world.grid,
        world.heights_m,
        world.sea_mask,
        flow_accumulation,
        precipitation_mm_yr=world.climate.annual_precipitation_mm_yr,
        channel_threshold_m2=_CHANNEL_THRESHOLD_CELL_MULTIPLE * _mean_cell_area(world.grid),
    )
    organisms, biomes = _content()
    species, resources, techs = _civ_content()
    below = LayersBelow(
        world.grid, world.climate, world.sea_mask, world.heights_m, world.biome_field
    )
    bio_ctx = build_biology_context(below, {org.species_id: org for org in organisms})
    civ_ctx = CivContext(
        grid=world.grid,
        heights_m=world.heights_m,
        sea_mask=world.sea_mask,
        species={spec.species_id: spec for spec in species},
        techs=techs,
        resources=resources,
        biomass_capacity_kg_m2=biomass_capacity_from_biomes(
            world.biome_field, biomes, _PEAK_BIOMASS_CAPACITY_KG_M2
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
    state = orchestrator.run(state, ticks=ticks)
    telemetry = orchestrator.telemetry

    channel_count = sum(1 for on in rivers.is_channel.values() if on)
    populations = {
        species_id: sum(field.values())
        for species_id, field in state.biology.surface_populations.items()
    }
    civ_population = {
        civ.civ_id: civ_population_total(world.grid, civ) for civ in state.civ.civs
    }
    return {
        "seed": seed,
        "channel_count": channel_count,
        "biology_ticks": ticks,
        "surviving_species": sum(1 for total in populations.values() if total > 0.0),
        "populations": populations,
        "civ_population": civ_population,
        "telemetry": telemetry.to_metrics() if telemetry is not None else {},
    }


def _seeds(base_seed: int, count: int) -> list[int]:
    """Return ``count`` consecutive seeds starting at ``base_seed``."""
    return [base_seed + offset for offset in range(count)]


def run_world_sweep(
    *,
    base_seed: int = 1,
    count: int = 6,
    keep_top_k: int = 2,
    resolution: int = _DEFAULT_RESOLUTION,
    deep_ticks: int = _DEFAULT_DEEP_TICKS,
    planet: PlanetConfig | None = None,
) -> SweepReport:
    """Sweep ``count`` seeds, keep the best ``keep_top_k``, and deep-sim them."""

    def evaluate(seed: int) -> tuple[float, dict[str, float]]:
        """Fast-score one seed at the sweep's resolution."""
        return evaluate_seed(seed, resolution=resolution, planet=planet)

    def deepen(seed: int) -> dict[str, object]:
        """Deep-sim one winning seed at the sweep's fidelity."""
        return deepen_seed(seed, resolution=resolution, ticks=deep_ticks, planet=planet)

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
