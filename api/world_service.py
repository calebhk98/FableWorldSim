"""Concrete world-sweep service: the composition root that wires the stack.

The pure sweep engine (:mod:`core.sim.world_sweep`) sequences generate ->
score -> select -> deepen over injected callables; this module supplies the
real callables, assembling grid + topography + climate + biome + rivers +
biology from concrete adapters. It is the "generate three dozen quick worlds,
keep the best, deep-sim the winners" workflow, runnable unattended and
exposed over the API (see the ``run_world_sweep`` command).

Fast preview: coarse grid, two seasons, biome classification -> habitability
score. Deep run (winners only): four seasons, a routed river network, and a
seeded biology run whose wall-clock telemetry is captured.
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
from core.biology.engine import BiologyProcess
from core.biology.organism import load_organisms
from core.biology.seed import seed_biosphere
from core.climate.model import simulate_climate
from core.hydrology.rivers import build_river_network
from core.hydrology.sea_mask import build_sea_mask
from core.sim.orchestrator import Orchestrator
from core.sim.presets import earth
from core.sim.world_sweep import SweepReport, habitability_score, run_sweep
from core.topography.procedural import ProceduralTopography

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.biology.biome import BiomeDefinition
    from core.biology.organism import Organism
    from core.biology.state import WorldBiologyState
    from core.climate.model import ClimateState
    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid

_REPO_CONTENT = Path(__file__).resolve().parents[1] / "content"
_FAST_SEASONS = 2
_DEEP_SEASONS = 4
_CHANNEL_THRESHOLD_CELL_MULTIPLE = 4.0
_DEFAULT_RESOLUTION = 1
_DEFAULT_DEEP_TICKS = 5


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


def build_world(seed: int, *, resolution: int, season_count: int) -> FastWorld:
    """Generate one world from a seed at the given grid + seasonal fidelity."""
    _organisms, biomes = _content()
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
    seed: int, *, resolution: int = _DEFAULT_RESOLUTION
) -> tuple[float, dict[str, float]]:
    """Fast-score a seed by habitability (the cheap preview)."""
    world = build_world(seed, resolution=resolution, season_count=_FAST_SEASONS)
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
) -> dict[str, object]:
    """Run the expensive simulation on one winner and summarize it.

    Rebuilds the world at higher seasonal fidelity, routes a river network,
    and steps a seeded biology run under the orchestrator — capturing its
    wall-clock telemetry. Returns a JSON-able summary.
    """
    world = build_world(seed, resolution=resolution, season_count=_DEEP_SEASONS)
    rivers = build_river_network(
        world.grid,
        world.heights_m,
        world.sea_mask,
        flow_accumulation,
        precipitation_mm_yr=world.climate.annual_precipitation_mm_yr,
        channel_threshold_m2=_CHANNEL_THRESHOLD_CELL_MULTIPLE * _mean_cell_area(world.grid),
    )
    organisms, _biomes = _content()
    below = LayersBelow(
        world.grid, world.climate, world.sea_mask, world.heights_m, world.biome_field
    )
    ctx = build_biology_context(below, {org.species_id: org for org in organisms})
    state = seed_biosphere(ctx)
    orchestrator: Orchestrator[WorldBiologyState] = Orchestrator()
    orchestrator.register(BiologyProcess(ctx, SeededRng(seed)))
    state = orchestrator.run(state, ticks=ticks)
    telemetry = orchestrator.telemetry

    channel_count = sum(1 for on in rivers.is_channel.values() if on)
    populations = {
        species: sum(field.values()) for species, field in state.surface_populations.items()
    }
    return {
        "seed": seed,
        "channel_count": channel_count,
        "biology_ticks": ticks,
        "surviving_species": sum(1 for total in populations.values() if total > 0.0),
        "populations": populations,
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
) -> SweepReport:
    """Sweep ``count`` seeds, keep the best ``keep_top_k``, and deep-sim them."""

    def evaluate(seed: int) -> tuple[float, dict[str, float]]:
        """Fast-score one seed at the sweep's resolution."""
        return evaluate_seed(seed, resolution=resolution)

    def deepen(seed: int) -> dict[str, object]:
        """Deep-sim one winning seed at the sweep's fidelity."""
        return deepen_seed(seed, resolution=resolution, ticks=deep_ticks)

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
