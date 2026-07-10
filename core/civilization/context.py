"""Shared invariants and tuning parameters for the civilization layer.

Mirrors the climate model's world-context pattern: per-run invariants are
bundled once instead of threading half a dozen arguments through every
function.  All knobs live on ``CivParams`` so tests and scenarios can retune
the layer without touching content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.civilization.terrain import TerrainCostParams

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.civilization.resources import ResourceDefinition
    from core.civilization.species import SpeciesDefinition
    from core.civilization.tech import TechDefinition
    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid


@dataclass(frozen=True)
class CivParams:
    """Engine-level tuning knobs for the civilization layer."""

    influence_range_m: float = 600_000.0
    supply_range_m: float = 900_000.0
    influence_floor: float = 1.0
    contest_ratio: float = 0.5
    initial_population: float = 50_000.0
    capacity_per_biomass: float = 0.002
    migration_per_year: float = 0.02
    workforce_fraction: float = 0.3
    biomass_regrowth_per_year: float = 0.08
    mine_rate_per_worker_year: float = 0.001
    food_value_per_kg: float = 1.0e-5
    food_per_person_year: float = 1.0
    farm_drawdown_fraction: float = 0.1
    stockpile_unit_kg: float = 1.0e6
    research_rate: float = 0.02
    arms_scale: float = 0.01
    settlement_population: float = 20_000.0
    settlement_spacing_m: float = 500_000.0
    city_population: float = 100_000.0
    terrain: TerrainCostParams = field(default_factory=TerrainCostParams)

    def __post_init__(self) -> None:
        """Reject scales that would divide by zero or invert decay."""
        if self.influence_range_m <= 0 or self.supply_range_m <= 0:
            msg = "influence and supply ranges must be > 0"
            raise ValueError(msg)
        if not 0.0 < self.contest_ratio <= 1.0:
            msg = "contest_ratio must be in (0, 1]"
            raise ValueError(msg)
        if self.stockpile_unit_kg <= 0 or self.food_per_person_year <= 0:
            msg = "economy unit scales must be > 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class CivContext:
    """Per-run invariants every civilization step shares."""

    grid: Grid
    heights_m: Mapping[CellId, float]
    sea_mask: SeaMask
    species: Mapping[str, SpeciesDefinition]
    techs: tuple[TechDefinition, ...]
    resources: tuple[ResourceDefinition, ...]
    biomass_capacity_kg_m2: Mapping[CellId, float]
    params: CivParams = field(default_factory=CivParams)
