"""Per-run invariants and tuning knobs for the biology layer.

Mirrors the climate and civilization world-context pattern: the fields the
biology step reads every tick (climate-derived trait axes, the biome map,
the subsurface temperature) are bundled once instead of threaded through
every function, and all tuning lives on :class:`BiologyParams` so tests and
scenarios retune the layer without touching content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.biology.suitability import ALTITUDE_AXIS, PRECIPITATION_AXIS, TEMPERATURE_AXIS
from core.biology.wildfire import FireParams

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.biology.organism import Organism
    from core.climate.model import ClimateState
    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid
    from ports.subsurface import NodeId, SubsurfaceGrid

_DEFAULT_WET_PRECIP_MM = 1000.0
"""Annual precipitation at and above which a cell is treated as fully wet."""


@dataclass(frozen=True)
class BiologyParams:
    """Engine-level tuning knobs for the biology layer."""

    intake_kg_per_kg_body_year: float = 8.0
    max_offtake_fraction: float = 0.5
    migration_per_year: float = 0.1
    migration_jitter: float = 0.2
    base_biome_preference: float = 0.5
    no_habitat_decay_per_year: float = 0.5
    extinction_epsilon_per_m2: float = 1.0e-15
    wet_precip_mm: float = _DEFAULT_WET_PRECIP_MM
    fire: FireParams = field(default_factory=FireParams)


@dataclass(frozen=True)
class LayersBelow:
    """The outputs of the layers beneath biology, bundled for the builder."""

    grid: Grid
    climate: ClimateState
    sea_mask: SeaMask
    heights_m: Mapping[CellId, float]
    biome_field: Mapping[CellId, str]


@dataclass(frozen=True)
class BiologyContext:
    """The fields and knobs every biology step shares for one run."""

    grid: Grid
    sea_mask: SeaMask
    organisms: Mapping[str, Organism]
    axis_fields: Mapping[str, Mapping[CellId, float]]
    biome_field: Mapping[CellId, str]
    dryness_by_cell: Mapping[CellId, float]
    land_cells: tuple[CellId, ...]
    subsurface: SubsurfaceGrid | None = None
    subsurface_temperature: Mapping[NodeId, float] = field(default_factory=dict)
    subsurface_diggability: Mapping[NodeId, float] = field(default_factory=dict)
    params: BiologyParams = field(default_factory=BiologyParams)


def _dryness_field(
    precipitation: Mapping[CellId, float],
    wet_precip_mm: float,
) -> dict[CellId, float]:
    """Return per-cell dryness in 0..1 (drier cells are more fire-prone)."""
    return {cell: max(0.0, 1.0 - precip / wet_precip_mm) for cell, precip in precipitation.items()}


def _subsurface_temperature(
    subsurface: SubsurfaceGrid,
    annual_temperature: Mapping[CellId, float],
) -> dict[NodeId, float]:
    """Map each node to the buffered (annual-mean) temperature above it."""
    return {
        node: annual_temperature[subsurface.surface_cell(node)]
        for node in subsurface.nodes()
        if subsurface.surface_cell(node) in annual_temperature
    }


def build_biology_context(
    below: LayersBelow,
    organisms: Mapping[str, Organism],
    subsurface: SubsurfaceGrid | None = None,
    params: BiologyParams | None = None,
) -> BiologyContext:
    """Assemble a :class:`BiologyContext` from the layers below biology."""
    params = params if params is not None else BiologyParams()
    climate = below.climate
    axis_fields = {
        TEMPERATURE_AXIS: dict(climate.annual_mean_temperature_k),
        PRECIPITATION_AXIS: dict(climate.annual_precipitation_mm_yr),
        ALTITUDE_AXIS: dict(below.heights_m),
    }
    land_cells = tuple(cell for cell in below.grid.cells() if below.sea_mask.is_land(cell))
    sub_temp: dict[NodeId, float] = {}
    sub_dig: dict[NodeId, float] = {}
    if subsurface is not None:
        sub_temp = _subsurface_temperature(subsurface, climate.annual_mean_temperature_k)
        sub_dig = {node: subsurface.diggability(node) for node in subsurface.nodes()}
    return BiologyContext(
        grid=below.grid,
        sea_mask=below.sea_mask,
        organisms=organisms,
        axis_fields=axis_fields,
        biome_field=below.biome_field,
        dryness_by_cell=_dryness_field(climate.annual_precipitation_mm_yr, params.wet_precip_mm),
        land_cells=land_cells,
        subsurface=subsurface,
        subsurface_temperature=sub_temp,
        subsurface_diggability=sub_dig,
        params=params,
    )
