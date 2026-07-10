"""Shared fixtures for civilization tests: a fake grid and world builders."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from core.civilization.context import CivContext, CivParams
from core.civilization.species import (
    SUBSURFACE_DOMAIN,
    SURFACE_DOMAIN,
    SpeciesDefinition,
)
from core.civilization.tech import TechDefinition, TechEffects
from core.civilization.terrain import TerrainCostParams
from core.hydrology.sea_mask import SeaMask
from ports.grid import CellId, Grid, LatLon


class FakeGrid(Grid):
    """A tiny deterministic rows-x-cols grid with 4-neighbour adjacency."""

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        spacing_deg: float = 2.0,
        radius_m: float = 6_371_000.0,
    ) -> None:
        self._rows = rows
        self._cols = cols
        self._spacing_deg = spacing_deg
        self._radius_m = radius_m
        self._cells = tuple(f"r{r}c{c}" for r in range(rows) for c in range(cols))

    @property
    def backend_name(self) -> str:
        return "fake"

    @property
    def resolution(self) -> int:
        return 0

    @property
    def radius_m(self) -> float:
        return self._radius_m

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    def cells(self) -> Iterator[CellId]:
        return iter(self._cells)

    def _parse(self, cell: CellId) -> tuple[int, int]:
        row, _, col = cell[1:].partition("c")
        return int(row), int(col)

    def neighbors(self, cell: CellId) -> Sequence[CellId]:
        row, col = self._parse(cell)
        found = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if 0 <= r < self._rows and 0 <= c < self._cols:
                found.append(f"r{r}c{c}")
        return found

    def area_m2(self, cell: CellId) -> float:
        arc = self._radius_m * math.radians(self._spacing_deg)
        return arc * arc

    def edge_length_m(self, cell: CellId, neighbor: CellId) -> float:
        if neighbor not in self.neighbors(cell):
            msg = f"{cell} and {neighbor} are not adjacent"
            raise ValueError(msg)
        return self._radius_m * math.radians(self._spacing_deg)

    def centroid(self, cell: CellId) -> LatLon:
        row, col = self._parse(cell)
        return LatLon(lat_deg=row * self._spacing_deg, lon_deg=col * self._spacing_deg)

    def cell_at(self, point: LatLon) -> CellId:
        row = min(self._rows - 1, max(0, round(point.lat_deg / self._spacing_deg)))
        col = min(self._cols - 1, max(0, round(point.lon_deg / self._spacing_deg)))
        return f"r{row}c{col}"

    def parent(self, cell: CellId) -> CellId | None:
        return None

    def children(self, cell: CellId) -> Sequence[CellId]:
        return ()


def two_island_world(grid: FakeGrid) -> tuple[dict[str, float], SeaMask]:
    """Heights and sea mask with an ocean strip splitting west and east islands."""
    heights: dict[str, float] = {}
    ocean: dict[str, bool] = {}
    for cell in grid.cells():
        _, _, col = cell[1:].partition("c")
        is_sea = int(col) in (3, 4)
        heights[cell] = -100.0 if is_sea else 100.0
        ocean[cell] = is_sea
    mask = SeaMask(
        sea_level_m=0.0,
        ocean=ocean,
        intertidal=dict.fromkeys(ocean, False),
    )
    return heights, mask


def flat_biomass(grid: FakeGrid, mask: SeaMask, kg_m2: float = 5.0) -> dict[str, float]:
    """A uniform plant-biomass capacity field over land."""
    return {cell: kg_m2 if mask.is_land(cell) else 0.0 for cell in grid.cells()}


def make_species(
    species_id: str,
    domain: str = SURFACE_DOMAIN,
    **overrides: object,
) -> SpeciesDefinition:
    """Build a synthetic species with sensible test defaults."""
    values: dict[str, object] = {
        "species_id": species_id,
        "sapient": True,
        "domain": domain,
        "growth_rate_per_year": 0.05,
        "commit_fraction": 0.1,
        "toponym_prefixes": ("New", "Old"),
        "toponym_suffixes": ("holm", "stead"),
    }
    values.update(overrides)
    return SpeciesDefinition(**values)  # type: ignore[arg-type]


def make_tech(tech_id: str, cost: float = 10.0, **effect_overrides: float) -> TechDefinition:
    """Build a synthetic tech whose effects come from keyword overrides."""
    return TechDefinition(
        tech_id=tech_id,
        cost=cost,
        effects=TechEffects(**effect_overrides),
    )


def make_context(
    grid: FakeGrid,
    species: Sequence[SpeciesDefinition],
    techs: Sequence[TechDefinition] = (),
    resources: Sequence[object] = (),
    params: CivParams | None = None,
) -> CivContext:
    """Assemble a CivContext over the two-island world."""
    heights, mask = two_island_world(grid)
    return CivContext(
        grid=grid,
        heights_m=heights,
        sea_mask=mask,
        species={spec.species_id: spec for spec in species},
        techs=tuple(techs),  # type: ignore[arg-type]
        resources=tuple(resources),  # type: ignore[arg-type]
        biomass_capacity_kg_m2=flat_biomass(grid, mask),
        params=params if params is not None else CivParams(),
    )


def fast_params(**overrides: object) -> CivParams:
    """CivParams tuned so interesting behavior shows up within a few ticks."""
    values: dict[str, object] = {
        "influence_range_m": 600_000.0,
        "supply_range_m": 900_000.0,
        "influence_floor": 1.0,
        "settlement_population": 5_000.0,
        "settlement_spacing_m": 500_000.0,
        "terrain": TerrainCostParams(water_cost=40.0),
    }
    values.update(overrides)
    return CivParams(**values)  # type: ignore[arg-type]


SUBSURFACE = SUBSURFACE_DOMAIN
SURFACE = SURFACE_DOMAIN
