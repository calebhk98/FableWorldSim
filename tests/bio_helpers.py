"""Shared fixtures for biology tests: synthetic organisms, grids, contexts."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from core.biology.context import BiologyContext, BiologyParams
from core.biology.organism import Organism
from core.biology.suitability import ALTITUDE_AXIS, PRECIPITATION_AXIS, TEMPERATURE_AXIS
from core.hydrology.sea_mask import SeaMask
from ports.grid import CellId
from ports.subsurface import NodeId, SubsurfaceGrid
from tests.civ_helpers import FakeGrid

_BAND = "@0"


def make_organism(species_id: str, **overrides: object) -> Organism:
    """Build a synthetic organism with sensible, sustainable defaults."""
    values: dict[str, object] = {
        "species_id": species_id,
        "medium": "terrestrial",
        "body_mass_kg": 1.0,
        "crowding_cap_per_m2": 1.0,
        "lifespan_years": 10.0,
        "maturity_years": 1.0,
        "litter_size": 6.0,
    }
    values.update(overrides)
    return Organism(**values)  # type: ignore[arg-type]


def all_land_mask(grid: FakeGrid) -> SeaMask:
    """A sea mask where every cell is dry land (no ocean, no intertidal)."""
    ocean = {cell: False for cell in grid.cells()}
    return SeaMask(sea_level_m=0.0, ocean=ocean, intertidal=dict.fromkeys(ocean, False))


@dataclass(frozen=True)
class UniformEnv:
    """The spatially uniform environment a test context is built over."""

    temperature_k: float = 290.0
    precipitation_mm_yr: float = 800.0
    altitude_m: float = 100.0
    biome_id: str = ""
    sea_mask: SeaMask | None = None


def uniform_context(
    grid: FakeGrid,
    organisms: Sequence[Organism],
    env: UniformEnv | None = None,
    subsurface: SubsurfaceGrid | None = None,
    params: BiologyParams | None = None,
) -> BiologyContext:
    """Assemble a BiologyContext with spatially uniform environmental fields."""
    env = env if env is not None else UniformEnv()
    mask = env.sea_mask if env.sea_mask is not None else all_land_mask(grid)
    cells = list(grid.cells())
    axis_fields = {
        TEMPERATURE_AXIS: dict.fromkeys(cells, env.temperature_k),
        PRECIPITATION_AXIS: dict.fromkeys(cells, env.precipitation_mm_yr),
        ALTITUDE_AXIS: dict.fromkeys(cells, env.altitude_m),
    }
    biome_field = {cell: env.biome_id for cell in cells if mask.is_land(cell)}
    land_cells = tuple(cell for cell in cells if mask.is_land(cell))
    annual = dict.fromkeys(cells, env.temperature_k)
    dryness = {cell: max(0.0, 1.0 - env.precipitation_mm_yr / 1000.0) for cell in cells}
    sub_temp: dict[NodeId, float] = {}
    sub_dig: dict[NodeId, float] = {}
    if subsurface is not None:
        sub_temp = {node: annual[subsurface.surface_cell(node)] for node in subsurface.nodes()}
        sub_dig = {node: subsurface.diggability(node) for node in subsurface.nodes()}
    return BiologyContext(
        grid=grid,
        sea_mask=mask,
        organisms={org.species_id: org for org in organisms},
        axis_fields=axis_fields,
        biome_field=biome_field,
        dryness_by_cell=dryness,
        land_cells=land_cells,
        subsurface=subsurface,
        subsurface_temperature=sub_temp,
        subsurface_diggability=sub_dig,
        params=params if params is not None else BiologyParams(),
    )


class FakeSubsurface(SubsurfaceGrid):
    """A single-band volume graph with hand-authored lateral tunnels.

    Lateral adjacency is given explicitly, so a test can connect two surface
    cells that are *not* surface-adjacent — the tunnel-under-the-massif case
    the design calls for — and verify digging life diffuses through it.
    """

    def __init__(self, grid: FakeGrid, tunnels: dict[CellId, tuple[CellId, ...]]) -> None:
        self._grid = grid
        self._tunnels = tunnels

    @property
    def surface_grid(self) -> FakeGrid:
        return self._grid

    @property
    def band_count(self) -> int:
        return 1

    def band_thickness_m(self, band: int) -> float:
        return 100.0

    def nodes(self) -> Iterator[NodeId]:
        return (f"{cell}{_BAND}" for cell in self._grid.cells())

    def node_at(self, cell: CellId, band: int) -> NodeId:
        return f"{cell}{_BAND}"

    def surface_cell(self, node: NodeId) -> CellId:
        return node.split(_BAND)[0]

    def depth_band(self, node: NodeId) -> int:
        return 0

    def vertical_neighbors(self, node: NodeId) -> Sequence[NodeId]:
        return ()

    def lateral_neighbors(self, node: NodeId) -> Sequence[NodeId]:
        cell = self.surface_cell(node)
        return tuple(f"{other}{_BAND}" for other in self._tunnels.get(cell, ()))

    def diggability(self, node: NodeId) -> float:
        return 1.0

    def rock_type(self, node: NodeId) -> str:
        return "rock"
