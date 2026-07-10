"""Minimal M1 subsurface: uniform depth bands under every surface cell.

A few depth bands per cell with vertical + lateral adjacency and a
per-band rock-type/diggability field — enough for subterranean species
and dwarf mining to route through a real volumetric graph.  Cave-network
topology, tunnel connectivity, and cross-massif digging are roadmap; they
replace this implementation behind the same port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ports.subsurface import NodeId, SubsurfaceGrid

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ports.grid import CellId, Grid

_SEPARATOR = "@"


@dataclass(frozen=True)
class BandSpec:
    """One depth band's uniform properties."""

    thickness_m: float
    rock_type: str
    diggability: float

    def __post_init__(self) -> None:
        """Validate thickness and diggability."""
        if self.thickness_m <= 0:
            msg = f"band thickness must be > 0, got {self.thickness_m}"
            raise ValueError(msg)
        if not 0.0 <= self.diggability <= 1.0:
            msg = f"diggability must be in [0, 1], got {self.diggability}"
            raise ValueError(msg)


class DepthBandSubsurface(SubsurfaceGrid):
    """Uniform depth bands hanging under a surface grid."""

    def __init__(self, grid: Grid, bands: Sequence[BandSpec]) -> None:
        """Create ``len(bands)`` bands under every surface cell."""
        if not bands:
            msg = "need at least one depth band"
            raise ValueError(msg)
        self._grid = grid
        self._bands = tuple(bands)

    @property
    def surface_grid(self) -> Grid:
        """Return the surface grid this volume hangs under."""
        return self._grid

    @property
    def band_count(self) -> int:
        """Return the number of depth bands."""
        return len(self._bands)

    def band_thickness_m(self, band: int) -> float:
        """Return the vertical extent of a depth band."""
        return self._band(band).thickness_m

    def _band(self, band: int) -> BandSpec:
        """Return a band spec, validating the index."""
        if not 0 <= band < len(self._bands):
            msg = f"band must be in [0, {len(self._bands) - 1}], got {band}"
            raise ValueError(msg)
        return self._bands[band]

    def nodes(self) -> Iterator[NodeId]:
        """Iterate every (cell, band) node, cells outer, bands inner."""
        for cell in self._grid.cells():
            for band in range(len(self._bands)):
                yield self.node_at(cell, band)

    def node_at(self, cell: CellId, band: int) -> NodeId:
        """Return the node under ``cell`` at depth ``band``."""
        self._band(band)
        return f"{cell}{_SEPARATOR}{band}"

    def surface_cell(self, node: NodeId) -> CellId:
        """Return the surface cell directly above ``node``."""
        cell, _, _ = node.rpartition(_SEPARATOR)
        return cell

    def depth_band(self, node: NodeId) -> int:
        """Return the depth band of ``node``."""
        _, _, band = node.rpartition(_SEPARATOR)
        return int(band)

    def vertical_neighbors(self, node: NodeId) -> Sequence[NodeId]:
        """Return the nodes directly above and/or below ``node``."""
        cell = self.surface_cell(node)
        band = self.depth_band(node)
        found = []
        if band > 0:
            found.append(self.node_at(cell, band - 1))
        if band < len(self._bands) - 1:
            found.append(self.node_at(cell, band + 1))
        return tuple(found)

    def lateral_neighbors(self, node: NodeId) -> Sequence[NodeId]:
        """Return same-band nodes under the surface cell's neighbors."""
        cell = self.surface_cell(node)
        band = self.depth_band(node)
        return tuple(self.node_at(neighbor, band) for neighbor in self._grid.neighbors(cell))

    def diggability(self, node: NodeId) -> float:
        """Return the band's diggability at ``node``."""
        return self._band(self.depth_band(node)).diggability

    def rock_type(self, node: NodeId) -> str:
        """Return the band's rock type at ``node``."""
        return self._band(self.depth_band(node)).rock_type
