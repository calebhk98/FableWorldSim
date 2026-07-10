"""SubsurfaceGrid port: the volumetric world under the surface shell.

The surface :class:`ports.grid.Grid` is a 2D spherical shell; this port
adds **depth** for creatures and civilizations that live *inside* the
world (dwarves, worms, sandworms, hobbit-holes).  It is a graph of
subsurface nodes — per surface cell x depth band in M1, cavern/tunnel
segments later — with vertical and lateral adjacency **distinct from
surface neighbors()**, so "tunnel under the mountain range to the other
side" is representable rather than "hop between adjacent mountain hexes".

Surface and subsurface territory can overlap (humans above, dwarves
below).  Full cave-network topology, tunnel connectivity, and
cross-massif digging are roadmap; the port exists now so they are not a
retrofit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ports.grid import CellId, Grid

NodeId = str
"""Opaque subsurface node identifier."""


class SubsurfaceGrid(ABC):
    """A volumetric graph beneath a surface grid."""

    @property
    @abstractmethod
    def surface_grid(self) -> Grid:
        """Return the surface grid this volume hangs under."""

    @property
    @abstractmethod
    def band_count(self) -> int:
        """Return the number of depth bands (band 0 is just below ground)."""

    @abstractmethod
    def band_thickness_m(self, band: int) -> float:
        """Return the vertical extent of a depth band in meters."""

    @abstractmethod
    def nodes(self) -> Iterator[NodeId]:
        """Iterate every subsurface node, in a stable order."""

    @abstractmethod
    def node_at(self, cell: CellId, band: int) -> NodeId:
        """Return the node under ``cell`` at depth ``band``."""

    @abstractmethod
    def surface_cell(self, node: NodeId) -> CellId:
        """Return the surface cell directly above ``node``."""

    @abstractmethod
    def depth_band(self, node: NodeId) -> int:
        """Return the depth band of ``node`` (0 = just below ground)."""

    @abstractmethod
    def vertical_neighbors(self, node: NodeId) -> Sequence[NodeId]:
        """Return the nodes directly above/below ``node`` (1 or 2)."""

    @abstractmethod
    def lateral_neighbors(self, node: NodeId) -> Sequence[NodeId]:
        """Return same-band nodes under the surface cell's neighbors."""

    @abstractmethod
    def diggability(self, node: NodeId) -> float:
        """Return how diggable the rock is, 0 (impenetrable) to 1 (loose)."""

    @abstractmethod
    def rock_type(self, node: NodeId) -> str:
        """Return the rock/material type at ``node`` (content-defined id)."""

    def neighbors(self, node: NodeId) -> Sequence[NodeId]:
        """Return all adjacent nodes (vertical + lateral)."""
        return (*self.vertical_neighbors(node), *self.lateral_neighbors(node))

    def volume_m3(self, node: NodeId) -> float:
        """Return the node's volume: surface-cell area x band thickness."""
        area = self.surface_grid.area_m2(self.surface_cell(node))
        return area * self.band_thickness_m(self.depth_band(node))
