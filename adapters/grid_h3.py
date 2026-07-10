"""H3 adapter for the grid port.

Wraps Uber's H3 hexagonal DGGS (``pip install 'fableworldsim[grid-h3]'``,
h3 v4 API).  H3 cells are hexagons plus exactly 12 pentagons per
resolution and are *not* equal-area, which is precisely why all
simulation math is area-weighted.

This module imports ``h3`` at module level (typed via
``adapters/stubs/h3.pyi``); the grid registry converts an ImportError
into a :class:`ports.grid.GridBackendUnavailableError` with an install
hint.  Cell areas and edge lengths are taken in radians/steradians and
scaled by the configured planet radius, so the same grid serves any
planet size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h3

from ports.grid import CellId, Grid, LatLon

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_FINEST_RESOLUTION = 15


class H3Grid(Grid):
    """A full-sphere H3 grid at one resolution, on a sphere of any radius."""

    def __init__(self, resolution: int, radius_m: float) -> None:
        """Validate and store the toggle parameters."""
        if resolution < 0:
            msg = f"h3 resolution must be >= 0, got {resolution}"
            raise ValueError(msg)
        if radius_m <= 0:
            msg = f"radius_m must be > 0, got {radius_m}"
            raise ValueError(msg)
        self._resolution = resolution
        self._radius_m = radius_m

    @property
    def backend_name(self) -> str:
        """Return the toggle name of this backend."""
        return "h3"

    @property
    def resolution(self) -> int:
        """Return the H3 resolution (0-15)."""
        return self._resolution

    @property
    def radius_m(self) -> float:
        """Return the sphere radius in meters."""
        return self._radius_m

    @property
    def cell_count(self) -> int:
        """Return the number of cells at this resolution."""
        return h3.get_num_cells(self._resolution)

    def cells(self) -> Iterator[CellId]:
        """Iterate all cells by descending from the 122 base cells."""
        for base in sorted(h3.get_res0_cells()):
            children = h3.cell_to_children(base, self._resolution)
            yield from sorted(children)

    def neighbors(self, cell: CellId) -> Sequence[CellId]:
        """Return edge-adjacent cells (6, or 5 around a pentagon)."""
        return tuple(other for other in h3.grid_disk(cell, 1) if other != cell)

    def edge_length_m(self, cell: CellId, neighbor: CellId) -> float:
        """Return the shared-edge length scaled to the planet radius."""
        try:
            edge = h3.cells_to_directed_edge(cell, neighbor)
        except h3.H3NotNeighborsError as exc:
            msg = f"cells {cell} and {neighbor} are not adjacent"
            raise ValueError(msg) from exc
        return h3.edge_length(edge, unit="rads") * self._radius_m

    def parent(self, cell: CellId) -> CellId | None:
        """Return the res-1 parent, or None at resolution 0."""
        if self._resolution == 0:
            return None
        return h3.cell_to_parent(cell, self._resolution - 1)

    def children(self, cell: CellId) -> Sequence[CellId]:
        """Return the res+1 children (7, or 6 for a pentagon)."""
        if self._resolution >= _FINEST_RESOLUTION:
            return ()
        return tuple(sorted(h3.cell_to_children(cell, self._resolution + 1)))

    def area_m2(self, cell: CellId) -> float:
        """Return the cell area scaled to the configured planet radius."""
        return h3.cell_area(cell, unit="rads^2") * self._radius_m**2

    def centroid(self, cell: CellId) -> LatLon:
        """Return the cell centroid in degrees."""
        lat, lon = h3.cell_to_latlng(cell)
        return LatLon(lat, lon)

    def cell_at(self, point: LatLon) -> CellId:
        """Return the cell containing a latitude/longitude point."""
        return h3.latlng_to_cell(point.lat_deg, point.lon_deg, self._resolution)


def create(resolution: int, radius_m: float) -> Grid:
    """Create an :class:`H3Grid`; registry entry point for backend 'h3'."""
    return H3Grid(resolution, radius_m)
