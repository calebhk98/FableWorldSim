"""S2 adapter for the grid port.

Wraps Google's S2 quad-cell DGGS via the pure-Python ``s2sphere`` package
(``pip install 'fableworldsim[grid-s2]'``).  S2 cells are quadrilaterals
from a cube projected onto the sphere; they are *not* equal-area, which
is why all simulation math is area-weighted.  Upstream is unmaintained,
so S2 is a display/interop toggle, not a recommended default.

This module imports ``s2sphere`` at module level (typed via
``adapters/stubs/s2sphere.pyi``); the grid registry converts an
ImportError into a backend-unavailable error with an install hint.
Cell ids are exposed as S2 tokens (strings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import s2sphere

from ports.grid import CellId, Grid, LatLon

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_FACES = 6
_MAX_LEVEL = 30
_VERTEX_TOL_RAD = 1e-9
_SHARED_VERTEX_COUNT = 2


class S2Grid(Grid):
    """A full-sphere S2 grid at one level, on a sphere of any radius."""

    def __init__(self, resolution: int, radius_m: float) -> None:
        """Validate and store the toggle parameters."""
        if resolution < 0:
            msg = f"s2 level must be >= 0, got {resolution}"
            raise ValueError(msg)
        if radius_m <= 0:
            msg = f"radius_m must be > 0, got {radius_m}"
            raise ValueError(msg)
        self._resolution = resolution
        self._radius_m = radius_m

    @property
    def backend_name(self) -> str:
        """Return the toggle name of this backend."""
        return "s2"

    @property
    def resolution(self) -> int:
        """Return the S2 cell level."""
        return self._resolution

    @property
    def radius_m(self) -> float:
        """Return the sphere radius in meters."""
        return self._radius_m

    @property
    def cell_count(self) -> int:
        """Return the number of cells at this level (6 * 4**level)."""
        return _FACES * int(4**self._resolution)

    def cells(self) -> Iterator[CellId]:
        """Iterate all cells in Hilbert-curve order."""
        cell_id = s2sphere.CellId.begin(self._resolution)
        end = s2sphere.CellId.end(self._resolution)
        while cell_id != end:
            yield cell_id.to_token()
            cell_id = cell_id.next()

    def neighbors(self, cell: CellId) -> Sequence[CellId]:
        """Return the four edge-adjacent cells."""
        cell_id = s2sphere.CellId.from_token(cell)
        return tuple(other.to_token() for other in cell_id.get_edge_neighbors())

    def edge_length_m(self, cell: CellId, neighbor: CellId) -> float:
        """Return the shared-edge length scaled to the planet radius.

        s2sphere exposes no direct shared-edge query, so this matches the
        two vertices the adjacent cells have in common and measures the
        great-circle arc between them.
        """
        mine = self._vertices(cell)
        theirs = self._vertices(neighbor)
        shared = [v for v in mine if any(v.angle(w) < _VERTEX_TOL_RAD for w in theirs)]
        if len(shared) != _SHARED_VERTEX_COUNT:
            msg = f"cells {cell} and {neighbor} are not adjacent"
            raise ValueError(msg)
        return shared[0].angle(shared[1]) * self._radius_m

    def _vertices(self, cell: CellId) -> list[s2sphere.Point]:
        """Return the four corner points of a cell."""
        s2_cell = s2sphere.Cell(s2sphere.CellId.from_token(cell))
        return [s2_cell.get_vertex(k) for k in range(4)]

    def parent(self, cell: CellId) -> CellId | None:
        """Return the level-1 parent, or None at level 0."""
        if self._resolution == 0:
            return None
        cell_id = s2sphere.CellId.from_token(cell)
        return cell_id.parent(self._resolution - 1).to_token()

    def children(self, cell: CellId) -> Sequence[CellId]:
        """Return the four level+1 children."""
        if self._resolution >= _MAX_LEVEL:
            return ()
        cell_id = s2sphere.CellId.from_token(cell)
        return tuple(kid.to_token() for kid in cell_id.children())

    def area_m2(self, cell: CellId) -> float:
        """Return the exact cell area scaled to the planet radius."""
        cell_id = s2sphere.CellId.from_token(cell)
        return s2sphere.Cell(cell_id).exact_area() * self._radius_m**2

    def centroid(self, cell: CellId) -> LatLon:
        """Return the cell center in degrees."""
        latlng = s2sphere.CellId.from_token(cell).to_lat_lng()
        return LatLon(float(latlng.lat().degrees), float(latlng.lng().degrees))

    def cell_at(self, point: LatLon) -> CellId:
        """Return the cell containing a latitude/longitude point."""
        latlng = s2sphere.LatLng.from_degrees(point.lat_deg, point.lon_deg)
        return s2sphere.CellId.from_lat_lng(latlng).parent(self._resolution).to_token()


def create(resolution: int, radius_m: float) -> Grid:
    """Create an :class:`S2Grid`; registry entry point for backend 's2'."""
    return S2Grid(resolution, radius_m)
