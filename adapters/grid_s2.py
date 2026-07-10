"""S2 adapter for the grid port.

Wraps Google's S2 quad-cell DGGS via the pure-Python ``s2sphere`` package.
S2 cells are quadrilaterals from a cube projected onto the sphere; they
are *not* equal-area, which is why all simulation math is area-weighted.

Cell ids are exposed as S2 tokens (strings).  Areas are exact spherical
areas in steradians scaled by the configured planet radius.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ports.grid import CellId, Grid, GridBackendUnavailableError, LatLon

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_FACES = 6


class S2Grid(Grid):
    """A full-sphere S2 grid at one level, on a sphere of any radius."""

    def __init__(self, resolution: int, radius_m: float) -> None:
        """Build the grid, importing optional ``s2sphere`` lazily."""
        try:
            self._s2 = import_module("s2sphere")
        except ImportError as exc:
            msg = "grid backend 's2' needs s2sphere: pip install s2sphere"
            raise GridBackendUnavailableError(msg) from exc
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
        cell_id = self._s2.CellId.begin(self._resolution)
        end = self._s2.CellId.end(self._resolution)
        while cell_id != end:
            yield str(cell_id.to_token())
            cell_id = cell_id.next()

    def neighbors(self, cell: CellId) -> Sequence[CellId]:
        """Return the four edge-adjacent cells."""
        cell_id = self._s2.CellId.from_token(cell)
        return tuple(str(other.to_token()) for other in cell_id.get_edge_neighbors())

    def area_m2(self, cell: CellId) -> float:
        """Return the exact cell area scaled to the planet radius."""
        cell_id = self._s2.CellId.from_token(cell)
        steradians = float(self._s2.Cell(cell_id).exact_area())
        return steradians * self._radius_m**2

    def centroid(self, cell: CellId) -> LatLon:
        """Return the cell center in degrees."""
        cell_id = self._s2.CellId.from_token(cell)
        latlng = cell_id.to_lat_lng()
        return LatLon(float(latlng.lat().degrees), float(latlng.lng().degrees))

    def cell_at(self, point: LatLon) -> CellId:
        """Return the cell containing a latitude/longitude point."""
        latlng = self._s2.LatLng.from_degrees(point.lat_deg, point.lon_deg)
        cell_id = self._s2.CellId.from_lat_lng(latlng).parent(self._resolution)
        return str(cell_id.to_token())


def create(resolution: int, radius_m: float) -> Grid:
    """Create an :class:`S2Grid`; registry entry point for backend 's2'."""
    return S2Grid(resolution, radius_m)
