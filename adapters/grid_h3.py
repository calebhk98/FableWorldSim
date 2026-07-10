"""H3 adapter for the grid port.

Wraps Uber's H3 hexagonal DGGS (``pip install h3``, v4 API).  H3 cells are
hexagons plus exactly 12 pentagons per resolution and are *not* equal-area,
which is precisely why all simulation math is area-weighted.

Cell areas are taken in steradians and scaled by the configured planet
radius, so the same grid serves any planet size.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ports.grid import CellId, Grid, GridBackendUnavailableError, LatLon

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class H3Grid(Grid):
    """A full-sphere H3 grid at one resolution, on a sphere of any radius."""

    def __init__(self, resolution: int, radius_m: float) -> None:
        """Build the grid, importing the optional ``h3`` dependency lazily."""
        try:
            self._h3 = import_module("h3")
        except ImportError as exc:
            msg = "grid backend 'h3' needs the h3 package: pip install 'h3>=4'"
            raise GridBackendUnavailableError(msg) from exc
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
        return int(self._h3.get_num_cells(self._resolution))

    def cells(self) -> Iterator[CellId]:
        """Iterate all cells by descending from the 122 base cells."""
        for base in sorted(self._h3.get_res0_cells()):
            children = self._h3.cell_to_children(base, self._resolution)
            yield from (str(cell) for cell in sorted(children))

    def neighbors(self, cell: CellId) -> Sequence[CellId]:
        """Return edge-adjacent cells (6, or 5 around a pentagon)."""
        ring = self._h3.grid_disk(cell, 1)
        return tuple(str(other) for other in ring if str(other) != cell)

    def area_m2(self, cell: CellId) -> float:
        """Return the cell area scaled to the configured planet radius."""
        steradians = float(self._h3.cell_area(cell, unit="rads^2"))
        return steradians * self._radius_m**2

    def centroid(self, cell: CellId) -> LatLon:
        """Return the cell centroid in degrees."""
        lat, lon = self._h3.cell_to_latlng(cell)
        return LatLon(float(lat), float(lon))

    def cell_at(self, point: LatLon) -> CellId:
        """Return the cell containing a latitude/longitude point."""
        return str(self._h3.latlng_to_cell(point.lat_deg, point.lon_deg, self._resolution))


def create(resolution: int, radius_m: float) -> Grid:
    """Create an :class:`H3Grid`; registry entry point for backend 'h3'."""
    return H3Grid(resolution, radius_m)
