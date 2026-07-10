"""DEM loader: a regular lat/lon elevation raster sampled onto the grid.

The dependency-free "load a real DEM" path: any global elevation model
resampled to a plain row-major raster (rows = latitudes north to south,
columns = longitudes west to east, meters) loads through this adapter —
Earth ETOPO, Mars MOLA, lunar LOLA all distribute in this shape.  The
GeoTIFF adapter (:mod:`adapters.dem_geotiff`) decodes into exactly this
class, keeping one sampling implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ports.topography import TopographySource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ports.grid import CellId, Grid


class LatLonGridDem(TopographySource):
    """Nearest-neighbor sampling of a global lat/lon raster."""

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        """Wrap a raster covering lat +90..-90, lon -180..+180.

        ``rows[0]`` is the northernmost latitude band; each row spans
        west to east.  All rows must be equal length.
        """
        if not rows or not rows[0]:
            msg = "DEM raster must have at least one row and column"
            raise ValueError(msg)
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            msg = "DEM raster rows must all have the same length"
            raise ValueError(msg)
        self._rows = [list(row) for row in rows]

    def heights(self, grid: Grid) -> Mapping[CellId, float]:
        """Sample the raster at every cell centroid."""
        return {cell: self.sample(*grid.centroid(cell)) for cell in grid.cells()}

    def sample(self, lat_deg: float, lon_deg: float) -> float:
        """Return the raster elevation nearest a lat/lon point."""
        row_count = len(self._rows)
        col_count = len(self._rows[0])
        row_pos = (90.0 - lat_deg) / 180.0 * row_count
        row = min(row_count - 1, max(0, int(row_pos)))
        lon_wrapped = ((lon_deg + 180.0) % 360.0) / 360.0 * col_count
        col = min(col_count - 1, max(0, int(lon_wrapped)))
        return float(self._rows[row][col])
