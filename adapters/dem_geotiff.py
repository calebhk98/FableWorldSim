"""GeoTIFF DEM adapter: real-world elevation files onto the grid.

Reads a global GeoTIFF (Earth/Mars/Moon DEMs as distributed by USGS and
friends) via the optional ``rasterio`` dependency
(``pip install 'fableworldsim[gis]'``) and delegates sampling to
:class:`adapters.dem_latlon.LatLonGridDem`.

A license/attribution manifest travels with imported real-world data:
pass ``attribution`` and it is preserved for the world recipe/wiki.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from adapters.dem_latlon import LatLonGridDem
from ports.topography import TopographySource

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ports.grid import CellId, Grid


class GisBackendUnavailableError(RuntimeError):
    """Raised when the optional GIS dependency is not installed."""


class GeoTiffDem(TopographySource):
    """A GeoTIFF elevation raster, decoded once and sampled per cell."""

    def __init__(self, path: Path | str, attribution: str = "") -> None:
        """Decode the raster (band 1) into the lat/lon sampler."""
        try:
            rasterio = import_module("rasterio")
        except ImportError as exc:
            msg = "GeoTIFF import needs rasterio: pip install 'fableworldsim[gis]'"
            raise GisBackendUnavailableError(msg) from exc
        with rasterio.open(str(path)) as dataset:
            band = dataset.read(1)
        self._dem = LatLonGridDem([[float(v) for v in row] for row in band])
        self.attribution = attribution
        """License/attribution text that travels with this data."""

    def heights(self, grid: Grid) -> Mapping[CellId, float]:
        """Sample the raster at every cell centroid."""
        return self._dem.heights(grid)
