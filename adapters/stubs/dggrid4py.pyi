"""Thin stub for the dggrid4py surface FableWorldSim calls.

dggrid4py shells out to the external DGGRID binary and returns
geopandas GeoDataFrames; upstream ships no type information of its own.
Deliberately explicit — no module ``__getattr__`` — so mypy flags any
call outside the vetted surface. Extend when the adapter grows.
"""

from typing import Any

import geopandas as gpd

class DGGRIDv7:
    def __init__(
        self,
        executable: str = ...,
        working_dir: str | None = ...,
        capture_logs: bool = ...,
        silent: bool = ...,
        tmp_geo_out_legacy: bool = ...,
        has_gdal: bool = ...,
        debug: bool = ...,
    ) -> None: ...
    def is_runnable(self) -> int: ...
    def grid_cell_polygons_for_extent(
        self,
        dggs_type: str,
        resolution: int,
        mixed_aperture_level: int | None = ...,
        clip_geom: Any | None = ...,
        split_dateline: bool = ...,
        output_address_type: str | None = ...,
        **conf_extra: Any,
    ) -> gpd.GeoDataFrame: ...
    def cells_for_geo_points(
        self,
        geodf_points_wgs84: gpd.GeoDataFrame,
        cell_ids_only: bool,
        dggs_type: str,
        resolution: int,
        mixed_aperture_level: int | None = ...,
        split_dateline: bool = ...,
        output_address_type: str | None = ...,
        **conf_extra: Any,
    ) -> gpd.GeoDataFrame: ...
