"""Thin stub for the geopandas surface FableWorldSim calls.

Only the constructor and attribute access the ISEA adapter touches;
upstream ships no type information of its own.
"""

from collections.abc import Iterator, Sequence
from typing import Any

class GeoSeries:
    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, index: int) -> Any: ...
    @property
    def iloc(self) -> GeoSeries: ...

class GeoDataFrame:
    geometry: GeoSeries
    columns: Sequence[str]

    def __init__(self, data: Any = ..., *, crs: Any = ..., geometry: str = ...) -> None: ...
    def __getitem__(self, key: str) -> GeoSeries: ...
