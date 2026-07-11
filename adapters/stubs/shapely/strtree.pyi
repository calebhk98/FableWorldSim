"""Thin stub for the shapely.strtree surface FableWorldSim calls.

Only the spatial index construction and bounding-box query the ISEA
adapter uses for neighbor detection; upstream ships no type information
of its own.
"""

from collections.abc import Sequence
from typing import Any

from shapely.geometry import BaseGeometry

class STRtree:
    def __init__(self, geoms: Sequence[BaseGeometry]) -> None: ...
    def query(self, geom: BaseGeometry) -> Sequence[int] | Any: ...
