"""Great-circle geometry on the simulated sphere."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ports.grid import LatLon


def great_circle_distance_m(a: LatLon, b: LatLon, radius_m: float) -> float:
    """Return the great-circle distance between two points (haversine)."""
    lat_a = math.radians(a.lat_deg)
    lat_b = math.radians(b.lat_deg)
    d_lat = lat_b - lat_a
    d_lon = math.radians(b.lon_deg - a.lon_deg)
    h = math.sin(d_lat / 2.0) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(d_lon / 2.0) ** 2
    return 2.0 * radius_m * math.asin(min(1.0, math.sqrt(h)))
