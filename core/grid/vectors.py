"""Local vector geometry on the sphere: directions and field gradients.

Vectors are (east, north) components in meters (or per-meter for
gradients) in each cell's local tangent plane — good at cell scale,
which is all the flux/wind/current math needs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports.grid import CellId, Grid, LatLon

Vector2 = tuple[float, float]
"""(east, north) components."""


def local_offset_m(origin: LatLon, target: LatLon, radius_m: float) -> Vector2:
    """Return the (east, north) offset from origin to target in meters."""
    d_lon = (target.lon_deg - origin.lon_deg + 180.0) % 360.0 - 180.0
    mean_lat = math.radians((origin.lat_deg + target.lat_deg) / 2.0)
    east = math.radians(d_lon) * math.cos(mean_lat) * radius_m
    north = math.radians(target.lat_deg - origin.lat_deg) * radius_m
    return (east, north)


def unit_direction(origin: LatLon, target: LatLon, radius_m: float) -> Vector2:
    """Return the unit (east, north) direction from origin to target."""
    east, north = local_offset_m(origin, target, radius_m)
    length = math.hypot(east, north)
    if length == 0.0:
        return (0.0, 0.0)
    return (east / length, north / length)


def gradient(
    field: Mapping[CellId, float],
    grid: Grid,
    cell: CellId,
) -> Vector2:
    """Return the local gradient of a field (per meter) at a cell.

    Averages the directional derivatives toward every neighbor —
    degree-agnostic, so it is correct at pentagons, hexes, and quads.
    """
    origin = grid.centroid(cell)
    east = 0.0
    north = 0.0
    neighbors = grid.neighbors(cell)
    for neighbor in neighbors:
        offset_e, offset_n = local_offset_m(origin, grid.centroid(neighbor), grid.radius_m)
        distance = math.hypot(offset_e, offset_n)
        if distance == 0.0:
            continue
        slope = (field[neighbor] - field[cell]) / distance
        east += slope * (offset_e / distance)
        north += slope * (offset_n / distance)
    count = max(1, len(neighbors))
    return (2.0 * east / count, 2.0 * north / count)


def divergence(
    field: Mapping[CellId, Vector2],
    grid: Grid,
    cell: CellId,
) -> float:
    """Return the local divergence (per second) of a vector field at a cell.

    Finite-volume / discrete Gauss form: for each shared edge, take the
    average of the two cells' vectors, project it onto the direction
    toward that neighbor (an outward-normal proxy that is exact for a
    Voronoi-like mesh), multiply by the edge length, sum over every
    neighbor, and divide by the cell's area — degree-agnostic, so it is
    correct at pentagons, hexes, and quads alike (mirrors the edge
    weighting in ``core.grid.flux.diffusion_step``).

    Positive means net outflow (divergence: subsiding, drying air);
    negative means net inflow (convergence: rising, moisture-piling
    air) — the sign convention ``core.climate.moisture`` relies on to
    turn wind convergence into rainout.
    """
    origin = grid.centroid(cell)
    value = field[cell]
    total_flux = 0.0
    for neighbor in grid.neighbors(cell):
        direction = unit_direction(origin, grid.centroid(neighbor), grid.radius_m)
        neighbor_value = field[neighbor]
        avg_east = (value[0] + neighbor_value[0]) / 2.0
        avg_north = (value[1] + neighbor_value[1]) / 2.0
        normal_component = avg_east * direction[0] + avg_north * direction[1]
        total_flux += normal_component * grid.edge_length_m(cell, neighbor)
    return total_flux / grid.area_m2(cell)


def rotate(vector: Vector2, angle_rad: float) -> Vector2:
    """Return the vector rotated counterclockwise by ``angle_rad``."""
    east, north = vector
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (east * cos_a - north * sin_a, east * sin_a + north * cos_a)


def magnitude(vector: Vector2) -> float:
    """Return the vector's length."""
    return math.hypot(vector[0], vector[1])
