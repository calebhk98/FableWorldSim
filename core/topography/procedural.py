"""Procedural terrain: multi-octave bumps on the sphere + plate-ish uplift.

Deterministic (all randomness through the Rng port): a few large
low-frequency bumps make continents and basins, higher octaves add
mountain-scale roughness, and a set of random great-circle "plate
boundaries" contribute ridge uplift — enough structure for coastlines,
ranges, and rain shadows without a real tectonics generator (roadmap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ports.topography import TopographySource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ports.grid import CellId, Grid, LatLon
    from ports.rng import Rng

_Vector = tuple[float, float, float]


def _unit_vector(point: LatLon) -> _Vector:
    """Return the 3D unit vector of a lat/lon point."""
    lat = math.radians(point.lat_deg)
    lon = math.radians(point.lon_deg)
    return (
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    )


def _angle_between(a: _Vector, b: _Vector) -> float:
    """Return the central angle (radians) between two unit vectors."""
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.acos(min(1.0, max(-1.0, dot)))


@dataclass(frozen=True)
class TerrainStyle:
    """Tunable knobs for procedural terrain (defaults are Earth-ish)."""

    octaves: int = 4
    base_bumps: int = 6
    amplitude_m: float = 2_500.0
    ridge_count: int = 4
    ridge_height_m: float = 1_500.0

    def __post_init__(self) -> None:
        """Validate the style."""
        if self.octaves < 1:
            msg = f"octaves must be >= 1, got {self.octaves}"
            raise ValueError(msg)


_BASIN_PROBABILITY = 0.5
"""Chance a bump is a basin (negative) instead of an uplift."""


class ProceduralTopography(TopographySource):
    """Seeded bump-mixture terrain generator."""

    def __init__(self, rng: Rng, style: TerrainStyle | None = None) -> None:
        """Draw all random structure up front from the given stream."""
        chosen = style if style is not None else TerrainStyle()
        self._bumps = self._draw_bumps(rng.fork("bumps"), chosen)
        self._ridges = self._draw_ridges(rng.fork("ridges"), chosen.ridge_count)
        self._ridge_height_m = chosen.ridge_height_m

    @staticmethod
    def _draw_bumps(rng: Rng, style: TerrainStyle) -> list[tuple[_Vector, float, float]]:
        """Return (center, amplitude, width) bump components, all octaves."""
        bumps = []
        for octave in range(style.octaves):
            count = style.base_bumps * 2**octave
            amplitude = style.amplitude_m / 2**octave
            width_rad = 1.2 / 2**octave
            for _ in range(count):
                center = _random_unit_vector(rng)
                sign = -1.0 if rng.random() < _BASIN_PROBABILITY else 1.0
                bumps.append((center, sign * amplitude, width_rad))
        return bumps

    @staticmethod
    def _draw_ridges(rng: Rng, count: int) -> list[tuple[_Vector, float]]:
        """Return (pole, width) great-circle ridge components."""
        return [(_random_unit_vector(rng), 0.06 + 0.06 * rng.random()) for _ in range(count)]

    def heights(self, grid: Grid) -> Mapping[CellId, float]:
        """Return the generated elevation for every cell."""
        return {cell: self._height_at(grid.centroid(cell)) for cell in grid.cells()}

    def _height_at(self, point: LatLon) -> float:
        """Evaluate the bump mixture + ridge uplift at one point."""
        position = _unit_vector(point)
        height = 0.0
        for center, amplitude, width in self._bumps:
            angle = _angle_between(position, center)
            height += amplitude * math.exp(-((angle / width) ** 2))
        for pole, width in self._ridges:
            off_circle = abs(_angle_between(position, pole) - math.pi / 2.0)
            height += self._ridge_height_m * math.exp(-((off_circle / width) ** 2))
        return height


def _random_unit_vector(rng: Rng) -> _Vector:
    """Return a uniformly distributed point on the unit sphere."""
    z = rng.uniform(-1.0, 1.0)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    radial = math.sqrt(max(0.0, 1.0 - z * z))
    return (radial * math.cos(theta), radial * math.sin(theta), z)


def field_stats(heights: Mapping[CellId, float]) -> tuple[float, float]:
    """Return (min, max) of a height field (handy for tests/goldens)."""
    values: Sequence[float] = list(heights.values())
    return (min(values), max(values))
