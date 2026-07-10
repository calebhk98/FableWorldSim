"""Insolation: top-of-atmosphere sunlight per cell, any planet.

Driven entirely by PlanetConfig: tilt + orbital position give the
seasonal daily-mean cycle by latitude; a tidally locked world instead
gets sunlight fixed by angular distance from the permanent substellar
point (hot eye, frozen night side) — no latitude bands at all.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.sim.planet_config import PlanetConfig
    from ports.grid import CellId, Grid, LatLon

SUBSTELLAR_POINT_LAT_DEG = 0.0
SUBSTELLAR_POINT_LON_DEG = 0.0
"""A tidally locked world's fixed substellar point (by convention)."""


def declination_deg(tilt_deg: float, orbit_phase: float) -> float:
    """Return solar declination at an orbit phase (0 = equinox).

    Phase 0.25 is northern summer solstice, 0.75 southern.
    """
    return tilt_deg * math.sin(2.0 * math.pi * orbit_phase)


def daily_mean_insolation_wm2(
    lat_deg: float,
    declination: float,
    solar_constant_wm2: float,
) -> float:
    """Return the 24h-mean TOA flux at a latitude for one declination.

    The standard day-length integral: polar night yields 0, polar day
    yields round-the-clock sunlight.
    """
    lat = math.radians(lat_deg)
    dec = math.radians(declination)
    cos_hour_angle = -math.tan(lat) * math.tan(dec)
    if cos_hour_angle >= 1.0:
        return 0.0
    hour_angle = math.acos(max(-1.0, cos_hour_angle))
    flux = (
        solar_constant_wm2
        / math.pi
        * (
            hour_angle * math.sin(lat) * math.sin(dec)
            + math.cos(lat) * math.cos(dec) * math.sin(hour_angle)
        )
    )
    return max(0.0, flux)


def substellar_insolation_wm2(
    point: LatLon,
    solar_constant_wm2: float,
) -> float:
    """Return flux on a tidally locked world: cosine of substellar angle."""
    lat = math.radians(point.lat_deg)
    lon = math.radians(point.lon_deg - SUBSTELLAR_POINT_LON_DEG)
    sub_lat = math.radians(SUBSTELLAR_POINT_LAT_DEG)
    cos_angle = math.sin(lat) * math.sin(sub_lat) + math.cos(lat) * math.cos(sub_lat) * math.cos(
        lon
    )
    return solar_constant_wm2 * max(0.0, cos_angle)


def insolation_field(
    grid: Grid,
    planet: PlanetConfig,
    orbit_phase: float,
) -> dict[CellId, float]:
    """Return per-cell TOA insolation for one orbit phase."""
    solar = planet.solar_constant_wm2
    if planet.is_tidally_locked:
        return {
            cell: substellar_insolation_wm2(grid.centroid(cell), solar) for cell in grid.cells()
        }
    dec = declination_deg(planet.axial_tilt_deg, orbit_phase)
    return {
        cell: daily_mean_insolation_wm2(grid.centroid(cell).lat_deg, dec, solar)
        for cell in grid.cells()
    }
