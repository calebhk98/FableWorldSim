"""Surface wind: zonal bands + thermal flow, all derived from PlanetConfig.

No wind pattern is hard-coded to Earth: band structure and deflection
come from the signed rotation rate (retrograde spin reverses the bands;
zero/tidally-locked spin removes them, leaving direct flow into the
warm hemisphere — substellar-to-antistellar circulation), and the
thermal component converges surface air into heat lows, which is what
lets monsoons emerge from seasonal land-sea contrast.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.grid.vectors import Vector2, gradient, rotate
from core.sim.constants import EARTH_SIDEREAL_DAY_S

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.sim.planet_config import PlanetConfig
    from ports.grid import CellId, Grid

_EARTH_ROTATION_RATE = 2.0 * math.pi / EARTH_SIDEREAL_DAY_S
_BAND_SPEED_M_S = 8.0
_THERMAL_SPEED_SCALE = 2.0e6
_MAX_THERMAL_SPEED_M_S = 15.0
_HADLEY_EDGE_DEG = 30.0
_FERREL_EDGE_DEG = 60.0
_MAX_DEFLECTION_RAD = 1.1


def rotation_strength(planet: PlanetConfig) -> float:
    """Return |spin| relative to Earth's, clamped to [0, 1].

    Scales the zonal-band machinery: 1 = full three-band structure,
    0 = no bands at all.  A tidally locked world is explicitly 0 even
    though it technically rotates once per orbit — its circulation is
    substellar-to-antistellar, not zonal.
    """
    if planet.is_tidally_locked:
        return 0.0
    return min(1.0, abs(planet.rotation_rate_rad_s) / _EARTH_ROTATION_RATE)


def zonal_band_wind(lat_deg: float, planet: PlanetConfig) -> Vector2:
    """Return the banded east-west wind component at a latitude.

    Trade easterlies to ~30 deg, westerlies to ~60 deg, polar
    easterlies beyond — reversed wholesale on a retrograde world and
    faded out as rotation slows.
    """
    strength = rotation_strength(planet)
    if strength == 0.0:
        return (0.0, 0.0)
    abs_lat = abs(lat_deg)
    if abs_lat <= _HADLEY_EDGE_DEG:
        east = -_BAND_SPEED_M_S * math.cos(math.radians(lat_deg) * 3.0)
    elif abs_lat <= _FERREL_EDGE_DEG:
        east = _BAND_SPEED_M_S
    else:
        east = -_BAND_SPEED_M_S / 2.0
    if planet.is_retrograde:
        east = -east
    return (east * strength, 0.0)


def thermal_wind(
    grid: Grid,
    cell: CellId,
    temps: Mapping[CellId, float],
    planet: PlanetConfig,
) -> Vector2:
    """Return the thermal component: surface flow into warm lows.

    Follows +grad(T) (cold high toward hot low), deflected by Coriolis
    with the *signed* rotation — retrograde worlds deflect the other
    way, non-rotating worlds flow straight down the gradient.
    """
    grad_e, grad_n = gradient(temps, grid, cell)
    east = grad_e * _THERMAL_SPEED_SCALE
    north = grad_n * _THERMAL_SPEED_SCALE
    speed = math.hypot(east, north)
    if speed > _MAX_THERMAL_SPEED_M_S:
        factor = _MAX_THERMAL_SPEED_M_S / speed
        east *= factor
        north *= factor
    coriolis = planet.coriolis_parameter(grid.centroid(cell).lat_deg)
    deflection = -_MAX_DEFLECTION_RAD * rotation_strength(planet) * _sign(coriolis)
    return rotate((east, north), deflection)


def _sign(value: float) -> float:
    """Return -1, 0, or +1."""
    if value > 0:
        return 1.0
    return -1.0 if value < 0 else 0.0


def wind_field(
    grid: Grid,
    planet: PlanetConfig,
    temps: Mapping[CellId, float],
) -> dict[CellId, Vector2]:
    """Return the per-cell surface wind vector (east, north) in m/s."""
    winds: dict[CellId, Vector2] = {}
    for cell in temps:
        band = zonal_band_wind(grid.centroid(cell).lat_deg, planet)
        thermal = thermal_wind(grid, cell, temps, planet)
        winds[cell] = (band[0] + thermal[0], band[1] + thermal[1])
    return winds
