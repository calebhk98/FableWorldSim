"""Physical and astronomical constants used across the simulation core.

Values are SI.  Earth/Moon/Sun figures exist only as convenient baselines
for presets and for normalizing first-order models (e.g. tidal range);
nothing in the simulation assumes the planet *is* Earth.
"""

from __future__ import annotations

AU_M = 1.495_978_707e11
"""One astronomical unit in meters."""

SOLAR_LUMINOSITY_W = 3.828e26
"""Nominal solar luminosity (IAU 2015) in watts."""

SECONDS_PER_HOUR = 3_600.0
"""Seconds in one hour."""

SECONDS_PER_DAY = 86_400.0
"""Seconds in one 24-hour day."""

SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
"""Seconds in one Julian year."""

EARTH_RADIUS_M = 6_371_008.8
"""IUGG mean Earth radius in meters."""

EARTH_GRAVITY_M_S2 = 9.806_65
"""Standard Earth surface gravity in m/s^2."""

EARTH_SIDEREAL_DAY_S = 86_164.0905
"""Earth's sidereal rotation period in seconds."""

EARTH_ORBITAL_PERIOD_S = 365.256_363 * SECONDS_PER_DAY
"""Earth's sidereal orbital period in seconds."""

EARTH_AXIAL_TILT_DEG = 23.44
"""Earth's obliquity in degrees."""

EARTH_OCEAN_FRACTION = 0.708
"""Fraction of Earth's surface covered by ocean."""

EARTH_SURFACE_PRESSURE_PA = 101_325.0
"""Earth's mean sea-level atmospheric pressure in pascals."""

EARTH_GREENHOUSE_OFFSET_K = 33.0
"""Earth's greenhouse warming over its airless equilibrium, in kelvin."""

MOON_MASS_KG = 7.342e22
"""Mass of Earth's Moon in kilograms."""

MOON_SEMI_MAJOR_AXIS_M = 3.844e8
"""Mean Earth-Moon distance in meters."""

LUNAR_EQUILIBRIUM_TIDE_M = 0.54
"""Peak-to-trough equilibrium ocean tide raised on Earth by the Moon, in
meters; the normalization baseline for the first-order tidal-range model."""
