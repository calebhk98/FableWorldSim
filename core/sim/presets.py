"""Solar-system preset worlds: Earth, Mars, Venus, and the Moon.

These are baseline configs proving that one code path serves very
different worlds — and convenient starting points to fork invented
planets from.  Earth and Mars specify orbital energy the *derived* way
(stellar luminosity + distance); the Moon uses a direct flux, exercising
both specification styles.
"""

from __future__ import annotations

from core.sim.constants import (
    AU_M,
    EARTH_AXIAL_TILT_DEG,
    EARTH_GRAVITY_M_S2,
    EARTH_GREENHOUSE_OFFSET_K,
    EARTH_OCEAN_FRACTION,
    EARTH_ORBITAL_PERIOD_S,
    EARTH_RADIUS_M,
    EARTH_SIDEREAL_DAY_S,
    EARTH_SURFACE_PRESSURE_PA,
    MOON_MASS_KG,
    MOON_SEMI_MAJOR_AXIS_M,
    SECONDS_PER_DAY,
    SOLAR_LUMINOSITY_W,
)
from core.sim.planet_config import Atmosphere, PlanetConfig, Satellite


def earth() -> PlanetConfig:
    """Return an Earth baseline (energy derived from Sun at 1 AU)."""
    return PlanetConfig(
        name="Earth",
        radius_m=EARTH_RADIUS_M,
        surface_gravity_m_s2=EARTH_GRAVITY_M_S2,
        axial_tilt_deg=EARTH_AXIAL_TILT_DEG,
        rotation_period_s=EARTH_SIDEREAL_DAY_S,
        orbital_period_s=EARTH_ORBITAL_PERIOD_S,
        ocean_fraction=EARTH_OCEAN_FRACTION,
        atmosphere=Atmosphere(
            surface_pressure_pa=EARTH_SURFACE_PRESSURE_PA,
            greenhouse_offset_k=EARTH_GREENHOUSE_OFFSET_K,
            composition={"N2": 0.781, "O2": 0.209, "Ar": 0.009, "CO2": 0.0004},
        ),
        satellites=(
            Satellite(
                name="Luna",
                mass_kg=MOON_MASS_KG,
                semi_major_axis_m=MOON_SEMI_MAJOR_AXIS_M,
            ),
        ),
        stellar_luminosity_w=SOLAR_LUMINOSITY_W,
        orbital_distance_m=AU_M,
    )


def mars() -> PlanetConfig:
    """Return a Mars baseline: thin CO2 air, bone-dry, two tiny moons."""
    return PlanetConfig(
        name="Mars",
        radius_m=3_389_500.0,
        surface_gravity_m_s2=3.720_76,
        axial_tilt_deg=25.19,
        rotation_period_s=88_642.66,
        orbital_period_s=686.980 * SECONDS_PER_DAY,
        ocean_fraction=0.0,
        atmosphere=Atmosphere(
            surface_pressure_pa=610.0,
            greenhouse_offset_k=5.0,
            composition={"CO2": 0.9532, "N2": 0.027, "Ar": 0.016},
        ),
        satellites=(
            Satellite(name="Phobos", mass_kg=1.0659e16, semi_major_axis_m=9.376e6),
            Satellite(name="Deimos", mass_kg=1.4762e15, semi_major_axis_m=2.346e7),
        ),
        stellar_luminosity_w=SOLAR_LUMINOSITY_W,
        orbital_distance_m=1.523_7 * AU_M,
    )


def venus() -> PlanetConfig:
    """Return a Venus baseline: retrograde spin, crushing greenhouse air."""
    return PlanetConfig(
        name="Venus",
        radius_m=6_051_800.0,
        surface_gravity_m_s2=8.87,
        axial_tilt_deg=2.64,
        rotation_period_s=-243.022_6 * SECONDS_PER_DAY,
        orbital_period_s=224.701 * SECONDS_PER_DAY,
        ocean_fraction=0.0,
        atmosphere=Atmosphere(
            surface_pressure_pa=9.3e6,
            greenhouse_offset_k=505.0,
            composition={"CO2": 0.965, "N2": 0.035},
        ),
        stellar_luminosity_w=SOLAR_LUMINOSITY_W,
        orbital_distance_m=0.723_3 * AU_M,
    )


def luna() -> PlanetConfig:
    """Return a Moon baseline: airless and tidally locked (day = year)."""
    period_s = 27.321_661 * SECONDS_PER_DAY
    return PlanetConfig(
        name="Luna",
        radius_m=1_737_400.0,
        surface_gravity_m_s2=1.62,
        axial_tilt_deg=1.5424,
        rotation_period_s=period_s,
        orbital_period_s=period_s,
        ocean_fraction=0.0,
        atmosphere=Atmosphere.airless(),
        insolation_wm2=1_361.0,
    )


def tidally_locked_ocean() -> PlanetConfig:
    """Return a tidally locked ocean world (eyeball-planet scenario).

    Rotation equals the short orbital period around a dim star: a
    permanent hot substellar point over a 90% ocean, a frozen night
    side, and no day-night cycle anywhere.
    """
    period_s = 20.0 * SECONDS_PER_DAY
    return PlanetConfig(
        name="Tidally Locked Ocean",
        radius_m=EARTH_RADIUS_M,
        surface_gravity_m_s2=EARTH_GRAVITY_M_S2,
        axial_tilt_deg=0.0,
        rotation_period_s=period_s,
        orbital_period_s=period_s,
        ocean_fraction=0.9,
        atmosphere=Atmosphere(
            surface_pressure_pa=EARTH_SURFACE_PRESSURE_PA,
            greenhouse_offset_k=30.0,
            composition={"N2": 0.9, "CO2": 0.1},
        ),
        insolation_wm2=1_100.0,
    )


def high_tilt() -> PlanetConfig:
    """Return a seasonal-extremes scenario: Uranus-grade axial tilt.

    At 60 degrees of tilt each pole spends part of the year as the
    substellar region — seasons dominate latitude.
    """
    return PlanetConfig(
        name="High Tilt",
        radius_m=EARTH_RADIUS_M,
        surface_gravity_m_s2=EARTH_GRAVITY_M_S2,
        axial_tilt_deg=60.0,
        rotation_period_s=EARTH_SIDEREAL_DAY_S,
        orbital_period_s=EARTH_ORBITAL_PERIOD_S,
        ocean_fraction=EARTH_OCEAN_FRACTION,
        atmosphere=Atmosphere(
            surface_pressure_pa=EARTH_SURFACE_PRESSURE_PA,
            greenhouse_offset_k=EARTH_GREENHOUSE_OFFSET_K,
            composition={"N2": 0.78, "O2": 0.21},
        ),
        stellar_luminosity_w=SOLAR_LUMINOSITY_W,
        orbital_distance_m=AU_M,
    )


def fantasy_default() -> PlanetConfig:
    """Return the fantasy default: gentle Earth-plus for storytelling.

    Slightly warmer and wetter than Earth with two small moons (busier
    tides and intertidal band), tuned for habitability rather than
    realism.  Doubles as a test fixture like every preset.
    """
    return PlanetConfig(
        name="Fantasia",
        radius_m=EARTH_RADIUS_M,
        surface_gravity_m_s2=EARTH_GRAVITY_M_S2,
        axial_tilt_deg=20.0,
        rotation_period_s=EARTH_SIDEREAL_DAY_S,
        orbital_period_s=EARTH_ORBITAL_PERIOD_S,
        ocean_fraction=0.65,
        atmosphere=Atmosphere(
            surface_pressure_pa=EARTH_SURFACE_PRESSURE_PA,
            greenhouse_offset_k=36.0,
            composition={"N2": 0.77, "O2": 0.22},
        ),
        satellites=(
            Satellite(name="Aster", mass_kg=4.0e22, semi_major_axis_m=3.0e8),
            Satellite(name="Corvin", mass_kg=1.0e22, semi_major_axis_m=4.5e8),
        ),
        insolation_wm2=1_400.0,
    )
