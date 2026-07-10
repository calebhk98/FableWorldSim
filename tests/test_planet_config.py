"""Tests for PlanetConfig derivations: flux, spin, tides, Coriolis."""

from __future__ import annotations

import math

import pytest

from core.sim.constants import (
    LUNAR_EQUILIBRIUM_TIDE_M,
    SECONDS_PER_DAY,
)
from core.sim.planet_config import Atmosphere, PlanetConfig
from core.sim.presets import earth, luna, mars, venus

_EARTH_SOLAR_CONSTANT_WM2 = 1_361.0
_EARTH_MEAN_SOLAR_DAY_S = 86_400.0
_VENUS_SOLAR_DAY_DAYS = 116.75
_EARTH_CORIOLIS_45N = 1.031e-4


def _bare_planet(**overrides: object) -> PlanetConfig:
    """Return a minimal valid planet, with keyword overrides applied."""
    fields: dict[str, object] = {
        "name": "Bare",
        "radius_m": 1.0e6,
        "surface_gravity_m_s2": 5.0,
        "axial_tilt_deg": 0.0,
        "rotation_period_s": 100_000.0,
        "orbital_period_s": 10_000_000.0,
        "ocean_fraction": 0.5,
        "atmosphere": Atmosphere.airless(),
        "insolation_wm2": 1_000.0,
    }
    fields.update(overrides)
    return PlanetConfig(**fields)  # type: ignore[arg-type]


def test_flux_derived_from_luminosity_and_distance() -> None:
    """Earth's preset derives ~1361 W/m^2 from the Sun at 1 AU."""
    assert earth().solar_constant_wm2 == pytest.approx(_EARTH_SOLAR_CONSTANT_WM2, rel=0.01)


def test_flux_given_directly_is_returned_verbatim() -> None:
    """A direct insolation spec is used as-is."""
    assert _bare_planet(insolation_wm2=42.0).solar_constant_wm2 == 42.0


def test_energy_spec_must_be_given_exactly_one_way() -> None:
    """Neither, both, or half an energy spec is rejected."""
    with pytest.raises(ValueError, match="exactly one way"):
        _bare_planet(insolation_wm2=None)
    with pytest.raises(ValueError, match="exactly one way"):
        _bare_planet(stellar_luminosity_w=3.8e26, orbital_distance_m=1.5e11)
    with pytest.raises(ValueError, match="together"):
        _bare_planet(insolation_wm2=None, stellar_luminosity_w=3.8e26)


def test_earth_solar_day_is_about_24_hours() -> None:
    """The sidereal-to-solar-day derivation lands on ~86400 s."""
    assert earth().solar_day_s == pytest.approx(_EARTH_MEAN_SOLAR_DAY_S, abs=5.0)


def test_venus_retrograde_solar_day() -> None:
    """Venus spins backwards and its solar day is ~116.75 Earth days."""
    config = venus()
    assert config.is_retrograde
    assert not config.is_tidally_locked
    assert config.solar_day_s / SECONDS_PER_DAY == pytest.approx(_VENUS_SOLAR_DAY_DAYS, rel=0.001)


def test_tidally_locked_moon_has_infinite_solar_day() -> None:
    """Rotation equal to orbital period means permanent day/night sides."""
    config = luna()
    assert config.is_tidally_locked
    assert math.isinf(config.solar_day_s)


def test_non_rotating_body_solar_day_is_the_year() -> None:
    """With zero spin the sun circles once per orbit."""
    config = _bare_planet(rotation_period_s=math.inf)
    assert config.rotation_rate_rad_s == 0.0
    assert not config.is_tidally_locked
    assert config.solar_day_s == pytest.approx(config.orbital_period_s)


def test_coriolis_parameter_matches_earth_midlatitude() -> None:
    """f = 2*omega*sin(45 deg) on Earth is about 1.03e-4 1/s."""
    assert earth().coriolis_parameter(45.0) == pytest.approx(_EARTH_CORIOLIS_45N, rel=0.01)
    assert earth().coriolis_parameter(0.0) == pytest.approx(0.0)
    assert earth().coriolis_parameter(-45.0) < 0


def test_tidal_range_normalized_to_earth_moon() -> None:
    """Earth's single moon reproduces the baseline equilibrium tide."""
    assert earth().tidal_range_m == pytest.approx(LUNAR_EQUILIBRIUM_TIDE_M)
    assert venus().tidal_range_m == 0.0
    assert 0.0 < mars().tidal_range_m < LUNAR_EQUILIBRIUM_TIDE_M


def test_validation_rejects_bad_geometry_and_spin() -> None:
    """Zero rotation period, bad ocean fraction, and bad radius all raise."""
    with pytest.raises(ValueError, match="rotation_period_s of 0"):
        _bare_planet(rotation_period_s=0.0)
    with pytest.raises(ValueError, match="ocean fraction"):
        _bare_planet(ocean_fraction=1.5)
    with pytest.raises(ValueError, match="radius"):
        _bare_planet(radius_m=-1.0)


def test_atmosphere_presets_span_thick_to_airless() -> None:
    """Venus-thick through airless atmospheres are all representable."""
    assert venus().atmosphere.surface_pressure_pa > earth().atmosphere.surface_pressure_pa
    assert mars().atmosphere.surface_pressure_pa < earth().atmosphere.surface_pressure_pa
    assert luna().atmosphere.is_airless
