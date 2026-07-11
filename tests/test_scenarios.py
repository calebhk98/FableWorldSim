"""Tests for scenario/preset API: discovery and world-building from presets."""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("h3")

from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings
from api.world_service import get_all_presets
from core.sim.planet_config import PlanetConfig
from core.sim.presets import earth, luna, mars, tidally_locked_ocean

_HTTP_OK = 200


def _client() -> TestClient:
    return TestClient(create_app(Settings()))


def test_get_all_presets_returns_seven_presets() -> None:
    """Verify all seven presets are discoverable."""
    presets = get_all_presets()
    assert len(presets) == 7
    expected_names = {
        "earth",
        "mars",
        "venus",
        "luna",
        "tidally_locked_ocean",
        "high_tilt",
        "fantasy_default",
    }
    assert set(presets.keys()) == expected_names


def test_each_preset_has_config_and_description() -> None:
    """Each preset entry is a (PlanetConfig, description) tuple."""
    presets = get_all_presets()
    for _name, (config, description) in presets.items():
        assert hasattr(config, "name")
        assert hasattr(config, "radius_m")
        assert isinstance(description, str)
        assert len(description) > 0


def test_scenarios_endpoint_lists_all_presets() -> None:
    """GET /scenarios lists all seven presets with descriptions."""
    client = _client()
    response = client.get("/scenarios")
    assert response.status_code == _HTTP_OK
    body = response.json()
    assert "scenarios" in body
    scenarios = body["scenarios"]
    assert len(scenarios) == 7
    by_name = {s["name"]: s for s in scenarios}
    assert "earth" in by_name
    assert "mars" in by_name
    assert by_name["earth"]["planet_name"] == "Earth"
    assert by_name["mars"]["planet_name"] == "Mars"


def test_list_scenarios_command_is_discoverable() -> None:
    """list_scenarios command appears in /commands."""
    client = _client()
    commands = client.get("/commands").json()
    by_name = {cmd["name"]: cmd for cmd in commands}
    assert "list_scenarios" in by_name
    assert by_name["list_scenarios"]["mutates"] is False


def test_list_scenarios_command_returns_presets() -> None:
    """POST /commands/list_scenarios returns all seven scenarios."""
    client = _client()
    response = client.post("/commands/list_scenarios", json={})
    assert response.status_code == _HTTP_OK
    result = response.json()["result"]
    assert "scenarios" in result
    scenarios = result["scenarios"]
    assert len(scenarios) == 7


def test_build_world_command_is_discoverable() -> None:
    """build_world command appears in /commands."""
    client = _client()
    commands = client.get("/commands").json()
    by_name = {cmd["name"]: cmd for cmd in commands}
    assert "build_world" in by_name
    assert by_name["build_world"]["mutates"] is True


def test_build_world_with_earth_preset() -> None:
    """Build a world with the earth preset."""
    client = _client()
    response = client.post(
        "/commands/build_world",
        json={
            "seed": 42,
            "preset_name": "earth",
            "resolution": 1,
            "season_count": 2,
        },
    )
    assert response.status_code == _HTTP_OK
    result = response.json()["result"]
    assert result["seed"] == 42
    assert result["preset"] == "earth"
    assert result["planet_name"] == "Earth"
    assert result["grid_cell_count"] > 0
    assert 0.0 <= result["ocean_fraction"] <= 1.0


def test_build_world_with_mars_preset() -> None:
    """Build a world with the mars preset."""
    client = _client()
    response = client.post(
        "/commands/build_world",
        json={
            "seed": 42,
            "preset_name": "mars",
            "resolution": 1,
            "season_count": 2,
        },
    )
    assert response.status_code == _HTTP_OK
    result = response.json()["result"]
    assert result["planet_name"] == "Mars"
    # Mars should have lower ocean fraction than Earth
    assert result["ocean_fraction"] < 0.5


def test_build_world_with_tidally_locked_preset() -> None:
    """Build a world with the tidally_locked_ocean preset (different characteristics)."""
    client = _client()
    response = client.post(
        "/commands/build_world",
        json={
            "seed": 42,
            "preset_name": "tidally_locked_ocean",
            "resolution": 1,
            "season_count": 2,
        },
    )
    assert response.status_code == _HTTP_OK
    result = response.json()["result"]
    assert result["planet_name"] == "Tidally Locked Ocean"
    # Tidally locked ocean should have high ocean fraction
    assert result["ocean_fraction"] > 0.8


def test_build_world_with_invalid_preset_name() -> None:
    """Building with an unknown preset name returns 422 or similar error."""
    client = _client()
    response = client.post(
        "/commands/build_world",
        json={
            "seed": 42,
            "preset_name": "invalid_planet",
            "resolution": 1,
            "season_count": 2,
        },
    )
    # Should be an error (422 or 500 depending on error handling)
    assert response.status_code >= 400


def test_build_world_defaults_to_earth() -> None:
    """Omitting preset_name defaults to earth."""
    client = _client()
    response = client.post(
        "/commands/build_world",
        json={
            "seed": 42,
            "resolution": 1,
            "season_count": 2,
        },
    )
    assert response.status_code == _HTTP_OK
    result = response.json()["result"]
    assert result["planet_name"] == "Earth"


def test_mars_differs_from_earth_rotation() -> None:
    """Mars and Earth presets have different rotation periods."""
    mars_config = mars()
    earth_config = earth()
    assert mars_config.rotation_period_s != earth_config.rotation_period_s
    # Mars rotation is ~24.6 hours, Earth is ~23.9 hours
    assert abs(mars_config.rotation_period_s - earth_config.rotation_period_s) > 100.0


def test_luna_is_tidally_locked() -> None:
    """Luna (Moon) preset is tidally locked."""
    config = luna()
    assert config.is_tidally_locked


def test_tidally_locked_ocean_has_high_ocean_fraction() -> None:
    """Tidally locked ocean preset has 90% ocean."""
    config = tidally_locked_ocean()
    assert config.ocean_fraction == 0.9


def test_build_world_with_custom_planet_config_round_trip() -> None:
    """Build a world from a full custom planet_config (round-tripped via asdict)."""
    # This is the bug scenario: user calls dataclasses.asdict on a preset,
    # which converts nested Atmosphere and Satellite objects to dicts.
    preset = mars()
    planet_dict = dataclasses.asdict(preset)

    # The API should accept this and reconstruct the planet correctly.
    client = _client()
    response = client.post(
        "/commands/build_world",
        json={
            "seed": 42,
            "planet_config": planet_dict,
            "resolution": 1,
            "season_count": 2,
        },
    )
    assert response.status_code == _HTTP_OK
    result = response.json()["result"]
    assert result["planet_name"] == "Mars"
    assert result["seed"] == 42
    assert result["grid_cell_count"] > 0


def test_build_world_with_earth_config_round_trip() -> None:
    """Build Earth from a custom config round-trip."""
    preset = earth()
    planet_dict = dataclasses.asdict(preset)

    client = _client()
    response = client.post(
        "/commands/build_world",
        json={
            "seed": 99,
            "planet_config": planet_dict,
            "resolution": 1,
            "season_count": 2,
        },
    )
    assert response.status_code == _HTTP_OK
    result = response.json()["result"]
    assert result["planet_name"] == "Earth"
    assert result["seed"] == 99


def test_invalid_planet_config_missing_required_field_returns_422() -> None:
    """Invalid planet_config (missing required field) returns 422, not 500."""
    client = _client()
    response = client.post(
        "/commands/build_world",
        json={
            "seed": 42,
            "planet_config": {
                # Missing many required fields, including name, radius_m, etc.
                "ocean_fraction": 0.5,
            },
            "resolution": 1,
            "season_count": 2,
        },
    )
    # Should be a client error (4xx), not a server error (5xx).
    assert response.status_code >= 400
    assert response.status_code < 500


def test_invalid_planet_config_bad_atmosphere_type_returns_422() -> None:
    """Invalid planet_config (bad atmosphere type) returns 422, not 500."""
    client = _client()
    response = client.post(
        "/commands/build_world",
        json={
            "seed": 42,
            "planet_config": {
                "name": "BadPlanet",
                "radius_m": 6.4e6,
                "surface_gravity_m_s2": 9.8,
                "axial_tilt_deg": 23.5,
                "rotation_period_s": 86400.0,
                "orbital_period_s": 365.25 * 86400.0,
                "ocean_fraction": 0.5,
                "atmosphere": "not a dict",  # Bad: should be a dict or Atmosphere
                "insolation_wm2": 1361.0,
            },
            "resolution": 1,
            "season_count": 2,
        },
    )
    assert response.status_code >= 400
    assert response.status_code < 500


def test_invalid_planet_config_bad_satellites_type_returns_422() -> None:
    """Invalid planet_config (bad satellites type) returns 422, not 500."""
    client = _client()
    response = client.post(
        "/commands/build_world",
        json={
            "seed": 42,
            "planet_config": {
                "name": "BadPlanet",
                "radius_m": 6.4e6,
                "surface_gravity_m_s2": 9.8,
                "axial_tilt_deg": 23.5,
                "rotation_period_s": 86400.0,
                "orbital_period_s": 365.25 * 86400.0,
                "ocean_fraction": 0.5,
                "atmosphere": {"surface_pressure_pa": 101325.0},
                "satellites": "not a list",  # Bad: should be a list/tuple
                "insolation_wm2": 1361.0,
            },
            "resolution": 1,
            "season_count": 2,
        },
    )
    assert response.status_code >= 400
    assert response.status_code < 500


def test_planet_config_from_dict_with_atmosphere_dict() -> None:
    """PlanetConfig.from_dict coerces atmosphere dict to Atmosphere."""
    config = earth()
    planet_dict = dataclasses.asdict(config)

    # Should not raise, and should return a valid PlanetConfig.
    reconstructed = PlanetConfig.from_dict(planet_dict)
    assert reconstructed.name == config.name
    assert reconstructed.radius_m == config.radius_m
    assert reconstructed.atmosphere.surface_pressure_pa == config.atmosphere.surface_pressure_pa


def test_planet_config_from_dict_with_satellites_dicts() -> None:
    """PlanetConfig.from_dict coerces satellite dicts to Satellite objects."""
    config = mars()
    planet_dict = dataclasses.asdict(config)

    # Mars has satellites (Phobos and Deimos).
    reconstructed = PlanetConfig.from_dict(planet_dict)
    assert len(reconstructed.satellites) == len(config.satellites)
    for orig, reconst in zip(config.satellites, reconstructed.satellites, strict=True):
        assert reconst.name == orig.name
        assert reconst.mass_kg == orig.mass_kg
        assert reconst.semi_major_axis_m == orig.semi_major_axis_m


def test_planet_config_from_dict_missing_required_field_raises_value_error() -> None:
    """PlanetConfig.from_dict raises ValueError when required field is missing."""
    bad_dict = {
        "name": "BadPlanet",
        "radius_m": 6.4e6,
        # Missing surface_gravity_m_s2 and other required fields
    }
    with pytest.raises(ValueError):
        PlanetConfig.from_dict(bad_dict)


def test_planet_config_from_dict_invalid_atmosphere_raises_value_error() -> None:
    """PlanetConfig.from_dict raises ValueError for invalid atmosphere."""
    bad_dict = {
        "name": "BadPlanet",
        "radius_m": 6.4e6,
        "surface_gravity_m_s2": 9.8,
        "axial_tilt_deg": 23.5,
        "rotation_period_s": 86400.0,
        "orbital_period_s": 365.25 * 86400.0,
        "ocean_fraction": 0.5,
        "atmosphere": {"surface_pressure_pa": -100},  # Invalid: negative pressure
        "insolation_wm2": 1361.0,
    }
    with pytest.raises(ValueError):
        PlanetConfig.from_dict(bad_dict)
