"""Tests for the single settings schema (the three-surface source of truth)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.settings import (
    Settings,
    get_setting,
    load_settings,
    planet_config_from,
    setting_paths,
    with_setting,
)

_REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"
_DEFAULT_PORT = 8080
_DEFAULT_RESOLUTION = 3
_NEW_RESOLUTION = 5


def test_defaults_hydrate_from_the_config_folder() -> None:
    settings = load_settings(_REPO_CONFIG)
    assert settings.grid.backend == "h3"
    assert settings.compute.backend == "auto"
    assert settings.i18n.locale == "en"
    assert settings.api.port == _DEFAULT_PORT
    assert settings.mods.paths == []


def test_local_toml_overrides_defaults(tmp_path: Path) -> None:
    (tmp_path / "default.toml").write_text('[grid]\nbackend = "h3"\n', encoding="utf-8")
    (tmp_path / "local.toml").write_text('[grid]\nbackend = "s2"\n', encoding="utf-8")
    assert load_settings(tmp_path).grid.backend == "s2"


def test_environment_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "default.toml").write_text('[grid]\nbackend = "h3"\n', encoding="utf-8")
    monkeypatch.setenv("FWS_GRID__BACKEND", "isea")
    assert load_settings(tmp_path).grid.backend == "isea"


def test_unknown_config_key_is_a_startup_error(tmp_path: Path) -> None:
    (tmp_path / "default.toml").write_text("[grid]\nbackedn = 3\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(tmp_path)


def test_every_option_is_reachable_by_dotted_path() -> None:
    settings = Settings()
    paths = setting_paths(settings)
    assert "grid.backend" in paths
    assert "planet.water_fraction" in paths
    for path in paths:
        get_setting(settings, path)


def test_with_setting_validates_and_does_not_mutate() -> None:
    settings = Settings()
    changed = with_setting(settings, "grid.resolution", 5)
    assert changed.grid.resolution == _NEW_RESOLUTION
    assert settings.grid.resolution == _DEFAULT_RESOLUTION
    with pytest.raises(ValidationError):
        with_setting(settings, "grid.backend", "square-lattice")
    with pytest.raises(KeyError, match="unknown setting"):
        with_setting(settings, "grid.nope", 1)


def test_planet_dials_build_a_planet_config() -> None:
    settings = Settings()
    baseline = planet_config_from(settings)
    assert baseline.ocean_fraction == pytest.approx(0.71)
    assert baseline.axial_tilt_deg == pytest.approx(23.44)

    slow = planet_config_from(with_setting(settings, "planet.rotation_rate", 0.5))
    assert slow.rotation_period_s == pytest.approx(2 * baseline.rotation_period_s)

    retrograde = planet_config_from(with_setting(settings, "planet.rotation_rate", -1.0))
    assert retrograde.is_retrograde

    still = planet_config_from(with_setting(settings, "planet.rotation_rate", 0.0))
    assert math.isinf(still.rotation_period_s)
    assert still.rotation_rate_rad_s == 0.0
