"""The single settings schema — source of truth for every option.

One pydantic-settings model backs all three control surfaces: the config
folder (TOML) hydrates it at startup, the API exposes get/set on the same
schema, and the UI is just an API client.  No option exists in only one
surface.  ``extra="forbid"`` makes a typoed config key a startup error
instead of a silent no-op.

Environment variables override TOML using the ``FWS_`` prefix with ``__``
as the section separator (e.g. ``FWS_GRID__BACKEND=s2``).
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from core.sim.constants import EARTH_SIDEREAL_DAY_S
from core.sim.planet_config import PlanetConfig
from core.sim.presets import earth

_FORBID = ConfigDict(extra="forbid")


class GridSettings(BaseModel):
    """Grid backend toggle (chosen at world-creation time) and resolution."""

    model_config = _FORBID

    backend: Literal["h3", "isea", "s2"] = "h3"
    resolution: int = Field(default=3, ge=0)


class ComputeSettings(BaseModel):
    """Array/compute backend selection and hardware budget."""

    model_config = _FORBID

    backend: Literal["auto", "numpy", "cupy", "jax", "dask"] = "auto"
    gpus: str | int | list[int] = "all"
    cpu_workers: str | int = "all"


class FidelitySettings(BaseModel):
    """The single accuracy<->speed dial."""

    model_config = _FORBID

    level: Literal["fast", "balanced", "accurate"] = "balanced"


class PlanetSettings(BaseModel):
    """World-creation dials layered over the Earth baseline.

    ``rotation_rate`` is a signed multiple of Earth's spin (negative =
    retrograde, 0 = non-rotating).
    """

    model_config = _FORBID

    axial_tilt_deg: float = 23.44
    rotation_rate: float = 1.0
    water_fraction: float = Field(default=0.71, ge=0.0, le=1.0)


class I18nSettings(BaseModel):
    """Locale selection; locales live in content/locales/<lang>.ftl."""

    model_config = _FORBID

    locale: str = "en"


class ModsSettings(BaseModel):
    """Extra content roots overlaying base content, in load order."""

    model_config = _FORBID

    paths: list[str] = Field(default_factory=list)


class ApiServerSettings(BaseModel):
    """API server binding."""

    model_config = _FORBID

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)


class Settings(BaseSettings):
    """All options, hydrated from config TOML and environment."""

    model_config = SettingsConfigDict(
        env_prefix="FWS_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    grid: GridSettings = Field(default_factory=GridSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    fidelity: FidelitySettings = Field(default_factory=FidelitySettings)
    planet: PlanetSettings = Field(default_factory=PlanetSettings)
    i18n: I18nSettings = Field(default_factory=I18nSettings)
    mods: ModsSettings = Field(default_factory=ModsSettings)
    api: ApiServerSettings = Field(default_factory=ApiServerSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order sources: explicit args > environment > config TOML."""
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )


def load_settings(config_dir: Path | str = Path("config")) -> Settings:
    """Return settings hydrated from ``<config_dir>/default.toml``.

    An optional ``local.toml`` next to it overrides per section (kept out
    of version control for machine-specific tweaks).
    """
    directory = Path(config_dir)
    files = [
        path for path in (directory / "default.toml", directory / "local.toml") if path.exists()
    ]

    class _FileSettings(Settings):
        """Settings bound to the discovered TOML files."""

        model_config = SettingsConfigDict(**{**Settings.model_config, "toml_file": files})

    return _FileSettings()


def setting_paths(settings: Settings) -> tuple[str, ...]:
    """Return every dotted option path (e.g. ``"grid.backend"``), sorted."""
    paths: list[str] = []
    for section, value in settings.model_dump().items():
        paths.extend(f"{section}.{option}" for option in value)
    return tuple(sorted(paths))


def get_setting(settings: Settings, path: str) -> object:
    """Return one option's current value by dotted path."""
    data = settings.model_dump()
    section, _, option = path.partition(".")
    if section not in data or option not in data[section]:
        known = ", ".join(setting_paths(settings))
        msg = f"unknown setting {path!r}; known: {known}"
        raise KeyError(msg)
    value: object = data[section][option]
    return value


def with_setting(settings: Settings, path: str, value: object) -> Settings:
    """Return a new validated Settings with one option changed.

    Raises ``KeyError`` for unknown paths and pydantic's
    ``ValidationError`` for values the schema rejects.
    """
    get_setting(settings, path)
    data = settings.model_dump()
    section, _, option = path.partition(".")
    data[section][option] = value
    return Settings.model_validate(data)


def planet_config_from(settings: Settings) -> PlanetConfig:
    """Build a PlanetConfig from the planet dials over the Earth baseline.

    ``rotation_rate`` scales Earth's spin (signed); 0 means non-rotating
    (infinite rotation period).
    """
    dials = settings.planet
    if dials.rotation_rate == 0:
        rotation_period_s = math.inf
    else:
        rotation_period_s = EARTH_SIDEREAL_DAY_S / dials.rotation_rate
    return replace(
        earth(),
        name="Configured Planet",
        axial_tilt_deg=dials.axial_tilt_deg,
        rotation_period_s=rotation_period_s,
        ocean_fraction=dials.water_fraction,
    )
