"""Tests for the Localizer port's Fluent adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fluent.runtime")

from adapters.i18n_fluent import FluentLocalizer

_REPO_LOCALES = Path(__file__).resolve().parents[1] / "content" / "locales"


def _locales(tmp_path: Path) -> Path:
    (tmp_path / "en.ftl").write_text(
        "biomes-tundra = Tundra\ngreeting = Hello { $name }!\nen-only = English only\n",
        encoding="utf-8",
    )
    (tmp_path / "es.ftl").write_text(
        "biomes-tundra = Tundra\ngreeting = ¡Hola { $name }!\n",
        encoding="utf-8",
    )
    return tmp_path


def test_translate_per_locale(tmp_path: Path) -> None:
    localizer = FluentLocalizer(_locales(tmp_path))
    assert localizer.available_locales() == ("en", "es")
    assert localizer.translate("greeting", {"name": "Ada"}) == "Hello Ada!"
    assert localizer.translate("greeting", {"name": "Ada"}, locale="es") == "¡Hola Ada!"


def test_fallback_chain_locale_then_default_then_key(tmp_path: Path) -> None:
    localizer = FluentLocalizer(_locales(tmp_path))
    assert localizer.translate("en-only", locale="es") == "English only"
    assert localizer.translate("totally-missing") == "totally-missing"
    assert localizer.translate("greeting", {"name": "Ada"}, locale="xx") == "Hello Ada!"


def test_default_locale_must_exist(tmp_path: Path) -> None:
    (tmp_path / "es.ftl").write_text("k = v\n", encoding="utf-8")
    with pytest.raises(ValueError, match="default locale"):
        FluentLocalizer(tmp_path, default_locale="en")


def test_repo_locales_load() -> None:
    localizer = FluentLocalizer(_REPO_LOCALES)
    assert "en" in localizer.available_locales()
