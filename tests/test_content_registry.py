"""Tests for the TOML content registry and mod overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from adapters.content_toml import TomlContentRegistry
from ports.content import ContentLoadError, ContentNotFoundError

if TYPE_CHECKING:
    from pathlib import Path


def _write(root: Path, kind: str, item_id: str, body: str) -> None:
    """Write one content item file under a root."""
    kind_dir = root / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    (kind_dir / f"{item_id}.toml").write_text(body, encoding="utf-8")


def test_items_load_by_kind_and_id(tmp_path: Path) -> None:
    """Species and biomes load from a base content root."""
    base = tmp_path / "base"
    _write(base, "species", "wolf", 'name = "Wolf"\ndiet = "carnivore"\n')
    _write(base, "biomes", "tundra", 'name = "Tundra"\n')
    registry = TomlContentRegistry([("base", base)])
    assert registry.kinds() == ("biomes", "species")
    assert registry.ids("species") == ("wolf",)
    wolf = registry.get("species", "wolf")
    assert wolf.data["diet"] == "carnivore"
    assert wolf.source == "base"


def test_later_mods_overlay_earlier_content(tmp_path: Path) -> None:
    """A mod replaces base items and adds new ones."""
    base = tmp_path / "base"
    mod = tmp_path / "coolmod"
    _write(base, "species", "wolf", 'name = "Wolf"\nlegs = 4\n')
    _write(mod, "species", "wolf", 'name = "Dire Wolf"\nlegs = 4\n')
    _write(mod, "species", "sandworm", 'name = "Sandworm"\nlegs = 0\n')
    registry = TomlContentRegistry([("base", base), ("coolmod", mod)])
    wolf = registry.get("species", "wolf")
    assert wolf.data["name"] == "Dire Wolf"
    assert wolf.source == "coolmod"
    assert registry.ids("species") == ("sandworm", "wolf")


def test_broken_mod_fails_cleanly_naming_mod_and_file(tmp_path: Path) -> None:
    """A malformed mod is a clean load error, never a corrupted world."""
    base = tmp_path / "base"
    _write(base, "species", "wolf", 'name = "Wolf"\n')
    broken = tmp_path / "badmod"
    _write(broken, "species", "glitch", "name = 'unclosed\n")
    with pytest.raises(ContentLoadError, match="badmod") as excinfo:
        TomlContentRegistry([("base", base), ("badmod", broken)])
    assert "glitch.toml" in str(excinfo.value)


def test_missing_content_raises(tmp_path: Path) -> None:
    """Unknown kinds, ids, and bad roots fail loudly."""
    base = tmp_path / "base"
    _write(base, "species", "wolf", 'name = "Wolf"\n')
    registry = TomlContentRegistry([("base", base)])
    with pytest.raises(ContentNotFoundError, match="kind"):
        registry.ids("techs")
    with pytest.raises(ContentNotFoundError, match="item"):
        registry.get("species", "dragon")
    with pytest.raises(ContentNotFoundError, match="not a directory"):
        TomlContentRegistry([("broken", tmp_path / "missing")])
