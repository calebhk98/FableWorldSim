"""Tests for mod discovery and load-order composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.content_toml import TomlContentRegistry
from adapters.mods import content_roots, roots_from_entry_points


class _FakeEntryPoint:
    def __init__(self, name: str, target: object) -> None:
        self.name = name
        self._target = target

    def load(self) -> object:
        return self._target


def test_entry_points_yield_paths_or_callables(tmp_path: Path) -> None:
    direct = _FakeEntryPoint("direct", str(tmp_path / "a"))
    lazy = _FakeEntryPoint("lazy", lambda: tmp_path / "b")
    roots = roots_from_entry_points([direct, lazy])
    assert roots == [("direct", tmp_path / "a"), ("lazy", tmp_path / "b")]
    bad = _FakeEntryPoint("bad", 42)
    with pytest.raises(TypeError, match="path"):
        roots_from_entry_points([bad])


def test_load_order_is_base_then_mods_then_config(tmp_path: Path) -> None:
    base = tmp_path / "base"
    late = tmp_path / "late"
    roots = content_roots(base, [late], include_entry_points=False)
    assert roots == [("base", base), ("late", late)]


def test_config_mod_overrides_base_content(tmp_path: Path) -> None:
    base = tmp_path / "base" / "species"
    base.mkdir(parents=True)
    (base / "wolf.toml").write_text('name = "Wolf"\n', encoding="utf-8")
    mod = tmp_path / "mymod" / "species"
    mod.mkdir(parents=True)
    (mod / "wolf.toml").write_text('name = "Dire Wolf"\n', encoding="utf-8")

    roots = content_roots(tmp_path / "base", [tmp_path / "mymod"], include_entry_points=False)
    registry = TomlContentRegistry(roots)
    assert registry.get("species", "wolf").data["name"] == "Dire Wolf"
    assert registry.get("species", "wolf").source == "mymod"
