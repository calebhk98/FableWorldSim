"""Tests for mod dependency/version resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from adapters.mod_manifest import ModResolutionError, read_manifest, resolve_load_order

if TYPE_CHECKING:
    from pathlib import Path


def _mod(tmp_path: Path, name: str, manifest: str | None = None) -> tuple[str, Path]:
    root = tmp_path / name
    root.mkdir()
    if manifest is not None:
        (root / "mod.toml").write_text(manifest, encoding="utf-8")
    return (name, root)


def test_missing_manifest_is_implicit(tmp_path: Path) -> None:
    _, root = _mod(tmp_path, "plain")
    manifest = read_manifest(root, "plain")
    assert manifest.name == "plain"
    assert manifest.requires == ()


def test_requires_orders_dependencies_first(tmp_path: Path) -> None:
    base = _mod(tmp_path, "base")
    addon = _mod(tmp_path, "addon", '[mod]\nname = "addon"\nrequires = ["base"]\n')
    ordered = resolve_load_order([addon, base])
    assert [name for name, _ in ordered] == ["base", "addon"]


def test_independent_mods_keep_configured_order(tmp_path: Path) -> None:
    mods = [_mod(tmp_path, "a"), _mod(tmp_path, "b"), _mod(tmp_path, "c")]
    assert [n for n, _ in resolve_load_order(mods)] == ["a", "b", "c"]


def test_conflicts_and_missing_requirements_fail_fast(tmp_path: Path) -> None:
    base = _mod(tmp_path, "base")
    grumpy = _mod(tmp_path, "grumpy", '[mod]\nname = "grumpy"\nconflicts = ["base"]\n')
    with pytest.raises(ModResolutionError, match="conflicts"):
        resolve_load_order([base, grumpy])
    needy = _mod(tmp_path, "needy", '[mod]\nname = "needy"\nrequires = ["ghost"]\n')
    with pytest.raises(ModResolutionError, match="requires"):
        resolve_load_order([needy])


def test_engine_version_and_cycles_fail_fast(tmp_path: Path) -> None:
    futuristic = _mod(
        tmp_path, "futuristic", '[mod]\nname = "futuristic"\nmin_engine_version = "999.0.0"\n'
    )
    with pytest.raises(ModResolutionError, match="engine"):
        resolve_load_order([futuristic])
    yin = _mod(tmp_path, "yin", '[mod]\nname = "yin"\nrequires = ["yang"]\n')
    yang = _mod(tmp_path, "yang", '[mod]\nname = "yang"\nrequires = ["yin"]\n')
    with pytest.raises(ModResolutionError, match="cycle"):
        resolve_load_order([yin, yang])


def test_malformed_manifest_fails_fast(tmp_path: Path) -> None:
    broken = _mod(tmp_path, "broken", "[mod\nname=")
    with pytest.raises(ModResolutionError, match="malformed"):
        resolve_load_order([broken])
