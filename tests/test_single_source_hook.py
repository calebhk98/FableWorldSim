"""Tests for the single-source-of-truth pre-commit hook."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HOOK = Path(__file__).resolve().parents[1] / "tools" / "hooks" / "check_single_source.py"


def _run(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_HOOK),
            "--content-dir",
            str(tmp_path / "content"),
            "--code-dir",
            str(tmp_path / "core"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _scaffold(tmp_path: Path, *, ftl: str, core_code: str) -> None:
    biomes = tmp_path / "content" / "biomes"
    biomes.mkdir(parents=True)
    (biomes / "tundra.toml").write_text('name-key = "biomes-tundra"\n', encoding="utf-8")
    locales = tmp_path / "content" / "locales"
    locales.mkdir()
    (locales / "en.ftl").write_text(ftl, encoding="utf-8")
    core = tmp_path / "core"
    core.mkdir()
    (core / "logic.py").write_text(core_code, encoding="utf-8")


def test_passes_when_ids_have_keys_and_code_is_clean(tmp_path: Path) -> None:
    _scaffold(tmp_path, ftl="biomes-tundra = Tundra\n", core_code="x = 1\n")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout


def test_fails_when_locale_key_is_missing(tmp_path: Path) -> None:
    _scaffold(tmp_path, ftl="something-else = Nope\n", core_code="x = 1\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "biomes-tundra" in result.stdout


def test_fails_when_core_hardcodes_a_content_id(tmp_path: Path) -> None:
    _scaffold(
        tmp_path,
        ftl="biomes-tundra = Tundra\n",
        core_code='biome = "tundra"\n',
    )
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "hard-coded" in result.stdout


def test_repo_is_currently_clean() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(_HOOK)],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
    )
    assert result.returncode == 0, result.stdout
