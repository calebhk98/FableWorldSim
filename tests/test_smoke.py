"""Scaffolding smoke tests.

These assert only that the project skeleton is wired up — packages import and the
default configuration parses. They contain no simulation logic; real behavior arrives
with M1.0 (the Grid port + equal-area conservation test), test-first.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PACKAGES = ["core", "ports", "adapters", "api"]


@pytest.mark.parametrize("name", PACKAGES)
def test_top_level_packages_import(name: str) -> None:
    """Each top-level package should import cleanly."""
    assert importlib.import_module(name) is not None


def test_default_config_parses() -> None:
    """The default config is valid TOML and exposes the expected top-level sections."""
    with (REPO_ROOT / "config" / "default.toml").open("rb") as handle:
        config = tomllib.load(handle)
    for section in ("grid", "compute", "fidelity", "planet", "i18n", "api"):
        assert section in config, f"missing [{section}] in default.toml"
