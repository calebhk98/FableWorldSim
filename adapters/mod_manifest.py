"""Mod manifests: dependency/version resolution with fail-fast loading.

Each mod root may carry a ``mod.toml``::

    [mod]
    name = "coolmod"
    version = "1.2.0"
    requires = ["base"]
    conflicts = ["oldmod"]
    min_engine_version = "0.1.0"

The resolver checks engine compatibility, rejects conflicts, and
topologically sorts by ``requires`` (stable: independent mods keep
their configured order).  A mod without a manifest is implicit: no
dependencies, no constraints.  The deterministic-mod contract (mods
route randomness/time through core Rng/Clock) gets its lint when
Python-hook mods land; M1 mods are data-only.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.version import engine_satisfies

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class ModResolutionError(RuntimeError):
    """Raised when the mod set cannot be loaded as configured."""


@dataclass(frozen=True)
class ModManifest:
    """One mod's declared identity and constraints."""

    name: str
    version: str = "0.0.0"
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    min_engine_version: str = "0.0.0"


def read_manifest(root: Path, fallback_name: str) -> ModManifest:
    """Return the mod's manifest (implicit when mod.toml is absent)."""
    manifest_file = root / "mod.toml"
    if not manifest_file.is_file():
        return ModManifest(name=fallback_name)
    try:
        data = tomllib.loads(manifest_file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        msg = f"mod {fallback_name!r} has a malformed manifest {manifest_file}: {exc}"
        raise ModResolutionError(msg) from exc
    section = data.get("mod", {})
    return ModManifest(
        name=str(section.get("name", fallback_name)),
        version=str(section.get("version", "0.0.0")),
        requires=tuple(str(r) for r in section.get("requires", [])),
        conflicts=tuple(str(c) for c in section.get("conflicts", [])),
        min_engine_version=str(section.get("min_engine_version", "0.0.0")),
    )


def _check_constraints(manifests: Sequence[ModManifest]) -> None:
    """Fail fast on engine mismatches, conflicts, and missing requires."""
    names = {m.name for m in manifests}
    for manifest in manifests:
        if not engine_satisfies(manifest.min_engine_version):
            msg = (
                f"mod {manifest.name!r} needs engine >= "
                f"{manifest.min_engine_version}; this engine is older"
            )
            raise ModResolutionError(msg)
        for enemy in manifest.conflicts:
            if enemy in names:
                msg = f"mod {manifest.name!r} conflicts with loaded mod {enemy!r}"
                raise ModResolutionError(msg)
        for need in manifest.requires:
            if need not in names:
                msg = f"mod {manifest.name!r} requires {need!r}, which is not loaded"
                raise ModResolutionError(msg)


def resolve_load_order(
    mods: Sequence[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    """Return mods sorted so every dependency loads before its dependent.

    Stable: mods with no ordering constraint keep their configured
    order.  Cycles fail fast with the members named.
    """
    manifests = [read_manifest(root, name) for name, root in mods]
    _check_constraints(manifests)
    by_name = {m.name: (name, root) for m, (name, root) in zip(manifests, mods, strict=True)}
    requires = {m.name: set(m.requires) for m in manifests}

    ordered: list[tuple[str, Path]] = []
    placed: set[str] = set()
    remaining = [m.name for m in manifests]
    while remaining:
        ready = [n for n in remaining if requires[n] <= placed]
        if not ready:
            msg = f"mod dependency cycle among: {sorted(remaining)}"
            raise ModResolutionError(msg)
        chosen = ready[0]
        remaining.remove(chosen)
        placed.add(chosen)
        ordered.append(by_name[chosen])
    return ordered
