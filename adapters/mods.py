"""Mod discovery: content roots in load order.

Base content is itself a mod (always first).  Installed mod packages are
discovered via the ``fableworldsim.mods`` entry-point group — each entry
point resolves to a content-root path, or a zero-arg callable returning
one.  Config-listed paths load last (later wins), so a local mod folder
can override anything.  Feed the result straight into
:class:`adapters.content_toml.TomlContentRegistry`.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

MOD_ENTRY_POINT_GROUP = "fableworldsim.mods"
"""Entry-point group installed mod packages register under."""


class EntryPointLike(Protocol):
    """Structural type for entry points (real or test doubles)."""

    @property
    def name(self) -> str:
        """Return the mod's entry-point name."""
        ...

    def load(self) -> object:
        """Return the entry point's target object."""
        ...


def _root_from_target(target: object) -> Path:
    """Return a content-root path from an entry point's target."""
    resolved = target() if callable(target) else target
    if not isinstance(resolved, (str, Path)):
        msg = f"mod entry point must yield a path, got {type(resolved).__name__}"
        raise TypeError(msg)
    return Path(resolved)


def roots_from_entry_points(eps: Iterable[EntryPointLike]) -> list[tuple[str, Path]]:
    """Return (mod name, content root) for each entry point, in order."""
    return [(ep.name, _root_from_target(ep.load())) for ep in eps]


def discover_entry_point_mods() -> list[tuple[str, Path]]:
    """Return content roots from installed ``fableworldsim.mods`` packages."""
    return roots_from_entry_points(entry_points(group=MOD_ENTRY_POINT_GROUP))


def content_roots(
    base_root: Path,
    config_paths: Sequence[str | Path] = (),
    *,
    include_entry_points: bool = True,
) -> list[tuple[str, Path]]:
    """Return all content roots in load order (later overlays earlier).

    Order: base content, then installed mod packages, then config-listed
    paths (the ``[mods] paths`` setting).
    """
    roots: list[tuple[str, Path]] = [("base", base_root)]
    if include_entry_points:
        roots.extend(discover_entry_point_mods())
    roots.extend((Path(p).name, Path(p)) for p in config_paths)
    return roots
