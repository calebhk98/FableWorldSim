"""TOML content adapter: loads moddable data from directory trees.

Layout: ``<root>/<kind>/<item_id>.toml`` — e.g.
``content/species/wolf.toml``.  Multiple roots overlay in order (base
content first, then mods), so a later root replaces items with the same
kind and id.  Uses stdlib ``tomllib``; a JSON loader can slot in behind
the same port.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

from ports.content import ContentItem, ContentNotFoundError, ContentRegistry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class TomlContentRegistry(ContentRegistry):
    """Content lookup over one or more TOML directory roots."""

    def __init__(self, roots: Sequence[tuple[str, Path]]) -> None:
        """Load all roots eagerly; each root is (mod name, directory).

        Later roots overlay earlier ones item-by-item.
        """
        self._items: dict[tuple[str, str], ContentItem] = {}
        for source, root in roots:
            self._load_root(source, root)

    def _load_root(self, source: str, root: Path) -> None:
        """Load every ``<kind>/<id>.toml`` under one root."""
        if not root.is_dir():
            msg = f"content root {root} of mod {source!r} is not a directory"
            raise ContentNotFoundError(msg)
        for kind_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for item_file in sorted(kind_dir.glob("*.toml")):
                data = tomllib.loads(item_file.read_text(encoding="utf-8"))
                item = ContentItem(
                    kind=kind_dir.name,
                    item_id=item_file.stem,
                    data=data,
                    source=source,
                )
                self._items[kind_dir.name, item_file.stem] = item

    def kinds(self) -> tuple[str, ...]:
        """Return the sorted content kinds."""
        return tuple(sorted({kind for kind, _ in self._items}))

    def ids(self, kind: str) -> tuple[str, ...]:
        """Return the sorted item ids for a kind."""
        found = tuple(sorted(i for k, i in self._items if k == kind))
        if not found:
            msg = f"no content of kind {kind!r}"
            raise ContentNotFoundError(msg)
        return found

    def get(self, kind: str, item_id: str) -> ContentItem:
        """Return one content item."""
        item = self._items.get((kind, item_id))
        if item is None:
            msg = f"no content item {kind!r}/{item_id!r}"
            raise ContentNotFoundError(msg)
        return item
