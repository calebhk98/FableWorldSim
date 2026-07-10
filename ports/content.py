"""ContentRegistry port: moddable data (species, biomes, tech, languages).

All game content — species, biomes, climate classes, tech, languages —
is data, not code.  The base game ships as the first "mod" under
``content/``; user mods overlay it.  Domain code looks content up
through this port and never parses files itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class ContentNotFoundError(LookupError):
    """Raised when a content kind or item id is unknown."""


class ContentLoadError(RuntimeError):
    """Raised when a mod's content cannot be loaded (malformed file, bad root).

    Raised at load time — before any world state is touched — naming the
    offending mod and file, so a broken mod fails cleanly instead of
    corrupting the world.
    """


@dataclass(frozen=True)
class ContentItem:
    """One piece of content: a species, biome, tech node, etc."""

    kind: str
    item_id: str
    data: Mapping[str, object]
    source: str
    """Which mod supplied this item (base content is a mod too)."""


class ContentRegistry(ABC):
    """Lookup over all loaded content, with mod overlays applied."""

    @abstractmethod
    def kinds(self) -> tuple[str, ...]:
        """Return the sorted content kinds (e.g. 'species', 'biomes')."""

    @abstractmethod
    def ids(self, kind: str) -> tuple[str, ...]:
        """Return the sorted item ids available for a kind."""

    @abstractmethod
    def get(self, kind: str, item_id: str) -> ContentItem:
        """Return one content item; raises ContentNotFoundError."""
