"""The world recipe: a reproducibility manifest.

Seed + settings + content/mod version pins + engine version — everything
needed to regenerate an identical world, so an interesting planet can be
shared as a few hundred bytes of JSON instead of a full snapshot.
Determinism (seeded Rng, deterministic Clock) makes this nearly free.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from core.version import ENGINE_VERSION

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class WorldRecipe:
    """Everything required to regenerate a world bit-for-bit."""

    world_name: str
    seed: int
    grid_backend: str
    grid_resolution: int
    settings: Mapping[str, object] = field(default_factory=dict)
    content_pins: Mapping[str, str] = field(default_factory=dict)
    """Mod name -> version for every loaded content root."""
    engine_version: str = ENGINE_VERSION

    def to_json(self) -> str:
        """Serialize to a stable, shareable JSON document."""
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> WorldRecipe:
        """Parse a recipe back from JSON."""
        data = json.loads(text)
        return cls(
            world_name=str(data["world_name"]),
            seed=int(data["seed"]),
            grid_backend=str(data["grid_backend"]),
            grid_resolution=int(data["grid_resolution"]),
            settings=dict(data.get("settings", {})),
            content_pins=dict(data.get("content_pins", {})),
            engine_version=str(data.get("engine_version", ENGINE_VERSION)),
        )
