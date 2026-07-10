"""Storage port: world snapshots and time series, with save migration.

Every snapshot carries a schema version.  A migration chain upgrades old
saves as the model evolves across milestones, so worlds created in M1
still load later.  Adapters (sqlite / parquet / zarr) only move bytes;
versioning and migration live here so every backend behaves identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

CURRENT_SCHEMA_VERSION = 1
"""Schema version written by this build; bump alongside a migration."""


@dataclass(frozen=True)
class Snapshot:
    """One saved world state.

    ``payload`` must be JSON-serializable (plain dicts/lists/scalars) so
    every storage adapter can persist it without custom encoders.
    """

    schema_version: int
    world_name: str
    tick: int
    payload: Mapping[str, object]


MigrationFn = Callable[[Snapshot], Snapshot]
"""Upgrades a snapshot from version N to version N + 1."""


class MigrationError(RuntimeError):
    """Raised when a snapshot cannot be upgraded to the current schema."""


class MigrationChain:
    """An ordered chain of per-version snapshot upgrades."""

    def __init__(self) -> None:
        """Create an empty chain."""
        self._steps: dict[int, MigrationFn] = {}

    def register(self, from_version: int, fn: MigrationFn) -> None:
        """Register the upgrade from ``from_version`` to the next version."""
        if from_version in self._steps:
            msg = f"migration from version {from_version} already registered"
            raise ValueError(msg)
        self._steps[from_version] = fn

    def migrate(self, snapshot: Snapshot) -> Snapshot:
        """Return ``snapshot`` upgraded to ``CURRENT_SCHEMA_VERSION``."""
        current = snapshot
        while current.schema_version < CURRENT_SCHEMA_VERSION:
            step = self._steps.get(current.schema_version)
            if step is None:
                msg = (
                    f"no migration from schema version {current.schema_version}"
                    f" toward {CURRENT_SCHEMA_VERSION}"
                )
                raise MigrationError(msg)
            upgraded = step(current)
            if upgraded.schema_version != current.schema_version + 1:
                msg = (
                    f"migration from {current.schema_version} produced version "
                    f"{upgraded.schema_version}, expected "
                    f"{current.schema_version + 1}"
                )
                raise MigrationError(msg)
            current = upgraded
        if current.schema_version > CURRENT_SCHEMA_VERSION:
            msg = (
                f"snapshot schema {current.schema_version} is newer than this "
                f"build's {CURRENT_SCHEMA_VERSION}"
            )
            raise MigrationError(msg)
        return current


def stamp_current(snapshot: Snapshot) -> Snapshot:
    """Return a copy of ``snapshot`` stamped with the current version."""
    return replace(snapshot, schema_version=CURRENT_SCHEMA_VERSION)


class Storage(ABC):
    """Persistence for snapshots; loads always return migrated snapshots."""

    @abstractmethod
    def save(self, snapshot: Snapshot) -> str:
        """Persist a snapshot and return its storage id."""

    @abstractmethod
    def load(self, snapshot_id: str) -> Snapshot:
        """Load a snapshot, migrated to ``CURRENT_SCHEMA_VERSION``."""

    @abstractmethod
    def list_snapshots(self, world_name: str | None = None) -> tuple[str, ...]:
        """Return snapshot ids, optionally filtered to one world."""
