"""Round-trip + migration tests for snapshot storage (guarded from day one)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from adapters.storage_sqlite import SqliteStorage
from ports.storage import (
    CURRENT_SCHEMA_VERSION,
    MigrationChain,
    MigrationError,
    Snapshot,
)

if TYPE_CHECKING:
    from pathlib import Path

_ANCIENT_VERSION = 0


def _snapshot() -> Snapshot:
    """Return a current-schema snapshot with a small payload."""
    return Snapshot(
        schema_version=CURRENT_SCHEMA_VERSION,
        world_name="testworld",
        tick=42,
        payload={"fields": {"elevation_m": [0.0, 10.0]}, "seed": 7},
    )


def _upgrade_v0(snapshot: Snapshot) -> Snapshot:
    """Sample migration: v0 payloads lacked the seed key."""
    payload = {**dict(snapshot.payload), "seed": 0}
    return replace(snapshot, schema_version=1, payload=payload)


def test_snapshot_round_trip(tmp_path: Path) -> None:
    """A saved snapshot loads back identical."""
    storage = SqliteStorage(tmp_path / "world.db")
    saved = _snapshot()
    snapshot_id = storage.save(saved)
    loaded = storage.load(snapshot_id)
    assert loaded == saved


def test_old_snapshot_is_migrated_on_load(tmp_path: Path) -> None:
    """A v0 save comes back upgraded to the current schema."""
    chain = MigrationChain()
    chain.register(_ANCIENT_VERSION, _upgrade_v0)
    storage = SqliteStorage(tmp_path / "world.db", migrations=chain)
    old = Snapshot(
        schema_version=_ANCIENT_VERSION,
        world_name="oldworld",
        tick=1,
        payload={"fields": {}},
    )
    loaded = storage.load(storage.save(old))
    assert loaded.schema_version == CURRENT_SCHEMA_VERSION
    assert loaded.payload["seed"] == 0


def test_unmigratable_snapshot_raises(tmp_path: Path) -> None:
    """Without a registered migration, an old save fails loudly."""
    storage = SqliteStorage(tmp_path / "world.db")
    old = Snapshot(
        schema_version=_ANCIENT_VERSION,
        world_name="oldworld",
        tick=1,
        payload={},
    )
    with pytest.raises(MigrationError, match="no migration"):
        storage.load(storage.save(old))


def test_newer_snapshot_raises(tmp_path: Path) -> None:
    """A save from a future build fails loudly instead of corrupting."""
    storage = SqliteStorage(tmp_path / "world.db")
    future = replace(_snapshot(), schema_version=CURRENT_SCHEMA_VERSION + 1)
    with pytest.raises(MigrationError, match="newer"):
        storage.load(storage.save(future))


def test_listing_filters_by_world(tmp_path: Path) -> None:
    """Snapshot listing can scope to one world."""
    storage = SqliteStorage(tmp_path / "world.db")
    first = storage.save(_snapshot())
    other = storage.save(replace(_snapshot(), world_name="otherworld"))
    assert storage.list_snapshots() == (first, other)
    assert storage.list_snapshots("testworld") == (first,)
    with pytest.raises(KeyError, match="no snapshot"):
        storage.load("999")
