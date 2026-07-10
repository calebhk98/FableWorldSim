"""sqlite storage adapter: snapshots in a single file, stdlib only.

Snapshots are stored as JSON payloads with their schema version; loads
run the migration chain so callers always see the current schema.
Parquet/zarr adapters slot in behind the same port for bulk time series.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from ports.storage import MigrationChain, Snapshot, Storage

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version INTEGER NOT NULL,
    world_name TEXT NOT NULL,
    tick INTEGER NOT NULL,
    payload TEXT NOT NULL
)
"""


class SqliteStorage(Storage):
    """Snapshot persistence in one sqlite database file."""

    def __init__(self, db_path: Path | str, migrations: MigrationChain | None = None) -> None:
        """Open (or create) the database and ensure the schema exists."""
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(_SCHEMA_SQL)
        self._conn.commit()
        self._migrations = migrations if migrations is not None else MigrationChain()

    def save(self, snapshot: Snapshot) -> str:
        """Persist a snapshot and return its storage id."""
        cursor = self._conn.execute(
            "INSERT INTO snapshots (schema_version, world_name, tick, payload) VALUES (?, ?, ?, ?)",
            (
                snapshot.schema_version,
                snapshot.world_name,
                snapshot.tick,
                json.dumps(dict(snapshot.payload)),
            ),
        )
        self._conn.commit()
        return str(cursor.lastrowid)

    def load(self, snapshot_id: str) -> Snapshot:
        """Load a snapshot, migrated to the current schema version."""
        row = self._conn.execute(
            "SELECT schema_version, world_name, tick, payload FROM snapshots WHERE id = ?",
            (int(snapshot_id),),
        ).fetchone()
        if row is None:
            msg = f"no snapshot with id {snapshot_id!r}"
            raise KeyError(msg)
        schema_version, world_name, tick, payload = row
        snapshot = Snapshot(
            schema_version=int(schema_version),
            world_name=str(world_name),
            tick=int(tick),
            payload=json.loads(payload),
        )
        return self._migrations.migrate(snapshot)

    def list_snapshots(self, world_name: str | None = None) -> tuple[str, ...]:
        """Return snapshot ids, optionally filtered to one world."""
        if world_name is None:
            rows = self._conn.execute("SELECT id FROM snapshots ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id FROM snapshots WHERE world_name = ? ORDER BY id",
                (world_name,),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
