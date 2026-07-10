"""The three independent version axes.

1. ``ENGINE_VERSION`` — the simulator build (SemVer; the public API
   contract follows it, so scripted integrations don't silently break).
2. Snapshot-schema version — ``ports.storage.CURRENT_SCHEMA_VERSION``,
   upgraded by the migration chain.
3. ``CONTENT_SCHEMA_VERSION`` — the shape of content/mod data files;
   mod manifests pin ``min_engine_version`` against axis 1.
"""

from __future__ import annotations

ENGINE_VERSION = "0.1.0"
"""Engine build version (SemVer)."""

CONTENT_SCHEMA_VERSION = 1
"""Version of the content/mod data-file schema."""


def parse_version(text: str) -> tuple[int, int, int]:
    """Parse ``major.minor.patch`` into a comparable tuple."""
    parts = text.strip().split(".")
    expected_parts = 3
    if len(parts) != expected_parts or not all(p.isdigit() for p in parts):
        msg = f"not a SemVer string: {text!r}"
        raise ValueError(msg)
    major, minor, patch = (int(p) for p in parts)
    return (major, minor, patch)


def engine_satisfies(minimum: str) -> bool:
    """Return whether this engine meets a mod's ``min_engine_version``."""
    return parse_version(ENGINE_VERSION) >= parse_version(minimum)
