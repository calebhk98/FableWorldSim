#!/usr/bin/env python3
"""Fail the commit if any source file exceeds the length limit.

Long files hide too much and resist decomposition, so we cap them and force a
split into smaller modules. The limit is intentionally strict; raise it only
with a deliberate team decision.
"""

from __future__ import annotations

import sys

MAX_LINES = 400


def _too_long(path: str) -> int | None:
    """Return the line count of ``path`` if it exceeds ``MAX_LINES``, else ``None``."""
    with open(path, encoding="utf-8") as handle:
        count = sum(1 for _ in handle)
    if count > MAX_LINES:
        return count
    return None


def main(paths: list[str]) -> int:
    """Check each path; print offenders and return a non-zero exit code if any."""
    offenders = [(path, count) for path in paths if (count := _too_long(path)) is not None]
    for path, count in offenders:
        print(f"{path}: {count} lines exceeds limit of {MAX_LINES}")
    return 1 if offenders else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
