"""Shared CLI output helpers: JSON pretty-printing and fatal-error exit.

Split out of :mod:`clients.cli.main` so both it and
:mod:`clients.cli.world_cli` can use the same helpers without either
importing the other (avoids a circular import between the two).
"""

from __future__ import annotations

import json
import sys
from typing import Any


def print_json(data: Any, indent: int = 2) -> None:
    """Pretty-print a JSON-serializable object."""
    print(json.dumps(data, indent=indent, default=str))


def fatal(msg: str) -> None:
    """Print an error and exit with status 1."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)
