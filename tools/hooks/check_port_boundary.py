#!/usr/bin/env python3
"""Enforce the hexagonal boundary: the domain core imports PORTS only.

Modules under ``core/`` must depend on abstract ``ports`` and the standard
library, never directly on concrete adapter packages or third-party libraries
(numpy is the one allowed numeric primitive). This is what keeps every library
swappable in minutes; see docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import ast
import sys

# Third-party / adapter packages the core must never import directly.
FORBIDDEN_PREFIXES = (
    "adapters",
    "h3",
    "s2sphere",
    "dggrid4py",
    "cupy",
    "jax",
    "dask",
    "fastapi",
    "zarr",
    "pyarrow",
)


def _imported_names(tree: ast.Module) -> list[str]:
    """Collect every top-level module name imported by ``tree``."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return names


def _violations(path: str) -> list[str]:
    """Return forbidden imports found in a ``core/`` file (empty for other files)."""
    if not path.startswith("core/"):
        return []
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    return [
        name
        for name in _imported_names(tree)
        if name.split(".")[0] in FORBIDDEN_PREFIXES
    ]


def main(paths: list[str]) -> int:
    """Check each path; print boundary violations and fail if any are found."""
    failed = False
    for path in paths:
        for name in _violations(path):
            failed = True
            print(f"{path}: core must not import '{name}' — depend on a port instead")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
