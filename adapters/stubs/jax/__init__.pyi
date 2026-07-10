"""Thin stub for the jax surface FableWorldSim calls.

jax ships its own types, but it is an optional extra that is absent in
lean environments (and on Windows CI); this stub keeps mypy deterministic
whether or not jax is installed. jax.numpy is consumed as an opaque
Array-API namespace.
"""

from typing import Any

def default_backend() -> str: ...
def __getattr__(name: str) -> Any: ...
