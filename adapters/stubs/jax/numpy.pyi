"""Thin stub for jax.numpy, consumed as an opaque Array-API namespace."""

from typing import Any

def __getattr__(name: str) -> Any: ...
