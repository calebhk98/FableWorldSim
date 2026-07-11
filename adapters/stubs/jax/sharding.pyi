"""Thin stub for the jax.sharding surface FableWorldSim calls.

jax ships its own types, but it is an optional extra that is absent in
lean environments (and on Windows CI); this stub keeps mypy deterministic
whether or not jax is installed. The sharding primitives are consumed
opaquely — construct a mesh, a spec, and a sharding, then hand them to
``jax.device_put`` — so every attribute resolves to ``Any``.
"""

from typing import Any

class Mesh:
    def __init__(self, devices: Any, axis_names: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class NamedSharding:
    def __init__(self, mesh: Any, spec: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class PartitionSpec:
    def __init__(self, *args: Any) -> None: ...

def __getattr__(name: str) -> Any: ...
