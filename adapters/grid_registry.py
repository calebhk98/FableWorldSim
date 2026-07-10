"""Grid backend registry: the single toggle that selects the DGGS.

``create_grid("h3" | "s2" | "isea", resolution=...)`` is the only place a
backend name means anything.  Everything downstream of the returned
:class:`ports.grid.Grid` is backend-agnostic, so toggling backends must
never change simulation results (all math is area-weighted).

Built-in backends are imported lazily so that optional dependencies are
only required for the backend actually toggled on.  Additional backends
(including test fakes) can be registered at runtime with
:func:`register_grid_backend`.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from ports.grid import EARTH_MEAN_RADIUS_M, Grid

GridFactory = Callable[[int, float], Grid]
"""Factory signature: ``(resolution, radius_m) -> Grid``."""

_BUILTIN_MODULES: dict[str, str] = {
    "h3": "adapters.grid_h3",
    "s2": "adapters.grid_s2",
    "isea": "adapters.grid_isea",
}

_registry: dict[str, GridFactory] = {}


def register_grid_backend(name: str, factory: GridFactory) -> None:
    """Register (or replace) a grid backend under a toggle name."""
    _registry[name.lower()] = factory


def available_backends() -> tuple[str, ...]:
    """Return the sorted toggle names of all known backends."""
    return tuple(sorted(set(_registry) | set(_BUILTIN_MODULES)))


def create_grid(
    backend: str,
    *,
    resolution: int,
    radius_m: float = EARTH_MEAN_RADIUS_M,
) -> Grid:
    """Build a grid from a backend toggle name.

    Raises ``ValueError`` for an unknown backend name and
    :class:`ports.grid.GridBackendUnavailableError` when the backend is
    known but its dependency is not installed in this environment.
    """
    key = backend.lower()
    factory = _registry.get(key)
    if factory is None and key in _BUILTIN_MODULES:
        module = import_module(_BUILTIN_MODULES[key])
        factory = module.create
        _registry[key] = factory
    if factory is None:
        known = ", ".join(available_backends())
        msg = f"unknown grid backend {backend!r}; known backends: {known}"
        raise ValueError(msg)
    return factory(resolution, radius_m)
