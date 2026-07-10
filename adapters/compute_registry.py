"""Compute backend registry: toggle + hardware auto-detection.

``create_array_backend("auto")`` prefers a GPU backend when one is
importable and falls back to numpy on CPU, so the same config runs on a
2010 laptop and a multi-GPU desktop.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from ports.array_backend import ArrayBackend, ComputeBackendUnavailableError

BackendFactory = Callable[[], ArrayBackend]
"""Factory signature for compute backends."""

_BUILTIN_MODULES: dict[str, str] = {
    "numpy": "adapters.compute_numpy",
    "cupy": "adapters.compute_cupy",
    "jax": "adapters.compute_jax",
}

_AUTO_ORDER = ("cupy", "jax", "numpy")

_registry: dict[str, BackendFactory] = {}


def register_array_backend(name: str, factory: BackendFactory) -> None:
    """Register (or replace) a compute backend under a toggle name."""
    _registry[name.lower()] = factory


def available_array_backends() -> tuple[str, ...]:
    """Return the sorted toggle names of all known compute backends."""
    return tuple(sorted(set(_registry) | set(_BUILTIN_MODULES)))


def create_array_backend(name: str = "auto") -> ArrayBackend:
    """Build a compute backend by toggle name (or auto-detect).

    ``"auto"`` tries GPU backends first (cupy, then jax) and falls back
    to numpy; an explicit name raises if that backend is unavailable.
    """
    key = name.lower()
    if key == "auto":
        return _detect()
    factory = _resolve(key)
    if factory is None:
        known = ", ".join((*available_array_backends(), "auto"))
        msg = f"unknown compute backend {name!r}; known: {known}"
        raise ValueError(msg)
    return factory()


def _resolve(key: str) -> BackendFactory | None:
    """Return the factory for a toggle name, importing lazily."""
    factory = _registry.get(key)
    if factory is None and key in _BUILTIN_MODULES:
        module = import_module(_BUILTIN_MODULES[key])
        factory = module.create
        _registry[key] = factory
    return factory


def _detect() -> ArrayBackend:
    """Return the best available backend, falling back to CPU numpy."""
    for key in _AUTO_ORDER:
        factory = _resolve(key)
        if factory is None:
            continue
        try:
            return factory()
        except ComputeBackendUnavailableError:
            continue
    msg = "no compute backend available; numpy failed to import"
    raise ComputeBackendUnavailableError(msg)
