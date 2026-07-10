"""Backend-conformance suite for ArrayBackend adapters.

"Same code, every backend" is a target enforced here, not a freebie:
each available backend must produce identical numerics through the
Array-API namespace.  GPU backends (cupy, jax) run when installed and
skip cleanly otherwise.
"""

from __future__ import annotations

import pytest

from adapters.compute_registry import (
    available_array_backends,
    create_array_backend,
)
from ports.array_backend import ArrayBackend, ComputeBackendUnavailableError

_VALUES = [1.0, 2.0, 3.0, 4.0]
_AREAS = [1.0, 1.0, 3.0, 5.0]
_EXPECTED_MEAN = (1.0 + 2.0 + 9.0 + 20.0) / 10.0
_BACKENDS = ("numpy", "cupy", "jax")


def _backend_or_skip(name: str) -> ArrayBackend:
    """Return a backend, skipping the test when it is unavailable."""
    try:
        return create_array_backend(name)
    except ComputeBackendUnavailableError as exc:
        pytest.skip(f"backend {name} unavailable: {exc}")


@pytest.mark.parametrize("name", _BACKENDS)
def test_conformance_area_weighted_mean(name: str) -> None:
    """Every backend computes the same area-weighted mean."""
    backend = _backend_or_skip(name)
    xp = backend.namespace()
    values = backend.asarray(_VALUES)
    areas = backend.asarray(_AREAS)
    mean = float(xp.sum(values * areas) / xp.sum(areas))
    assert mean == pytest.approx(_EXPECTED_MEAN)


@pytest.mark.parametrize("name", _BACKENDS)
def test_conformance_roundtrip(name: str) -> None:
    """Arrays round-trip to plain Python floats identically."""
    backend = _backend_or_skip(name)
    assert backend.to_list(backend.asarray(_VALUES)) == _VALUES


def test_auto_detection_always_yields_a_backend() -> None:
    """'auto' falls back to CPU numpy when no GPU backend exists."""
    backend = create_array_backend("auto")
    assert backend.name in _BACKENDS
    assert backend.device in {"cpu", "gpu"}


def test_registry_lists_and_rejects() -> None:
    """Toggle names are discoverable; unknown names fail loudly."""
    names = available_array_backends()
    for expected in _BACKENDS:
        assert expected in names
    with pytest.raises(ValueError, match="unknown compute backend"):
        create_array_backend("abacus")


def test_jax_declares_immutability() -> None:
    """The functional backend advertises no in-place mutation."""
    backend = _backend_or_skip("jax")
    assert backend.supports_inplace_mutation is False
