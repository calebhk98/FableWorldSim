"""Kernel registry tests: stable signatures, swappable implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from adapters.kernels_python import (
    NO_RECEIVER,
    flow_accumulation,
    install_python_kernels,
)
from ports.kernel import KernelNotFoundError, KernelRegistry

if TYPE_CHECKING:
    from collections.abc import Sequence

_ELEVATIONS = [100.0, 50.0, 10.0, 200.0]
_AREAS = [1.0, 1.0, 1.0, 1.0]
_RECEIVERS = [1, 2, NO_RECEIVER, 0]
_EXPECTED_DRAINAGE = [2.0, 3.0, 4.0, 1.0]


def test_flow_accumulation_reference_semantics() -> None:
    """Drainage area accumulates downhill: 3 -> 0 -> 1 -> 2 (outlet)."""
    result = flow_accumulation(_RECEIVERS, _AREAS, _ELEVATIONS)
    assert result == pytest.approx(_EXPECTED_DRAINAGE)


def test_flow_accumulation_validates_inputs() -> None:
    """Mismatched lengths and self-receivers raise."""
    with pytest.raises(ValueError, match="same cells"):
        flow_accumulation([NO_RECEIVER], _AREAS, _ELEVATIONS)
    with pytest.raises(ValueError, match="invalid receiver"):
        flow_accumulation([0], [1.0], [5.0])


def test_swapping_implementations_preserves_results() -> None:
    """An alternative impl registers in minutes and must agree exactly."""
    registry = KernelRegistry()
    install_python_kernels(registry)

    def alternative(
        receivers: Sequence[int],
        areas_m2: Sequence[float],
        elevations_m: Sequence[float],
    ) -> list[float]:
        """A second implementation with the same stable signature."""
        return flow_accumulation(receivers, areas_m2, elevations_m)

    registry.register("flow_accumulation", "alt", alternative)
    default_fn = registry.get("flow_accumulation")
    alt_fn = registry.get("flow_accumulation", "alt")
    assert default_fn(_RECEIVERS, _AREAS, _ELEVATIONS) == alt_fn(_RECEIVERS, _AREAS, _ELEVATIONS)
    assert registry.implementations("flow_accumulation") == ("alt", "python")


def test_default_selection_and_missing_kernels() -> None:
    """First registration is the default; unknown lookups fail loudly."""
    registry = KernelRegistry()
    registry.register("demo", "one", lambda: 1)
    registry.register("demo", "two", lambda: 2)
    assert registry.get("demo")() == 1
    registry.register("demo", "two", lambda: 2, make_default=True)
    assert registry.get("demo")() == 2
    assert registry.kernels() == ("demo",)
    with pytest.raises(KernelNotFoundError, match="no implementations"):
        registry.get("warp_drive")
    with pytest.raises(KernelNotFoundError, match="no impl"):
        registry.get("demo", "gpu")
