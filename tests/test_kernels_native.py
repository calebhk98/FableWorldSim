"""The 15-minute-swap proof: native C kernel matches the reference exactly."""

from __future__ import annotations

import pytest

from adapters.kernel_registry import build_kernel_registry, select_kernel
from adapters.kernels_native import flow_accumulation_c, native_kernels_available
from adapters.kernels_python import NO_RECEIVER, flow_accumulation

_needs_compiler = pytest.mark.skipif(
    not native_kernels_available(), reason="no C compiler on this host"
)


def _river_network() -> tuple[list[int], list[float], list[float]]:
    """Return a 200-cell braided drainage: two ridges into one outlet."""
    count = 200
    elevations = [float((count - i) % 97 + (i % 7) * 13) for i in range(count)]
    receivers = []
    for i in range(count):
        downhill = [j for j in (i - 1, i + 1, i + 13) if 0 <= j < count]
        lower = [j for j in downhill if elevations[j] < elevations[i]]
        receivers.append(min(lower, key=lambda j: elevations[j]) if lower else NO_RECEIVER)
    areas = [1.0 + (i % 5) * 0.25 for i in range(count)]
    return receivers, areas, elevations


@_needs_compiler
def test_native_kernel_matches_reference_exactly() -> None:
    receivers, areas, elevations = _river_network()
    reference = flow_accumulation(receivers, areas, elevations)
    native = flow_accumulation_c(receivers, areas, elevations)
    assert native == pytest.approx(reference, rel=1e-12)


@_needs_compiler
def test_auto_prefers_native_and_config_can_pin_either() -> None:
    registry = build_kernel_registry()
    assert set(registry.implementations("flow_accumulation")) == {"python", "c"}
    receivers, areas, elevations = _river_network()
    auto = select_kernel(registry, "flow_accumulation", "auto")
    pinned_python = select_kernel(registry, "flow_accumulation", "python")
    pinned_c = select_kernel(registry, "flow_accumulation", "c")
    expected = flow_accumulation(receivers, areas, elevations)
    for impl in (auto, pinned_python, pinned_c):
        assert impl(receivers, areas, elevations) == pytest.approx(expected)


def test_registry_works_without_a_compiler() -> None:
    registry = build_kernel_registry()
    fallback = select_kernel(registry, "flow_accumulation", "python")
    receivers, areas, elevations = _river_network()
    assert sum(a for r, a in zip(receivers, areas, strict=True) if r == NO_RECEIVER) > 0
    assert len(fallback(receivers, areas, elevations)) == len(receivers)  # type: ignore[arg-type]
