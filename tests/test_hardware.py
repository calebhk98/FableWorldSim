"""Tests for the host probe and auto-scaling recommendations."""

from __future__ import annotations

import pytest

from adapters.hardware import (
    HostCapabilities,
    downgrade_fidelity,
    probe_host,
    recommend,
)

_GIB = 1024**3
_LAPTOP_4GB = HostCapabilities(cpu_cores=2, ram_bytes=4 * _GIB, gpu_count=0)
_DESKTOP_64GB = HostCapabilities(cpu_cores=16, ram_bytes=64 * _GIB, gpu_count=0)
_DUAL_GPU = HostCapabilities(
    cpu_cores=16,
    ram_bytes=64 * _GIB,
    gpu_count=2,
    gpu_vram_bytes=(24 * _GIB, 24 * _GIB),
)


def test_probe_never_raises_and_sees_the_machine() -> None:
    caps = probe_host()
    assert caps.cpu_cores >= 1
    assert caps.ram_bytes >= 0
    assert caps.gpu_count >= 0


def test_recommendations_scale_with_hardware() -> None:
    laptop = recommend(_LAPTOP_4GB)
    assert laptop.compute_backend == "numpy"
    assert laptop.fidelity_level == "fast"

    desktop = recommend(_DESKTOP_64GB)
    assert desktop.fidelity_level == "balanced"

    gpu_box = recommend(_DUAL_GPU)
    assert gpu_box.fidelity_level == "accurate"
    assert gpu_box.grid_resolution > laptop.grid_resolution


def test_low_memory_downgrades_gracefully() -> None:
    assert downgrade_fidelity("accurate") == "balanced"
    assert downgrade_fidelity("balanced") == "fast"
    assert downgrade_fidelity("fast") is None
    with pytest.raises(ValueError, match="fidelity"):
        downgrade_fidelity("ludicrous")
