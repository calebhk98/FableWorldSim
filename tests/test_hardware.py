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
_SINGLE_GPU = HostCapabilities(
    cpu_cores=16,
    ram_bytes=64 * _GIB,
    gpu_count=1,
    gpu_vram_bytes=(24 * _GIB,),
)
_DUAL_GPU = HostCapabilities(
    cpu_cores=16,
    ram_bytes=64 * _GIB,
    gpu_count=2,
    gpu_vram_bytes=(24 * _GIB, 24 * _GIB),
)
# AWS p4de.24xlarge: 8x A100 80 GB = 640 GiB aggregate VRAM, 96 vCPU.
_P4DE = HostCapabilities(
    cpu_cores=96,
    ram_bytes=1152 * _GIB,
    gpu_count=8,
    gpu_vram_bytes=(80 * _GIB,) * 8,
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


def test_gpu_profile_scales_with_count_and_vram() -> None:
    # The old recommender flatlined the moment it saw one GPU: a single
    # 3090 and an 8x A100 box got the identical profile. Resolution must
    # now climb with aggregate VRAM.
    single = recommend(_SINGLE_GPU)
    dual = recommend(_DUAL_GPU)
    p4de = recommend(_P4DE)

    assert single.grid_resolution < dual.grid_resolution < p4de.grid_resolution
    assert all(r.fidelity_level == "accurate" for r in (single, dual, p4de))


def test_scale_out_plan_targets_every_device_and_core() -> None:
    # device_count tells the sharding layer how many GPUs to span;
    # cpu_workers tells the numpy path how many cores to use.
    single = recommend(_SINGLE_GPU)
    assert single.compute_backend == "auto"
    assert single.device_count == 1

    p4de = recommend(_P4DE)
    assert p4de.compute_backend == "jax"  # multi-GPU -> device-mesh path
    assert p4de.device_count == 8
    assert p4de.cpu_workers == 96

    laptop = recommend(_LAPTOP_4GB)
    assert laptop.device_count == 0
    assert laptop.cpu_workers == 2


def test_low_memory_downgrades_gracefully() -> None:
    assert downgrade_fidelity("accurate") == "balanced"
    assert downgrade_fidelity("balanced") == "fast"
    assert downgrade_fidelity("fast") is None
    with pytest.raises(ValueError, match="fidelity"):
        downgrade_fidelity("ludicrous")
