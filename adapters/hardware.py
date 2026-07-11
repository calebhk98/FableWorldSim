"""Host hardware probe: use the whole machine, auto-detected.

At startup the sim probes GPU count/VRAM, CPU cores, and total RAM, then
recommends a fidelity profile (grid resolution x timestep x model detail),
a compute backend, and a scale-out plan (``device_count`` GPUs to shard
across, ``cpu_workers`` cores to parallelize the CPU path over).  The
profile scales with GPU *count and aggregate VRAM*, so a single 3090, both
3090s, and an 8x A100 box (p4de/p5) each get a distinct answer rather than
one flat "has a GPU" tier.  Auto is only the default — everything is
pinnable via config/API.  Detected RAM caps resident resolution; a
low-memory situation downgrades fidelity gracefully instead of crashing
(see :func:`downgrade_fidelity`).
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from importlib import import_module

_GIB = 1024**3
_LOW_RAM_GIB = 8
_HIGH_RAM_GIB = 32

# At or above this GPU count, prefer jax (device-mesh sharding is its
# multi-GPU story); a single GPU stays on the cupy-preferred "auto" path.
_MULTI_GPU_THRESHOLD = 2

# When a GPU backend reports no per-device VRAM (e.g. the jax probe path),
# assume this conservative floor per device so resolution still scales with
# GPU *count* instead of collapsing to the single-GPU baseline.
_DEFAULT_GPU_VRAM_GIB = 8

# Grid resolution by aggregate GPU VRAM (GiB): each H3/S2 level multiplies
# cell count ~7x, so more resident VRAM buys another refinement level. A
# single 24 GB card holds the "accurate" baseline (5); a p4de/p5 box
# (8x80 GB = 640 GiB) earns the top tier. Descending; first threshold that
# fits the aggregate wins.
_GPU_VRAM_RESOLUTION_LADDER: tuple[tuple[float, int], ...] = (
    (480, 8),
    (160, 7),
    (48, 6),
    (0, 5),
)


@dataclass(frozen=True)
class HostCapabilities:
    """What the machine offers: cores, memory, GPUs."""

    cpu_cores: int
    ram_bytes: int
    gpu_count: int
    gpu_vram_bytes: tuple[int, ...] = ()

    @property
    def ram_gib(self) -> float:
        """Return total RAM in GiB (0.0 when detection failed)."""
        return self.ram_bytes / _GIB


@dataclass(frozen=True)
class Recommendation:
    """First-pass auto-scaling profile derived from the probe.

    The full fidelity autotuner (resolution/timestep from benchmarks) is
    roadmap; these heuristics give sane startup defaults.

    ``device_count`` and ``cpu_workers`` are the *scale-out plan*: how many
    GPUs the array backend should shard fields across, and how many CPU
    cores the numpy path should parallelize over.  They let the recommender
    differentiate every tier — a single 3090, both 3090s, and an 8x A100
    box no longer collapse to one profile — and hand the execution layers a
    concrete target instead of defaulting to one device / one core.
    """

    compute_backend: str
    fidelity_level: str
    grid_resolution: int
    device_count: int = 0
    cpu_workers: int = 1


def _total_ram_bytes() -> int:
    """Return total physical RAM, or 0 when the platform hides it."""
    if sys.platform == "win32":
        return _windows_ram_bytes()
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


def _windows_ram_bytes() -> int:
    """Return total physical RAM on Windows via GlobalMemoryStatusEx."""
    if sys.platform != "win32":  # platform gate doubles as a mypy narrower
        return 0

    class _MemoryStatusEx(ctypes.Structure):
        """ctypes mirror of the Win32 MEMORYSTATUSEX struct."""

        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) == 0:
        return 0
    return int(status.ullTotalPhys)


def _probe_gpus() -> tuple[int, tuple[int, ...]]:
    """Return (gpu count, per-device VRAM bytes); (0, ()) without GPUs.

    Tries cupy first (exact VRAM), then jax device enumeration.  Any
    probe failure means "no usable GPU" — never a crash.
    """
    try:
        cupy = import_module("cupy")
        count = int(cupy.cuda.runtime.getDeviceCount())
        vram = []
        for index in range(count):
            with cupy.cuda.Device(index):
                _free, total = cupy.cuda.runtime.memGetInfo()
                vram.append(int(total))
        return count, tuple(vram)
    except Exception:  # cupy absent or CUDA broken - fall through
        pass
    try:
        jax = import_module("jax")
        devices = [d for d in jax.devices() if d.platform != "cpu"]
        return len(devices), ()
    except Exception:
        return 0, ()


def probe_host() -> HostCapabilities:
    """Return what this machine offers (never raises)."""
    gpu_count, gpu_vram = _probe_gpus()
    return HostCapabilities(
        cpu_cores=os.cpu_count() or 1,
        ram_bytes=_total_ram_bytes(),
        gpu_count=gpu_count,
        gpu_vram_bytes=gpu_vram,
    )


def recommend(caps: HostCapabilities) -> Recommendation:
    """Return a startup profile for the probed hardware.

    GPU boxes get the accurate tier, with resolution scaled by *aggregate
    VRAM* and a ``device_count`` equal to the GPUs to shard across — so a
    single 3090, both 3090s, and an 8x A100 (p4de/p5) box each get a
    distinct profile instead of one flat "has a GPU" answer.  Big-RAM CPU
    boxes get the balanced tier; small machines (the 4 GB no-GPU laptop)
    the fast tier at a coarse resolution.  Every profile carries
    ``cpu_workers`` so the CPU path can use the whole Ryzen 9, not one
    core.  The conservation rule makes results comparable across all tiers.
    """
    workers = max(1, caps.cpu_cores)
    if caps.gpu_count >= 1:
        # Multiple GPUs -> jax, whose device-mesh sharding is the design's
        # multi-GPU path; a single GPU stays on "auto" (cupy preferred).
        backend = "jax" if caps.gpu_count >= _MULTI_GPU_THRESHOLD else "auto"
        resolution = _gpu_resolution(caps)
        return Recommendation(
            backend, "accurate", resolution, device_count=caps.gpu_count, cpu_workers=workers
        )
    if caps.ram_gib >= _HIGH_RAM_GIB:
        return Recommendation("numpy", "balanced", 4, device_count=0, cpu_workers=workers)
    if caps.ram_gib >= _LOW_RAM_GIB:
        return Recommendation("numpy", "balanced", 3, device_count=0, cpu_workers=workers)
    return Recommendation("numpy", "fast", 2, device_count=0, cpu_workers=workers)


def _aggregate_gpu_vram_gib(caps: HostCapabilities) -> float:
    """Return total GPU VRAM in GiB, estimating when the probe hid it.

    The cupy probe reports exact per-device VRAM; the jax fallback reports
    none, so assume a conservative floor per device to keep resolution
    scaling with GPU count instead of collapsing to the baseline.
    """
    if caps.gpu_vram_bytes:
        return sum(caps.gpu_vram_bytes) / _GIB
    return caps.gpu_count * _DEFAULT_GPU_VRAM_GIB


def _gpu_resolution(caps: HostCapabilities) -> int:
    """Return the accurate-tier grid resolution for the GPUs present."""
    aggregate = _aggregate_gpu_vram_gib(caps)
    for threshold, resolution in _GPU_VRAM_RESOLUTION_LADDER:
        if aggregate >= threshold:
            return resolution
    return _GPU_VRAM_RESOLUTION_LADDER[-1][1]


def downgrade_fidelity(level: str) -> str | None:
    """Return the next-lower fidelity level, or None at the floor.

    The graceful answer to running low on memory on a large grid:
    accurate -> balanced -> fast -> (None: nothing left to shed).
    """
    ladder = {"accurate": "balanced", "balanced": "fast", "fast": None}
    if level not in ladder:
        msg = f"unknown fidelity level {level!r}"
        raise ValueError(msg)
    return ladder[level]
