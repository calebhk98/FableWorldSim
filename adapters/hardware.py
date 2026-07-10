"""Host hardware probe: use the whole machine, auto-detected.

At startup the sim probes GPU count/VRAM, CPU cores, and total RAM, then
recommends a fidelity profile (grid resolution x timestep x model detail)
and compute backend.  Auto is only the default — everything is pinnable
via config/API.  Detected RAM caps resident resolution; a low-memory
situation downgrades fidelity gracefully instead of crashing (see
:func:`downgrade_fidelity`).
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
    """

    compute_backend: str
    fidelity_level: str
    grid_resolution: int


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

    GPU boxes get the accurate tier; big-RAM CPU boxes the balanced
    tier; small machines (the 4 GB no-GPU laptop) the fast tier at a
    coarse resolution.  The conservation rule makes results comparable
    across all of them.
    """
    if caps.gpu_count >= 1:
        return Recommendation("auto", "accurate", 5)
    if caps.ram_gib >= _HIGH_RAM_GIB:
        return Recommendation("numpy", "balanced", 4)
    if caps.ram_gib >= _LOW_RAM_GIB:
        return Recommendation("numpy", "balanced", 3)
    return Recommendation("numpy", "fast", 2)


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
