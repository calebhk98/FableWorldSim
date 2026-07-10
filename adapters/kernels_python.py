"""Pure-Python kernel implementations: the always-available baseline.

Every kernel here defines the *reference semantics* for its name; faster
numpy/GPU/native implementations must match these outputs (the kernel
conformance tests compare against them).

Kernel signatures are index-based (cells 0..n-1 with adjacency lists),
so implementations can be swapped for array/native code without the
domain layer changing how it calls them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ports.kernel import KernelRegistry

NO_RECEIVER = -1
"""Receiver value marking a cell that drains nowhere (pit or ocean)."""


def flow_accumulation(
    receivers: Sequence[int],
    areas_m2: Sequence[float],
    elevations_m: Sequence[float],
) -> list[float]:
    """Return per-cell drainage area (m^2), including the cell itself.

    ``receivers[i]`` is the index each cell drains to (its steepest
    downhill neighbor) or ``NO_RECEIVER``.  Cells are processed from
    highest to lowest so every donor is accumulated before its receiver.
    """
    if not len(receivers) == len(areas_m2) == len(elevations_m):
        msg = "receivers, areas, and elevations must describe the same cells"
        raise ValueError(msg)
    accumulated = list(areas_m2)
    order = sorted(range(len(receivers)), key=lambda i: -elevations_m[i])
    for index in order:
        receiver = receivers[index]
        if receiver == NO_RECEIVER:
            continue
        if receiver == index or not 0 <= receiver < len(receivers):
            msg = f"cell {index} has invalid receiver {receiver}"
            raise ValueError(msg)
        accumulated[receiver] += accumulated[index]
    return accumulated


def install_python_kernels(registry: KernelRegistry) -> None:
    """Register all pure-Python kernels on a registry."""
    registry.register("flow_accumulation", "python", flow_accumulation)
