"""Kernel wiring: install every implementation, honor the config choice.

``[kernels] flow_accumulation = "auto" | "python" | "c"`` is the
one-line swap: auto takes the registry default (native when a compiler
exists, reference otherwise), an explicit name pins one implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from adapters.kernels_native import install_native_kernels
from adapters.kernels_python import install_python_kernels
from ports.kernel import KernelRegistry

if TYPE_CHECKING:
    from ports.kernel import KernelFn


def build_kernel_registry() -> KernelRegistry:
    """Return a registry with all usable implementations installed.

    Pure-Python reference kernels always register first (and are always
    present); native kernels overlay as defaults only where the host can
    actually build them.
    """
    registry = KernelRegistry()
    install_python_kernels(registry)
    install_native_kernels(registry)
    return registry


def select_kernel(registry: KernelRegistry, kernel: str, choice: str) -> KernelFn:
    """Return the implementation the config asks for.

    ``"auto"`` means the registry default; anything else is an explicit
    implementation name (KernelNotFoundError if absent on this host).
    """
    return registry.get(kernel, None if choice == "auto" else choice)
