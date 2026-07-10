"""Kernel port: hot compute kernels behind stable signatures.

Each kernel name (e.g. ``"flow_accumulation"``) has a documented, stable
call signature; implementations — pure Python, numpy, GPU, or native —
register under that name and are interchangeable.  This is the concrete
proof of the 15-minute-swap claim: swapping a hot loop's language means
registering a new implementation, not touching domain code.

Domain code receives a :class:`KernelRegistry` (dependency injection)
and never imports an implementation module directly.
"""

from __future__ import annotations

from collections.abc import Callable

KernelFn = Callable[..., object]
"""A kernel implementation; the signature is fixed per kernel name."""


class KernelNotFoundError(LookupError):
    """Raised when no implementation is registered for a kernel."""


class KernelRegistry:
    """Registry mapping (kernel name, implementation name) to callables."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._impls: dict[str, dict[str, KernelFn]] = {}
        self._defaults: dict[str, str] = {}

    def register(
        self,
        kernel: str,
        impl: str,
        fn: KernelFn,
        *,
        make_default: bool = False,
    ) -> None:
        """Register an implementation of ``kernel`` under ``impl``.

        The first implementation registered for a kernel becomes its
        default; pass ``make_default=True`` to take over the default.
        """
        by_impl = self._impls.setdefault(kernel, {})
        by_impl[impl] = fn
        if make_default or kernel not in self._defaults:
            self._defaults[kernel] = impl

    def get(self, kernel: str, impl: str | None = None) -> KernelFn:
        """Return an implementation (the default when ``impl`` is None)."""
        by_impl = self._impls.get(kernel)
        if not by_impl:
            msg = f"no implementations registered for kernel {kernel!r}"
            raise KernelNotFoundError(msg)
        chosen = impl if impl is not None else self._defaults[kernel]
        if chosen not in by_impl:
            known = ", ".join(sorted(by_impl))
            msg = f"kernel {kernel!r} has no impl {chosen!r}; known: {known}"
            raise KernelNotFoundError(msg)
        return by_impl[chosen]

    def implementations(self, kernel: str) -> tuple[str, ...]:
        """Return the sorted implementation names for a kernel."""
        return tuple(sorted(self._impls.get(kernel, {})))

    def kernels(self) -> tuple[str, ...]:
        """Return the sorted names of all registered kernels."""
        return tuple(sorted(self._impls))
