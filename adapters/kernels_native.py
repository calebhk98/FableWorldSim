"""Out-of-Python flow-accumulation kernel: the 15-minute-swap proof.

The same ``Kernel`` port signature served by a compiled C implementation
(built on demand with the host's C compiler, called through ctypes).
This demonstrates the port boundary genuinely permits non-Python
implementations — swapping the sim onto it is a one-line config change
(``[kernels] flow_accumulation = "c"``) — and gives the single most
performance-critical kernel a native path on capable hardware while the
pure-Python reference keeps every laptop working.

A matching-output test runs both implementations on identical inputs.
Machines without a C compiler simply stay on the reference path.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from adapters.kernels_python import NO_RECEIVER, validate_flow_inputs
from ports.kernel import KernelUnavailableError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ports.kernel import KernelRegistry

_C_SOURCE = """
void flow_accumulation(int n, const int *receivers, const int *order,
                       const double *areas, double *out)
{
    for (int i = 0; i < n; i++) {
        out[i] = areas[i];
    }
    for (int k = 0; k < n; k++) {
        int donor = order[k];
        int receiver = receivers[donor];
        if (receiver >= 0) {
            out[receiver] += out[donor];
        }
    }
}
"""

_lib_cache: ctypes.CDLL | None = None


def _find_compiler() -> str | None:
    """Return an available C compiler executable, or None."""
    for name in ("cc", "gcc", "clang"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _compile_library() -> ctypes.CDLL:
    """Compile the C kernel into a shared library (cached per process)."""
    global _lib_cache  # noqa: PLW0603 - one compilation per process by design
    if _lib_cache is not None:
        return _lib_cache
    compiler = _find_compiler()
    if compiler is None:
        msg = (
            "native flow_accumulation needs a C compiler (cc/gcc/clang); "
            "the pure-Python reference implementation remains available"
        )
        raise KernelUnavailableError(msg)
    build_dir = Path(tempfile.mkdtemp(prefix="fws-kernels-"))
    source = build_dir / "flow_accumulation.c"
    library = build_dir / "flow_accumulation.so"
    source.write_text(_C_SOURCE, encoding="utf-8")
    result = subprocess.run(
        [compiler, "-O2", "-shared", "-fPIC", "-o", str(library), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = f"C kernel compilation failed: {result.stderr.strip()}"
        raise KernelUnavailableError(msg)
    _lib_cache = ctypes.CDLL(str(library))
    return _lib_cache


def flow_accumulation_c(
    receivers: Sequence[int],
    areas_m2: Sequence[float],
    elevations_m: Sequence[float],
) -> list[float]:
    """Native implementation of the flow_accumulation kernel signature.

    Same contract as :func:`adapters.kernels_python.flow_accumulation`;
    ordering (highest cell first) stays in Python, the accumulation loop
    runs in C.
    """
    validate_flow_inputs(receivers, areas_m2, elevations_m)
    library = _compile_library()
    count = len(receivers)
    order = sorted(range(count), key=lambda i: -elevations_m[i])
    receivers_arr = (ctypes.c_int * count)(*receivers)
    order_arr = (ctypes.c_int * count)(*order)
    areas_arr = (ctypes.c_double * count)(*areas_m2)
    out_arr = (ctypes.c_double * count)()
    library.flow_accumulation(ctypes.c_int(count), receivers_arr, order_arr, areas_arr, out_arr)
    return list(out_arr)


def native_kernels_available() -> bool:
    """Return whether the native path can be used on this host."""
    return _find_compiler() is not None


def install_native_kernels(registry: KernelRegistry) -> None:
    """Register native kernels, making them the default where usable.

    No-op on hosts without a C compiler: the registry keeps the
    pure-Python defaults and nothing breaks.
    """
    if not native_kernels_available():
        return
    registry.register("flow_accumulation", "c", flow_accumulation_c, make_default=True)


__all__ = [
    "NO_RECEIVER",
    "flow_accumulation_c",
    "install_native_kernels",
    "native_kernels_available",
]
