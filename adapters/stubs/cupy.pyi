"""Thin stub for cupy (optional GPU dependency, absent on CI runners).

cupy is consumed as an opaque Array-API namespace, so a module-level
__getattr__ is the honest shape; only the device probe is spelled out.
"""

from typing import Any

cuda: Any

def __getattr__(name: str) -> Any: ...
