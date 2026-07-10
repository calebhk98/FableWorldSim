"""Refinement policy port: interest-driven adaptive LOD.

The grid's ``parent``/``children`` give the mechanism; this hook gives
the *policy* — refine where the world is interesting (coastlines, biome
and border boundaries, steep gradients).  Designed in at M1 so
incremental dirty-region recompute and the fidelity autotuner (both
roadmap) plug into an existing seam.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports.grid import CellId, Grid


class RefinementPolicy(Protocol):
    """Decides which cells deserve children at the next resolution."""

    def should_refine(
        self,
        cell: CellId,
        grid: Grid,
        fields: Mapping[str, Mapping[CellId, float]],
    ) -> bool:
        """Return whether ``cell`` should be refined given the fields."""
        ...
