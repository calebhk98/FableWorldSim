"""Default refinement policies: refine where gradients are steep.

Steep height gradients catch coastlines and mountains; the same policy
pointed at a biome or border field catches ecological/political edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ports.grid import CellId, Grid
    from ports.refinement import RefinementPolicy


@dataclass(frozen=True)
class GradientRefinementPolicy:
    """Refine cells whose field contrast with any neighbor exceeds a threshold."""

    field_name: str
    min_contrast: float

    def should_refine(
        self,
        cell: CellId,
        grid: Grid,
        fields: Mapping[str, Mapping[CellId, float]],
    ) -> bool:
        """Return whether the watched field jumps sharply at this cell."""
        values = fields[self.field_name]
        here = values[cell]
        return any(
            abs(values[neighbor] - here) >= self.min_contrast for neighbor in grid.neighbors(cell)
        )


def refinement_targets(
    grid: Grid,
    policy: RefinementPolicy,
    fields: Mapping[str, Mapping[CellId, float]],
) -> Iterator[CellId]:
    """Yield the cells a policy wants refined, in grid order."""
    for cell in grid.cells():
        if policy.should_refine(cell, grid, fields):
            yield cell
