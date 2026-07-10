"""Reprojection: moving a world's fields onto a different grid backend.

Backends' cells are not spatially congruent, so the backend is chosen at
world-creation time; switching an existing world is this explicit
resampling operation — supported but not free, and surfaced in the API
as a distinct operation from the create-time choice.

M1 uses nearest-centroid sampling: first-order accurate and unbiased for
intensive fields.  Extensive amounts are converted to densities, sampled,
and re-integrated, which approximately (not exactly) preserves global
totals; an exactly conservative overlap-area remap is roadmap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports.grid import CellId, Grid


def reproject_intensive(
    values: Mapping[CellId, float],
    source: Grid,
    target: Grid,
) -> dict[CellId, float]:
    """Resample an intensive (per-area) field onto another grid.

    Each target cell takes the source value at its centroid.
    """
    return {cell: values[source.cell_at(target.centroid(cell))] for cell in target.cells()}


def reproject_extensive(
    amounts: Mapping[CellId, float],
    source: Grid,
    target: Grid,
) -> dict[CellId, float]:
    """Resample an extensive (total-per-cell) field onto another grid.

    Converts to density (amount / area), samples, and multiplies by the
    target cell's area, so totals are approximately preserved.
    """
    densities = {cell: amounts[cell] / source.area_m2(cell) for cell in amounts}
    resampled = reproject_intensive(densities, source, target)
    return {cell: density * target.area_m2(cell) for cell, density in resampled.items()}
