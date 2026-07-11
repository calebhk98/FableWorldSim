"""Wire lakes back into the river flow graph.

Splitting a filled depression (:mod:`core.hydrology.lakes`) into a lake
changes how water routes: a flat lake surface stops routing internally,
while an overflowing lake drains onward from its rim. These helpers rebuild
the steepest-descent receiver graph over the lake-adjusted terrain so a
fresh ``flow_accumulation`` pass carries every drop the lake absorbs
downstream of the original pit instead of dead-ending there.

Kept separate from :mod:`core.hydrology.lakes` (which only depends on
:class:`~core.hydrology.lakes.LakeNetwork`, one-directionally) so neither
module grows past the file-length limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.hydrology.lakes import LakeNetwork
from core.hydrology.rivers import NO_RECEIVER, steepest_descent_receivers

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid


def lake_adjusted_heights(
    heights_m: Mapping[CellId, float], lakes: LakeNetwork
) -> dict[CellId, float]:
    """Return ``heights_m`` with every lake cell raised to its water level.

    Feed the result back into
    :func:`core.hydrology.rivers.steepest_descent_receivers` (with the same
    ``grid``/``sea_mask``) to get a lake-aware receiver graph, ready for a
    fresh ``flow_accumulation`` pass: a flat lake surface has no strictly
    lower neighbor *within* the lake, so its interior (correctly) stops
    routing internally, while an overflowing lake's rim cell — now tied
    with, rather than higher than, the raised lake surface — again has a
    strictly lower neighbor *outside* the basin and so drains onward,
    continuing the river network downstream of the filled depression.
    Terminal (endorheic) lakes simply stay a sink, since their level never
    reaches a rim cell.
    """
    adjusted = dict(heights_m)
    for lake in lakes.lakes:
        for cell in lake.cells:
            adjusted[cell] = lake.level_m
    return adjusted


_FLAT_TIE_BREAK_EPSILON_M = 1e-3
"""Elevation nudge (1 mm) added to an overflowing lake's cells so its flat
surface sits infinitesimally *above*, rather than tied with, its rim —
standard practice in DEM depression filling (e.g. Garbrecht & Martz 1997).
Without it, a tie leaves the rim's drainage direction undecided and
violates ``flow_accumulation``'s requirement that a donor's elevation
strictly exceed its receiver's, without perceptibly changing the level."""


def lake_aware_receivers(
    grid: Grid, heights_m: Mapping[CellId, float], sea_mask: SeaMask, lakes: LakeNetwork
) -> tuple[list[CellId], list[int], list[float], list[float]]:
    """Return a lake-aware flow graph, ready for a fresh ``flow_accumulation`` pass.

    Same shape as :func:`core.hydrology.rivers.steepest_descent_receivers`
    (``cells``, ``receivers``, ``areas_m2``, ``elevations_m``). Receivers
    are recomputed over :func:`lake_adjusted_heights`, nudged up by
    :data:`_FLAT_TIE_BREAK_EPSILON_M` for overflowing lakes so their rim
    correctly drains outward instead of the rim draining inward as it did
    on the raw terrain (terminal lakes skip the nudge: a real neighbor
    could otherwise, by coincidence, sit just below their non-pour-point
    level and leak a closed basin). Any lake cell still a sink afterwards
    (a multi-cell lake's low point, bordering only other lake cells) is
    then redirected straight to its lake's ``outlet`` when one exists, so
    every drop the lake absorbs keeps flowing downstream of the filled
    depression instead of dead-ending at the original pit.
    """
    adjusted = lake_adjusted_heights(heights_m, lakes)
    for lake in lakes.lakes:
        if lake.outlet is None:
            continue
        for cell in lake.cells:
            adjusted[cell] += _FLAT_TIE_BREAK_EPSILON_M
    cells, receivers, areas_m2, elevations_m = steepest_descent_receivers(grid, adjusted, sea_mask)
    index_of = {cell: i for i, cell in enumerate(cells)}
    for lake in lakes.lakes:
        if lake.outlet is None:
            continue
        outlet_index = index_of[lake.outlet]
        for cell in lake.cells:
            i = index_of[cell]
            if receivers[i] == NO_RECEIVER:
                receivers[i] = outlet_index
    return cells, receivers, areas_m2, elevations_m
