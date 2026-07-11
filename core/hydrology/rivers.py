"""Hydrology part B: river networks from steepest-descent flow routing.

Runs after climate (M1.2 sea level/land mask) and after precipitation is
available, because channel sizing draws on runoff to estimate discharge.

Flow routing here is a single-flow-direction scheme: each land cell drains
to its one steepest strictly-lower neighbor. This is the classic "D8"
scheme adapted to a variable-degree hex/DGG grid — instead of picking
among eight fixed compass directions on a square raster, a cell picks the
steepest of however many neighbors ``ports/grid.py`` gives it (6 for most
H3/ISEA3H hexes, 5 at their 12 pentagons, 4 for S2 quads). Ocean cells and
local minima ("pits" — land cells with no strictly-lower neighbor, i.e.
endorheic basins) are terminal sinks that do not route further.

The resulting receiver graph is index-based (cells 0..n-1, matching the
``flow_accumulation`` kernel contract in ``ports/kernel.py`` /
``adapters/kernels_python.py``) so it can be handed straight to an
injected accumulation kernel to get per-cell drainage area. Cells whose
drainage area clears a threshold are classified as channels and sized via
downstream hydraulic geometry (Leopold & Maddock 1953): width and depth
grow with discharge along the network, not with elevation drop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.sim.constants import SECONDS_PER_YEAR

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid

NO_RECEIVER = -1
"""Sentinel for a cell that drains nowhere (ocean cell or pit).

Mirrors ``adapters.kernels_python.NO_RECEIVER`` and the ``flow_accumulation``
kernel-port contract (receiver == -1 means "no downstream cell"); defined
locally because core must not import adapters.
"""

FlowAccumulator = Callable[["Sequence[int]", "Sequence[float]", "Sequence[float]"], "list[float]"]
"""Injected drainage-accumulation kernel: ``(receivers, areas_m2, elevations_m) ->
per-cell accumulated drainage area in m^2``. Matches the ``flow_accumulation``
signature in ``ports/kernel.py``; production callers inject
``adapters.kernels_python.flow_accumulation`` (or a faster registered impl)."""

_CHANNEL_THRESHOLD_CELL_MULTIPLE = 50.0
"""Default channel-initiation threshold, as a multiple of the grid's mean
cell area. Real drainage networks need dozens to hundreds of upstream
cells' worth of catchment before a permanent channel forms (headwater
hillslopes are unchannelized); 50x the mean cell area is a middle-of-the-
road pick that yields a sparse dendritic network rather than "every cell
with any uphill neighbor is a river" or "almost nothing channelizes"."""

_DEFAULT_PRECIPITATION_MM_YR = 1_000.0
"""Global-average terrestrial precipitation baseline (~Earth's land mean),
used to derive a spatially uniform runoff rate when no precipitation
field is supplied."""

_HYDRAULIC_WIDTH_EXPONENT = 0.5
"""Leopold & Maddock (1953) downstream hydraulic-geometry width exponent:
width w = K_w * Q^0.5 along a channel network as discharge Q grows."""

_HYDRAULIC_DEPTH_EXPONENT = 0.4
"""Leopold & Maddock (1953) downstream hydraulic-geometry depth exponent:
depth d = K_d * Q^0.4 (velocity absorbs the remaining ~0.1 so that
continuity w * d * v = Q holds approximately)."""

_HYDRAULIC_WIDTH_COEFFICIENT_M = 2.0
"""Width prefactor K_w (SI: meters, with Q in m^3/s). Real-world prefactors
vary hugely with climate, sediment supply, and bed/bank material; this
value is an order-of-magnitude pick, not calibrated to any real basin —
it puts a modest headwater channel (Q ~ 1 m^3/s) around 2 m wide and a
large trunk river (Q ~ 10,000 m^3/s) around 200 m wide."""

_HYDRAULIC_DEPTH_COEFFICIENT_M = 0.3
"""Depth prefactor K_d (SI: meters, with Q in m^3/s); see
``_HYDRAULIC_WIDTH_COEFFICIENT_M`` for the same caveats. Paired with
``_HYDRAULIC_WIDTH_COEFFICIENT_M`` this keeps width > depth across the
plausible discharge range, as in real channels."""


def _steepest_descent_receiver(
    cell: CellId,
    grid: Grid,
    heights_m: Mapping[CellId, float],
    index_of: Mapping[CellId, int],
) -> int:
    """Return the index of ``cell``'s steepest strictly-downhill neighbor.

    Returns ``NO_RECEIVER`` when no neighbor sits strictly lower (a pit /
    local minimum, i.e. an endorheic sink).
    """
    lower = [n for n in grid.neighbors(cell) if heights_m[n] < heights_m[cell]]
    if not lower:
        return NO_RECEIVER
    return index_of[min(lower, key=lambda n: heights_m[n])]


def steepest_descent_receivers(
    grid: Grid,
    heights_m: Mapping[CellId, float],
    sea_mask: SeaMask,
) -> tuple[list[CellId], list[int], list[float], list[float]]:
    """Build the steepest-descent flow-direction graph for a height field.

    Returns ``(cells, receivers, areas_m2, elevations_m)``, all aligned by
    index and ready to feed straight into a ``flow_accumulation`` kernel:
    ``cells`` is a stable ordering (the iteration order of ``heights_m``),
    ``receivers[i]`` is the index each cell drains to (or ``NO_RECEIVER``),
    ``areas_m2[i]`` is that cell's surface area, and ``elevations_m[i]`` is
    its height. Ocean cells are terminal sinks (water that reached the sea
    has arrived; it does not need to keep draining within the ocean), and
    land pits with no strictly-lower neighbor are terminal endorheic sinks.
    """
    cells = list(heights_m)
    index_of = {cell: i for i, cell in enumerate(cells)}
    receivers = [
        NO_RECEIVER
        if sea_mask.ocean[cell]
        else _steepest_descent_receiver(cell, grid, heights_m, index_of)
        for cell in cells
    ]
    areas_m2 = [grid.area_m2(cell) for cell in cells]
    elevations_m = [heights_m[cell] for cell in cells]
    return cells, receivers, areas_m2, elevations_m


def terminal_index(start: int, receivers: Sequence[int]) -> int:
    """Follow a receiver chain from ``start`` to its terminal sink index.

    A terminal sink is a cell with ``NO_RECEIVER`` (an ocean cell or a
    pit). Steepest-descent receivers strictly decrease in elevation, so a
    valid graph cannot cycle; the visited-set guard is a defensive stop in
    case a caller hands in a hand-built or corrupted receiver graph.
    """
    seen: set[int] = set()
    current = start
    while receivers[current] != NO_RECEIVER and current not in seen:
        seen.add(current)
        current = receivers[current]
    return current


def river_outlets(cells: Sequence[CellId], receivers: Sequence[int]) -> dict[CellId, CellId]:
    """Return each cell's terminal sink cell (its river's outlet).

    The outlet is where a cell's receiver chain stops draining — an ocean
    shoreline cell for a river that reaches the sea, or a pit cell for an
    endorheic basin. Callers combine this with a ``SeaMask`` to check
    whether drainage actually reaches the ocean.
    """
    index_of = {cell: i for i, cell in enumerate(cells)}
    return {cell: cells[terminal_index(index_of[cell], receivers)] for cell in cells}


@dataclass(frozen=True)
class RiverNetwork:
    """A solved river network: drainage areas, channel mask, and geometry.

    All mappings are keyed by cell id and cover every cell in the grid.
    Non-channel cells have ``width_m`` and ``depth_m`` of ``0.0``.
    """

    drainage_area_m2: Mapping[CellId, float]
    is_channel: Mapping[CellId, bool]
    width_m: Mapping[CellId, float]
    depth_m: Mapping[CellId, float]
    channel_threshold_m2: float


def _runoff_rate_m_s(
    cell: CellId,
    precipitation_mm_yr: Mapping[CellId, float] | None,
    runoff_coefficient: float,
) -> float:
    """Return a cell's runoff rate in meters/second (depth of water per second).

    Uses the cell's own precipitation when a field is supplied (spatially
    varying runoff); otherwise falls back to a uniform global-average
    baseline, so the whole grid gets the same rate. Either way,
    ``runoff_coefficient`` is the fraction of precipitation that becomes
    channeled runoff rather than infiltrating or evaporating.
    """
    if precipitation_mm_yr is not None:
        precip_mm_yr = precipitation_mm_yr[cell]
    else:
        precip_mm_yr = _DEFAULT_PRECIPITATION_MM_YR
    precip_m_yr = precip_mm_yr / 1_000.0
    return precip_m_yr / SECONDS_PER_YEAR * runoff_coefficient


def _channel_geometry(discharge_m3_s: float) -> tuple[float, float]:
    """Return ``(width_m, depth_m)`` for a channel via hydraulic geometry.

    Leopold & Maddock (1953) downstream hydraulic geometry: at increasing
    discharge along a network, width and depth grow as power laws of Q
    (see the module-level exponent/coefficient constants for the SI
    prefactors chosen and their caveats).
    """
    width_m = _HYDRAULIC_WIDTH_COEFFICIENT_M * discharge_m3_s**_HYDRAULIC_WIDTH_EXPONENT
    depth_m = _HYDRAULIC_DEPTH_COEFFICIENT_M * discharge_m3_s**_HYDRAULIC_DEPTH_EXPONENT
    return width_m, depth_m


def build_river_network(  # noqa: PLR0913 - matches the injected-kernel + tunable-params contract
    grid: Grid,
    heights_m: Mapping[CellId, float],
    sea_mask: SeaMask,
    accumulate: FlowAccumulator,
    *,
    precipitation_mm_yr: Mapping[CellId, float] | None = None,
    channel_threshold_m2: float | None = None,
    runoff_coefficient: float = 0.3,
) -> RiverNetwork:
    """Route flow, accumulate drainage area, and size the channel network.

    Steps: build steepest-descent receivers (:func:`steepest_descent_receivers`),
    call the injected ``accumulate`` kernel (e.g.
    ``adapters.kernels_python.flow_accumulation``) to get per-cell drainage
    area, classify a cell as a channel where its drainage area clears
    ``channel_threshold_m2`` (default: :data:`_CHANNEL_THRESHOLD_CELL_MULTIPLE`
    times the grid's mean cell area), and size each channel cell from an
    estimated discharge ``Q = drainage_area_m2 * runoff_rate_m_s``: a
    first-order approximation that treats the whole upstream catchment as
    contributing at this cell's local runoff rate (real discharge would
    accumulate runoff cell-by-cell up the network; this module trades that
    precision for simplicity — see :func:`_runoff_rate_m_s`). Non-channel
    cells get ``width_m = depth_m = 0.0``.

    Raises:
        ValueError: if ``runoff_coefficient`` or a resolved
            ``channel_threshold_m2`` is not strictly positive.
    """
    if runoff_coefficient <= 0.0:
        msg = f"runoff_coefficient must be > 0, got {runoff_coefficient}"
        raise ValueError(msg)

    cells, receivers, areas_m2, elevations_m = steepest_descent_receivers(grid, heights_m, sea_mask)
    drainage_m2 = accumulate(receivers, areas_m2, elevations_m)

    threshold_m2 = channel_threshold_m2
    if threshold_m2 is None:
        threshold_m2 = _CHANNEL_THRESHOLD_CELL_MULTIPLE * (sum(areas_m2) / len(areas_m2))
    if threshold_m2 <= 0.0:
        msg = f"channel_threshold_m2 must be > 0, got {threshold_m2}"
        raise ValueError(msg)

    # A "channel" is a river reach on land; the ocean is already water via
    # SeaMask, so ocean cells never count as channels even though a river
    # mouth cell can accumulate large drainage from the land draining into it.
    is_channel = {
        cell: not sea_mask.ocean[cell] and drainage_m2[i] >= threshold_m2
        for i, cell in enumerate(cells)
    }

    width_m: dict[CellId, float] = {}
    depth_m: dict[CellId, float] = {}
    for i, cell in enumerate(cells):
        if not is_channel[cell]:
            width_m[cell] = 0.0
            depth_m[cell] = 0.0
            continue
        runoff_rate = _runoff_rate_m_s(cell, precipitation_mm_yr, runoff_coefficient)
        discharge_m3_s = drainage_m2[i] * runoff_rate
        width_m[cell], depth_m[cell] = _channel_geometry(discharge_m3_s)

    return RiverNetwork(
        drainage_area_m2=dict(zip(cells, drainage_m2, strict=True)),
        is_channel=is_channel,
        width_m=width_m,
        depth_m=depth_m,
        channel_threshold_m2=threshold_m2,
    )
