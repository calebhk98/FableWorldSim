"""Hydrology part C: depression filling into lakes.

Rivers (``core.hydrology.rivers``) route flow by steepest descent and stop
at any "pit" (a land cell with no strictly-lower neighbor). Real terrain
pits do not just vanish: they fill with runoff until either the basin
overflows a low point on its rim (a lake with an outlet, continuing the
river network downstream) or the water surface settles at an equilibrium
level where evaporation balances inflow (a terminal/endorheic lake, or no
lake at all if evaporation wins even at the smallest possible extent).

This module has two phases:

1. **Depression filling** (:func:`fill_depressions`): a priority-flood
   scan (Barnes, Lehman & Mulla 2014; the graph generalization of
   Planchon & Darboux 2002) over the DGGS neighbor graph. Seeded from the
   ocean boundary, it floods inward and assigns every land cell the
   lowest possible "pour point" elevation reachable on a path to the
   ocean — i.e. the level the cell's basin would reach if filled all the
   way to the rim. This is backend-agnostic: it only ever calls
   ``grid.neighbors``, so it is correct at 5-edge pentagons as well as
   6-edge hexes.
2. **Water balance** (:func:`build_lake_network`): for each basin found
   by phase 1, compares the catchment's accumulated runoff (inflow)
   against evaporation from the lake's surface area at increasing
   candidate levels, to decide whether a lake actually forms and at what
   level — full pour-point level and overflowing, a lower equilibrium
   level and endorheic, or dry.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.hydrology.rivers import NO_RECEIVER, river_outlets, steepest_descent_receivers
from core.sim.constants import SECONDS_PER_YEAR

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid

_DEFAULT_PRECIPITATION_MM_YR = 1_000.0
"""Global-average terrestrial precipitation baseline, mirroring
``core.hydrology.rivers._DEFAULT_PRECIPITATION_MM_YR`` — used to derive a
spatially uniform catchment runoff rate when no precipitation field is
supplied."""

_DEFAULT_EVAPORATION_MM_YR = 1_000.0
"""Global-average open-water evaporation baseline (order-of-magnitude pick
for a temperate lake), used when no evaporation field is supplied. Real
lake evaporation varies hugely with climate (a few hundred mm/yr in cold
or humid regions to several meters/yr in hot deserts); callers with a
climate model should pass a spatially varying field instead."""

_LAKE_LEVEL_BISECTION_STEPS = 60
"""Iterations for the terminal-lake level bisection. Each step halves the
search interval, so 60 steps resolve the level to far better than
floating-point-meaningful precision for any plausible basin depth."""


def fill_depressions(
    grid: Grid, heights_m: Mapping[CellId, float], sea_mask: SeaMask
) -> dict[CellId, float]:
    """Return each land cell's pour-point elevation via priority-flood.

    A min-priority queue, seeded with every ocean cell at sea level (the
    fixed water boundary all drainage ultimately reaches), repeatedly
    pops the globally lowest not-yet-visited frontier cell; a running
    ``water_level`` (initialized to sea level) tracks the highest
    elevation popped so far and never decreases, so every land cell is
    assigned ``max(water_level_when_reached, own_height)`` — the lowest
    rim its basin would have to fill to before spilling toward the ocean.

    A cell's result equals its own height whenever a path to the ocean
    exists that never rises above it (it drains freely); it is strictly
    higher only inside a depression, which is exactly what
    :func:`build_lake_network` uses to find candidate lake basins.
    """
    visited = {cell for cell in heights_m if sea_mask.ocean[cell]}
    heap: list[tuple[float, CellId]] = [(sea_mask.sea_level_m, cell) for cell in visited]
    heapq.heapify(heap)

    fill_level: dict[CellId, float] = {}
    water_level = sea_mask.sea_level_m
    while heap:
        popped_height, cell = heapq.heappop(heap)
        water_level = max(water_level, popped_height)
        if not sea_mask.ocean[cell]:
            fill_level[cell] = water_level
        for neighbor in grid.neighbors(cell):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            heapq.heappush(heap, (heights_m[neighbor], neighbor))
    return fill_level


def _basins(
    grid: Grid, heights_m: Mapping[CellId, float], fill_level: Mapping[CellId, float]
) -> list[tuple[frozenset[CellId], float]]:
    """Group flooded cells into basins: connected regions at one shared level.

    Priority-flood assigns a uniform pour-point elevation to every cell in
    one filled depression, so grouping adjacent cells that share the exact
    same ``fill_level`` value (a connected-components search) recovers
    each basin. Two basins that happen to reach the same numeric level
    without touching stay separate components.
    """
    wet = {cell: level for cell, level in fill_level.items() if level > heights_m[cell]}
    visited: set[CellId] = set()
    basins: list[tuple[frozenset[CellId], float]] = []
    for start, level in wet.items():
        if start in visited:
            continue
        visited.add(start)
        component = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for neighbor in grid.neighbors(current):
                if neighbor in visited or wet.get(neighbor) != level:
                    continue
                visited.add(neighbor)
                component.add(neighbor)
                frontier.append(neighbor)
        basins.append((frozenset(component), level))
    return basins


def _basin_outlet(
    grid: Grid, heights_m: Mapping[CellId, float], basin: frozenset[CellId]
) -> CellId:
    """Return the lowest cell bordering ``basin`` from outside it.

    This is the basin's pour point: the neighbor water spills into once
    the lake tops out at the basin's fill level, and the cell downstream
    flow routing should continue from.
    """
    rim = {neighbor for cell in basin for neighbor in grid.neighbors(cell) if neighbor not in basin}
    return min(rim, key=lambda cell: heights_m[cell])


def _catchment_of(basin: frozenset[CellId], outlets: Mapping[CellId, CellId]) -> list[CellId]:
    """Return every cell whose pre-fill drainage terminates inside ``basin``.

    Includes the basin's own cells (direct precipitation/runoff on the
    lake surface) as well as surrounding hillslope cells that drain into
    it, matching ``core.hydrology.rivers.river_outlets`` semantics.
    """
    return [cell for cell, outlet in outlets.items() if outlet in basin]


def _runoff_rate_m_s(
    cell: CellId,
    rate_mm_yr: Mapping[CellId, float] | None,
    default_mm_yr: float,
    coefficient: float,
) -> float:
    """Return a per-cell rate (m/s) from an mm/yr field or a uniform default.

    Shared shape with ``core.hydrology.rivers._runoff_rate_m_s``: used
    here for both catchment runoff (``coefficient`` = the fraction of
    precipitation that becomes inflow) and lake evaporation (``coefficient
    = 1.0``, the whole field is already a water-loss rate).
    """
    rate_mm = rate_mm_yr[cell] if rate_mm_yr is not None else default_mm_yr
    return (rate_mm / 1_000.0) / SECONDS_PER_YEAR * coefficient


def _area_at_level(
    grid: Grid, heights_m: Mapping[CellId, float], basin: frozenset[CellId], level_m: float
) -> float:
    """Return the surface area (m^2) of ``basin`` cells at or below ``level_m``."""
    return sum(grid.area_m2(cell) for cell in basin if heights_m[cell] <= level_m)


@dataclass(frozen=True)
class Lake:
    """One basin holding water: its extent, level, balance, and outlet.

    ``cells`` covers only the flooded footprint (the deepest cell(s) of
    the basin up to ``level_m``, not the whole rim-to-rim depression).
    ``outlet`` is the downstream cell the lake spills into when
    ``level_m >= spill_level_m`` (an overflowing lake); it is ``None``
    for a terminal (endorheic) lake whose level sits below the basin's
    pour point.
    """

    cells: frozenset[CellId]
    level_m: float
    spill_level_m: float
    inflow_m3_s: float
    evaporation_m3_s: float
    outlet: CellId | None


@dataclass(frozen=True)
class LakeNetwork:
    """Solved lake set: per-cell water-body classification, for every cell.

    Mirrors ``core.hydrology.rivers.RiverNetwork``'s shape — mappings
    keyed by cell id and covering every grid cell, with a neutral default
    (``is_lake=False``, ``level_m=0.0``, ``lake_id=None``) off any lake —
    so biology (aquatic habitat suitability) and climate (an evaporation
    source distinct from the ocean) can consume it the same way they
    already consume ``SeaMask``.
    """

    is_lake: Mapping[CellId, bool]
    level_m: Mapping[CellId, float]
    lake_id: Mapping[CellId, int | None]
    lakes: tuple[Lake, ...]


def _solve_basin(  # noqa: PLR0913 - one param per physical quantity being balanced
    grid: Grid,
    heights_m: Mapping[CellId, float],
    basin: frozenset[CellId],
    spill_level_m: float,
    inflow_m3_s: float,
    evaporation_rate_m_s: float,
) -> tuple[frozenset[CellId], float, float, CellId | None] | None:
    """Solve one basin's water balance; return ``None`` if it stays dry.

    Returns ``(wet_cells, level_m, evaporation_m3_s, outlet)``. The lake
    fills to the basin's pour point (``spill_level_m``) and gains an
    ``outlet`` when inflow still exceeds evaporation there (net-positive
    water balance); otherwise it settles, dry-free, at whatever lower
    level makes evaporation equal inflow (a terminal lake with no
    outlet); it stays entirely dry when even the deepest single cell's
    puddle would evaporate faster than the catchment can refill it.
    """
    floor_m = min(heights_m[cell] for cell in basin)
    if inflow_m3_s <= evaporation_rate_m_s * _area_at_level(grid, heights_m, basin, floor_m):
        return None

    evap_at_spill = evaporation_rate_m_s * _area_at_level(grid, heights_m, basin, spill_level_m)
    if inflow_m3_s >= evap_at_spill:
        wet_cells = frozenset(cell for cell in basin if heights_m[cell] <= spill_level_m)
        outlet = _basin_outlet(grid, heights_m, basin)
        return wet_cells, spill_level_m, evap_at_spill, outlet

    low, high = floor_m, spill_level_m
    for _ in range(_LAKE_LEVEL_BISECTION_STEPS):
        mid = (low + high) / 2.0
        evap_mid = evaporation_rate_m_s * _area_at_level(grid, heights_m, basin, mid)
        if evap_mid > inflow_m3_s:
            high = mid
        else:
            low = mid
    level_m = low
    wet_cells = frozenset(cell for cell in basin if heights_m[cell] <= level_m)
    evaporation_m3_s = evaporation_rate_m_s * _area_at_level(grid, heights_m, basin, level_m)
    return wet_cells, level_m, evaporation_m3_s, None


def build_lake_network(  # noqa: PLR0913 - matches build_river_network's injected-graph + tunable-params shape
    grid: Grid,
    heights_m: Mapping[CellId, float],
    sea_mask: SeaMask,
    receivers: Sequence[int],
    cells: Sequence[CellId],
    *,
    precipitation_mm_yr: Mapping[CellId, float] | None = None,
    evaporation_mm_yr: Mapping[CellId, float] | float = _DEFAULT_EVAPORATION_MM_YR,
    runoff_coefficient: float = 0.3,
) -> LakeNetwork:
    """Fill depressions and decide which ones hold a lake.

    ``cells``/``receivers`` are the steepest-descent output of
    ``core.hydrology.rivers.steepest_descent_receivers`` for the same
    ``heights_m``/``sea_mask`` — reused here (rather than recomputed) so
    catchments are defined by the identical flow-direction graph rivers
    use. For every basin found by :func:`fill_depressions`, the
    catchment's runoff (precipitation * ``runoff_coefficient``, summed
    over cells that drain into the basin) is weighed against evaporation
    from the lake surface at increasing levels (see :func:`_solve_basin`)
    to decide whether it forms a lake, and if so whether it overflows
    (gets an outlet) or sits at a lower endorheic equilibrium.

    Raises:
        ValueError: if ``runoff_coefficient`` is not strictly positive.
    """
    if runoff_coefficient <= 0.0:
        msg = f"runoff_coefficient must be > 0, got {runoff_coefficient}"
        raise ValueError(msg)

    outlets = river_outlets(cells, receivers)
    fill_level = fill_depressions(grid, heights_m, sea_mask)

    if isinstance(evaporation_mm_yr, int | float):
        evap_field, evap_default = None, float(evaporation_mm_yr)
    else:
        evap_field, evap_default = evaporation_mm_yr, _DEFAULT_EVAPORATION_MM_YR

    is_lake = dict.fromkeys(heights_m, False)
    level_m: dict[CellId, float] = dict.fromkeys(heights_m, 0.0)
    lake_id: dict[CellId, int | None] = dict.fromkeys(heights_m, None)
    lakes: list[Lake] = []

    for basin, spill_level_m in _basins(grid, heights_m, fill_level):
        catchment = _catchment_of(basin, outlets)
        inflow_m3_s = sum(
            grid.area_m2(cell)
            * _runoff_rate_m_s(
                cell, precipitation_mm_yr, _DEFAULT_PRECIPITATION_MM_YR, runoff_coefficient
            )
            for cell in catchment
        )
        # Evaporation is a water-loss rate already (not scaled by a runoff
        # coefficient), so every cell just contributes its own rate; using
        # the basin's lowest cell is representative since evaporation_mm_yr
        # is typically near-uniform over the small footprint of one lake.
        evaporation_rate_m_s = _runoff_rate_m_s(
            min(basin, key=lambda cell: heights_m[cell]), evap_field, evap_default, 1.0
        )

        solved = _solve_basin(
            grid, heights_m, basin, spill_level_m, inflow_m3_s, evaporation_rate_m_s
        )
        if solved is None:
            continue
        wet_cells, level, evaporation_m3_s, outlet = solved
        this_id = len(lakes)
        lakes.append(
            Lake(
                cells=wet_cells,
                level_m=level,
                spill_level_m=spill_level_m,
                inflow_m3_s=inflow_m3_s,
                evaporation_m3_s=evaporation_m3_s,
                outlet=outlet,
            )
        )
        for cell in wet_cells:
            is_lake[cell] = True
            level_m[cell] = level
            lake_id[cell] = this_id

    return LakeNetwork(is_lake=is_lake, level_m=level_m, lake_id=lake_id, lakes=tuple(lakes))


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
