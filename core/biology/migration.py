"""Migration: suitability-gradient local diffusion across the grid graph.

Populations drift toward better neighbouring habitat: each cell sends a
small, count-conserving share of its people/biomass to adjacent cells in
proportion to how suitable each neighbour is, with a little stochastic
jitter so spread is not perfectly smooth.  The same routine runs over the
surface grid and, for subterranean species, over the ``SubsurfaceGrid`` —
only the neighbour and area functions differ, so digging life diffuses
through the volume graph rather than surface adjacency.

M1 models gradual local spread; true long-range seasonal migration
(birds jumping between two distant known regions) is roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ports.rng import Rng

_MAX_MOVE_FRACTION = 0.5
"""Cap so a cell never sends away more than half its population in a step."""


@dataclass(frozen=True)
class Geometry:
    """The graph a diffusion runs over: adjacency and per-location area."""

    neighbors_of: Callable[[str], Sequence[str]]
    area_of: Callable[[str], float]


@dataclass(frozen=True)
class DiffusionParams:
    """How strongly and how noisily a population diffuses in one step."""

    move_fraction: float
    jitter: float = 0.0


def diffuse(
    density: Mapping[str, float],
    geometry: Geometry,
    suitability: Mapping[str, float],
    params: DiffusionParams,
    rng: Rng,
) -> dict[str, float]:
    """Return the density field after one suitability-weighted diffusion step.

    Movement conserves head count: densities are converted to counts via
    the per-location area, moved, and converted back, so a coarse and a
    fine cell exchange people correctly.
    """
    fraction = min(_MAX_MOVE_FRACTION, params.move_fraction)
    if fraction <= 0.0:
        return dict(density)
    result = dict(density)
    for loc, value in density.items():
        weights = _neighbor_weights(loc, value, geometry, suitability)
        if weights is None:
            continue
        moved = _jittered_fraction(fraction, params.jitter, rng)
        moving_count = value * geometry.area_of(loc) * moved
        result[loc] -= moving_count / geometry.area_of(loc)
        _deposit(result, weights, moving_count, geometry.area_of)
    return result


def _neighbor_weights(
    loc: str,
    value: float,
    geometry: Geometry,
    suitability: Mapping[str, float],
) -> dict[str, float] | None:
    """Return suitable-neighbour weights, or ``None`` if nothing attracts."""
    if value <= 0.0:
        return None
    weights = {nb: suitability.get(nb, 0.0) for nb in geometry.neighbors_of(loc)}
    if sum(weights.values()) <= 0.0:
        return None
    return weights


def _jittered_fraction(fraction: float, jitter: float, rng: Rng) -> float:
    """Return the move fraction perturbed by symmetric stochastic jitter."""
    moved = fraction * (1.0 + jitter * (rng.random() - 0.5))
    return min(_MAX_MOVE_FRACTION, max(0.0, moved))


def _deposit(
    result: dict[str, float],
    weights: Mapping[str, float],
    moving_count: float,
    area_of: Callable[[str], float],
) -> None:
    """Spread ``moving_count`` head across neighbours in proportion to weight."""
    total = sum(weights.values())
    for nb, weight in weights.items():
        share = moving_count * (weight / total)
        result[nb] = result.get(nb, 0.0) + share / area_of(nb)
