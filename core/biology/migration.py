"""Migration: local diffusion, plus a bounded multi-hop seasonal pull.

Populations drift toward better neighbouring habitat: each cell sends a
small, count-conserving share of its people/biomass to adjacent cells in
proportion to how suitable each neighbour is, with a little stochastic
jitter so spread is not perfectly smooth.  The same routine (:func:`diffuse`)
runs over the surface grid and, for subterranean species, over the
``SubsurfaceGrid`` — only the neighbour and area functions differ, so
digging life diffuses through the volume graph rather than surface
adjacency.

On top of that, a species flagged ``seasonal_migration`` (see
:class:`core.biology.organism.Organism`) additionally runs
:func:`seasonal_pull` each step: the *same* suitability-weighted movement
repeated for several hops within one tick, so a migratory population
redistributes several cells toward better current habitat in a single
step instead of creeping one cell at a time.  This is a bounded M1
mechanic — a multi-hop pull toward whatever is suitable *right now* — not
true path-memory migration between two remembered regions (a bird
returning to a specific nesting ground); that two-region, path-aware
model remains roadmap.
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


@dataclass(frozen=True)
class SeasonalPullParams:
    """Tuning for the bounded multi-hop seasonal pull (see module docstring).

    ``hops`` repeats of a :func:`diffuse`-shaped move happen within one
    step, each moving ``move_fraction`` of what remains at a location —
    additive to, and independent of, the local :func:`diffuse` call the
    same step already ran.
    """

    hops: int
    move_fraction: float
    jitter: float = 0.0


def seasonal_pull(
    density: Mapping[str, float],
    geometry: Geometry,
    suitability: Mapping[str, float],
    params: SeasonalPullParams,
    rng: Rng,
) -> dict[str, float]:
    """Return density after a bounded multi-hop seasonal redistribution.

    Applies :func:`diffuse` ``params.hops`` times in a row against the
    *same* suitability field, each hop forked to its own RNG stream so the
    sequence stays deterministic.  Repeating the suitability-weighted move
    lets population reach a neighbour's neighbour (and beyond) within a
    single tick — a bird flying several cells toward better habitat in one
    season, not just drifting into the adjacent cell — while still
    conserving head count at every hop.  A non-migratory species never
    calls this; it only ever gets the single local :func:`diffuse` step.
    """
    if params.hops <= 0 or params.move_fraction <= 0.0:
        return dict(density)
    hop_params = DiffusionParams(move_fraction=params.move_fraction, jitter=params.jitter)
    result = dict(density)
    for hop in range(params.hops):
        result = diffuse(result, geometry, suitability, hop_params, rng.fork(f"hop:{hop}"))
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
