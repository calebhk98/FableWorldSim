"""Batch world sweep: generate many worlds, score them, deepen the best.

This is the "make three dozen quick worlds while you sleep, keep the few
you like, then deep-sim those" workflow. This module is the pure engine:
it sequences generate -> score -> rank -> select -> deepen over *injected*
callables, so the identical logic drives a real run (concrete builders in
the app layer) and a fast, deterministic unit test (fakes). It also ships
:func:`habitability_score`, a content-agnostic 0..1 metric over a finished
climate+biome world, used as the default fast score.

Determinism: ranking breaks ties by seed, so the same seeds + evaluator
always produce the same ranking and the same selected winners.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.grid.area_weighted import area_fraction

if TYPE_CHECKING:
    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid

# Default comfortable surface-temperature band (K): liquid water up to a
# warm-but-survivable ceiling. Widened/narrowed by callers per taste.
_COMFORTABLE_MIN_K = 273.15
_COMFORTABLE_MAX_K = 305.0

# Blend weights for the default habitability score (sum to 1).
_W_COMFORT = 0.4
_W_DIVERSITY = 0.3
_W_WATER = 0.3

# Evenness is undefined below this many distinct land biomes.
_MIN_BIOMES_FOR_DIVERSITY = 2


Evaluate = Callable[[int], "tuple[float, Mapping[str, float]]"]
"""Seed -> (score, named component breakdown). Higher score = more desirable."""

Deepen = Callable[[int], object]
"""Seed -> deep-run result (any JSON-able summary); called only on winners."""


@dataclass(frozen=True)
class SweepCandidate:
    """One evaluated world: its seed, fast score, and score breakdown."""

    seed: int
    score: float
    details: Mapping[str, float]


@dataclass(frozen=True)
class SweepReport:
    """The outcome of a sweep: every world ranked, the winners, deep results."""

    ranked: tuple[SweepCandidate, ...]
    """All candidates, best score first (ties broken by seed)."""
    selected: tuple[SweepCandidate, ...]
    """The kept top-K candidates."""
    deepened: Mapping[int, object]
    """Winner seed -> deep-run result (empty when no ``deepen`` was given)."""


def run_sweep(
    seeds: Sequence[int],
    evaluate: Evaluate,
    keep_top_k: int,
    deepen: Deepen | None = None,
) -> SweepReport:
    """Evaluate every seed, keep the top ``keep_top_k``, and deepen the winners.

    ``evaluate`` runs the fast preview per seed; ``deepen`` (optional) runs
    the expensive simulation on the winners only. Both are injected so this
    engine stays pure and backend-agnostic.

    Raises:
        ValueError: if ``seeds`` is empty or ``keep_top_k`` is below 1.
    """
    if not seeds:
        msg = "seeds must be non-empty"
        raise ValueError(msg)
    if keep_top_k < 1:
        msg = f"keep_top_k must be >= 1, got {keep_top_k}"
        raise ValueError(msg)
    candidates = []
    for seed in seeds:
        score, details = evaluate(seed)
        candidates.append(SweepCandidate(seed=seed, score=score, details=dict(details)))
    ranked = tuple(sorted(candidates, key=lambda c: (-c.score, c.seed)))
    selected = ranked[: min(keep_top_k, len(ranked))]
    deepened: dict[int, object] = {}
    if deepen is not None:
        for candidate in selected:
            deepened[candidate.seed] = deepen(candidate.seed)
    return SweepReport(ranked=ranked, selected=selected, deepened=deepened)


def _biome_evenness(biome_field: Mapping[CellId, str], areas: Mapping[CellId, float]) -> float:
    """Return Shannon evenness (0..1) of land-biome area shares.

    1.0 = many biomes in equal proportion; 0.0 = a single biome (or none).
    """
    by_biome: dict[str, float] = defaultdict(float)
    for cell, biome in biome_field.items():
        by_biome[biome] += areas.get(cell, 0.0)
    total = sum(by_biome.values())
    if total <= 0.0 or len(by_biome) < _MIN_BIOMES_FOR_DIVERSITY:
        return 0.0
    entropy = -sum((a / total) * math.log(a / total) for a in by_biome.values() if a > 0.0)
    return entropy / math.log(len(by_biome))


def habitability_score(
    grid: Grid,
    biome_field: Mapping[CellId, str],
    annual_mean_temperature_k: Mapping[CellId, float],
    sea_mask: SeaMask,
    *,
    comfortable_k: tuple[float, float] = (_COMFORTABLE_MIN_K, _COMFORTABLE_MAX_K),
) -> tuple[float, dict[str, float]]:
    """Return a 0..1 habitability score plus its component breakdown.

    Blends three content-agnostic signals over the finished world:
    ``comfort`` (area fraction of land in the comfortable temperature band),
    ``diversity`` (Shannon evenness of land-biome area shares), and
    ``water`` (a land/ocean balance term, ``4*f*(1-f)``, peaking at a
    half-ocean world and zero for all-land or all-ocean). The returned dict
    always includes those three components and the overall ``score``.
    """
    cells = list(grid.cells())
    areas = {cell: grid.area_m2(cell) for cell in cells}
    ocean_fraction = area_fraction(
        [sea_mask.ocean[cell] for cell in cells], [areas[cell] for cell in cells]
    )

    land_cells = [cell for cell in cells if sea_mask.is_land(cell)]
    lo, hi = comfortable_k
    if land_cells:
        comfort = area_fraction(
            [lo <= annual_mean_temperature_k[cell] <= hi for cell in land_cells],
            [areas[cell] for cell in land_cells],
        )
    else:
        comfort = 0.0
    diversity = _biome_evenness(biome_field, areas)
    water = 4.0 * ocean_fraction * (1.0 - ocean_fraction)

    score = _W_COMFORT * comfort + _W_DIVERSITY * diversity + _W_WATER * water
    breakdown = {
        "comfort": comfort,
        "diversity": diversity,
        "water": water,
        "ocean_fraction": ocean_fraction,
        "score": score,
    }
    return score, breakdown
