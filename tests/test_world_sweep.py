"""Pure batch-sweep engine and the habitability scorer.

The engine is exercised with fake evaluate/deepen callables (no real
worlds), and the scorer with hand-built synthetic worlds — both fast and
deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from core.hydrology.sea_mask import SeaMask
from core.sim.world_sweep import SweepReport, habitability_score, run_sweep
from tests.civ_helpers import FakeGrid


def _evaluate(scores: Mapping[int, float]):
    """Return an evaluate() that reads a fixed score per seed."""

    def evaluate(seed: int) -> tuple[float, dict[str, float]]:
        return scores[seed], {"score": scores[seed]}

    return evaluate


def test_run_sweep_ranks_by_score_and_keeps_top_k() -> None:
    scores = {1: 0.2, 2: 0.9, 3: 0.5, 4: 0.7}
    report = run_sweep([1, 2, 3, 4], _evaluate(scores), keep_top_k=2)
    assert isinstance(report, SweepReport)
    assert [c.seed for c in report.ranked] == [2, 4, 3, 1]
    assert [c.seed for c in report.selected] == [2, 4]
    assert report.deepened == {}


def test_run_sweep_breaks_ties_by_seed_deterministically() -> None:
    scores = {5: 0.5, 3: 0.5, 8: 0.5}
    report = run_sweep([8, 5, 3], _evaluate(scores), keep_top_k=3)
    assert [c.seed for c in report.ranked] == [3, 5, 8]


def test_run_sweep_deepens_only_the_winners() -> None:
    scores = {1: 0.1, 2: 0.9, 3: 0.5}
    deepened_seeds = []

    def deepen(seed: int) -> str:
        deepened_seeds.append(seed)
        return f"deep-{seed}"

    report = run_sweep([1, 2, 3], _evaluate(scores), keep_top_k=1, deepen=deepen)
    assert deepened_seeds == [2]  # only the top world was deep-simmed
    assert report.deepened == {2: "deep-2"}


def test_run_sweep_validates_inputs() -> None:
    with pytest.raises(ValueError, match="seeds"):
        run_sweep([], _evaluate({}), keep_top_k=1)
    with pytest.raises(ValueError, match="keep_top_k"):
        run_sweep([1], _evaluate({1: 0.5}), keep_top_k=0)


def test_keep_top_k_larger_than_the_field_is_clamped() -> None:
    report = run_sweep([1, 2], _evaluate({1: 0.5, 2: 0.6}), keep_top_k=10)
    assert [c.seed for c in report.selected] == [2, 1]


def _mask(ocean: dict[str, bool]) -> SeaMask:
    return SeaMask(sea_level_m=0.0, ocean=ocean, intertidal=dict.fromkeys(ocean, False))


def test_habitability_prefers_a_diverse_comfortable_watery_world() -> None:
    grid = FakeGrid(rows=2, cols=2)
    cells = list(grid.cells())
    # Good world: half ocean, comfortable land, two distinct biomes.
    good_ocean = {cells[0]: True, cells[1]: True, cells[2]: False, cells[3]: False}
    good = habitability_score(
        grid,
        biome_field={cells[2]: "temperate_forest", cells[3]: "savanna"},
        annual_mean_temperature_k=dict.fromkeys(cells, 290.0),
        sea_mask=_mask(good_ocean),
    )[0]
    # Bad world: all land, frozen, single biome.
    bad = habitability_score(
        grid,
        biome_field=dict.fromkeys(cells, "polar_desert"),
        annual_mean_temperature_k=dict.fromkeys(cells, 220.0),
        sea_mask=_mask(dict.fromkeys(cells, False)),
    )[0]
    assert 0.0 <= bad < good <= 1.0


def test_habitability_breakdown_reports_components() -> None:
    grid = FakeGrid(rows=1, cols=2)
    cells = list(grid.cells())
    ocean = {cells[0]: True, cells[1]: False}
    score, details = habitability_score(
        grid,
        biome_field={cells[1]: "savanna"},
        annual_mean_temperature_k=dict.fromkeys(cells, 295.0),
        sea_mask=_mask(ocean),
    )
    assert set(details) >= {"comfort", "diversity", "water", "ocean_fraction", "score"}
    assert details["score"] == pytest.approx(score)
    assert details["comfort"] == pytest.approx(1.0)  # the one land cell is comfortable
    assert details["water"] == pytest.approx(1.0)  # exactly half ocean
