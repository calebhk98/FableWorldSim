"""Density-dependent disease: threshold outbreaks, die-offs, and the toggle."""

from __future__ import annotations

from adapters.rng_seeded import SeededRng
from core.biology.context import BiologyParams
from core.biology.disease import DiseaseParams, Syndrome, apply_disease
from core.biology.engine import step_biology
from core.biology.state import WorldBiologyState
from core.chronicle.log import Chronicle
from core.sim.constants import SECONDS_PER_YEAR
from tests.bio_helpers import make_organism, uniform_context
from tests.civ_helpers import FakeGrid

_YEAR = SECONDS_PER_YEAR


def _total(field: dict[str, float]) -> float:
    return sum(field.values())


# --- unit tests on the disease module directly ---------------------------


def test_dense_population_suffers_a_die_off_an_identical_sparse_one_does_not() -> None:
    syndrome = Syndrome(
        transmission_rate_per_year=5.0, reference_density_per_m2=1.0, mortality_fraction=0.5
    )
    params = DiseaseParams(syndrome=syndrome)
    dense_after, dense_cells = apply_disease("critter", {"c": 5.0}, params, 1.0, SeededRng(7))
    sparse_after, sparse_cells = apply_disease("critter", {"c": 0.5}, params, 1.0, SeededRng(7))
    assert dense_cells == ["c"]
    assert dense_after["c"] == 2.5
    assert sparse_cells == []
    assert sparse_after == {"c": 0.5}


def test_disease_never_fires_at_or_below_the_threshold_density_regardless_of_rng() -> None:
    syndrome = Syndrome(transmission_rate_per_year=1000.0, reference_density_per_m2=2.0)
    params = DiseaseParams(syndrome=syndrome)
    field = {"a": 2.0, "b": 1.0}
    for seed in range(10):
        _, cells = apply_disease("x", field, params, 1.0, SeededRng(seed))
        assert cells == []


def test_disease_toggle_disables_the_effect() -> None:
    syndrome = Syndrome(transmission_rate_per_year=1000.0, reference_density_per_m2=0.1)
    params = DiseaseParams(enabled=False, syndrome=syndrome)
    field = {"c": 5.0}
    after, cells = apply_disease("critter", field, params, 1.0, SeededRng(1))
    assert cells == []
    assert after == field


def test_immune_species_never_suffers_an_outbreak() -> None:
    syndrome = Syndrome(
        transmission_rate_per_year=1000.0,
        reference_density_per_m2=0.1,
        immune_species=frozenset({"critter"}),
    )
    params = DiseaseParams(syndrome=syndrome)
    field = {"c": 5.0}
    after, cells = apply_disease("critter", field, params, 1.0, SeededRng(1))
    assert cells == []
    assert after == field


def test_disease_is_deterministic_under_a_fixed_seed() -> None:
    syndrome = Syndrome(
        transmission_rate_per_year=0.3, reference_density_per_m2=1.0, mortality_fraction=0.4
    )
    params = DiseaseParams(syndrome=syndrome)
    field = {"a": 1.5, "b": 1.2, "c": 3.0, "d": 1.05}
    first = apply_disease("critter", field, params, 1.0, SeededRng(42))
    second = apply_disease("critter", field, params, 1.0, SeededRng(42))
    assert first == second


# --- integration through the full biology step ---------------------------


def _critter_context(grid: FakeGrid, params: BiologyParams) -> tuple[object, object]:
    """Build a species with plenty of headroom under its crowding cap."""
    organism = make_organism(
        "critter", crowding_cap_per_m2=20.0, reproduction="asexual", min_viable_population=1
    )
    ctx = uniform_context(grid, [organism], params=params)  # type: ignore[list-item]
    return organism, ctx


def test_engine_dense_population_die_off_sparse_unaffected() -> None:
    grid = FakeGrid(rows=2, cols=2)
    syndrome = Syndrome(
        transmission_rate_per_year=50.0, reference_density_per_m2=1.0, mortality_fraction=0.5
    )
    params = BiologyParams(disease=DiseaseParams(syndrome=syndrome))
    _, ctx = _critter_context(grid, params)
    land = list(grid.cells())
    dense = WorldBiologyState(
        surface_populations={"critter": dict.fromkeys(land, 10.0)}, subsurface_populations={}
    )
    sparse = WorldBiologyState(
        surface_populations={"critter": dict.fromkeys(land, 0.05)}, subsurface_populations={}
    )
    dense_after = step_biology(dense, ctx, SeededRng(3), _YEAR)
    sparse_after = step_biology(sparse, ctx, SeededRng(3), _YEAR)
    # The dense herd's growth is more than wiped out by the outbreak...
    assert _total(dense_after.surface_populations["critter"]) < _total(
        dense.surface_populations["critter"]
    )
    # ...while the sparse one, staying under the threshold density, simply grows.
    assert _total(sparse_after.surface_populations["critter"]) > _total(
        sparse.surface_populations["critter"]
    )


def test_engine_toggling_disease_off_disables_the_effect() -> None:
    grid = FakeGrid(rows=2, cols=2)
    syndrome = Syndrome(
        transmission_rate_per_year=50.0, reference_density_per_m2=1.0, mortality_fraction=0.5
    )
    on_params = BiologyParams(disease=DiseaseParams(syndrome=syndrome))
    off_params = BiologyParams(disease=DiseaseParams(enabled=False, syndrome=syndrome))
    land = list(grid.cells())
    state = WorldBiologyState(
        surface_populations={"critter": dict.fromkeys(land, 10.0)}, subsurface_populations={}
    )
    _, ctx_on = _critter_context(grid, on_params)
    _, ctx_off = _critter_context(grid, off_params)
    with_disease = step_biology(state, ctx_on, SeededRng(3), _YEAR)
    without_disease = step_biology(state, ctx_off, SeededRng(3), _YEAR)
    assert _total(with_disease.surface_populations["critter"]) < _total(
        without_disease.surface_populations["critter"]
    )


def test_engine_disease_die_off_is_chronicled() -> None:
    grid = FakeGrid(rows=2, cols=2)
    syndrome = Syndrome(
        transmission_rate_per_year=50.0, reference_density_per_m2=1.0, mortality_fraction=0.5
    )
    params = BiologyParams(disease=DiseaseParams(syndrome=syndrome))
    _, ctx = _critter_context(grid, params)
    land = list(grid.cells())
    state = WorldBiologyState(
        surface_populations={"critter": dict.fromkeys(land, 10.0)}, subsurface_populations={}
    )
    chronicle = Chronicle()
    step_biology(state, ctx, SeededRng(3), _YEAR, chronicle)
    disease_events = [event for event in chronicle.events() if event.kind == "disease"]
    assert disease_events
    assert disease_events[0].subject == "critter"
    assert disease_events[0].payload["cells"] > 0


def test_engine_same_seed_replays_identically() -> None:
    grid = FakeGrid(rows=2, cols=2)
    syndrome = Syndrome(
        transmission_rate_per_year=0.4, reference_density_per_m2=1.0, mortality_fraction=0.4
    )
    params = BiologyParams(disease=DiseaseParams(syndrome=syndrome))
    _, ctx = _critter_context(grid, params)
    land = list(grid.cells())
    state = WorldBiologyState(
        surface_populations={"critter": dict.fromkeys(land, 3.0)}, subsurface_populations={}
    )
    first = step_biology(state, ctx, SeededRng(11), _YEAR)
    second = step_biology(state, ctx, SeededRng(11), _YEAR)
    assert first.surface_populations == second.surface_populations
