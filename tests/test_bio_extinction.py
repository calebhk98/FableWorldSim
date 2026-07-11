"""Extinction and minimum-viable-population viability gates."""

from __future__ import annotations

from adapters.rng_seeded import SeededRng
from core.biology.context import BiologyParams
from core.biology.domain import DomainRunner
from core.biology.extinction import enforce_viability, total_individuals
from core.chronicle.log import Chronicle
from tests.bio_helpers import make_organism


def _unit_area(_node: str) -> float:
    return 1.0


def _no_neighbors(_node: str) -> tuple[str, ...]:
    return ()


def _runner(organism: object, chronicle: Chronicle, tick: int = 3) -> DomainRunner:
    return DomainRunner(
        organisms={organism.species_id: organism},  # type: ignore[attr-defined]
        neighbors_of=_no_neighbors,
        area_of=_unit_area,
        params=BiologyParams(),
        rng=SeededRng(0),
        dt_years=1.0,
        tick=tick,
        domain="surface",
        chronicle=chronicle,
    )


def test_total_individuals_is_area_weighted() -> None:
    assert total_individuals({"a": 2.0, "b": 3.0}, _unit_area) == 5.0


def test_sexual_species_below_mvp_goes_extinct() -> None:
    deer = make_organism("deer", reproduction="sexual", min_viable_population=2)
    culled, alive = enforce_viability({"c": 1.5}, deer, _unit_area, 1e-15)
    assert not alive
    assert all(value == 0.0 for value in culled.values())


def test_asexual_species_rebounds_from_a_single_survivor() -> None:
    grass = make_organism("grass", reproduction="asexual", min_viable_population=1)
    culled, alive = enforce_viability({"c": 0.5}, grass, _unit_area, 1e-15)
    assert alive
    assert culled["c"] == 0.5


def test_numerical_dust_is_culled_to_zero() -> None:
    organism = make_organism("x", reproduction="asexual", min_viable_population=1)
    culled, alive = enforce_viability({"c": 1e-20}, organism, _unit_area, 1e-15)
    assert culled["c"] == 0.0
    assert not alive


def test_a_cell_a_species_vanishes_from_but_survives_elsewhere_logs_a_local_event() -> None:
    deer = make_organism("deer", reproduction="sexual", min_viable_population=1)
    chronicle = Chronicle()
    runner = _runner(deer, chronicle)
    previous = {"deer": {"a": 1.5, "b": 1.5}}
    pops = {"deer": {"a": 0.0, "b": 1.5}}
    runner.finalize(["deer"], pops, previous)
    events = list(chronicle.events())
    local = [e for e in events if e.kind == "local_extinction"]
    assert len(local) == 1
    assert local[0].subject == "deer"
    assert local[0].payload["cell"] == "a"
    assert not any(e.kind == "extinction" for e in events)


def test_a_whole_domain_extinction_logs_only_the_global_event_not_per_cell() -> None:
    doomed = make_organism("doomed", reproduction="sexual", min_viable_population=10**9)
    chronicle = Chronicle()
    runner = _runner(doomed, chronicle)
    previous = {"doomed": {"a": 1.5, "b": 1.5}}
    pops = {"doomed": {"a": 1.4, "b": 1.4}}
    runner.finalize(["doomed"], pops, previous)
    events = list(chronicle.events())
    assert any(e.kind == "extinction" and e.subject == "doomed" for e in events)
    assert not any(e.kind == "local_extinction" for e in events)
