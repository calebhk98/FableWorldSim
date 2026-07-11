"""One ecological domain's per-step update (surface shell or the volume).

Surface life and subterranean life run the *same* pipeline — feed and
compete, remove what was eaten, grow toward carrying capacity, diffuse
toward better habitat, then check viability — differing only in the
neighbour and area functions the runner is built with.  Keeping the
geometry injected means the food web and population math are written once
and reused, so subterranean species genuinely diffuse through the
``SubsurfaceGrid`` rather than surface adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.biology.extinction import enforce_viability
from core.biology.foodweb import FeedingParams, apply_offtake, feed_location
from core.biology.migration import (
    DiffusionParams,
    Geometry,
    SeasonalPullParams,
    diffuse,
    seasonal_pull,
)
from core.biology.population import GrowthParams, env_capacity, grow

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from core.biology.context import BiologyParams
    from core.biology.organism import Organism
    from core.chronicle.log import Chronicle
    from ports.rng import Rng

_Field = dict[str, float]
_Fields = dict[str, _Field]


@dataclass(frozen=True)
class DomainRunner:
    """Runs one domain's populations, bound to that domain's geometry."""

    organisms: Mapping[str, Organism]
    neighbors_of: Callable[[str], Sequence[str]]
    area_of: Callable[[str], float]
    params: BiologyParams
    rng: Rng
    dt_years: float
    tick: int
    domain: str
    chronicle: Chronicle | None = None

    def step(
        self,
        species_ids: Sequence[str],
        pops: Mapping[str, Mapping[str, float]],
        suitability: Mapping[str, Mapping[str, float]],
    ) -> _Fields:
        """Feed, thin by offtake, grow, and diffuse — without the viability gate."""
        fed, offtaken = self._feed(species_ids, pops)
        result: _Fields = {}
        for sid in species_ids:
            grown = self._grow(sid, offtaken[sid], suitability.get(sid, {}), fed.get(sid, {}))
            result[sid] = self._migrate(sid, grown, suitability.get(sid, {}))
        return result

    def finalize(
        self,
        species_ids: Sequence[str],
        pops: Mapping[str, _Field],
        previous: Mapping[str, Mapping[str, float]],
    ) -> _Fields:
        """Cull dust, apply viability gates, and log any fresh extinction."""
        result: _Fields = {}
        for sid in species_ids:
            culled, alive = enforce_viability(
                pops[sid], self.organisms[sid], self.area_of, self.params.extinction_epsilon_per_m2
            )
            result[sid] = culled
            was_alive = any(value > 0.0 for value in previous.get(sid, {}).values())
            if was_alive and not alive and self.chronicle is not None:
                self.chronicle.append(
                    tick=self.tick, kind="extinction", subject=sid, payload={"domain": self.domain}
                )
        return result

    def _feed(
        self,
        species_ids: Sequence[str],
        pops: Mapping[str, Mapping[str, float]],
    ) -> tuple[_Fields, _Fields]:
        """Resolve feeding everywhere; return fed fractions and post-offtake pops."""
        feeding = FeedingParams(
            intake_kg_per_kg_body_year=self.params.intake_kg_per_kg_body_year,
            max_offtake_fraction=self.params.max_offtake_fraction,
            dt_years=self.dt_years,
        )
        org_map = {sid: self.organisms[sid] for sid in species_ids}
        fed: _Fields = {sid: {} for sid in species_ids}
        new_pops: _Fields = {sid: dict(pops.get(sid, {})) for sid in species_ids}
        for loc in _locations(species_ids, pops):
            at_loc = {
                sid: pops[sid][loc] for sid in species_ids if pops.get(sid, {}).get(loc, 0.0) > 0.0
            }
            result = feed_location(at_loc, org_map, feeding)
            for sid, fraction in result.fed_fraction.items():
                fed[sid][loc] = fraction
            for prey_id, removed in result.offtake.items():
                new_pops[prey_id][loc] = apply_offtake(
                    new_pops[prey_id][loc], self.organisms[prey_id], removed
                )
        return fed, new_pops

    def _grow(
        self,
        species_id: str,
        field: Mapping[str, float],
        suitability: Mapping[str, float],
        fed_fraction: Mapping[str, float],
    ) -> _Field:
        """Grow one species' field toward its suitability-scaled capacity."""
        organism = self.organisms[species_id]
        capacity = env_capacity(organism, suitability)
        params = GrowthParams(
            dt_years=self.dt_years,
            no_habitat_decay_per_year=self.params.no_habitat_decay_per_year,
        )
        return grow(field, capacity, organism, fed_fraction, params)

    def _migrate(
        self,
        species_id: str,
        field: Mapping[str, float],
        suitability: Mapping[str, float],
    ) -> _Field:
        """Diffuse toward better neighbouring habitat, then a seasonal pull if flagged.

        Every species gets the local :func:`diffuse` step; a species with
        ``seasonal_migration=True`` additionally gets :func:`seasonal_pull`
        on top, so it relocates several cells toward current good habitat
        in the same tick a non-migratory species only creeps one cell.
        """
        geometry = Geometry(neighbors_of=self.neighbors_of, area_of=self.area_of)
        params = DiffusionParams(
            move_fraction=self.params.migration_per_year * self.dt_years,
            jitter=self.params.migration_jitter,
        )
        moved = diffuse(
            field, geometry, suitability, params, self.rng.fork(f"migrate:{species_id}:{self.tick}")
        )
        if not self.organisms[species_id].seasonal_migration:
            return moved
        pull_params = SeasonalPullParams(
            hops=self.params.seasonal_migration_hops,
            move_fraction=self.params.seasonal_migration_fraction,
            jitter=self.params.migration_jitter,
        )
        return seasonal_pull(
            moved,
            geometry,
            suitability,
            pull_params,
            self.rng.fork(f"seasonal-pull:{species_id}:{self.tick}"),
        )


def _locations(
    species_ids: Sequence[str],
    pops: Mapping[str, Mapping[str, float]],
) -> set[str]:
    """Return every location any of the domain's species currently occupies."""
    found: set[str] = set()
    for sid in species_ids:
        found |= set(pops.get(sid, {}))
    return found
