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

from core.biology.disease import apply_disease
from core.biology.extinction import enforce_viability
from core.biology.foodweb import (
    FeedingParams,
    apply_offtake,
    feed_location,
    reachable_food_biomass,
)
from core.biology.migration import (
    DiffusionParams,
    Geometry,
    SeasonalPullParams,
    diffuse,
    seasonal_pull,
)
from core.biology.population import FoodCapacityParams, GrowthParams, env_capacity, grow

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
        """Feed, thin by offtake, grow, cull by disease, and diffuse (no viability gate)."""
        fed, offtaken = self._feed(species_ids, pops)
        result: _Fields = {}
        disease_hits: dict[str, int] = {}
        for sid in species_ids:
            grown = self._grow(
                sid, offtaken[sid], suitability.get(sid, {}), fed.get(sid, {}), offtaken
            )
            diseased, cells = self._disease(sid, grown)
            if cells:
                disease_hits[sid] = len(cells)
            result[sid] = self._migrate(sid, diseased, suitability.get(sid, {}))
        self._log_disease(disease_hits)
        return result

    def finalize(
        self,
        species_ids: Sequence[str],
        pops: Mapping[str, _Field],
        previous: Mapping[str, Mapping[str, float]],
    ) -> _Fields:
        """Cull dust, apply viability gates, and log extinction (global or local)."""
        result: _Fields = {}
        for sid in species_ids:
            culled, alive = enforce_viability(
                pops[sid], self.organisms[sid], self.area_of, self.params.extinction_epsilon_per_m2
            )
            result[sid] = culled
            if self.chronicle is None:
                continue
            prior_field = previous.get(sid, {})
            was_alive = any(value > 0.0 for value in prior_field.values())
            if was_alive and not alive:
                self.chronicle.append(
                    tick=self.tick, kind="extinction", subject=sid, payload={"domain": self.domain}
                )
            elif alive:
                self._log_local_extinctions(sid, culled, prior_field)
        return result

    def _log_local_extinctions(
        self,
        species_id: str,
        current: Mapping[str, float],
        prior_field: Mapping[str, float],
    ) -> None:
        """Append a local-extinction event per cell a surviving species vanished from.

        Distinct from the domain-wide ``extinction`` event: this only fires
        when the species is still alive elsewhere this tick, so a genuine
        global extinction is reported exactly once (by :meth:`finalize`)
        rather than echoed as a flood of per-cell events for the same
        underlying loss.  Cheap and append-only: one comparison per
        previously-occupied location, one event per zero-crossing.
        """
        if self.chronicle is None:
            return
        for loc, prior_value in prior_field.items():
            if prior_value > 0.0 and current.get(loc, 0.0) <= 0.0:
                self.chronicle.append(
                    tick=self.tick,
                    kind="local_extinction",
                    subject=species_id,
                    payload={"domain": self.domain, "cell": loc},
                )

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
        post_offtake_pops: Mapping[str, Mapping[str, float]],
    ) -> _Field:
        """Grow one species' field toward its food- and suitability-limited capacity.

        ``post_offtake_pops`` is every species' field *after* this tick's
        feeding pass — the current standing food stock a consumer's
        food-limited capacity term is drawn from (see
        :func:`core.biology.foodweb.reachable_food_biomass`).
        """
        organism = self.organisms[species_id]
        food_biomass = reachable_food_biomass(organism, post_offtake_pops, self.organisms)
        food_params = FoodCapacityParams(
            intake_kg_per_kg_body_year=self.params.intake_kg_per_kg_body_year,
            max_offtake_fraction=self.params.max_offtake_fraction,
        )
        capacity = env_capacity(organism, suitability, food_biomass, food_params)
        params = GrowthParams(
            dt_years=self.dt_years,
            no_habitat_decay_per_year=self.params.no_habitat_decay_per_year,
        )
        return grow(field, capacity, organism, fed_fraction, params)

    def _disease(self, species_id: str, field: _Field) -> tuple[_Field, list[str]]:
        """Run the density-dependent disease pass for one species.

        Returns the (possibly culled) field and the cells an outbreak hit,
        so the caller can log a die-off event only when one actually
        happened.
        """
        return apply_disease(
            species_id,
            field,
            self.params.disease,
            self.dt_years,
            self.rng.fork(f"disease:{species_id}:{self.tick}"),
        )

    def _log_disease(self, hits: Mapping[str, int]) -> None:
        """Append one chronicle event per species an outbreak hit this step."""
        if self.chronicle is None:
            return
        for sid, cell_count in hits.items():
            self.chronicle.append(
                tick=self.tick,
                kind="disease",
                subject=sid,
                payload={"domain": self.domain, "cells": cell_count},
            )

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
