"""Emergent disease: density-dependent outbreaks that cull dense populations.

Disease is the biology layer's other emergent hazard (see
:mod:`core.biology.wildfire` for the sibling mechanic): instead of a fixed
per-cell probability, the chance of an outbreak grows with local
**density** — a crowded herd or a packed settlement spreads a pathogen far
faster than a sparse one, so disease is a natural brake on unchecked
crowding exactly where wildfire is a brake on unchecked fuel.

M1 deliberately collapses the classic SIR compartments (susceptible /
infected / recovered) to a single stateless **prevalence-and-cull** step
each tick: a cell's local density sets an outbreak probability, and an
outbreak that fires removes a mortality fraction of the population there,
same tick — no persistent "currently infected" field to thread through
:class:`~core.biology.state.WorldBiologyState`.  This is intentionally the
minimal instance of a much richer space (see :class:`Syndrome`): a full
roadmap SIR model would carry infected/recovered fractions forward and let
them decay/recover between outbreaks, and a "syndrome catalog" would add
vectors (a disease that spreads *through* another species) and per-species
immunity.  Both are represented as inert, roadmap fields on
:class:`Syndrome` today, read by the die-off check below but not yet
populated by any content — so richer disease content is a data drop later,
not a retrofit of this module's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ports.rng import Rng

_Field = dict[str, float]


@dataclass(frozen=True)
class Syndrome:
    """One disease process's epidemiological knobs (a "syndrome" in DF parlance).

    ``reference_density_per_m2`` is a **threshold density**: the classic
    epidemiological result that a pathogen needs a large enough host
    density to sustain transmission at all (the "critical community
    size").  At or below the threshold the outbreak probability is exactly
    zero; above it, ``transmission_rate_per_year`` scales the hazard by how
    far density sits past the threshold, so a cell at twice the excess
    density is twice as likely to see an outbreak this step — density-
    dependent transmission with a built-in sparse-population immunity, not
    just noisier stochastics.  A firing outbreak removes
    ``mortality_fraction`` of that cell's population, same step, mirroring
    wildfire's burn fraction.

    ``recovery_rate_per_year``, ``immune_species``, and ``vector_species``
    are the roadmap extension points: a future SIR-style syndrome catalog
    would use ``recovery_rate_per_year`` to move survivors from "infected"
    to "recovered/immune" between outbreaks, and ``vector_species`` to
    route transmission through a carrier species instead of same-species
    density. ``immune_species`` is already load-bearing in M1 — any species
    id it lists is fully exempt from this syndrome, so a species can be
    hand-authored disease-proof without touching the outbreak math.
    """

    syndrome_id: str = "generic-crowd-disease"
    transmission_rate_per_year: float = 0.8
    reference_density_per_m2: float = 1.0
    mortality_fraction: float = 0.3
    recovery_rate_per_year: float = 2.0
    immune_species: frozenset[str] = field(default_factory=frozenset)
    vector_species: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject malformed rates at construction time."""
        if self.transmission_rate_per_year < 0.0:
            msg = "syndrome transmission_rate_per_year must be >= 0"
            raise ValueError(msg)
        if self.reference_density_per_m2 <= 0.0:
            msg = "syndrome reference_density_per_m2 must be > 0"
            raise ValueError(msg)
        if not 0.0 <= self.mortality_fraction <= 1.0:
            msg = "syndrome mortality_fraction must be in [0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True)
class DiseaseParams:
    """Engine-level toggle and default syndrome for the disease pass.

    ``enabled`` is the config/mod toggle the plan calls for: a scenario or
    mod can set it ``False`` to switch disease off wholesale (or tune
    ``syndrome`` down to a near-zero transmission rate for the same
    effect) without touching any code.
    """

    enabled: bool = True
    syndrome: Syndrome = field(default_factory=Syndrome)


def outbreak_cells(
    species_id: str,
    field_: Mapping[str, float],
    syndrome: Syndrome,
    dt_years: float,
    rng: Rng,
) -> list[str]:
    """Return the cells where an outbreak fires this step.

    Density-dependent with a hard threshold: a cell at or below the
    syndrome's reference (threshold) density never ignites; above it, the
    outbreak probability grows with the excess density, so denser
    populations are reliably more outbreak-prone rather than merely
    stochastically noisier.  An immune species never rolls an outbreak.
    """
    if species_id in syndrome.immune_species:
        return []
    ignited: list[str] = []
    for cell, density in field_.items():
        if density <= syndrome.reference_density_per_m2:
            continue
        excess = (density - syndrome.reference_density_per_m2) / syndrome.reference_density_per_m2
        probability = min(1.0, syndrome.transmission_rate_per_year * excess * dt_years)
        if rng.random() < probability:
            ignited.append(cell)
    return ignited


def cull(field_: Mapping[str, float], cells: Iterable[str], syndrome: Syndrome) -> _Field:
    """Return the field after each ignited cell loses its mortality fraction."""
    culled = dict(field_)
    for cell in cells:
        if cell in culled:
            culled[cell] = culled[cell] * (1.0 - syndrome.mortality_fraction)
    return culled


def apply_disease(
    species_id: str,
    field_: Mapping[str, float],
    params: DiseaseParams,
    dt_years: float,
    rng: Rng,
) -> tuple[_Field, list[str]]:
    """Run one species' disease step; return the culled field and any hit cells.

    A no-op (field unchanged, no cells reported) when disease is toggled
    off or nothing ignites, so callers can log a die-off event only when
    one actually happened.
    """
    if not params.enabled:
        return dict(field_), []
    cells = outbreak_cells(species_id, field_, params.syndrome, dt_years, rng)
    if not cells:
        return dict(field_), []
    return cull(field_, cells, params.syndrome), cells
