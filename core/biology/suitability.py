"""Per-cell habitat suitability: the medium gate times the graded bands.

Suitability is a **graded** 0..1 field per organism.  The habitat medium
is a categorical gate applied first (a land animal scores 0 over water);
then each environmental trait contributes its graded comfort/tolerance
response, combined by Liebig's law of the minimum — the single most
limiting factor governs, so one hostile axis is enough to exclude a cell
even if the others are ideal.  A land-cover (biome) preference scales the
result, so deer favour woods and grazers favour plains without a hard
gate.  Subterranean organisms are scored over the buffered underground
instead of the surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.biology.medium import medium_allows

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from core.biology.organism import Organism
    from core.hydrology.sea_mask import SeaMask

# Axis ids the base game feeds into the trait bands; a mod may add more,
# and any axis a species declares but that a cell lacks is simply skipped.
TEMPERATURE_AXIS = "temperature_k"
PRECIPITATION_AXIS = "precipitation_mm_yr"
ALTITUDE_AXIS = "altitude_m"
OXYGEN_AXIS = "oxygen_kpa"
"""Partial pressure of O2 in kilopascals (sea level ~21 kPa; thins with altitude).

A species only feels this axis if it declares an ``[traits.oxygen_kpa]``
band; one that does not is unaffected, exactly like any other axis a cell
happens to lack (see :func:`env_response`).
"""
COLDEST_SEASON_AXIS = "coldest_month_temperature_k"
"""The coldest of a cell's seasonal mean temperatures (a hard-winter signal).

Distinct from :data:`TEMPERATURE_AXIS` (the annual mean): two cells can
share an annual mean while one has a much colder winter, and a
cold-intolerant species that declares this axis is excluded from that one
even though the mean looks comfortable.
"""
TEMPERATURE_RANGE_AXIS = "temperature_range_k"
"""A cell's seasonal temperature swing (warmest season minus coldest).

Optional continental-climate signal alongside :data:`COLDEST_SEASON_AXIS`;
most species will not declare a band for it.
"""

_DEFAULT_BASE_BIOME_PREFERENCE = 0.5
"""Preference an organism gives a biome it does not explicitly list."""


@dataclass(frozen=True)
class SurfaceEnvironment:
    """The surface fields suitability is scored against, bundled once.

    ``axis_fields`` maps each environmental axis to its per-cell values;
    ``biome_field`` gives each land cell's biome id; ``sea_mask`` drives the
    medium gate; ``base_preference`` is the weight for an unlisted biome.
    """

    axis_fields: Mapping[str, Mapping[str, float]]
    biome_field: Mapping[str, str]
    sea_mask: SeaMask
    base_preference: float = _DEFAULT_BASE_BIOME_PREFERENCE


def env_response(organism: Organism, axis_values: Mapping[str, float]) -> float:
    """Return the graded environmental response (Liebig minimum over bands)."""
    responses = [
        band.response(axis_values[axis])
        for axis, band in organism.traits.items()
        if axis in axis_values
    ]
    if not responses:
        return 1.0
    return min(responses)


def biome_factor(
    organism: Organism,
    biome_id: str | None,
    base_preference: float,
) -> float:
    """Return the land-cover preference multiplier for a cell's biome."""
    if not organism.biome_preference or biome_id is None:
        return 1.0
    return organism.biome_preference.get(biome_id, base_preference)


def surface_suitability(
    organism: Organism,
    cells: Iterable[str],
    environment: SurfaceEnvironment,
) -> dict[str, float]:
    """Return the suitability of every allowed surface cell for an organism.

    Cells the organism's medium forbids are omitted (suitability 0); the
    caller treats an absent cell as uninhabitable.
    """
    result: dict[str, float] = {}
    for cell in cells:
        if not medium_allows(organism.medium, cell, environment.sea_mask):
            continue
        values = {
            axis: field[cell] for axis, field in environment.axis_fields.items() if cell in field
        }
        score = env_response(organism, values) * biome_factor(
            organism, environment.biome_field.get(cell), environment.base_preference
        )
        if score > 0.0:
            result[cell] = score
    return result


def subsurface_suitability(
    organism: Organism,
    nodes: Iterable[str],
    temperature_by_node: Mapping[str, float],
    diggability_by_node: Mapping[str, float],
) -> dict[str, float]:
    """Return suitability over the volume graph for a subterranean organism.

    Underground temperature is buffered (the surface annual mean), and how
    loose the rock is (``diggability``) scales how habitable a node is, so
    dwarves and cave flora favour workable ground.
    """
    result: dict[str, float] = {}
    for node in nodes:
        values = (
            {TEMPERATURE_AXIS: temperature_by_node[node]} if node in temperature_by_node else {}
        )
        score = env_response(organism, values) * diggability_by_node.get(node, 1.0)
        if score > 0.0:
            result[node] = score
    return result
