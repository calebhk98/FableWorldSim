"""Organism definitions: every plant and animal as moddable data.

An organism is data, not code — plants, animals, aquatic, aerial,
subterranean, and civilization-founding species all share this one
framework, so adding a race or a creature is a content drop, never a code
change.  Organisms live under the same ``species`` content kind that the
civilization race view (:mod:`core.civilization.species`) reads: the two
are different *projections* of one TOML file — the ecology fields here,
the civ-founding fields there.

Trophic level is **derived**, not declared: an organism with an empty diet
is an **autotroph** (a plant, whose population is standing biomass in
kg/m2); one with diet edges is a **heterotroph** (an animal, whose
population is a count density in individuals/m2).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.biology.bands import EnvBand
from core.biology.medium import MEDIA
from core.content_fields import read_bool, read_float, read_int, read_str, read_weight_table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ports.content import ContentItem, ContentRegistry

SPECIES_KIND = "species"

SEXUAL = "sexual"
ASEXUAL = "asexual"
_REPRODUCTION_MODES = (SEXUAL, ASEXUAL)

_BAND_KEYS = ("comfort_min", "comfort_max", "tolerance_min", "tolerance_max")


@dataclass(frozen=True)
class Organism:
    """One species' ecology: habitat, body plan, life history, and diet.

    ``crowding_cap_per_m2`` is the body-size crowding cap — the maximum
    sustainable density independent of food (individuals/m2 for animals,
    peak standing biomass kg/m2 for plants).  ``traits`` maps an
    environmental axis (see :mod:`core.biology.suitability`) to its
    comfort/tolerance band; ``diet`` maps each eaten species id to a
    preference weight; ``biome_preference`` maps a biome id to a land-cover
    preference (unlisted biomes fall back to a base weight).
    ``seasonal_migration`` opts a species into the bounded multi-hop
    seasonal pull on top of local diffusion (see
    :mod:`core.biology.migration`); it is a flag, not a route — birds
    redistribute toward wherever suitability is currently highest, not
    along a remembered path.
    """

    species_id: str
    medium: str
    body_mass_kg: float
    crowding_cap_per_m2: float
    lifespan_years: float
    maturity_years: float
    litter_size: float
    reproduction: str = SEXUAL
    min_viable_population: int = 2
    oscillatory: bool = False
    sapient: bool = False
    seasonal_migration: bool = False
    traits: Mapping[str, EnvBand] = field(default_factory=dict)
    biome_preference: Mapping[str, float] = field(default_factory=dict)
    diet: Mapping[str, float] = field(default_factory=dict)
    color: str = "#888888"

    def __post_init__(self) -> None:
        """Reject malformed organisms at load time."""
        if self.medium not in MEDIA:
            msg = f"organism {self.species_id!r} has unknown medium {self.medium!r}"
            raise ValueError(msg)
        if self.reproduction not in _REPRODUCTION_MODES:
            msg = f"organism {self.species_id!r} has unknown reproduction {self.reproduction!r}"
            raise ValueError(msg)
        if min(self.body_mass_kg, self.crowding_cap_per_m2, self.litter_size) <= 0.0:
            msg = f"organism {self.species_id!r} body/crowding/litter must be > 0"
            raise ValueError(msg)
        if min(self.lifespan_years, self.maturity_years) <= 0.0:
            msg = f"organism {self.species_id!r} lifespan and maturity must be > 0"
            raise ValueError(msg)
        if self.min_viable_population < 1:
            msg = f"organism {self.species_id!r} min_viable_population must be >= 1"
            raise ValueError(msg)

    @property
    def is_autotroph(self) -> bool:
        """Return whether this organism makes its own food (empty diet)."""
        return not self.diet

    @property
    def name_key(self) -> str:
        """Return the Fluent key for the organism's display name."""
        return f"{SPECIES_KIND}-{self.species_id}"


def _env_band(table: Mapping[str, object]) -> EnvBand:
    """Build one trait's :class:`EnvBand` from its content sub-table."""
    return EnvBand(
        comfort_min=read_float(table, "comfort_min"),
        comfort_max=read_float(table, "comfort_max"),
        tolerance_min=read_float(table, "tolerance_min"),
        tolerance_max=read_float(table, "tolerance_max"),
        spread=read_float(table, "spread", 0.0),
    )


def _read_traits(data: Mapping[str, object]) -> dict[str, EnvBand]:
    """Parse the nested ``[traits.<axis>]`` tables into env bands."""
    raw = data.get("traits")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        msg = "field 'traits' must be a table of trait bands"
        raise ValueError(msg)
    bands: dict[str, EnvBand] = {}
    for axis, table in raw.items():
        if not isinstance(table, Mapping):
            msg = f"trait {axis!r} must be a table of band edges"
            raise ValueError(msg)
        bands[str(axis)] = _env_band(table)
    return bands


def organism_from_content(item: ContentItem) -> Organism:
    """Build an organism from one content item's ecology fields."""
    data = item.data
    try:
        return Organism(
            species_id=item.item_id,
            medium=read_str(data, "medium"),
            body_mass_kg=read_float(data, "body_mass_kg"),
            crowding_cap_per_m2=read_float(data, "crowding_cap_per_m2"),
            lifespan_years=read_float(data, "lifespan_years"),
            maturity_years=read_float(data, "maturity_years"),
            litter_size=read_float(data, "litter_size"),
            reproduction=read_str(data, "reproduction", SEXUAL),
            min_viable_population=read_int(data, "min_viable_population", 2),
            oscillatory=read_bool(data, "oscillatory", default=False),
            sapient=read_bool(data, "sapient", default=False),
            seasonal_migration=read_bool(data, "seasonal_migration", default=False),
            traits=_read_traits(data),
            biome_preference=read_weight_table(data, "biome_preference"),
            diet=read_weight_table(data, "diet"),
            color=read_str(data, "color", "#888888"),
        )
    except KeyError as exc:
        msg = f"organism {item.item_id!r} is missing field {exc.args[0]!r}"
        raise ValueError(msg) from exc


def load_organisms(registry: ContentRegistry) -> tuple[Organism, ...]:
    """Load every organism from the content registry's species kind."""
    return tuple(
        organism_from_content(registry.get(SPECIES_KIND, item_id))
        for item_id in registry.ids(SPECIES_KIND)
    )


def autotrophs(organisms: Sequence[Organism]) -> tuple[Organism, ...]:
    """Return only the plants (autotrophs) from a roster."""
    return tuple(org for org in organisms if org.is_autotroph)


def organism_name_keys(organisms: Sequence[Organism]) -> dict[str, str]:
    """Map each organism id to its Fluent display-name key (legend helper)."""
    return {org.species_id: org.name_key for org in organisms}
