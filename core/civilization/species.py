"""Sapient species definitions loaded from moddable content.

A species is data, not code: any species flagged ``sapient`` can found a
civilization, so adding a new playable race (elves, dwarves, centaurs) is a
content drop — a TOML file plus a locale key — never a code change.  The
``domain`` field decides which territory layer the race occupies: surface
races and subsurface races share planet columns without contesting them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.civilization.content_fields import (
    read_bool,
    read_float,
    read_str,
    read_str_tuple,
    read_weight_table,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ports.content import ContentItem, ContentRegistry

SPECIES_KIND = "species"

SURFACE_DOMAIN = "surface"
SUBSURFACE_DOMAIN = "subsurface"
DOMAINS = (SURFACE_DOMAIN, SUBSURFACE_DOMAIN)


@dataclass(frozen=True)
class SpeciesDefinition:
    """One species: civ-founding rights plus its per-race tuning knobs."""

    species_id: str
    sapient: bool
    domain: str
    growth_rate_per_year: float
    commit_fraction: float
    research_aptitude: float = 1.0
    mine_affinity: float = 1.0
    color: str = "#888888"
    propensity: Mapping[str, float] = field(default_factory=dict)
    toponym_prefixes: tuple[str, ...] = ()
    toponym_suffixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate ranges so malformed content fails at load time."""
        if self.domain not in DOMAINS:
            msg = f"species {self.species_id!r} has unknown domain {self.domain!r}"
            raise ValueError(msg)
        if self.growth_rate_per_year < 0:
            msg = f"species {self.species_id!r} growth rate must be >= 0"
            raise ValueError(msg)
        if not 0.0 <= self.commit_fraction <= 1.0:
            msg = f"species {self.species_id!r} commit_fraction must be in [0, 1]"
            raise ValueError(msg)
        if self.research_aptitude <= 0 or self.mine_affinity <= 0:
            msg = f"species {self.species_id!r} multipliers must be > 0"
            raise ValueError(msg)

    @property
    def name_key(self) -> str:
        """Return the Fluent key for the species' display name."""
        return f"{SPECIES_KIND}-{self.species_id}"


def species_from_content(item: ContentItem) -> SpeciesDefinition:
    """Build a species definition from one content item."""
    data = item.data
    try:
        return SpeciesDefinition(
            species_id=item.item_id,
            sapient=read_bool(data, "sapient"),
            domain=read_str(data, "domain"),
            growth_rate_per_year=read_float(data, "growth_rate_per_year"),
            commit_fraction=read_float(data, "commit_fraction"),
            research_aptitude=read_float(data, "research_aptitude", 1.0),
            mine_affinity=read_float(data, "mine_affinity", 1.0),
            color=read_str(data, "color", "#888888"),
            propensity=read_weight_table(data, "propensity"),
            toponym_prefixes=read_str_tuple(data, "toponym_prefixes"),
            toponym_suffixes=read_str_tuple(data, "toponym_suffixes"),
        )
    except KeyError as exc:
        msg = f"species {item.item_id!r} is missing field {exc.args[0]!r}"
        raise ValueError(msg) from exc


def load_species(registry: ContentRegistry) -> tuple[SpeciesDefinition, ...]:
    """Load every species definition from the content registry."""
    return tuple(
        species_from_content(registry.get(SPECIES_KIND, item_id))
        for item_id in registry.ids(SPECIES_KIND)
    )


def sapient_species(species: Sequence[SpeciesDefinition]) -> tuple[SpeciesDefinition, ...]:
    """Return only the species that can found civilizations."""
    return tuple(spec for spec in species if spec.sapient)


def species_name_keys(species: Sequence[SpeciesDefinition]) -> dict[str, str]:
    """Map each species id to its Fluent display-name key."""
    return {spec.species_id: spec.name_key for spec in species}
