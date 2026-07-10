"""Technology definitions, the prerequisite DAG, and research selection.

Tech progression is driven, not free: civs accumulate research points and
advance through a prerequisite DAG, gated by the resources they can actually
reach.  Effects are plain multipliers read from content, so domain code never
special-cases a tech id (single-source rule): naval tech "works" because its
content lowers the water-crossing cost multiplier, not because code knows
about boats, and an isolated civ with no coal or iron simply never sees the
techs those resources gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.civilization.content_fields import read_float, read_str_tuple

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from ports.content import ContentItem, ContentRegistry
    from ports.rng import Rng

TECH_KIND = "techs"


@dataclass(frozen=True)
class TechEffects:
    """Multiplicative modifiers one tech contributes once unlocked."""

    strength: float = 1.0
    water_cost: float = 1.0
    research: float = 1.0
    extraction: float = 1.0
    farm_yield: float = 1.0
    supply_range: float = 1.0

    def combine(self, other: TechEffects) -> TechEffects:
        """Compose two effect sets by multiplying each modifier."""
        return TechEffects(
            strength=self.strength * other.strength,
            water_cost=self.water_cost * other.water_cost,
            research=self.research * other.research,
            extraction=self.extraction * other.extraction,
            farm_yield=self.farm_yield * other.farm_yield,
            supply_range=self.supply_range * other.supply_range,
        )


IDENTITY_EFFECTS = TechEffects()


@dataclass(frozen=True)
class TechDefinition:
    """One node of the technology DAG."""

    tech_id: str
    cost: float
    prerequisites: tuple[str, ...] = ()
    requires_resources: tuple[str, ...] = ()
    effects: TechEffects = IDENTITY_EFFECTS

    def __post_init__(self) -> None:
        """Reject nonsense costs and self-referencing prerequisites."""
        if self.cost <= 0:
            msg = f"tech {self.tech_id!r} must have a positive research cost"
            raise ValueError(msg)
        if self.tech_id in self.prerequisites:
            msg = f"tech {self.tech_id!r} cannot be its own prerequisite"
            raise ValueError(msg)

    @property
    def name_key(self) -> str:
        """Return the Fluent key for the tech's display name."""
        return f"{TECH_KIND}-{self.tech_id}"


def tech_from_content(item: ContentItem) -> TechDefinition:
    """Build a tech definition from one content item."""
    data = item.data
    try:
        effects = TechEffects(
            strength=read_float(data, "strength", 1.0),
            water_cost=read_float(data, "water_cost", 1.0),
            research=read_float(data, "research", 1.0),
            extraction=read_float(data, "extraction", 1.0),
            farm_yield=read_float(data, "farm_yield", 1.0),
            supply_range=read_float(data, "supply_range", 1.0),
        )
        return TechDefinition(
            tech_id=item.item_id,
            cost=read_float(data, "cost"),
            prerequisites=read_str_tuple(data, "prerequisites"),
            requires_resources=read_str_tuple(data, "requires_resources"),
            effects=effects,
        )
    except KeyError as exc:
        msg = f"tech {item.item_id!r} is missing field {exc.args[0]!r}"
        raise ValueError(msg) from exc


def load_techs(registry: ContentRegistry) -> tuple[TechDefinition, ...]:
    """Load every tech definition and validate the prerequisite DAG."""
    techs = tuple(
        tech_from_content(registry.get(TECH_KIND, item_id)) for item_id in registry.ids(TECH_KIND)
    )
    validate_tech_dag(techs)
    return techs


def validate_tech_dag(techs: Sequence[TechDefinition]) -> None:
    """Reject unknown prerequisites and cycles (Kahn's algorithm)."""
    by_id = {tech.tech_id: tech for tech in techs}
    for tech in techs:
        for prereq in tech.prerequisites:
            if prereq not in by_id:
                msg = f"tech {tech.tech_id!r} requires unknown tech {prereq!r}"
                raise ValueError(msg)
    remaining = {tech.tech_id: set(tech.prerequisites) for tech in techs}
    while remaining:
        ready = [tech_id for tech_id, prereqs in remaining.items() if not prereqs]
        if not ready:
            cycle = ", ".join(sorted(remaining))
            msg = f"tech prerequisites contain a cycle among: {cycle}"
            raise ValueError(msg)
        for tech_id in ready:
            del remaining[tech_id]
        for prereqs in remaining.values():
            prereqs.difference_update(ready)


def combined_effects(
    unlocked: AbstractSet[str],
    techs: Sequence[TechDefinition],
) -> TechEffects:
    """Fold the effects of every unlocked tech into one modifier set."""
    effects = IDENTITY_EFFECTS
    for tech in techs:
        if tech.tech_id in unlocked:
            effects = effects.combine(tech.effects)
    return effects


def available_techs(
    techs: Sequence[TechDefinition],
    unlocked: AbstractSet[str],
    accessible_resources: AbstractSet[str],
) -> tuple[TechDefinition, ...]:
    """Return techs whose prerequisites are met and whose gating resources are reachable."""
    return tuple(
        tech
        for tech in techs
        if tech.tech_id not in unlocked
        and all(prereq in unlocked for prereq in tech.prerequisites)
        and all(res in accessible_resources for res in tech.requires_resources)
    )


def choose_research(
    rng: Rng,
    candidates: Sequence[TechDefinition],
    propensity: Mapping[str, float],
) -> TechDefinition | None:
    """Pick the next research target, weighted by the species' propensities."""
    if not candidates:
        return None
    weights = [max(0.0, propensity.get(tech.tech_id, 1.0)) for tech in candidates]
    total = sum(weights)
    if total <= 0:
        return candidates[0]
    threshold = rng.random() * total
    cumulative = 0.0
    for tech, weight in zip(candidates, weights, strict=True):
        cumulative += weight
        if threshold < cumulative:
            return tech
    return candidates[-1]


def tech_dag(techs: Sequence[TechDefinition]) -> dict[str, list[dict[str, object]]]:
    """Emit the DAG as nodes and edges for client-side graph layout (elkjs)."""
    nodes: list[dict[str, object]] = [
        {
            "id": tech.tech_id,
            "cost": tech.cost,
            "requires_resources": list(tech.requires_resources),
            "name_key": tech.name_key,
        }
        for tech in techs
    ]
    edges: list[dict[str, object]] = [
        {"source": prereq, "target": tech.tech_id}
        for tech in techs
        for prereq in tech.prerequisites
    ]
    return {"nodes": nodes, "edges": edges}
