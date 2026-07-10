"""Resource definitions: mines, forestry, and farmland behave differently.

Three structurally different extraction modes ship in M1: mines hold a
finite stock that depletes and runs out; forestry harvests the shared
plant-biomass field and can be over-harvested faster than regrowth; farmland
draws on managed plant biomass that renews while the underlying biology
holds.  Gameplay roles (feeding population, arming frontier contests) are
content fields — ``feeds_population`` and ``arms_value`` — so no resource id
is ever special-cased in domain code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.civilization.content_fields import read_bool, read_float, read_str

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core.hydrology.sea_mask import SeaMask
    from ports.content import ContentItem, ContentRegistry
    from ports.grid import CellId, Grid
    from ports.rng import Rng

RESOURCE_KIND = "resources"

MINE_MODE = "mine"
FORESTRY_MODE = "forestry"
FARMLAND_MODE = "farmland"
MODES = (MINE_MODE, FORESTRY_MODE, FARMLAND_MODE)


@dataclass(frozen=True)
class ResourceDefinition:
    """One extractable resource and its gameplay role."""

    resource_id: str
    mode: str
    subsurface: bool = False
    deposit_fraction: float = 0.0
    deposit_stock: float = 0.0
    harvest_fraction_per_year: float = 0.0
    feeds_population: bool = False
    arms_value: float = 0.0

    def __post_init__(self) -> None:
        """Validate the mode and per-mode numeric ranges."""
        if self.mode not in MODES:
            msg = f"resource {self.resource_id!r} has unknown mode {self.mode!r}"
            raise ValueError(msg)
        if not 0.0 <= self.deposit_fraction <= 1.0:
            msg = f"resource {self.resource_id!r} deposit_fraction must be in [0, 1]"
            raise ValueError(msg)
        if self.deposit_stock < 0 or self.arms_value < 0:
            msg = f"resource {self.resource_id!r} stocks and values must be >= 0"
            raise ValueError(msg)
        if not 0.0 <= self.harvest_fraction_per_year <= 1.0:
            msg = f"resource {self.resource_id!r} harvest fraction must be in [0, 1]"
            raise ValueError(msg)

    @property
    def name_key(self) -> str:
        """Return the Fluent key for the resource's display name."""
        return f"{RESOURCE_KIND}-{self.resource_id}"


def resource_from_content(item: ContentItem) -> ResourceDefinition:
    """Build a resource definition from one content item."""
    data = item.data
    try:
        return ResourceDefinition(
            resource_id=item.item_id,
            mode=read_str(data, "mode"),
            subsurface=read_bool(data, "subsurface", default=False),
            deposit_fraction=read_float(data, "deposit_fraction", 0.0),
            deposit_stock=read_float(data, "deposit_stock", 0.0),
            harvest_fraction_per_year=read_float(data, "harvest_fraction_per_year", 0.0),
            feeds_population=read_bool(data, "feeds_population", default=False),
            arms_value=read_float(data, "arms_value", 0.0),
        )
    except KeyError as exc:
        msg = f"resource {item.item_id!r} is missing field {exc.args[0]!r}"
        raise ValueError(msg) from exc


def load_resources(registry: ContentRegistry) -> tuple[ResourceDefinition, ...]:
    """Load every resource definition from the content registry."""
    return tuple(
        resource_from_content(registry.get(RESOURCE_KIND, item_id))
        for item_id in registry.ids(RESOURCE_KIND)
    )


def generate_mine_deposits(
    grid: Grid,
    rng: Rng,
    resources: Sequence[ResourceDefinition],
    sea_mask: SeaMask,
) -> dict[str, dict[CellId, float]]:
    """Seed finite mine deposits on land cells, deterministically per resource.

    Each mine resource forks its own RNG stream, so adding a resource never
    perturbs where the others spawn.
    """
    deposits: dict[str, dict[CellId, float]] = {}
    for resource in resources:
        if resource.mode != MINE_MODE:
            continue
        stream = rng.fork(f"deposits:{resource.resource_id}")
        deposits[resource.resource_id] = {
            cell: resource.deposit_stock
            for cell in grid.cells()
            if sea_mask.is_land(cell) and stream.random() < resource.deposit_fraction
        }
    return deposits
