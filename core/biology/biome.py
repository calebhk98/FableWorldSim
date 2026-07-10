"""Biome registry: content-defined life zones over the Holdridge chart.

Each biome is a content item declaring the rectangle of Holdridge cells
(``belt`` x ``province``) it occupies plus display/ecology data.  The
classifier maps a cell's climate to the biome whose rectangle contains
its :class:`~core.biology.holdridge.LifeZoneCoord`.  Because the biomes
come from :class:`ports.content.ContentRegistry`, the whole scheme is
mod-tunable: this module holds the classification *logic*, never a biome
id.

The base biomes ship in ``content/biomes`` and tile the chart with no
gaps or overlaps (guarded by a test), so every land cell classifies to
exactly one biome.  Ocean cells are not land life zones and are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.biology.holdridge import (
    BELT_COUNT,
    PROVINCE_COUNT,
    LifeZoneCoord,
    biotemperature_c,
    life_zone,
)
from core.sim.units import kelvin_to_celsius

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from core.climate.model import ClimateState
    from core.hydrology.sea_mask import SeaMask
    from ports.content import ContentItem, ContentRegistry
    from ports.grid import CellId

BIOME_KIND = "biomes"
"""Content kind under which biome items live."""


class BiomeClassificationError(RuntimeError):
    """Raised when a climate has no biome (a gap in the registry tiling)."""


@dataclass(frozen=True)
class BiomeDefinition:
    """One biome's Holdridge footprint and display data.

    The footprint is the inclusive rectangle
    ``[belt_min, belt_max] x [province_min, province_max]`` on the
    Holdridge chart.  ``vegetation_density`` (0..1) and ``color`` are
    display/ecology data other layers and overlays consume.
    """

    biome_id: str
    belt_min: int
    belt_max: int
    province_min: int
    province_max: int
    vegetation_density: float
    color: str

    def __post_init__(self) -> None:
        """Validate the footprint lies on the chart and is non-empty."""
        if not 0 <= self.belt_min <= self.belt_max < BELT_COUNT:
            msg = f"biome {self.biome_id!r} has an invalid belt range"
            raise ValueError(msg)
        if not 0 <= self.province_min <= self.province_max < PROVINCE_COUNT:
            msg = f"biome {self.biome_id!r} has an invalid province range"
            raise ValueError(msg)
        if not 0.0 <= self.vegetation_density <= 1.0:
            msg = f"biome {self.biome_id!r} vegetation_density must be in [0, 1]"
            raise ValueError(msg)

    @property
    def name_key(self) -> str:
        """Return the i18n key for this biome's display name."""
        return f"{BIOME_KIND}-{self.biome_id}"

    def contains(self, coord: LifeZoneCoord) -> bool:
        """Return whether a Holdridge coordinate falls in this footprint."""
        return (
            self.belt_min <= coord.belt <= self.belt_max
            and self.province_min <= coord.province <= self.province_max
        )


def biome_from_content(item: ContentItem) -> BiomeDefinition:
    """Build a :class:`BiomeDefinition` from a content item."""
    data = item.data
    try:
        return BiomeDefinition(
            biome_id=item.item_id,
            belt_min=int(str(data["belt_min"])),
            belt_max=int(str(data["belt_max"])),
            province_min=int(str(data["province_min"])),
            province_max=int(str(data["province_max"])),
            vegetation_density=float(str(data["vegetation_density"])),
            color=str(data["color"]),
        )
    except KeyError as exc:
        msg = f"biome {item.item_id!r} is missing field {exc.args[0]!r}"
        raise ValueError(msg) from exc


def load_biomes(registry: ContentRegistry) -> tuple[BiomeDefinition, ...]:
    """Return every biome definition from the content registry."""
    return tuple(
        biome_from_content(registry.get(BIOME_KIND, item_id))
        for item_id in registry.ids(BIOME_KIND)
    )


def classify(
    coord: LifeZoneCoord,
    biomes: Sequence[BiomeDefinition],
) -> str:
    """Return the id of the biome whose footprint contains ``coord``.

    Raises :class:`BiomeClassificationError` when no biome matches — a
    tiling gap, which the base content is tested never to have but a mod
    could introduce.
    """
    for biome in biomes:
        if biome.contains(coord):
            return biome.biome_id
    msg = f"no biome covers Holdridge cell belt={coord.belt} province={coord.province}"
    raise BiomeClassificationError(msg)


def biome_field(
    climate: ClimateState,
    sea_mask: SeaMask,
    biomes: Sequence[BiomeDefinition],
) -> dict[CellId, str]:
    """Return the biome id for every land cell.

    Biotemperature comes from the seasonal temperatures (each clamped to
    the growing-season window), precipitation from the annual total.
    Ocean cells are omitted — Holdridge zones are land life zones.
    """
    seasonal_c: dict[CellId, tuple[float, ...]] = {}
    for cell in climate.annual_mean_temperature_k:
        seasonal_c[cell] = tuple(
            kelvin_to_celsius(season.temperature_k[cell]) for season in climate.seasons
        )
    result: dict[CellId, str] = {}
    for cell, temps_c in seasonal_c.items():
        if sea_mask.ocean[cell]:
            continue
        biotemp = biotemperature_c(temps_c)
        precip = climate.annual_precipitation_mm_yr[cell]
        result[cell] = classify(life_zone(biotemp, precip), biomes)
    return result


def biome_name_keys(biomes: Sequence[BiomeDefinition]) -> Mapping[str, str]:
    """Return each biome id mapped to its i18n name key (legend helper)."""
    return {biome.biome_id: biome.name_key for biome in biomes}
