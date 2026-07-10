"""Tests for the content-driven Holdridge biome registry and classifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.content_toml import TomlContentRegistry
from core.biology.biome import (
    BiomeClassificationError,
    BiomeDefinition,
    biome_field,
    classify,
    load_biomes,
)
from core.biology.holdridge import BELT_COUNT, PROVINCE_COUNT, LifeZoneCoord, life_zone

_REPO_CONTENT = Path(__file__).resolve().parents[1] / "content"
_EXPECTED_BIOME_COUNT = 16


def _base_biomes() -> tuple[BiomeDefinition, ...]:
    registry = TomlContentRegistry([("base", _REPO_CONTENT)])
    return load_biomes(registry)


def test_base_content_loads_every_biome() -> None:
    biomes = _base_biomes()
    assert len(biomes) == _EXPECTED_BIOME_COUNT
    taiga = next(b for b in biomes if b.biome_id == "taiga")
    assert taiga.name_key == "biomes-taiga"
    assert taiga.color.startswith("#")
    assert 0.0 <= taiga.vegetation_density <= 1.0


def test_base_content_tiles_the_chart_with_no_gaps_or_overlaps() -> None:
    """Every Holdridge cell maps to exactly one base biome."""
    biomes = _base_biomes()
    for belt in range(BELT_COUNT):
        for province in range(PROVINCE_COUNT):
            coord = LifeZoneCoord(belt=belt, province=province)
            matches = [b.biome_id for b in biomes if b.contains(coord)]
            assert len(matches) == 1, f"cell {coord} matched {matches}"


def test_classification_of_recognizable_climates() -> None:
    biomes = _base_biomes()

    def biome_of(biotemp_c: float, precip_mm: float) -> str:
        return classify(life_zone(biotemp_c, precip_mm), biomes)

    # Hot and soaked -> tropical rainforest; hot and parched -> desert.
    # Holdridge's "rain forest" province needs a very low PET ratio, so it
    # takes more than a moist forest's rainfall to reach it.
    assert biome_of(27.0, 9_000.0) == "tropical_rainforest"
    assert biome_of(27.0, 4_000.0) == "tropical_moist_forest"
    assert biome_of(27.0, 100.0) == "tropical_desert"
    # Warm, seasonal-dry grassland belt -> savanna.
    assert biome_of(20.0, 700.0) == "savanna"
    # Cold -> tundra / polar desert regardless of moisture.
    assert biome_of(2.0, 3_000.0) == "tundra"
    assert biome_of(0.5, 3_000.0) == "polar_desert"
    # Mid-latitude moist forest.
    assert biome_of(9.0, 1_200.0) == "temperate_forest"


def test_classification_gap_fails_loudly() -> None:
    """A mod that leaves a chart cell uncovered raises, not silently wrong."""
    only_tropics = [b for b in _base_biomes() if b.belt_min >= BELT_COUNT - 1]
    with pytest.raises(BiomeClassificationError, match="no biome covers"):
        classify(LifeZoneCoord(belt=0, province=0), only_tropics)


def test_biome_definition_rejects_invalid_footprints() -> None:
    with pytest.raises(ValueError, match="belt range"):
        BiomeDefinition("bad", -1, 0, 0, 0, 0.5, "#000000")
    with pytest.raises(ValueError, match="province range"):
        BiomeDefinition("bad", 0, 0, 3, 1, 0.5, "#000000")
    with pytest.raises(ValueError, match="vegetation_density"):
        BiomeDefinition("bad", 0, 0, 0, 0, 1.5, "#000000")


def test_biome_field_on_earth_climate_is_land_only_and_sensible() -> None:
    pytest.importorskip("h3")
    from adapters.grid_registry import create_grid
    from adapters.rng_seeded import SeededRng
    from core.climate.model import simulate_climate
    from core.hydrology.sea_mask import build_sea_mask
    from core.sim.presets import earth
    from core.topography.procedural import ProceduralTopography

    planet = earth()
    grid = create_grid("h3", resolution=1, radius_m=planet.radius_m)
    heights = dict(ProceduralTopography(SeededRng(42)).heights(grid))
    mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    climate = simulate_climate(grid, planet, heights, mask, season_count=2)

    biomes = _base_biomes()
    field = biome_field(climate, mask, biomes)

    # Only land cells, and every one classified (no gaps on a real planet).
    land = [c for c in grid.cells() if not mask.ocean[c]]
    assert set(field) == set(land)
    assert field

    # A structured planet shows more than one life zone...
    assert len(set(field.values())) >= 3
    # ...and cold high-latitude land trends to cold biomes.
    cold_ids = {"polar_desert", "tundra", "taiga", "cold_desert"}
    polar_land = [c for c in land if abs(grid.centroid(c).lat_deg) > 70.0]
    if polar_land:
        cold_share = sum(field[c] in cold_ids for c in polar_land) / len(polar_land)
        assert cold_share > 0.5
