"""Per-cell suitability: medium gate x Liebig-minimum bands x biome preference."""

from __future__ import annotations

from core.biology.bands import EnvBand
from core.biology.suitability import (
    ALTITUDE_AXIS,
    TEMPERATURE_AXIS,
    SurfaceEnvironment,
    biome_factor,
    env_response,
    surface_suitability,
)
from tests.bio_helpers import all_land_mask, make_organism
from tests.civ_helpers import FakeGrid

_WARM = EnvBand(comfort_min=285.0, comfort_max=300.0, tolerance_min=270.0, tolerance_max=315.0)
_LOW = EnvBand(comfort_min=0.0, comfort_max=500.0, tolerance_min=-100.0, tolerance_max=2000.0)


def test_env_response_takes_the_most_limiting_axis() -> None:
    organism = make_organism("x", traits={TEMPERATURE_AXIS: _WARM, ALTITUDE_AXIS: _LOW})
    comfortable = env_response(organism, {TEMPERATURE_AXIS: 292.0, ALTITUDE_AXIS: 100.0})
    one_hostile = env_response(organism, {TEMPERATURE_AXIS: 292.0, ALTITUDE_AXIS: 5000.0})
    assert comfortable == 1.0
    assert one_hostile == 0.0


def test_biome_factor_prefers_listed_biomes() -> None:
    organism = make_organism("deer", biome_preference={"woods": 1.0})
    assert biome_factor(organism, "woods", 0.5) == 1.0
    assert biome_factor(organism, "desert", 0.5) == 0.5


def test_surface_suitability_gates_by_medium() -> None:
    grid = FakeGrid(rows=2, cols=2)
    mask = all_land_mask(grid)
    fish = make_organism("fish", medium="aquatic")
    env = SurfaceEnvironment(axis_fields={}, biome_field={}, sea_mask=mask)
    # All land, so an aquatic species is suitable nowhere.
    assert surface_suitability(fish, list(grid.cells()), env) == {}
