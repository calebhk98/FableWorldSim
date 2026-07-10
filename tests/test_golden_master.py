"""Golden-master test: catches behavioral drift in emergent output.

A known-seed world's summary metrics are checked in and diffed with
tolerance — unit tests catch broken functions, this catches "the code
still runs but the planet quietly changed".  When a change is
*intentional*, regenerate with:

    FWS_UPDATE_GOLDEN=1 pytest tests/test_golden_master.py

and commit the updated JSON alongside the change that caused it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("h3")

from adapters.grid_registry import create_grid
from adapters.rng_seeded import SeededRng
from core.climate.model import simulate_climate
from core.grid.area_weighted import area_fraction, area_weighted_mean
from core.hydrology.sea_mask import build_sea_mask
from core.sim.presets import earth
from core.topography.procedural import ProceduralTopography

_GOLDEN = Path(__file__).parent / "golden" / "earth_res0.json"
_SEED = 42
_TOLERANCE = 1e-6


def _summarize() -> dict[str, float]:
    """Build the known-seed world and reduce it to summary metrics."""
    planet = earth()
    grid = create_grid("h3", resolution=0, radius_m=planet.radius_m)
    heights = ProceduralTopography(SeededRng(_SEED)).heights(grid)
    mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    state = simulate_climate(grid, planet, heights, mask, season_count=2)

    cells = sorted(state.annual_mean_temperature_k)
    areas = [grid.area_m2(c) for c in cells]
    return {
        "sea_level_m": mask.sea_level_m,
        "ocean_area_fraction": area_fraction([mask.ocean[c] for c in cells], areas),
        "mean_temperature_k": area_weighted_mean(
            [state.annual_mean_temperature_k[c] for c in cells], areas
        ),
        "mean_precipitation_mm_yr": area_weighted_mean(
            [state.annual_precipitation_mm_yr[c] for c in cells], areas
        ),
        "permanent_ice_fraction": area_fraction([state.permanent_ice[c] for c in cells], areas),
        "min_height_m": min(heights.values()),
        "max_height_m": max(heights.values()),
        "probe_temperature_k": state.annual_mean_temperature_k[cells[7]],
        "probe_precipitation_mm_yr": state.annual_precipitation_mm_yr[cells[21]],
    }


def test_known_seed_world_matches_the_golden_master() -> None:
    actual = _summarize()
    if os.environ.get("FWS_UPDATE_GOLDEN"):
        _GOLDEN.parent.mkdir(exist_ok=True)
        _GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True), encoding="utf-8")
    expected = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=_TOLERANCE, abs=1e-9), key
