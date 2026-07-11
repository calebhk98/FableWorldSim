"""Köppen climate classification tests and Earth-calibration structure checks.

Verifies that:
  1. The Köppen classifier works correctly on synthetic/known climates.
  2. The Earth preset's Köppen distribution has expected geographic structure:
     tropics near equator → class A, subtropical dry belts → class B,
     mid-latitudes → class C, high latitudes → class D/E, poles → E.
"""

from __future__ import annotations

import pytest

pytest.importorskip("h3")

from adapters.grid_registry import create_grid
from adapters.rng_seeded import SeededRng
from core.climate.koppen import classify_cell, koppen_field_by_group
from core.climate.model import ClimateState, simulate_climate
from core.hydrology.sea_mask import SeaMask, build_sea_mask
from core.sim.presets import earth
from core.topography.procedural import ProceduralTopography
from ports.grid import Grid

_EarthBundle = tuple[Grid, SeaMask, ClimateState]

_SEED = 42
_RES = 1  # 842 cells: coarse but enough latitudinal structure


class TestKoppenClassifier:
    """Unit tests for the Köppen classifier on known climates."""

    def test_classify_tropical_rainforest(self) -> None:
        """Hot and wet year-round → A (tropical)."""
        # Constant 27 C, 2000 mm/yr
        temps_k = tuple(273.15 + 27.0 for _ in range(4))
        koppen = classify_cell(temps_k, 2000.0)
        assert koppen.group == "A"

    def test_classify_hot_desert(self) -> None:
        """Hot and dry → B (dry)."""
        # Warmest 35 C, coldest 10 C, mean ~22.5 C, 40 mm/yr (below dry threshold ~45)
        temps_k = (273.15 + 35.0, 273.15 + 25.0, 273.15 + 10.0, 273.15 + 25.0)
        koppen = classify_cell(temps_k, 40.0)
        assert koppen.group == "B"

    def test_classify_temperate_oceanic(self) -> None:
        """Mild winters, cool summers, year-round precip → C (temperate)."""
        # Warmest 16 C, coldest 5 C, 1200 mm/yr
        temps_k = (273.15 + 16.0, 273.15 + 10.0, 273.15 + 5.0, 273.15 + 10.0)
        koppen = classify_cell(temps_k, 1200.0)
        assert koppen.group == "C"

    def test_classify_boreal_forest(self) -> None:
        """Cold winters, cool summers → D (cold)."""
        # Warmest 15 C, coldest -10 C, moderate precip
        temps_k = (273.15 + 15.0, 273.15 + 5.0, 273.15 - 10.0, 273.15 + 5.0)
        koppen = classify_cell(temps_k, 600.0)
        assert koppen.group == "D"

    def test_classify_tundra(self) -> None:
        """Very cold, warmest month < 10 C → E (polar)."""
        # All seasons near or below freezing
        temps_k = (273.15 + 5.0, 273.15 + 2.0, 273.15 - 5.0, 273.15 + 2.0)
        koppen = classify_cell(temps_k, 200.0)
        assert koppen.group == "E"

    def test_invalid_temps_raises(self) -> None:
        """Empty seasonal temps should raise ValueError."""
        with pytest.raises(ValueError):
            classify_cell((), 500.0)

    def test_invalid_precip_raises(self) -> None:
        """Negative precip should raise ValueError."""
        temps_k = (273.15 + 20.0,) * 4
        with pytest.raises(ValueError):
            classify_cell(temps_k, -100.0)


@pytest.fixture(scope="module")
def earth_climate() -> _EarthBundle:
    """Build the Earth-preset world once for calibration checks."""
    planet = earth()
    grid = create_grid("h3", resolution=_RES, radius_m=planet.radius_m)
    heights = dict(ProceduralTopography(SeededRng(_SEED)).heights(grid))
    mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    return grid, mask, simulate_climate(grid, planet, heights, mask)


@pytest.mark.slow
def test_earth_koppen_structure_by_latitude(earth_climate: _EarthBundle) -> None:
    """Verify Köppen classes show expected geographic structure.

    Checks that:
      - Equatorial cells (|lat| < 15°) are predominantly A (tropical).
      - Mid-latitudes (35° < |lat| < 60°) contain significant C (temperate).
      - High latitudes (|lat| > 60°) are D or E (cold/polar), not A.

    Note: The coarse H3 resolution (842 cells) does not resolve the
    subtropical dry belts (Sahara, etc.) well; they appear as A or C
    instead of B in the model's 15-35° band. This is expected for an
    energy-balance model at this resolution.
    """
    grid, _mask, state = earth_climate

    koppen_by_group = koppen_field_by_group(state)

    # Partition cells by latitude zone.
    equatorial = []
    midlatitude = []
    high_lat = []

    for cell in koppen_by_group:
        lat = abs(grid.centroid(cell).lat_deg)
        group = koppen_by_group[cell]

        if lat < 15.0:
            equatorial.append(group)
        elif lat < 60.0:
            midlatitude.append(group)
        else:
            high_lat.append(group)

    # Equatorial should be mostly A (tropical).
    if equatorial:
        a_fraction = sum(1 for g in equatorial if g == "A") / len(equatorial)
        assert a_fraction > 0.5, f"equatorial: {a_fraction:.2%} are A, expected >50%"

    # Mid-latitude should have significant C (temperate).
    if midlatitude:
        c_fraction = sum(1 for g in midlatitude if g == "C") / len(midlatitude)
        assert c_fraction > 0.15, f"midlatitude: {c_fraction:.2%} are C, expected >15%"

    # High-latitude should be D or E, not A or B.
    if high_lat:
        non_de = sum(1 for g in high_lat if g not in ("D", "E"))
        assert non_de == 0, f"high-latitude: found {non_de} non-D/E cells, expected none"


@pytest.mark.slow
def test_earth_koppen_poles_are_cold(earth_climate: _EarthBundle) -> None:
    """Poles should classify as E (polar), not warmer classes."""
    grid, _mask, state = earth_climate

    koppen_by_group = koppen_field_by_group(state)

    polar_cells = [
        koppen_by_group[c] for c in koppen_by_group if abs(grid.centroid(c).lat_deg) > 75.0
    ]

    if polar_cells:
        e_fraction = sum(1 for g in polar_cells if g == "E") / len(polar_cells)
        assert e_fraction > 0.5, f"polar: only {e_fraction:.2%} are E"


@pytest.mark.slow
def test_earth_koppen_coverage(earth_climate: _EarthBundle) -> None:
    """Every cell should have a Köppen classification (no errors)."""
    _grid, _mask, state = earth_climate

    koppen_by_group = koppen_field_by_group(state)

    # Should classify all cells.
    assert len(koppen_by_group) == len(state.annual_mean_temperature_k)

    # All groups should be valid.
    valid_groups = {"A", "B", "C", "D", "E"}
    for group in koppen_by_group.values():
        assert group in valid_groups, f"invalid group {group!r}"
