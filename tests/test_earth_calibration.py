"""Earth-calibration test: the model's Earth preset checked against real climatology.

Verification-checklist item: compare the Earth preset's climate against
real-world climatology, but only within a STATED, GENEROUS band, and only
for DIRECTIONAL / STRUCTURAL agreement (zone ordering, wet-tropics vs.
dry-poles, an ice cap without a snowball). ``core/climate`` is a
simplified energy-balance model (see its module docstring: seasons
resolved, day-to-day weather is roadmap), so it will not reproduce GCM- or
reanalysis-exact temperatures. Asserting numeric closeness to real
observations would be testing "did we hand-tune constants to 2026
reanalysis data", not "does the physics produce an Earth-like regime".

Bands below were chosen by first running the Earth preset at H3
resolution 1 (842 cells, seed 42, quarterly seasons — see
``tests/test_climate.py``'s ``earth_climate`` fixture for the same
construction) and recording the actual area-weighted zonal output, then
picking bounds that bracket the observed value with a wide, documented
margin. Observed reference run (kept here for anyone re-deriving the
bands):

    tropical  (|lat| < 23.5):  mean_temp_k=307.31  mean_precip_mm_yr=1032.54
    temperate (23.5<=|lat|<66.5): mean_temp_k=274.04  mean_precip_mm_yr=226.94
    polar     (|lat| >= 66.5): mean_temp_k=219.72  mean_precip_mm_yr=4.99
    global area-weighted mean temperature: 282.62 K   (real Earth ~288 K)
    permanent-ice area fraction, polar zone: 0.988
    permanent-ice area fraction, whole planet: 0.146

The final test in this module (``test_earth_koppen_matches_reference_climatology``)
is the map-diff item from the calibration checklist: it classifies every
cell of the same Earth preset with the Köppen classifier
(``core/climate/koppen.py``) and diffs the per-cell *group* against
``tests/data/koppen_reference.json``, a coarse hand-encoded stand-in for
real Köppen-Geiger climatology (see that file's own header for what it is
and, importantly, what it is *not*). See that test's docstring for why the
diff is band/land-ocean based rather than exact-longitude, and for the
observed agreement fraction the chosen threshold is calibrated against.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("h3")

from adapters.grid_registry import create_grid
from adapters.rng_seeded import SeededRng
from core.climate.koppen import koppen_field_by_group
from core.climate.model import ClimateState, simulate_climate
from core.grid.area_weighted import area_fraction, area_weighted_mean
from core.hydrology.sea_mask import SeaMask, build_sea_mask
from core.sim.presets import earth
from core.topography.procedural import ProceduralTopography
from ports.grid import CellId, Grid

_EarthBundle = tuple[Grid, SeaMask, ClimateState]

_SEED = 42
_RES = 1  # 842 cells: coarse but enough latitudinal structure for zonal means

_TROPICAL_MAX_ABS_LAT = 23.5
_TEMPERATE_MAX_ABS_LAT = 66.5

# Generous margin around the observed global mean (282.62 K), chosen to
# bracket both the model's own output and real Earth's ~288 K while still
# excluding an obviously broken regime (e.g. a runaway freeze or greenhouse).
_GLOBAL_TEMP_BAND_K = (257.0, 308.0)

# Ordering margins: observed zone-to-zone gaps are ~33 K and ~54 K. Ten
# kelvin is a small fraction of either gap, so this asserts "meaningfully
# monotone by latitude", not "matches the reference run to the kelvin".
_MIN_ZONE_TEMP_GAP_K = 10.0

# Observed tropical/polar precip ratio is roughly 200:1; a factor of 2 is a
# deliberately weak floor that only checks the Hadley-cell direction holds.
_MIN_TROPICAL_TO_POLAR_PRECIP_RATIO = 2.0

# "Contains permanent ice" / "not a snowball": small floor, generous ceiling.
_MIN_POLAR_ICE_FRACTION = 0.3
_MIN_GLOBAL_ICE_FRACTION = 0.02
_MAX_GLOBAL_ICE_FRACTION = 0.6

_KOPPEN_REFERENCE_PATH = Path(__file__).parent / "data" / "koppen_reference.json"

# Observed cell-count group match against tests/data/koppen_reference.json for
# the same seed-42/res-1 Earth preset run (842 cells): 388/842 = 46.1%. For
# context, always guessing the single most common group the model actually
# produced (D, 265/842 cells) would score 31.5% -- a "no-skill" ceiling this
# reference-informed comparison must clear. The floor below sits comfortably
# above that no-skill baseline and a few points under the observed value, so
# the test asserts "the classifier's latitude/land-ocean structure is doing
# real work", not "this exact seed reproduces 46.1% forever".
_MIN_REFERENCE_GROUP_AGREEMENT = 0.40


def _zone(abs_lat_deg: float) -> str:
    if abs_lat_deg < _TROPICAL_MAX_ABS_LAT:
        return "tropical"
    if abs_lat_deg < _TEMPERATE_MAX_ABS_LAT:
        return "temperate"
    return "polar"


def _cells_by_zone(grid: Grid, cells: list[CellId]) -> dict[str, list[CellId]]:
    zones: dict[str, list[CellId]] = {"tropical": [], "temperate": [], "polar": []}
    for cell in cells:
        zones[_zone(abs(grid.centroid(cell).lat_deg))].append(cell)
    return zones


def _zonal_mean(grid: Grid, state: ClimateState, cells: list[CellId], field: str) -> float:
    values = [getattr(state, field)[c] for c in cells]
    areas = [grid.area_m2(c) for c in cells]
    return area_weighted_mean(values, areas)


@pytest.fixture(scope="module")
def earth_climate() -> _EarthBundle:
    """Build the known-seed Earth-preset world once for every test here."""
    planet = earth()
    grid = create_grid("h3", resolution=_RES, radius_m=planet.radius_m)
    heights = dict(ProceduralTopography(SeededRng(_SEED)).heights(grid))
    mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    return grid, mask, simulate_climate(grid, planet, heights, mask)


@pytest.mark.slow
def test_zonal_temperature_is_monotone_tropical_to_polar(earth_climate: _EarthBundle) -> None:
    """Warmest at the equator, coldest at the poles, temperate in between.

    This is the basic thermal signature of a sphere lit unevenly by
    latitude; any Earth-like preset must reproduce the ordering even
    though its absolute values will not match a GCM.
    """
    grid, _mask, state = earth_climate
    cells = list(state.annual_mean_temperature_k)
    zones = _cells_by_zone(grid, cells)

    tropical = _zonal_mean(grid, state, zones["tropical"], "annual_mean_temperature_k")
    temperate = _zonal_mean(grid, state, zones["temperate"], "annual_mean_temperature_k")
    polar = _zonal_mean(grid, state, zones["polar"], "annual_mean_temperature_k")

    assert tropical - temperate > _MIN_ZONE_TEMP_GAP_K, (tropical, temperate)
    assert temperate - polar > _MIN_ZONE_TEMP_GAP_K, (temperate, polar)


@pytest.mark.slow
def test_global_mean_temperature_is_earth_like(earth_climate: _EarthBundle) -> None:
    """The planet lands in a plausible Earth-like thermal regime.

    Guards that rotation, tilt, and atmosphere composition combine into
    something in Earth's ballpark, not a snowball or a runaway
    greenhouse — within a band wide enough to absorb this being a
    simplified energy-balance model rather than a GCM.
    """
    grid, _mask, state = earth_climate
    cells = list(state.annual_mean_temperature_k)
    areas = [grid.area_m2(c) for c in cells]
    global_mean = area_weighted_mean([state.annual_mean_temperature_k[c] for c in cells], areas)
    low, high = _GLOBAL_TEMP_BAND_K
    assert low < global_mean < high, global_mean


@pytest.mark.slow
def test_tropics_are_wetter_than_the_poles(earth_climate: _EarthBundle) -> None:
    """ITCZ / Hadley-cell rainfall vs. the cold, dry polar zone.

    Real climatology puts the wettest annual-mean band near the equator
    (rising, moisture-laden air) and the driest near the poles (a cold,
    high-pressure desert). The ratio is asserted with a very weak floor
    since only the direction is a structural guarantee.
    """
    grid, _mask, state = earth_climate
    cells = list(state.annual_precipitation_mm_yr)
    zones = _cells_by_zone(grid, cells)

    tropical = _zonal_mean(grid, state, zones["tropical"], "annual_precipitation_mm_yr")
    polar = _zonal_mean(grid, state, zones["polar"], "annual_precipitation_mm_yr")

    assert polar > 0.0, "expected some finite polar precipitation, not a degenerate zero"
    assert tropical > polar * _MIN_TROPICAL_TO_POLAR_PRECIP_RATIO, (tropical, polar)


@pytest.mark.slow
def test_polar_ice_cap_without_a_global_snowball(earth_climate: _EarthBundle) -> None:
    """Permanent ice sits at the poles, but it never eats the planet.

    Checks both ends of the failure mode: an Earth preset with no ice at
    all (greenhouse runaway) and one where ice-albedo feedback spirals
    into a snowball are both wrong regimes.
    """
    grid, _mask, state = earth_climate
    cells = list(state.permanent_ice)
    zones = _cells_by_zone(grid, cells)

    polar_ice_fraction = area_fraction(
        [state.permanent_ice[c] for c in zones["polar"]],
        [grid.area_m2(c) for c in zones["polar"]],
    )
    global_ice_fraction = area_fraction(
        [state.permanent_ice[c] for c in cells],
        [grid.area_m2(c) for c in cells],
    )

    assert polar_ice_fraction > _MIN_POLAR_ICE_FRACTION, polar_ice_fraction
    assert _MIN_GLOBAL_ICE_FRACTION < global_ice_fraction < _MAX_GLOBAL_ICE_FRACTION, (
        global_ice_fraction
    )


def _load_koppen_reference_bands() -> list[dict[str, float | str]]:
    """Load the coarse reference bands from ``tests/data/koppen_reference.json``."""
    with _KOPPEN_REFERENCE_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    bands: list[dict[str, float | str]] = payload["bands"]
    return bands


def _reference_group(bands: list[dict[str, float | str]], abs_lat_deg: float, is_land: bool) -> str:
    """Return the reference's dominant Köppen group for one (lat, land/ocean) cell."""
    key = "land_group" if is_land else "ocean_group"
    for band in bands:
        if band["lat_min_deg"] <= abs_lat_deg < band["lat_max_deg"]:
            return str(band[key])
    msg = f"no reference band covers |lat|={abs_lat_deg}"
    raise ValueError(msg)


@pytest.mark.slow
def test_earth_koppen_matches_reference_climatology(earth_climate: _EarthBundle) -> None:
    """Diff the Earth preset's Köppen map against coarse reference climatology.

    This is the calibration-checklist item: classify every cell with the
    real Köppen classifier and compare its *group* letter against
    ``tests/data/koppen_reference.json`` -- a coarse, hand-encoded stand-in
    for real Köppen-Geiger climatology (see that file's own disclaimer).

    The comparison is by (latitude band, land-vs-ocean), not exact
    longitude: the Earth preset's continents are procedurally generated
    (``core/topography/procedural.py``, seeded noise) and bear no
    resemblance to real coastlines, so indexing the reference by this
    world's actual longitude would mostly score where the procedural
    generator happened to place land relative to a real desert or
    rainforest -- geography-placement luck, not climate physics. Latitude
    band and this world's own land/ocean mask are properties the diff can
    legitimately compare.

    This is *not* a claim of GCM- or reanalysis-grade accuracy: the
    simplified energy-balance model has known biases relevant here (a
    ~5 K global cold bias vs. real Earth -- see the module docstring's
    observed reference run -- and a tendency for procedurally-generated
    high terrain to push land cells at many latitudes into the cold D
    class via lapse-rate cooling, an effect real alpine climates share but
    this coarse reference does not encode). The threshold
    (``_MIN_REFERENCE_GROUP_AGREEMENT``, see its definition for the
    observed value and the no-skill baseline it must clear) is chosen to
    demand real structural agreement without pretending this coarse setup
    can reproduce a 1-km satellite-era climate raster.
    """
    grid, mask, state = earth_climate
    bands = _load_koppen_reference_bands()
    koppen_by_group = koppen_field_by_group(state)

    matches = sum(
        1
        for cell, group in koppen_by_group.items()
        if group == _reference_group(bands, abs(grid.centroid(cell).lat_deg), mask.is_land(cell))
    )
    agreement = matches / len(koppen_by_group)

    assert agreement > _MIN_REFERENCE_GROUP_AGREEMENT, (
        f"group agreement {agreement:.2%} against coarse reference climatology, "
        f"expected > {_MIN_REFERENCE_GROUP_AGREEMENT:.0%}"
    )
