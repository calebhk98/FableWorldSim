"""Köppen-Geiger climate classification for Earth-calibration verification.

The Köppen-Geiger scheme groups the world's climates into five main classes
(A: tropical, B: dry, C: temperate, D: cold, E: polar) with subdivisions
based on temperature and precipitation seasonality. This module implements
a simplified version approximated from seasonal/annual data the climate
model provides (seasonal temps, seasonal precip, annual mean temp, annual
precip).

Approximations vs textbook Köppen:
  - Classic Köppen uses monthly data; we derive seasonal extremes (and,
    for the A/C subtypes below, seasonal precipitation) from
    quarterly/bi-monthly seasonal fields. This captures the annual range
    (warmest/coldest season) used for temperature class boundaries and
    the wet/dry season contrast used for precipitation-seasonality
    subtypes, but at coarser time resolution than the textbook monthly
    normals.
  - The "threshold" between dry and wet (B class) uses a simplified
    relationship depending on annual temp.
  - A-group subtypes (Af/Am/Aw) use the standard Af "driest month" and
    Am monsoon-deficit formulas, but "driest month" is approximated as
    (driest season's annualized rate) / 12 — a quarterly stand-in for a
    true monthly minimum, which can miss short, sharp dry spells a
    monthly record would catch.
  - C-group subtypes (Csa/Csb/Cwa/Cwb/Cfa/Cfb) use the season with the
    highest/lowest temperature as a proxy for "summer"/"winter" (correct
    regardless of hemisphere) and the classic wet/dry-season precip
    ratios, but again at quarterly rather than monthly resolution, and
    the "b" vs "a" warm-season letter uses a season-count analog of the
    textbook "4 months >= 10 C" rule (see ``_c_warmth_letter``).
  - When seasonal precipitation is not supplied (e.g. direct unit-test
    calls that only pass seasonal temperatures), A and C fall back to
    the coarsest possible subtype (Af, Cfb) since there is no seasonality
    signal to derive one from.

Classification relies on:
  - Seasonal temperatures from the ClimateState.seasons tuple
  - Seasonal precipitation from the ClimateState.seasons tuple (optional;
    enables A/C subtype derivation)
  - Annual mean temperature (for reference)
  - Annual total precipitation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.sim.units import kelvin_to_celsius

if TYPE_CHECKING:
    from core.climate.model import ClimateState
    from ports.grid import CellId

# Köppen temperature thresholds (°C)
_WARMEST_POLAR_THRESHOLD_C = 10.0
_WARMEST_PERPETUAL_FROST_THRESHOLD_C = 0.0
_COLDEST_COLD_CLASS_THRESHOLD_C = -3.0
_COLDEST_VERY_COLD_THRESHOLD_C = -38.0
_TROPICAL_COLDEST_MONTH_THRESHOLD_C = 18.0

# A-group (tropical) subtype thresholds.
_AF_DRIEST_MONTH_MM = 60.0
"""Af (rainforest) requires every month >= this many mm."""
_MONTHS_PER_YEAR = 12.0

# C-group (temperate) subtype thresholds.
_C_HOT_SUMMER_THRESHOLD_C = 22.0
"""'a' (hot summer) requires warmest month >= this; else 'b'/'c'."""
_C_MILD_MONTH_FRACTION_FOR_B = 4.0 / 12.0
"""'b' requires >= ~4 of 12 months >= 10 C; approximated per season count."""
_DRY_SUMMER_MAX_MONTH_MM = 40.0
"""'s' (dry summer) requires the driest summer month below this."""
_DRY_SUMMER_WET_WINTER_RATIO = 3.0
"""'s' also requires the wettest winter month >= this many times the driest summer month."""
_DRY_WINTER_WET_SUMMER_RATIO = 10.0
"""'w' (dry winter) requires the wettest summer month >= this many times the driest winter one."""
_MIN_SEASONS_FOR_PRECIP_LETTER = 2
"""Need at least two distinct seasons to compare a "summer" against a "winter"."""


@dataclass(frozen=True)
class KoppenClass:
    """A Köppen climate class with group and subdivisions.

    ``group`` is the main class letter (A, B, C, D, or E).
    ``full_code`` is the complete classification (e.g., "Csa", "BWh").
    """

    group: str
    full_code: str


def _seasonality_stats(
    temps_c: list[float],
) -> tuple[float, float, float]:
    """Return (warmest_C, coldest_C, annual_mean_C) from per-season Celsius temps."""
    return max(temps_c), min(temps_c), sum(temps_c) / len(temps_c)


def _dry_threshold_mm(
    annual_mean_c: float,
) -> float:
    """Return the annual precipitation threshold (mm/yr) for B vs non-B class.

    Simplification: If most precip falls in summer (derived from seasonal
    temp structure), use a lower threshold; if winter-dominant, use higher.
    For now, use the temperate-latitude threshold as default.
    """
    # Classic Köppen: if precip is concentrated in summer, threshold ~2*T;
    # if spread/winter-dominant, ~2*T + 28 (where T is annual mean in C).
    # Default to summer-dominant approximation: 2*T.
    return max(0.0, 2.0 * annual_mean_c)


def _tropical_subtype(
    annual_precip_mm: float,
    seasonal_precip_mm_yr: tuple[float, ...] | None,
) -> str:
    """Return Af/Am/Aw from the driest-season precip and the Am monsoon test.

    Standard formulas (with "month" approximated by "season" — see the
    module docstring):
      - Af: driest month >= 60 mm.
      - Am: driest month below 60 mm but the annual total is high enough
        to satisfy ``Pdry >= 100 - MAP / 25`` (MAP = mean annual precip,
        both in mm) — a short dry season the wet months compensate for.
      - Aw: neither condition holds (a real dry-winter savanna season).
    """
    if seasonal_precip_mm_yr is None:
        return "Af"
    driest_season_mm_yr = min(seasonal_precip_mm_yr)
    driest_month_mm = driest_season_mm_yr / _MONTHS_PER_YEAR
    if driest_month_mm >= _AF_DRIEST_MONTH_MM:
        return "Af"
    monsoon_threshold_mm = 100.0 - annual_precip_mm / 25.0
    if driest_month_mm >= monsoon_threshold_mm:
        return "Am"
    return "Aw"


def _c_precip_letter(
    temps_c: list[float],
    seasonal_precip_mm_yr: tuple[float, ...] | None,
) -> str:
    """Return 's' (dry summer), 'w' (dry winter), or 'f' (no dry season).

    Uses the hottest/coldest season as a hemisphere-agnostic proxy for
    "summer"/"winter" and the classic wet:dry ratio tests.
    """
    if seasonal_precip_mm_yr is None or len(seasonal_precip_mm_yr) < _MIN_SEASONS_FOR_PRECIP_LETTER:
        return "f"
    warm_idx = temps_c.index(max(temps_c))
    cold_idx = temps_c.index(min(temps_c))
    summer_mm_yr = seasonal_precip_mm_yr[warm_idx]
    winter_mm_yr = seasonal_precip_mm_yr[cold_idx]
    driest_summer_month_mm = summer_mm_yr / _MONTHS_PER_YEAR
    if (
        summer_mm_yr * _DRY_SUMMER_WET_WINTER_RATIO < winter_mm_yr
        and driest_summer_month_mm < _DRY_SUMMER_MAX_MONTH_MM
    ):
        return "s"
    if winter_mm_yr * _DRY_WINTER_WET_SUMMER_RATIO < summer_mm_yr:
        return "w"
    return "f"


def _c_warmth_letter(warmest_c: float, temps_c: list[float]) -> str:
    """Return 'a' (hot summer), 'b' (warm summer), or 'c' (cool/short summer).

    'a' requires a warmest month >= 22 C. Otherwise the textbook test is
    "at least 4 of 12 months >= 10 C" for 'b'; with only a handful of
    seasons we scale that fraction (~1/3 of the year) to however many
    seasons this run has, rounding up so 'b' still means "most of the
    year is mild", not "barely one season touches it".
    """
    if warmest_c >= _C_HOT_SUMMER_THRESHOLD_C:
        return "a"
    mild_seasons = sum(1 for t in temps_c if t >= _WARMEST_POLAR_THRESHOLD_C)
    threshold = max(1, math.ceil(len(temps_c) * _C_MILD_MONTH_FRACTION_FOR_B))
    return "b" if mild_seasons >= threshold else "c"


def _temperate_subtype(
    warmest_c: float,
    temps_c: list[float],
    seasonal_precip_mm_yr: tuple[float, ...] | None,
) -> str:
    """Return the full Cxy code: precip letter + warmth letter."""
    if seasonal_precip_mm_yr is None:
        return "Cfb"
    precip_letter = _c_precip_letter(temps_c, seasonal_precip_mm_yr)
    warmth_letter = _c_warmth_letter(warmest_c, temps_c)
    return f"C{precip_letter}{warmth_letter}"


def _classify_group(
    temps_c: list[float],
    annual_precip_mm: float,
    seasonal_precip_mm_yr: tuple[float, ...] | None,
) -> KoppenClass:
    """Classify into Köppen group and return a KoppenClass code.

    Main logic (in priority order):
      - E: warmest month < 10 C (polar)
      - B: precipitation below threshold (dry), given temp constraints
      - D: warmest month >= 10 C AND coldest month < -3 C (cold/boreal)
      - C: warmest month >= 10 C AND coldest month in [-3, 18] C (temperate)
      - A: coldest month >= 18 C (tropical)
    """
    warmest_c, coldest_c, annual_mean_c = _seasonality_stats(temps_c)

    # Polar (E) — checked first since it overrides all others
    if warmest_c < _WARMEST_POLAR_THRESHOLD_C:
        code = "EF" if warmest_c < _WARMEST_PERPETUAL_FROST_THRESHOLD_C else "ET"
        return KoppenClass("E", code)

    # Dry (B) — checked early since low precip overrides temperature
    threshold = _dry_threshold_mm(annual_mean_c)
    if annual_precip_mm < threshold:
        code = "BWh" if annual_mean_c > _TROPICAL_COLDEST_MONTH_THRESHOLD_C else "BWk"
        return KoppenClass("B", code)

    # Cold (D)
    if coldest_c < _COLDEST_COLD_CLASS_THRESHOLD_C:
        code = "Dfd" if coldest_c < _COLDEST_VERY_COLD_THRESHOLD_C else "Dfc"
        return KoppenClass("D", code)

    # Tropical (A)
    if coldest_c >= _TROPICAL_COLDEST_MONTH_THRESHOLD_C:
        return KoppenClass("A", _tropical_subtype(annual_precip_mm, seasonal_precip_mm_yr))

    # Temperate (C) — fallback for remaining valid cases
    return KoppenClass("C", _temperate_subtype(warmest_c, temps_c, seasonal_precip_mm_yr))


def classify_cell(
    temps_k: tuple[float, ...],
    annual_precip_mm: float,
    seasonal_precip_mm_yr: tuple[float, ...] | None = None,
) -> KoppenClass:
    """Classify a single cell's climate into a Köppen class.

    Args:
        temps_k: Seasonal temperatures (K) — tuple from ClimateState.seasons.
        annual_precip_mm: Annual total precipitation (mm/yr).
        seasonal_precip_mm_yr: Optional per-season precipitation rates
            (mm/yr, same order/length as ``temps_k``) — tuple from
            ``ClimateState.seasons``. When given, enables real A-group
            (Af/Am/Aw) and C-group (Csa/Csb/Cwa/Cwb/Cfa/Cfb) subtype
            derivation instead of the coarsest fallback; see the module
            docstring for the approximations involved.

    Returns:
        A :class:`KoppenClass` with group and full code.
    """
    if not temps_k or annual_precip_mm < 0.0:
        msg = f"invalid input: temps_k={temps_k}, precip={annual_precip_mm}"
        raise ValueError(msg)
    if seasonal_precip_mm_yr is not None and len(seasonal_precip_mm_yr) != len(temps_k):
        msg = (
            f"seasonal_precip_mm_yr length {len(seasonal_precip_mm_yr)} "
            f"must match temps_k length {len(temps_k)}"
        )
        raise ValueError(msg)

    temps_c = [kelvin_to_celsius(t) for t in temps_k]
    return _classify_group(temps_c, annual_precip_mm, seasonal_precip_mm_yr)


def koppen_field(
    climate: ClimateState,
) -> dict[CellId, KoppenClass]:
    """Return the Köppen class for every cell (including ocean).

    Args:
        climate: The full climate state with seasonal and annual fields.

    Returns:
        A mapping from cell ID to its Köppen classification.
    """
    result: dict[CellId, KoppenClass] = {}

    # Extract seasonal temperatures and precip for each cell.
    for cell in climate.annual_mean_temperature_k:
        temps_k = tuple(season.temperature_k[cell] for season in climate.seasons)
        seasonal_precip = tuple(season.precipitation_mm_yr[cell] for season in climate.seasons)
        precip_mm = climate.annual_precipitation_mm_yr[cell]

        result[cell] = classify_cell(temps_k, precip_mm, seasonal_precip)

    return result


def koppen_field_by_group(
    climate: ClimateState,
) -> dict[CellId, str]:
    """Return the Köppen group letter (A/B/C/D/E) for every cell.

    Convenience function; calls :func:`koppen_field` and extracts the group.
    """
    koppen = koppen_field(climate)
    return {cell: kc.group for cell, kc in koppen.items()}
