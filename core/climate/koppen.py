"""Köppen-Geiger climate classification for Earth-calibration verification.

The Köppen-Geiger scheme groups the world's climates into five main classes
(A: tropical, B: dry, C: temperate, D: cold, E: polar) with subdivisions
based on temperature and precipitation seasonality. This module implements
a simplified version approximated from seasonal/annual data the climate
model provides (seasonal temps, annual mean temp, annual precip).

Approximations vs textbook Köppen:
  - Classic Köppen uses monthly data; we derive seasonal extremes from
    quarterly/bi-monthly seasonal fields. This captures the annual range
    (warmest/coldest season) used for temperature class boundaries.
  - Precipitation seasonality is not modeled: we use only annual total,
    not month-to-month patterns. This prevents finer subdivisions (e.g.,
    Csa vs Csb), but main class and winter/summer dominance are derived
    from seasonal temperatures.
  - The "threshold" between dry and wet (B class) uses a simplified
    relationship depending on annual temp.

Classification relies on:
  - Seasonal temperatures from the ClimateState.seasons tuple
  - Annual mean temperature (for reference)
  - Annual total precipitation
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class KoppenClass:
    """A Köppen climate class with group and subdivisions.

    ``group`` is the main class letter (A, B, C, D, or E).
    ``full_code`` is the complete classification (e.g., "Csa", "BWh").
    """

    group: str
    full_code: str


def _seasonality_stats(
    temps_k: tuple[float, ...],
) -> tuple[float, float, float]:
    """Return (warmest_C, coldest_C, annual_mean_C) from seasonal temps."""
    if not temps_k:
        msg = "need at least one seasonal temperature"
        raise ValueError(msg)
    temps_c = [kelvin_to_celsius(t) for t in temps_k]
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


def _classify_group(
    warmest_c: float,
    coldest_c: float,
    annual_mean_c: float,
    annual_precip_mm: float,
) -> KoppenClass:
    """Classify into Köppen group and return a KoppenClass code.

    Main logic (in priority order):
      - E: warmest month < 10 C (polar)
      - B: precipitation below threshold (dry), given temp constraints
      - D: warmest month >= 10 C AND coldest month < -3 C (cold/boreal)
      - C: warmest month >= 10 C AND coldest month in [-3, 18] C (temperate)
      - A: coldest month >= 18 C (tropical)
    """
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
        return KoppenClass("A", "Af")

    # Temperate (C) — fallback for remaining valid cases
    return KoppenClass("C", "Cfb")


def classify_cell(
    temps_k: tuple[float, ...],
    annual_precip_mm: float,
) -> KoppenClass:
    """Classify a single cell's climate into a Köppen class.

    Args:
        temps_k: Seasonal temperatures (K) — tuple from ClimateState.seasons.
        annual_precip_mm: Annual total precipitation (mm/yr).

    Returns:
        A :class:`KoppenClass` with group and full code.
    """
    if not temps_k or annual_precip_mm < 0.0:
        msg = f"invalid input: temps_k={temps_k}, precip={annual_precip_mm}"
        raise ValueError(msg)

    warmest_c, coldest_c, annual_mean_c = _seasonality_stats(temps_k)
    return _classify_group(warmest_c, coldest_c, annual_mean_c, annual_precip_mm)


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

    # Extract seasonal temperatures for each cell.
    for cell in climate.annual_mean_temperature_k:
        temps_k = tuple(season.temperature_k[cell] for season in climate.seasons)
        precip_mm = climate.annual_precipitation_mm_yr[cell]

        result[cell] = classify_cell(temps_k, precip_mm)

    return result


def koppen_field_by_group(
    climate: ClimateState,
) -> dict[CellId, str]:
    """Return the Köppen group letter (A/B/C/D/E) for every cell.

    Convenience function; calls :func:`koppen_field` and extracts the group.
    """
    koppen = koppen_field(climate)
    return {cell: kc.group for cell, kc in koppen.items()}
