"""Holdridge life-zone axes: biotemperature, precipitation, PET ratio.

The Holdridge scheme is the science-grounded biome classifier the design
calls for (temperature x precipitation x humidity -> life zone), chosen
over an ad-hoc scheme because it is established and mod-friendly.  This
module owns only the *axes* and the log-scale bucketing into a
(latitudinal belt, humidity province) coordinate; the mapping from that
coordinate to a named biome lives in **content** (``content/biomes``),
so a modder retunes biomes by editing data, not code — which is why this
module contains no biome ids.

Definitions (all SI; classification works on any planet):

* **Biotemperature** — mean surface temperature with sub-freezing and
  super-30 C values clamped, since plant growth neither happens below
  0 C nor speeds up above ~30 C.  Clamping (rather than the textbook
  "set to 0") keeps hot deserts in the tropical belt where they belong.
* **Potential evapotranspiration (PET)** ~ ``biotemperature * 58.93``
  mm/yr (the Holdridge constant).
* **PET ratio** = PET / annual precipitation.  Higher = drier.
"""

from __future__ import annotations

from dataclasses import dataclass

BIOTEMP_MIN_C = 0.0
BIOTEMP_MAX_C = 30.0
PET_CONSTANT_MM_PER_C = 58.93
"""Holdridge PET ~ biotemperature * this (mm per year per deg C)."""

# Latitudinal-belt boundaries by biotemperature (deg C), cold -> hot.
# Six thresholds -> seven belts, index 0 (polar) .. 6 (tropical).
_BELT_THRESHOLDS_C = (1.5, 3.0, 6.0, 12.0, 18.0, 24.0)
BELT_COUNT = len(_BELT_THRESHOLDS_C) + 1

# Humidity-province boundaries by PET ratio, dry -> wet (descending PET).
# Province index 0 (arid) .. 5 (rain); five thresholds -> six provinces.
_PROVINCE_THRESHOLDS_PET = (4.0, 2.0, 1.0, 0.5, 0.25)
PROVINCE_COUNT = len(_PROVINCE_THRESHOLDS_PET) + 1

_MIN_PRECIP_MM = 1.0e-6
"""Floor on precipitation so PET ratio never divides by zero."""


@dataclass(frozen=True)
class LifeZoneCoord:
    """A cell's position on the Holdridge chart.

    ``belt`` is 0 (polar) .. ``BELT_COUNT - 1`` (tropical); ``province``
    is 0 (arid) .. ``PROVINCE_COUNT - 1`` (rain).
    """

    belt: int
    province: int


def biotemperature_c(temperatures_c: tuple[float, ...]) -> float:
    """Return mean biotemperature from period temperatures in Celsius.

    Each value is clamped to ``[0, 30]`` before averaging.  Pass monthly
    or seasonal means; more periods track the growing season better.
    """
    if not temperatures_c:
        msg = "need at least one temperature sample"
        raise ValueError(msg)
    clamped = [min(BIOTEMP_MAX_C, max(BIOTEMP_MIN_C, t)) for t in temperatures_c]
    return sum(clamped) / len(clamped)


def potential_evapotranspiration_mm(biotemperature_c: float) -> float:
    """Return annual PET in mm from biotemperature."""
    return biotemperature_c * PET_CONSTANT_MM_PER_C


def pet_ratio(biotemperature_c: float, annual_precip_mm: float) -> float:
    """Return the PET ratio (PET / precipitation); higher means drier."""
    precip = max(_MIN_PRECIP_MM, annual_precip_mm)
    return potential_evapotranspiration_mm(biotemperature_c) / precip


def belt_index(biotemperature_c: float) -> int:
    """Return the latitudinal-belt index (0 polar .. 6 tropical)."""
    return sum(1 for threshold in _BELT_THRESHOLDS_C if biotemperature_c >= threshold)


def province_index(ratio: float) -> int:
    """Return the humidity-province index (0 arid .. 5 rain)."""
    return sum(1 for threshold in _PROVINCE_THRESHOLDS_PET if ratio < threshold)


def life_zone(biotemperature_c: float, annual_precip_mm: float) -> LifeZoneCoord:
    """Return the Holdridge coordinate for a cell's climate."""
    return LifeZoneCoord(
        belt=belt_index(biotemperature_c),
        province=province_index(pet_ratio(biotemperature_c, annual_precip_mm)),
    )
