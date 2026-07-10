"""Tests for the Holdridge life-zone axes (biotemperature, PET, buckets)."""

from __future__ import annotations

import pytest

from core.biology.holdridge import (
    BELT_COUNT,
    PROVINCE_COUNT,
    belt_index,
    biotemperature_c,
    pet_ratio,
    potential_evapotranspiration_mm,
    province_index,
)


def test_biotemperature_clamps_below_freezing_and_above_thirty() -> None:
    # -10 -> 0, 5 -> 5, 20 -> 20, 40 -> 30; mean of [0, 5, 20, 30] = 13.75.
    assert biotemperature_c((-10.0, 5.0, 20.0, 40.0)) == pytest.approx(13.75)
    # Hot deserts stay hot (clamp to 30, not to 0) so they land tropical.
    assert biotemperature_c((35.0, 35.0)) == pytest.approx(30.0)
    with pytest.raises(ValueError, match="at least one"):
        biotemperature_c(())


def test_pet_scales_with_biotemperature() -> None:
    assert potential_evapotranspiration_mm(10.0) == pytest.approx(589.3)
    # ratio = PET / precip; 25 C over 2000 mm is humid (ratio < 1).
    assert pet_ratio(25.0, 2000.0) == pytest.approx(0.7366, rel=1e-3)
    # Bone-dry precipitation does not divide by zero.
    assert pet_ratio(20.0, 0.0) > 1_000.0


def test_belt_index_spans_polar_to_tropical() -> None:
    assert belt_index(0.5) == 0  # polar
    assert belt_index(2.0) == 1  # subpolar
    assert belt_index(10.0) == 3  # cool temperate
    assert belt_index(30.0) == BELT_COUNT - 1  # tropical
    # Boundaries are inclusive at the lower edge.
    assert belt_index(6.0) == 3
    assert belt_index(5.999) == 2


def test_province_index_spans_arid_to_rain() -> None:
    assert province_index(5.0) == 0  # arid (very dry)
    assert province_index(3.0) == 1  # semiarid
    assert province_index(0.7366) == 3  # humid / moist
    assert province_index(0.1) == PROVINCE_COUNT - 1  # rain
    # Boundaries: ratio just under a threshold moves one step wetter.
    assert province_index(1.0) == 2
    assert province_index(0.999) == 3
