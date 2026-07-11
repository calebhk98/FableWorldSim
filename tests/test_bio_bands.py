"""Graded comfort/tolerance response with intra-population spread."""

from __future__ import annotations

import pytest

from core.biology.bands import EnvBand


def _band(spread: float = 0.0) -> EnvBand:
    return EnvBand(
        comfort_min=10.0, comfort_max=20.0, tolerance_min=0.0, tolerance_max=30.0, spread=spread
    )


def test_comfort_is_full_and_beyond_tolerance_is_zero() -> None:
    band = _band()
    assert band.response(15.0) == 1.0
    assert band.response(10.0) == 1.0
    assert band.response(20.0) == 1.0
    assert band.response(-5.0) == 0.0
    assert band.response(35.0) == 0.0


def test_tolerance_edge_is_graded_between_zero_and_one() -> None:
    band = _band()
    midpoint = band.response(5.0)
    assert 0.0 < midpoint < 1.0
    assert band.response(5.0) == pytest.approx(0.5)


def test_spread_lets_some_individuals_survive_past_the_tolerance_edge() -> None:
    just_past_tolerance = -2.0
    sharp = _band(spread=0.0).response(just_past_tolerance)
    soft = _band(spread=4.0).response(just_past_tolerance)
    assert sharp == 0.0
    assert soft > 0.0


def test_inverted_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance_min"):
        EnvBand(comfort_min=20.0, comfort_max=10.0, tolerance_min=0.0, tolerance_max=30.0)


def test_non_nested_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance_min"):
        EnvBand(comfort_min=5.0, comfort_max=20.0, tolerance_min=10.0, tolerance_max=30.0)
