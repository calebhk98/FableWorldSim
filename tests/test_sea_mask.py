"""Tests for sea level, land/ocean mask, and the intertidal band (M1.2)."""

from __future__ import annotations

import pytest

from core.hydrology.sea_mask import build_sea_mask
from tests.test_grid_port import FakeGrid

_HEIGHTS = {"c0": -100.0, "c1": -1.0, "c2": 1.0, "c3": 500.0}


def _grid() -> FakeGrid:
    return FakeGrid(0, 1_000.0)


def test_mask_matches_the_target_fraction() -> None:
    mask = build_sea_mask(_grid(), _HEIGHTS, ocean_fraction=0.5)
    assert mask.ocean["c0"] and mask.ocean["c1"]
    assert not mask.ocean["c2"] and not mask.ocean["c3"]
    assert mask.is_land("c3")
    assert mask.sea_level_m == pytest.approx(-1.0)


def test_intertidal_band_follows_tidal_range() -> None:
    mask = build_sea_mask(_grid(), _HEIGHTS, ocean_fraction=0.5, tidal_range_m=6.0)
    assert mask.intertidal["c1"]
    assert mask.intertidal["c2"]
    assert not mask.intertidal["c0"]
    assert not mask.intertidal["c3"]


def test_no_tides_means_no_intertidal_band() -> None:
    mask = build_sea_mask(_grid(), _HEIGHTS, ocean_fraction=0.5, tidal_range_m=0.0)
    assert not any(mask.intertidal.values())


def test_bone_dry_world_is_all_land() -> None:
    mask = build_sea_mask(_grid(), _HEIGHTS, ocean_fraction=0.0)
    assert not any(mask.ocean.values())
