"""The habitat-medium gate: a categorical filter before the trait bands."""

from __future__ import annotations

from core.biology.medium import (
    AERIAL,
    AQUATIC,
    INTERTIDAL,
    SUBTERRANEAN,
    TERRESTRIAL,
    is_surface_medium,
    medium_allows,
)
from core.hydrology.sea_mask import SeaMask

_LAND = "land"
_SEA = "sea"
_SHORE = "shore"


def _mask() -> SeaMask:
    return SeaMask(
        sea_level_m=0.0,
        ocean={_LAND: False, _SEA: True, _SHORE: False},
        intertidal={_LAND: False, _SEA: False, _SHORE: True},
    )


def test_terrestrial_only_on_land() -> None:
    mask = _mask()
    assert medium_allows(TERRESTRIAL, _LAND, mask)
    assert not medium_allows(TERRESTRIAL, _SEA, mask)


def test_aquatic_only_on_water() -> None:
    mask = _mask()
    assert medium_allows(AQUATIC, _SEA, mask)
    assert not medium_allows(AQUATIC, _LAND, mask)


def test_aerial_flies_over_land_and_water() -> None:
    mask = _mask()
    assert medium_allows(AERIAL, _LAND, mask)
    assert medium_allows(AERIAL, _SEA, mask)


def test_intertidal_only_in_the_tidal_band() -> None:
    mask = _mask()
    assert medium_allows(INTERTIDAL, _SHORE, mask)
    assert not medium_allows(INTERTIDAL, _LAND, mask)


def test_subterranean_is_never_a_surface_medium() -> None:
    mask = _mask()
    assert not medium_allows(SUBTERRANEAN, _LAND, mask)
    assert not is_surface_medium(SUBTERRANEAN)
    assert is_surface_medium(TERRESTRIAL)
