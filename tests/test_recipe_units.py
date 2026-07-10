"""Tests for the world recipe and the dimensional-analysis guard."""

from __future__ import annotations

import pytest

from core.sim.recipe import WorldRecipe
from core.sim.units import (
    JOULE,
    KELVIN,
    SQUARE_METER,
    WATT,
    WATT_PER_M2,
    Quantity,
    UnitMismatchError,
    kelvin_to_celsius,
    kelvin_to_fahrenheit,
)
from core.version import ENGINE_VERSION, engine_satisfies, parse_version


def test_recipe_round_trips_through_json() -> None:
    recipe = WorldRecipe(
        world_name="testworld",
        seed=42,
        grid_backend="h3",
        grid_resolution=3,
        settings={"fidelity": {"level": "fast"}},
        content_pins={"base": "0.1.0", "coolmod": "1.2.0"},
    )
    again = WorldRecipe.from_json(recipe.to_json())
    assert again == recipe
    assert again.engine_version == ENGINE_VERSION


def test_units_block_the_unit_bug_class() -> None:
    flux = Quantity(340.0, WATT_PER_M2)
    energy = Quantity(1000.0, JOULE)
    with pytest.raises(UnitMismatchError):
        _ = flux + energy
    with pytest.raises(UnitMismatchError):
        _ = energy - Quantity(1.0, KELVIN)


def test_units_compose_through_multiplication() -> None:
    flux = Quantity(340.0, WATT_PER_M2)
    area = Quantity(2.0, SQUARE_METER)
    power = flux * area
    assert power.unit == WATT
    assert power.value == pytest.approx(680.0)
    assert (power / area).unit == WATT_PER_M2


def test_display_conversions_are_edge_only() -> None:
    assert kelvin_to_celsius(273.15) == pytest.approx(0.0)
    assert kelvin_to_fahrenheit(300.0) == pytest.approx(80.33)


def test_version_axis_parsing() -> None:
    assert parse_version("1.2.3") == (1, 2, 3)
    assert engine_satisfies("0.0.1")
    assert not engine_satisfies("999.0.0")
    with pytest.raises(ValueError, match="SemVer"):
        parse_version("not-a-version")
