"""Tests for the disturbance mechanism (scripted + emergent share it)."""

from __future__ import annotations

import pytest

from core.hazards.perturb import Disturbance, apply_disturbance, ash_cloud, radial_deltas
from ports.grid import Grid


def _fields() -> dict[str, dict[str, float]]:
    pytest.importorskip("h3")
    from adapters.grid_registry import create_grid

    grid = create_grid("h3", resolution=0)
    return {"temperature_k": dict.fromkeys(grid.cells(), 280.0)}


def _grid() -> Grid:
    from adapters.grid_registry import create_grid

    return create_grid("h3", resolution=0)


def test_radial_deltas_peak_at_center_and_decay() -> None:
    pytest.importorskip("h3")
    grid = _grid()
    center = next(iter(grid.cells()))
    deltas = radial_deltas(grid, center, radius_m=2_000_000.0, peak_delta=-10.0)
    assert deltas[center] == pytest.approx(-10.0)
    neighbor_values = [deltas[n] for n in grid.neighbors(center) if n in deltas]
    assert neighbor_values
    assert all(-10.0 < value < 0.0 for value in neighbor_values)


def test_ash_cloud_cools_a_region_and_only_that_region() -> None:
    pytest.importorskip("h3")
    grid = _grid()
    fields = _fields()
    center = next(iter(grid.cells()))
    disturbance = ash_cloud(grid, center, radius_m=2_000_000.0, cooling_k=5.0)
    assert disturbance.trigger == "scripted:ash_cloud"
    after = apply_disturbance(fields, disturbance)
    assert after["temperature_k"][center] == pytest.approx(275.0)
    untouched = [
        cell
        for cell in after["temperature_k"]
        if cell not in disturbance.field_deltas["temperature_k"]
    ]
    assert all(after["temperature_k"][c] == 280.0 for c in untouched)
    assert fields["temperature_k"][center] == 280.0


def test_disturbances_cannot_invent_state() -> None:
    fields = {"temperature_k": {"c0": 280.0}}
    bad_field = Disturbance("x", "scripted:test", {"unobtainium": {"c0": 1.0}})
    with pytest.raises(KeyError, match="unknown fields"):
        apply_disturbance(fields, bad_field)
    bad_cell = Disturbance("x", "scripted:test", {"temperature_k": {"nope": 1.0}})
    with pytest.raises(KeyError, match="unknown cell"):
        apply_disturbance(fields, bad_cell)
