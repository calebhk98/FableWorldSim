"""Tests for topography: generate, load, and edit behind one port."""

from __future__ import annotations

import pytest

from adapters.dem_latlon import LatLonGridDem
from adapters.rng_seeded import SeededRng
from core.chronicle.log import Chronicle
from core.topography.edit import (
    EDIT_EVENT_KIND,
    EditRejectedError,
    EditRequest,
    apply_edit,
    preview_edit,
)
from core.topography.editable import EditableTopography
from core.topography.procedural import ProceduralTopography
from tests.test_grid_port import FakeGrid

_SEED = 20260710


def _fake_grid() -> FakeGrid:
    return FakeGrid(0, 6_371_000.0)


def test_procedural_terrain_is_deterministic_per_seed() -> None:
    grid = _fake_grid()
    first = ProceduralTopography(SeededRng(_SEED)).heights(grid)
    again = ProceduralTopography(SeededRng(_SEED)).heights(grid)
    other = ProceduralTopography(SeededRng(_SEED + 1)).heights(grid)
    assert first == again
    assert first != other


def test_procedural_terrain_has_relief() -> None:
    pytest.importorskip("h3")
    from adapters.grid_registry import create_grid

    grid = create_grid("h3", resolution=1)
    heights = ProceduralTopography(SeededRng(_SEED)).heights(grid)
    assert len(heights) == grid.cell_count
    assert max(heights.values()) - min(heights.values()) > 1_000.0


def test_dem_raster_samples_by_latitude_and_longitude() -> None:
    # 3 latitude bands (N pole / equator / S pole), 4 longitude columns.
    dem = LatLonGridDem(
        [
            [100.0, 100.0, 100.0, 100.0],
            [0.0, 10.0, 20.0, 30.0],
            [-50.0, -50.0, -50.0, -50.0],
        ]
    )
    assert dem.sample(85.0, 0.0) == 100.0
    assert dem.sample(-85.0, 0.0) == -50.0
    assert dem.sample(0.0, -170.0) == 0.0
    assert dem.sample(0.0, 170.0) == 30.0
    grid = _fake_grid()
    assert set(dem.heights(grid)) == set(grid.cells())


def test_dem_rejects_ragged_rasters() -> None:
    with pytest.raises(ValueError, match="same length"):
        LatLonGridDem([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError, match="at least one"):
        LatLonGridDem([])


def test_edit_preview_shows_without_applying() -> None:
    heights = {"c0": 100.0, "c1": 200.0, "c2": 0.0, "c3": 50.0}
    request = EditRequest(deltas_m={"c0": 50.0, "c1": -25.0}, editor="player:ada")
    preview = preview_edit(heights, request)
    assert preview.rejected is None
    assert preview.cell_count == 2
    assert preview.changes["c0"] == (100.0, 150.0)
    assert heights["c0"] == 100.0


def test_edit_applies_and_chronicles_who_did_it() -> None:
    heights = {"c0": 100.0, "c1": 200.0}
    chronicle = Chronicle()
    request = EditRequest(deltas_m={"c0": 50.0}, editor="script:terraform", reason="test")
    edited = apply_edit(heights, request, chronicle, tick=7)
    assert edited["c0"] == 150.0
    assert heights["c0"] == 100.0
    events = list(chronicle.events(kind=EDIT_EVENT_KIND))
    assert len(events) == 1
    assert events[0].subject == "script:terraform"
    assert events[0].tick == 7


def test_guardrail_rejects_oversized_edits() -> None:
    heights = {"c0": 100.0}
    chronicle = Chronicle()
    huge = EditRequest(deltas_m={"c0": 1_000.0}, editor="script:chaos")
    assert preview_edit(heights, huge).rejected is not None
    with pytest.raises(EditRejectedError, match="guardrail"):
        apply_edit(heights, huge, chronicle, tick=1)
    assert len(chronicle) == 0
    unknown = EditRequest(deltas_m={"nope": 1.0}, editor="script:lost")
    with pytest.raises(EditRejectedError, match="unknown"):
        apply_edit(heights, unknown, chronicle, tick=1)


def test_editable_topography_is_a_source() -> None:
    base = ProceduralTopography(SeededRng(_SEED))
    editable = EditableTopography(base)
    grid = _fake_grid()
    heights = editable.heights(grid)
    assert len(heights) == grid.cell_count
    assert max(heights.values()) - min(heights.values()) > 0.0


def test_editable_applies_edits_to_heights() -> None:
    base = ProceduralTopography(SeededRng(_SEED))
    editable = EditableTopography(base)
    grid = _fake_grid()
    base_heights = base.heights(grid)
    cells = list(grid.cells())
    c0 = cells[0]
    original = base_heights[c0]
    request = EditRequest(deltas_m={c0: 50.0}, editor="player:test")
    editable.apply_edit(base_heights, request, tick=1)
    edited_heights = editable.heights(grid)
    assert edited_heights[c0] == original + 50.0


def test_editable_preview_doesnt_mutate() -> None:
    base = ProceduralTopography(SeededRng(_SEED))
    editable = EditableTopography(base)
    grid = _fake_grid()
    heights = base.heights(grid)
    cells = list(grid.cells())
    c0 = cells[0]
    original = heights[c0]
    request = EditRequest(deltas_m={c0: 50.0}, editor="player:test")
    preview = editable.preview_edit(heights, request)
    assert preview["rejected"] is None
    assert preview["changes"][c0] == (original, original + 50.0)
    edited_heights = editable.heights(grid)
    assert edited_heights[c0] == original


def test_editable_respects_magnitude_cap() -> None:
    base = ProceduralTopography(SeededRng(_SEED))
    editable = EditableTopography(base)
    grid = _fake_grid()
    heights = base.heights(grid)
    cells = list(grid.cells())
    c0 = cells[0]
    huge = EditRequest(deltas_m={c0: 1_000.0}, editor="script:chaos")
    preview = editable.preview_edit(heights, huge)
    assert preview["rejected"] is not None
    with pytest.raises(EditRejectedError, match="guardrail"):
        editable.apply_edit(heights, huge, tick=1)


def test_editable_chronicles_edits_when_chronicle_provided() -> None:
    base = ProceduralTopography(SeededRng(_SEED))
    chronicle = Chronicle()
    editable = EditableTopography(base, chronicle=chronicle)
    grid = _fake_grid()
    heights = base.heights(grid)
    cells = list(grid.cells())
    c0 = cells[0]
    request = EditRequest(deltas_m={c0: 50.0}, editor="script:terraform", reason="testing")
    editable.apply_edit(heights, request, tick=7)
    events = list(chronicle.events(kind=EDIT_EVENT_KIND))
    assert len(events) == 1
    assert events[0].subject == "script:terraform"
    assert events[0].tick == 7


def test_editable_accumulates_multiple_edits() -> None:
    base = ProceduralTopography(SeededRng(_SEED))
    editable = EditableTopography(base)
    grid = _fake_grid()
    base_heights = base.heights(grid)
    cells = list(grid.cells())
    c0 = cells[0]
    original = base_heights[c0]
    req1 = EditRequest(deltas_m={c0: 50.0}, editor="player:test")
    editable.apply_edit(base_heights, req1, tick=1)
    req2 = EditRequest(deltas_m={c0: 30.0}, editor="player:test")
    editable.apply_edit(base_heights, req2, tick=2)
    edited_heights = editable.heights(grid)
    assert edited_heights[c0] == original + 80.0
