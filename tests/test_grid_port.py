"""Tests for the grid port contract and the backend toggle registry."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from adapters.grid_registry import (
    available_backends,
    create_grid,
    register_grid_backend,
)
from ports.grid import CellId, Grid, GridBackendUnavailableError, LatLon

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_FAKE_CELLS = ("c0", "c1", "c2", "c3")


class FakeGrid(Grid):
    """A four-cell tetrahedron-style grid used to exercise the port."""

    def __init__(self, resolution: int, radius_m: float) -> None:
        """Store the toggle parameters."""
        self._resolution = resolution
        self._radius_m = radius_m

    @property
    def backend_name(self) -> str:
        """Return the toggle name of this backend."""
        return "fake"

    @property
    def resolution(self) -> int:
        """Return the configured resolution."""
        return self._resolution

    @property
    def radius_m(self) -> float:
        """Return the sphere radius in meters."""
        return self._radius_m

    @property
    def cell_count(self) -> int:
        """Return the number of cells."""
        return len(_FAKE_CELLS)

    def cells(self) -> Iterator[CellId]:
        """Iterate the four cells."""
        yield from _FAKE_CELLS

    def neighbors(self, cell: CellId) -> Sequence[CellId]:
        """Return every other cell (tetrahedron adjacency)."""
        return tuple(other for other in _FAKE_CELLS if other != cell)

    def edge_length_m(self, cell: CellId, neighbor: CellId) -> float:
        """Return a symmetric constant edge length (radius-scaled)."""
        if neighbor not in self.neighbors(cell):
            msg = f"cells {cell} and {neighbor} are not adjacent"
            raise ValueError(msg)
        return self._radius_m

    def parent(self, cell: CellId) -> CellId | None:
        """Return None: the fake grid has a single resolution."""
        return None

    def children(self, cell: CellId) -> Sequence[CellId]:
        """Return no children: the fake grid has a single resolution."""
        return ()

    def area_m2(self, cell: CellId) -> float:
        """Return an equal share of the sphere's surface."""
        return 4.0 * math.pi * self._radius_m**2 / len(_FAKE_CELLS)

    def centroid(self, cell: CellId) -> LatLon:
        """Return an arbitrary but distinct centroid per cell."""
        index = _FAKE_CELLS.index(cell)
        return LatLon(lat_deg=float(-45 + 30 * index), lon_deg=float(90 * index))

    def cell_at(self, point: LatLon) -> CellId:
        """Return the first cell (adequate for port tests)."""
        return _FAKE_CELLS[0]

    def boundary(self, cell: CellId) -> Sequence[LatLon]:
        """Return a small triangle of vertices straddling the centroid."""
        center = self.centroid(cell)
        offsets = ((0.1, 0.0), (-0.05, 0.1), (-0.05, -0.1))
        return tuple(
            LatLon(lat_deg=center.lat_deg + dlat, lon_deg=center.lon_deg + dlon)
            for dlat, dlon in offsets
        )


def _fake_factory(resolution: int, radius_m: float) -> Grid:
    """Build a FakeGrid; registered under the 'fake' toggle name."""
    return FakeGrid(resolution, radius_m)


def test_registry_toggle_builds_registered_backend() -> None:
    """A registered backend is constructed by its toggle name."""
    register_grid_backend("fake", _fake_factory)
    grid = create_grid("fake", resolution=2, radius_m=1_000.0)
    assert grid.backend_name == "fake"
    assert grid.resolution == 2
    assert grid.radius_m == 1_000.0


def test_registry_lists_builtin_and_registered_backends() -> None:
    """The three built-in DGGS toggles and runtime additions are listed."""
    register_grid_backend("fake", _fake_factory)
    names = available_backends()
    for expected in ("h3", "s2", "isea", "fake"):
        assert expected in names


def test_registry_rejects_unknown_backend() -> None:
    """An unknown toggle name raises with the known names listed."""
    with pytest.raises(ValueError, match="unknown grid backend"):
        create_grid("square-lattice", resolution=1)


def test_isea_names_the_missing_binary_when_unavailable() -> None:
    """'isea' fails loudly and namedly when the DGGRID binary is missing.

    This only certifies the graceful-degradation path for environments
    without a ``dggrid`` executable (this repo's dev/CI extras deliberately
    don't guarantee one — see ``pyproject.toml``'s ``grid-isea`` extra); it
    is not ISEA coverage. Real backend behavior is exercised in
    ``test_grid_backends.py`` and ``test_conservation.py``, which run
    whenever ``dggrid`` actually is on ``PATH``.
    """
    try:
        create_grid("isea", resolution=1)
    except GridBackendUnavailableError as exc:
        assert "dggrid" in str(exc).lower()
    else:
        pytest.skip("a working 'dggrid' binary is installed in this environment")


def test_total_area_matches_sphere() -> None:
    """Cell areas sum to the full sphere surface for the planet radius."""
    radius_m = 2_500.0
    grid = FakeGrid(0, radius_m)
    expected = 4.0 * math.pi * radius_m**2
    assert grid.total_area_m2() == pytest.approx(expected)


def test_latlng_is_the_centroid_alias() -> None:
    """The doc's latlng(cell) accessor mirrors centroid(cell)."""
    grid = FakeGrid(0, 1_000.0)
    cell = next(iter(grid.cells()))
    assert grid.latlng(cell) == grid.centroid(cell)


def test_edge_length_requires_adjacency() -> None:
    """Asking for the shared edge of non-neighbors raises."""
    grid = FakeGrid(0, 1_000.0)
    with pytest.raises(ValueError, match="not adjacent"):
        grid.edge_length_m("c0", "c0")


def test_boundary_is_a_ring_of_at_least_three_vertices_near_the_centroid() -> None:
    """Every cell's boundary has >=3 vertices, each close to its centroid."""
    grid = FakeGrid(0, 1_000.0)
    for cell in grid.cells():
        boundary = grid.boundary(cell)
        assert len(boundary) >= 3
        centroid = grid.centroid(cell)
        for vertex in boundary:
            assert abs(vertex.lat_deg - centroid.lat_deg) < 1.0
            assert abs(vertex.lon_deg - centroid.lon_deg) < 1.0
