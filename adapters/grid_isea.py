"""ISEA/DGGRID adapter for the grid port (equal-area backend).

ISEA (Icosahedral Snyder Equal Area, via the external DGGRID tool) is the
one built-in backend whose cells genuinely have equal area, making it the
reference for verifying that area-weighted simulation math is invariant
to the backend toggle.  We standardize on the **ISEA3H** topology: hex
cells plus 12 pentagons (exactly 5/6 of a hex in area), giving H3-like
adjacency with strictly equal-area hexes.

Wiring DGGRID (an external binary driven through ``dggrid4py``) is part of
the grid milestone; until then this adapter keeps the toggle name and the
full port shape but reports itself unavailable at construction time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ports.grid import CellId, Grid, GridBackendUnavailableError, LatLon

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_UNAVAILABLE_MSG = (
    "grid backend 'isea' is not wired up yet: it requires the external "
    "DGGRID tool (via dggrid4py) and lands with the grid milestone; "
    "toggle 'h3' or 's2' in the meantime"
)


class IseaGrid(Grid):
    """Placeholder ISEA grid: full port shape, construction unavailable."""

    def __init__(self, resolution: int, radius_m: float) -> None:
        """Always raise: the DGGRID integration is not wired up yet."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    @property
    def backend_name(self) -> str:
        """Return the toggle name of this backend."""
        return "isea"

    @property
    def resolution(self) -> int:
        """Return the ISEA aperture level."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    @property
    def radius_m(self) -> float:
        """Return the sphere radius in meters."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    @property
    def cell_count(self) -> int:
        """Return the number of cells at this level."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    def cells(self) -> Iterator[CellId]:
        """Iterate all cells."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    def neighbors(self, cell: CellId) -> Sequence[CellId]:
        """Return edge-adjacent cells (ISEA3H: 6, or 5 at the 12 pentagons)."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    def edge_length_m(self, cell: CellId, neighbor: CellId) -> float:
        """Return the shared-edge length scaled to the planet radius."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    def parent(self, cell: CellId) -> CellId | None:
        """Return the parent cell one aperture level coarser."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    def children(self, cell: CellId) -> Sequence[CellId]:
        """Return the child cells one aperture level finer."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    def area_m2(self, cell: CellId) -> float:
        """Return the (equal) cell area scaled to the planet radius."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    def centroid(self, cell: CellId) -> LatLon:
        """Return the cell centroid in degrees."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)

    def cell_at(self, point: LatLon) -> CellId:
        """Return the cell containing a latitude/longitude point."""
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)


def create(resolution: int, radius_m: float) -> Grid:
    """Create an :class:`IseaGrid`; registry entry point for backend 'isea'."""
    return IseaGrid(resolution, radius_m)
