"""Grid port: the sphere-tessellation contract every simulation layer uses.

The world surface is a Discrete Global Grid System (DGGS) — cells that
tessellate a true sphere — never a flat lat/lon square lattice.  Concrete
backends (H3, S2, ISEA/DGGRID) live in ``adapters`` and are selected by a
single toggle; simulation code depends only on this interface.

Because H3 and S2 cells are *not* equal-area, all simulation math must be
area-weighted / per-unit-area (see ``core.grid.area_weighted``) so results
stay invariant to the backend toggle and to cell-size variation within a
backend.

Toggle semantics: backends' cells are not spatially congruent, so the
backend is chosen at **world-creation time**.  Moving an existing world to
another backend is an explicit **reprojection** (resample every field; see
``core.grid.reproject``), never a silent switch.

Contract — variable degree: neighbor count is topology-dependent (H3 and
ISEA3H hexes have 6 but their 12 pentagons have 5; S2 quads have 4), so
flux/diffusion math must iterate ``neighbors(cell)`` and weight by
``edge_length_m``/``area_m2`` — never hardcode a degree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

CellId = str
"""Opaque, backend-specific cell identifier (stable within one backend)."""

EARTH_MEAN_RADIUS_M = 6_371_008.8
"""IUGG mean Earth radius in meters, the default sphere radius for grids."""


class LatLon(NamedTuple):
    """A geodetic point on the sphere, in degrees."""

    lat_deg: float
    lon_deg: float


class GridBackendUnavailableError(RuntimeError):
    """Raised when a known grid backend cannot run in this environment.

    Typically the backend's optional dependency (e.g. ``h3``, ``s2sphere``,
    or the external DGGRID tool) is not installed.
    """


class Grid(ABC):
    """A tessellation of a sphere of radius ``radius_m`` into cells.

    Implementations wrap one DGGS backend at one fixed resolution.  The
    meaning of ``resolution`` is backend-specific (H3 res, S2 level, ISEA
    aperture level); simulation code must never interpret it — it only
    enumerates cells, walks adjacency, and weights by per-cell area.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the toggle name of the backing DGGS (e.g. ``"h3"``)."""

    @property
    @abstractmethod
    def resolution(self) -> int:
        """Return the backend-specific resolution this grid was built at."""

    @property
    @abstractmethod
    def radius_m(self) -> float:
        """Return the sphere radius in meters used for areas/distances."""

    @property
    @abstractmethod
    def cell_count(self) -> int:
        """Return the total number of cells tiling the sphere."""

    @abstractmethod
    def cells(self) -> Iterator[CellId]:
        """Iterate over every cell id on the sphere, in a stable order."""

    @abstractmethod
    def neighbors(self, cell: CellId) -> Sequence[CellId]:
        """Return the ids of cells sharing an edge with ``cell``.

        The count varies by backend and cell (e.g. 6 for H3 hexes but 5
        for its 12 pentagons); simulation code must not assume a fixed
        neighbor count.
        """

    @abstractmethod
    def area_m2(self, cell: CellId) -> float:
        """Return the surface area of ``cell`` in square meters.

        This is the weight for all per-cell aggregation; it already
        accounts for ``radius_m``.
        """

    @abstractmethod
    def edge_length_m(self, cell: CellId, neighbor: CellId) -> float:
        """Return the length of the edge shared by two adjacent cells.

        Fluxes between cells (heat, moisture, migration) cross this edge,
        so flow math weights by edge length — never by a per-cell
        constant.  Raises ``ValueError`` when the cells are not adjacent.
        """

    @abstractmethod
    def centroid(self, cell: CellId) -> LatLon:
        """Return the centroid of ``cell`` as latitude/longitude degrees."""

    @abstractmethod
    def boundary(self, cell: CellId) -> Sequence[LatLon]:
        """Return ``cell``'s boundary vertices, in winding order.

        The closing edge (last vertex back to the first) is implicit --
        the sequence does not repeat the first point at the end. Vertex
        count is topology-dependent (6 for H3/ISEA hexes, 5 for their 12
        pentagons, 4 for S2 quads), so callers must not assume a fixed
        count, mirroring the variable-degree contract of ``neighbors``.
        """

    @abstractmethod
    def cell_at(self, point: LatLon) -> CellId:
        """Return the id of the cell containing ``point``."""

    @abstractmethod
    def parent(self, cell: CellId) -> CellId | None:
        """Return the containing cell one resolution coarser (LOD).

        Returns ``None`` at the coarsest resolution.
        """

    @abstractmethod
    def children(self, cell: CellId) -> Sequence[CellId]:
        """Return the cells one resolution finer that cover ``cell`` (LOD).

        Returns an empty sequence when the backend cannot refine further.
        """

    def latlng(self, cell: CellId) -> LatLon:
        """Return the cell's representative point (alias of ``centroid``)."""
        return self.centroid(cell)

    def total_area_m2(self) -> float:
        """Return the summed area of all cells (approximately 4*pi*r^2)."""
        return sum(self.area_m2(cell) for cell in self.cells())
