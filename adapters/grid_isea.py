"""ISEA/DGGRID adapter for the grid port (equal-area backend).

ISEA (Icosahedral Snyder Equal Area, via the external DGGRID tool) is the
one built-in backend whose cells genuinely have equal area, making it the
reference for verifying that area-weighted simulation math is invariant
to the backend toggle. We standardize on the **ISEA3H** topology: hex
cells plus 12 pentagons (exactly 5/6 of a hex in area), giving H3-like
adjacency with strictly equal-area hexes.

DGGRID itself is an external C++ binary driven through ``dggrid4py``
(``pip install 'fableworldsim[grid-isea]'`` *and* a working ``dggrid``
executable on ``PATH`` — see https://github.com/sahrk/DGGRID). Unlike H3
or S2, the Python package always imports cleanly; the missing piece is
usually the binary, which is only discoverable at runtime.  So this
module imports ``dggrid4py`` at module level but never touches the
binary until a grid is actually constructed, at which point a missing
binary raises :class:`ports.grid.GridBackendUnavailableError` instead of
crashing.

DGGRID is a batch tool with no incremental query API, so an
:class:`IseaGrid` materializes the whole-sphere cell set once (lazily, on
first use) by shelling out to ``dggrid`` and parsing back its cell
polygons; everything else (area, neighbors, centroid, point lookup) is
served from that in-memory cache or a further, targeted ``dggrid`` call.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

import dggrid4py
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree

from ports.grid import CellId, Grid, GridBackendUnavailableError, LatLon

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from shapely.geometry import BaseGeometry, Polygon

_DGGS_TYPE = "ISEA3H"
_SEQNUM = "SEQNUM"
_ID_COLUMNS = ("name", "Name", "global_id")
_UNAVAILABLE_MSG = (
    "grid backend 'isea' needs the external DGGRID binary (via dggrid4py): "
    "install DGGRID (https://github.com/sahrk/DGGRID), put 'dggrid' on "
    "PATH, or toggle 'h3'/'s2' in the meantime"
)


def _id_column(gdf: gpd.GeoDataFrame) -> str:
    """Return whichever cell-id column name this dggrid4py output used."""
    for candidate in _ID_COLUMNS:
        if candidate in gdf.columns:
            return candidate
    msg = f"dggrid output has none of the expected id columns {_ID_COLUMNS}"
    raise GridBackendUnavailableError(msg)


def _wrapped_delta_deg(lon2: float, lon1: float) -> float:
    """Return ``lon2 - lon1`` wrapped into ``(-180, 180]`` degrees."""
    return (lon2 - lon1 + 180) % 360 - 180


def _as_lines(geometry: BaseGeometry) -> list[BaseGeometry]:
    """Return the line components of a boundary intersection, or ``[]``.

    Two adjacent hex/pentagon cells share exactly one edge, so the
    intersection is normally a single ``LineString``; a ``MultiLineString``
    (e.g. a numerically split edge) is unpacked into its parts. Anything
    else (empty, or a lone shared vertex) means the cells are not adjacent.
    """
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    return []


def _ring_area_m2(ring: Sequence[tuple[float, float]], radius_m: float) -> float:
    """Return the area enclosed by a closed geodetic ring, in square meters.

    ``ring`` is a sequence of ``(lon_deg, lat_deg)`` vertices with the first
    point repeated as the last (the shapely/GeoJSON convention). This is the
    spherical form of the shoelace formula: by Stokes' theorem the surface
    integral of ``cos(lat) dlat dlon`` (the area element) equals the line
    integral ``-sin(lat) dlon`` around the boundary, which the usual
    trapezoidal rule turns into a per-edge average of ``sin(lat)`` — exact
    for graticule (constant-lon/constant-lat) edges and an excellent
    approximation for the short geodesic edges of a DGGS cell.
    """
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in pairwise(ring):
        dlon = math.radians(_wrapped_delta_deg(lon2, lon1))
        total += dlon * (math.sin(math.radians(lat1)) + math.sin(math.radians(lat2)))
    return abs(total) * radius_m**2 / 2.0


def _haversine_m(a: LatLon, b: LatLon, radius_m: float) -> float:
    """Return the great-circle distance between two points, in meters."""
    lat1, lat2 = math.radians(a.lat_deg), math.radians(b.lat_deg)
    dlat = lat2 - lat1
    dlon = math.radians(_wrapped_delta_deg(b.lon_deg, a.lon_deg))
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius_m * math.asin(min(1.0, math.sqrt(hav)))


def _dggrid_runner() -> dggrid4py.DGGRIDv7:
    """Return a runner bound to the ``dggrid`` executable, checked but unused.

    Raises :class:`GridBackendUnavailableError` when the binary cannot be
    found or is not executable; never invokes it.
    """
    runner = dggrid4py.DGGRIDv7(executable="dggrid", capture_logs=True, silent=True)
    if not runner.is_runnable():
        raise GridBackendUnavailableError(_UNAVAILABLE_MSG)
    return runner


class IseaGrid(Grid):
    """A full-sphere ISEA3H grid at one resolution, on a sphere of any radius.

    Construction only checks that ``dggrid`` is on ``PATH``; the (possibly
    expensive) whole-earth cell generation happens lazily on first use of
    any method that needs cell data, and is cached for the grid's lifetime.
    """

    def __init__(self, resolution: int, radius_m: float) -> None:
        """Validate the toggle parameters and confirm ``dggrid`` is runnable."""
        if resolution < 0:
            msg = f"isea resolution must be >= 0, got {resolution}"
            raise ValueError(msg)
        if radius_m <= 0:
            msg = f"radius_m must be > 0, got {radius_m}"
            raise ValueError(msg)
        self._resolution = resolution
        self._radius_m = radius_m
        self._runner = _dggrid_runner()
        self._cell_ids: tuple[CellId, ...] | None = None
        self._polygons: dict[CellId, Polygon] = {}
        self._centroids: dict[CellId, LatLon] = {}
        self._areas_m2: dict[CellId, float] = {}
        self._tree: STRtree | None = None
        self._tree_ids: tuple[CellId, ...] = ()

    @property
    def backend_name(self) -> str:
        """Return the toggle name of this backend."""
        return "isea"

    @property
    def resolution(self) -> int:
        """Return the ISEA3H resolution."""
        return self._resolution

    @property
    def radius_m(self) -> float:
        """Return the sphere radius in meters."""
        return self._radius_m

    @property
    def cell_count(self) -> int:
        """Return the number of cells at this resolution."""
        return len(self._materialize())

    def _materialize(self) -> tuple[CellId, ...]:
        """Generate (once) and cache every whole-earth cell at this resolution."""
        if self._cell_ids is not None:
            return self._cell_ids
        gdf = self._runner.grid_cell_polygons_for_extent(
            dggs_type=_DGGS_TYPE,
            resolution=self._resolution,
            output_address_type=_SEQNUM,
        )
        id_col = _id_column(gdf)
        cell_ids = []
        for cell_id, geometry in zip(gdf[id_col], gdf.geometry, strict=True):
            cid = str(cell_id)
            cell_ids.append(cid)
            self._polygons[cid] = geometry
            ring = list(geometry.exterior.coords)
            self._areas_m2[cid] = _ring_area_m2(ring, self._radius_m)
            lon, lat = geometry.centroid.x, geometry.centroid.y
            self._centroids[cid] = LatLon(lat_deg=float(lat), lon_deg=float(lon))
        self._cell_ids = tuple(sorted(cell_ids, key=int))
        return self._cell_ids

    def _index(self) -> tuple[STRtree, tuple[CellId, ...]]:
        """Return (building it once) a spatial index over every cell polygon."""
        cell_ids = self._materialize()
        if self._tree is None:
            self._tree_ids = cell_ids
            self._tree = STRtree([self._polygons[cid] for cid in cell_ids])
        return self._tree, self._tree_ids

    def _lookup_cell(self, point: LatLon, resolution: int) -> CellId:
        """Ask dggrid which cell at ``resolution`` contains ``point``."""
        points_gdf = gpd.GeoDataFrame({"geometry": [Point(point.lon_deg, point.lat_deg)]}, crs=4326)
        result = self._runner.cells_for_geo_points(
            geodf_points_wgs84=points_gdf,
            cell_ids_only=True,
            dggs_type=_DGGS_TYPE,
            resolution=resolution,
            output_address_type=_SEQNUM,
        )
        return str(result[_id_column(result)].iloc[0])

    def cells(self) -> Iterator[CellId]:
        """Iterate all cells in ascending SEQNUM order."""
        yield from self._materialize()

    def neighbors(self, cell: CellId) -> Sequence[CellId]:
        """Return edge-adjacent cells (6, or 5 around a pentagon).

        In a proper hex/pentagon tessellation every pair of cells that
        touch at all share a full edge (three cells meet at each vertex,
        each pair of them sharing one of the three edges there), so a
        bounding-box index plus a boundary-touch test is exact.
        """
        tree, cell_ids = self._index()
        polygon = self._polygons[cell]
        candidates = tree.query(polygon)
        found = []
        for index in candidates:
            other = cell_ids[index]
            if other == cell:
                continue
            if polygon.touches(self._polygons[other]):
                found.append(other)
        return tuple(found)

    def edge_length_m(self, cell: CellId, neighbor: CellId) -> float:
        """Return the shared-edge length scaled to the planet radius."""
        shared = self._polygons[cell].intersection(self._polygons[neighbor])
        lines = _as_lines(shared)
        if not lines:
            msg = f"cells {cell} and {neighbor} are not adjacent"
            raise ValueError(msg)
        total_m = 0.0
        for line in lines:
            for (lon1, lat1), (lon2, lat2) in pairwise(line.coords):
                a = LatLon(lat_deg=lat1, lon_deg=lon1)
                b = LatLon(lat_deg=lat2, lon_deg=lon2)
                total_m += _haversine_m(a, b, self._radius_m)
        return total_m

    def parent(self, cell: CellId) -> CellId | None:
        """Return the cell one resolution coarser whose area contains it.

        Computed by asking dggrid which coarser cell contains this cell's
        centroid; ISEA3H's aperture-3 hierarchy is not a perfect nesting,
        so this is a containment relation rather than an exact index-tree
        parent (mirrors how ``cell_at`` classifies any point).
        """
        if self._resolution == 0:
            return None
        return self._lookup_cell(self.centroid(cell), self._resolution - 1)

    def children(self, cell: CellId) -> Sequence[CellId]:
        """Return the finer cells whose centroid falls within ``cell``."""
        finer = self._runner.grid_cell_polygons_for_extent(
            dggs_type=_DGGS_TYPE,
            resolution=self._resolution + 1,
            clip_geom=self._polygons[cell],
            output_address_type=_SEQNUM,
        )
        id_col = _id_column(finer)
        parent_polygon = self._polygons[cell]
        kids = [
            str(cid)
            for cid, geometry in zip(finer[id_col], finer.geometry, strict=True)
            if parent_polygon.contains(geometry.centroid)
        ]
        return tuple(sorted(kids, key=int))

    def area_m2(self, cell: CellId) -> float:
        """Return the cell's exact geodetic area, scaled to the planet radius."""
        self._materialize()
        return self._areas_m2[cell]

    def centroid(self, cell: CellId) -> LatLon:
        """Return the cell centroid in degrees."""
        self._materialize()
        return self._centroids[cell]

    def cell_at(self, point: LatLon) -> CellId:
        """Return the cell containing a latitude/longitude point."""
        return self._lookup_cell(point, self._resolution)

    def boundary(self, cell: CellId) -> Sequence[LatLon]:
        """Return the ordered polygon vertices dggrid materialized for ``cell``.

        The exterior ring shapely/dggrid4py hand back repeats its first
        vertex as the last (the GeoJSON closing convention); that repeat is
        dropped here so the count matches the port's "no repeated closing
        point" contract.
        """
        self._materialize()
        ring = list(self._polygons[cell].exterior.coords)
        return tuple(LatLon(lat_deg=lat, lon_deg=lon) for lon, lat in ring[:-1])


def create(resolution: int, radius_m: float) -> Grid:
    """Create an :class:`IseaGrid`; registry entry point for backend 'isea'."""
    return IseaGrid(resolution, radius_m)
