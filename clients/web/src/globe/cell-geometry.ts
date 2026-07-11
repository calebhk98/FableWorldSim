/**
 * Converts the live world's real per-cell centroid geometry
 * (`grid_geometry`: cell_id -> {lat, lng}, see `src/api/world.ts`) into
 * unit-sphere points for the Three.js point cloud in `scene.ts`.
 *
 * Every point here is one actual DGGS cell of the server's grid — not a
 * synthetic sampling. The one honesty caveat, inherited straight from the
 * API (`api/world_state.py` `grid_geometry`'s docstring): the `Grid` port
 * exposes centroids only, not cell-boundary vertices, so cells render as
 * points/discs rather than true polygons. See README.md's "Remaining
 * limits" section.
 */

import type { GridGeometry } from "../api/world";

export interface CellPoint {
  cellId: string;
  /** Unit-sphere position. */
  x: number;
  y: number;
  z: number;
  latDeg: number;
  lonDeg: number;
}

/** Same lat/lon -> xyz convention as scene.ts's graticule, so both agree on where "up" is. */
export function cellPointsFromGeometry(geometry: GridGeometry): CellPoint[] {
  const points: CellPoint[] = [];
  for (const [cellId, { lat, lng }] of Object.entries(geometry)) {
    const latRad = (lat * Math.PI) / 180;
    const lonRad = (lng * Math.PI) / 180;
    points.push({
      cellId,
      x: Math.cos(latRad) * Math.cos(lonRad),
      y: Math.sin(latRad),
      z: Math.cos(latRad) * Math.sin(lonRad),
      latDeg: lat,
      lonDeg: lng,
    });
  }
  return points;
}
