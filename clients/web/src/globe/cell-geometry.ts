/**
 * Converts the live world's real per-cell geometry (`grid_geometry`:
 * cell_id -> {lat, lng, boundary}, see `src/api/world.ts`) into unit-sphere
 * points and boundary polygons for the Three.js layer in `scene.ts`.
 *
 * Every cell here is one actual DGGS cell of the server's grid — not a
 * synthetic sampling. `grid_geometry` returns each cell's ordered boundary
 * vertices (6 per H3/ISEA hex, 5 for the 12 pentagons, 4 per S2 quad) as well
 * as its centroid, so `scene.ts` draws true cell-shaped polygons (a fan of
 * triangles from the centroid to consecutive boundary vertices) rather than
 * billboarded discs. The centroid is still carried on every `CellPoint`: it's
 * the fan's origin vertex, and the fallback point cloud `scene.ts` uses if a
 * world's cells ever come back without boundary data.
 */

import type { GridGeometry } from "../api/world";

/** A position on the unit sphere. */
export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface CellPoint extends Vec3 {
  cellId: string;
  latDeg: number;
  lonDeg: number;
  /** Ordered unit-sphere boundary vertices for this cell; empty if the
   *  server didn't provide one (see the module docstring's fallback note). */
  boundary: Vec3[];
}

/** Same lat/lon -> xyz convention as scene.ts's graticule, so both agree on where "up" is. */
export function latLonToUnitSphere(latDeg: number, lonDeg: number): Vec3 {
  const latRad = (latDeg * Math.PI) / 180;
  const lonRad = (lonDeg * Math.PI) / 180;
  return {
    x: Math.cos(latRad) * Math.cos(lonRad),
    y: Math.sin(latRad),
    z: Math.cos(latRad) * Math.sin(lonRad),
  };
}

export function cellPointsFromGeometry(geometry: GridGeometry): CellPoint[] {
  const points: CellPoint[] = [];
  for (const [cellId, cell] of Object.entries(geometry)) {
    const { x, y, z } = latLonToUnitSphere(cell.lat, cell.lng);
    points.push({
      cellId,
      x,
      y,
      z,
      latDeg: cell.lat,
      lonDeg: cell.lng,
      boundary: (cell.boundary ?? []).map(([lat, lng]) => latLonToUnitSphere(lat, lng)),
    });
  }
  return points;
}
