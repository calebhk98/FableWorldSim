/**
 * A generic-sphere point sampling used to *stand in* for the real DGGS cell
 * tessellation until the API exposes one.
 *
 * IMPORTANT — this is presentational only, not simulation logic and not a
 * reproduction of any grid backend's math: it is a Fibonacci-sphere point
 * distribution (a well-known, backend-agnostic way to scatter N points
 * roughly evenly on a sphere), sized by an arbitrary density knob. It does
 * NOT reproduce H3/S2/ISEA cell shapes, counts, or ids — those live only in
 * adapters/grid_*.py behind the Grid port (ports/grid.py) and are not
 * exposed by any endpoint today. See clients/web/README.md's "API gaps"
 * section and src/globe/field-layer.ts.
 */

export interface SpherePoint {
  /** Unit-sphere position. */
  x: number;
  y: number;
  z: number;
  /** Latitude in [-90, 90] and longitude in [-180, 180], degrees. */
  latDeg: number;
  lonDeg: number;
}

const MIN_POINTS = 400;
const POINTS_PER_RESOLUTION_STEP = 900;
const GOLDEN_ANGLE_RAD = Math.PI * (3 - Math.sqrt(5));

/**
 * Return a Fibonacci-sphere point cloud whose density scales with
 * `resolution` (the grid.resolution setting) — purely so turning that dial
 * visibly changes the globe's density, echoing (without reproducing) how a
 * finer DGGS resolution means more, smaller cells.
 */
export function placeholderSpherePoints(resolution: number): SpherePoint[] {
  const count = MIN_POINTS + Math.max(0, resolution) * POINTS_PER_RESOLUTION_STEP;
  const points: SpherePoint[] = [];
  for (let i = 0; i < count; i += 1) {
    // Standard Fibonacci-lattice sphere parametrization.
    const y = 1 - (2 * i) / (count - 1);
    const radiusAtY = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = GOLDEN_ANGLE_RAD * i;
    const x = Math.cos(theta) * radiusAtY;
    const z = Math.sin(theta) * radiusAtY;
    points.push({
      x,
      y,
      z,
      latDeg: (Math.asin(y) * 180) / Math.PI,
      lonDeg: (Math.atan2(z, x) * 180) / Math.PI,
    });
  }
  return points;
}
