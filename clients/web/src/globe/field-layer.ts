/**
 * Builds a `FieldLayer` (per-point values + legend metadata) for one of the
 * live world's REAL per-cell fields, fetched via `query_field` (see
 * `src/api/world.ts`). Replaces the old synthetic latitude-shading
 * placeholder now that the API exposes `create_world`/`grid_geometry`/
 * `query_field` (issue #10's persisted, steppable world).
 *
 * `query_field` returns `cell_id -> value`, which this module aligns to a
 * given `CellPoint[]` order (from `cell-geometry.ts`) so `scene.ts` can zip
 * positions and colors index-for-index.
 */

import type { FieldValues } from "../api/world";
import type { CellPoint } from "./cell-geometry";

/** A field this client's layer switcher offers, in switcher order. */
export interface FieldDescriptor {
  /** The exact `query_field` `field` param value. */
  name: string;
  label: string;
  unit: string;
}

/**
 * Fields query_field understands (api/world_state.py FIELD_NAMES), minus
 * "elevation" (an alias for "height" server-side — offering both would just
 * be the same values twice in the switcher).
 */
export const FIELD_DESCRIPTORS: FieldDescriptor[] = [
  { name: "height", label: "Height", unit: "m" },
  { name: "temperature", label: "Temperature", unit: "K" },
  { name: "precipitation", label: "Precipitation", unit: "mm/yr" },
];

/** Restrict FIELD_DESCRIPTORS to what this world's get_world actually reports. */
export function availableFieldDescriptors(fieldsAvailable: string[]): FieldDescriptor[] {
  return FIELD_DESCRIPTORS.filter((d) => fieldsAvailable.includes(d.name));
}

export interface FieldLayer {
  title: string;
  unit: string;
  /** One value per point, same order/length as the CellPoint[] it colors. */
  values: number[];
  min: number;
  max: number;
}

/** Align a query_field response to `points`' order and compute its domain. */
export function buildFieldLayer(
  descriptor: FieldDescriptor,
  values: FieldValues,
  points: CellPoint[],
): FieldLayer {
  const aligned = points.map((point) => values[point.cellId] ?? 0);
  return {
    title: descriptor.label,
    unit: descriptor.unit,
    values: aligned,
    min: aligned.length ? Math.min(...aligned) : 0,
    max: aligned.length ? Math.max(...aligned) : 0,
  };
}
