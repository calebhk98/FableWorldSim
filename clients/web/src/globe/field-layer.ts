/**
 * The per-cell data layer (height or temperature) overlaid on the globe.
 *
 * API GAP: as of this build, no REST route or command returns a per-cell
 * field for a generated world — `run_world_sweep` (the only command that
 * runs a simulation) returns aggregate habitability scores and non-spatial
 * deep-run summaries (population totals, river channel count), never a
 * cell -> value mapping, and there is no per-cell geometry endpoint to
 * anchor one to either. See clients/web/README.md for the full note.
 *
 * Rather than hardcode that absence, this module *asks* the server (via
 * the already-fetched `/commands` list) whether a matching command exists,
 * and only falls back to a clearly-labeled synthetic demo layer when it
 * doesn't. The moment a server build adds e.g. `get_cell_field`, this seam
 * is where it plugs in — no other module needs to change.
 */

import type { CommandDescriptor } from "../api/types";
import { findCommand } from "../api/discovery";
import type { SpherePoint } from "./placeholder-sphere";

/** Command names this client would recognize as a per-cell field source. */
const CANDIDATE_FIELD_COMMANDS = ["get_cell_field", "get_field", "query_cell_field"];

export interface FieldLayer {
  /** Human label, e.g. "Temperature (demo)". */
  title: string;
  unit: string;
  /** One value per point, same order/length as the sphere points it colors. */
  values: number[];
  min: number;
  max: number;
  /** False when this is the synthetic fallback, not server data. */
  isLiveServerData: boolean;
}

/** Look for a server-side per-cell field command; returns undefined if none exists. */
export function detectFieldCommand(
  commands: CommandDescriptor[],
): CommandDescriptor | undefined {
  for (const name of CANDIDATE_FIELD_COMMANDS) {
    const found = findCommand(commands, name);
    if (found) return found;
  }
  return undefined;
}

/**
 * A synthetic, deterministic "temperature-like" layer: warm at the equator,
 * cold at the poles. This is generic textbook latitude-band shading, not a
 * reproduction of FableWorldSim's climate model (core/climate/model.py) —
 * it exists only so the palette/legend machinery has something to render
 * while no real field endpoint is exposed. Always labeled "(demo)" in the UI.
 */
export function syntheticDemoLayer(points: SpherePoint[]): FieldLayer {
  const values = points.map((p) => Math.cos((p.latDeg * Math.PI) / 180));
  return {
    title: "Latitude shading (demo)",
    unit: "unitless",
    values,
    min: Math.min(...values),
    max: Math.max(...values),
    isLiveServerData: false,
  };
}
