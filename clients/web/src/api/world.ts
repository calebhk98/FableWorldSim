/**
 * Typed wrappers over the persisted, steppable world commands
 * (`api/world_commands.py`, delegating to `api/world_state.py`). This is the
 * client's one source of REAL (non-synthetic) globe data: a world is built
 * once via `create_world` and kept live on the server, so the globe can
 * query its per-cell geometry/fields and step it forward across many
 * requests. See `src/globe/cell-geometry.ts` and `src/globe/field-layer.ts`
 * for how the results here become points and colors on the sphere.
 */

import { runCommand } from "./commands";
import type { CommandResult } from "./types";

/** api/world_state.py world_summary(): GET-like result of create_world/get_world. */
export interface WorldSummary {
  exists: boolean;
  tick?: number;
  seed?: number;
  planet_name?: string;
  grid_backend?: string;
  grid_resolution?: number;
  cell_count?: number;
  /** Field names this world's query_field understands, e.g. height/temperature/precipitation. */
  fields_available?: string[];
}

/** api/world_commands.py CreateWorldParams (extends the shared WorldBuildParams). */
export interface CreateWorldParams {
  seed: number;
  preset_name?: string | null;
  planet_config?: Record<string, unknown> | null;
  /** 0-4; 0 = 122 h3 cells (coarsest/fastest), matching the server default. */
  resolution?: number;
  season_count?: number;
  grid_backend?: string;
}

/** api/world_state.py step_persisted_world(): POST /commands/step_world result. */
export interface StepWorldResult {
  tick: number;
  ticks_advanced: number;
  surviving_species: number;
  populations: Record<string, number>;
  civ_population: Record<string, number>;
  telemetry: Record<string, unknown>;
}

/**
 * cell_id -> {lat, lng, boundary}, in degrees (api/world_state.py grid_geometry()).
 * `boundary` is the cell's ordered [lat, lng] boundary vertices — 6 per H3/ISEA
 * hex, 5 for the 12 pentagons, 4 per S2 quad — so the globe can draw true cell
 * polygons instead of just centroid points; see `src/globe/cell-geometry.ts`.
 */
export type GridGeometry = Record<
  string,
  { lat: number; lng: number; boundary: [number, number][] }
>;

/** cell_id -> field value (api/world_state.py query_field()). */
export type FieldValues = Record<string, number>;

/** POST /commands/create_world — build a world and store it as the live, steppable world. */
export function createWorld(params: CreateWorldParams): Promise<CommandResult<WorldSummary>> {
  return runCommand("create_world", params);
}

/** POST /commands/get_world — summary of the current live world (exists:false if none yet). */
export function getWorld(): Promise<CommandResult<WorldSummary>> {
  return runCommand("get_world");
}

/** POST /commands/step_world — advance the live world by `ticks` orchestrator ticks. */
export function stepWorld(ticks: number): Promise<CommandResult<StepWorldResult>> {
  return runCommand("step_world", { ticks });
}

/** POST /commands/query_field — cell_id -> value for one field of the live world. */
export function queryField(field: string): Promise<CommandResult<FieldValues>> {
  return runCommand("query_field", { field });
}

/** POST /commands/grid_geometry — cell_id -> centroid + boundary for every cell of the live world's grid. */
export function fetchGridGeometry(): Promise<CommandResult<GridGeometry>> {
  return runCommand("grid_geometry");
}

/** POST /commands/export_recipe — the reproducibility recipe for the live world. */
export function exportRecipe(): Promise<CommandResult<Record<string, unknown>>> {
  return runCommand("export_recipe");
}
