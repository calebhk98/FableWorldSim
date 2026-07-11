/**
 * The three self-description surfaces the API advertises (see api/app.py's
 * module docstring): OpenAPI for REST, the command registry, and the WS
 * event schemas. A thin client's first move is always to ask, never to
 * assume — this module is that "ask".
 */

import { getJson } from "./http";
import type { CommandDescriptor, JsonSchema } from "./types";

/** GET /schema — the full OpenAPI document. */
export function getOpenApiSchema(): Promise<Record<string, unknown>> {
  return getJson("/schema");
}

/** GET /commands — every invocable command with its params JSON Schema. */
export function getCommands(): Promise<CommandDescriptor[]> {
  return getJson("/commands");
}

/** GET /ws/schema — one JSON Schema per streamable WS event type. */
export function getWsSchema(): Promise<Record<string, JsonSchema>> {
  return getJson("/ws/schema");
}

/**
 * Look up a command by name in an already-fetched descriptor list.
 *
 * Used for capability detection: rather than hardcoding a belief about what
 * the server can do, callers that need an optional/future command check
 * for it here first and degrade gracefully when it is absent. See
 * `src/globe/field-layer.ts` for the concrete case this exists for (no
 * per-cell field endpoint exists in this API build).
 */
export function findCommand(
  commands: CommandDescriptor[],
  name: string,
): CommandDescriptor | undefined {
  return commands.find((c) => c.name === name);
}
