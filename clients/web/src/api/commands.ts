/**
 * Typed wrappers over POST /commands/{name} for the built-in registry
 * (api/commands.py `build_default_registry`). Each function here mirrors
 * one real, self-described command — nothing invented. `runCommand` stays
 * exported for the generic command-runner panel, which drives *any*
 * command (including future ones) straight from `/commands` metadata.
 */

import { postJson } from "./http";
import type {
  CommandResult,
  ComputeBackendInfo,
  HardwareProbe,
  SweepReport,
} from "./types";

/** POST /commands/{name} with an arbitrary JSON body; the generic escape hatch. */
export function runCommand<T = unknown>(name: string, params?: unknown): Promise<CommandResult<T>> {
  return postJson(`/commands/${encodeURIComponent(name)}`, params);
}

export function ping(): Promise<CommandResult<{ pong: boolean }>> {
  return runCommand("ping");
}

export function listGridBackends(): Promise<CommandResult<string[]>> {
  return runCommand("list_grid_backends");
}

export function probeHardware(): Promise<CommandResult<HardwareProbe>> {
  return runCommand("probe_hardware");
}

export function getRunTelemetry(): Promise<CommandResult<Record<string, unknown>>> {
  return runCommand("get_run_telemetry");
}

export function getComputeBackend(): Promise<CommandResult<ComputeBackendInfo>> {
  return runCommand("get_compute_backend");
}

export function setComputeBackend(
  backend: "auto" | "numpy" | "cupy" | "jax",
): Promise<CommandResult<ComputeBackendInfo>> {
  return runCommand("set_compute_backend", { backend });
}

/** Params accepted by run_world_sweep (api/commands.py SweepParams); all optional server-side. */
export interface SweepParams {
  base_seed?: number;
  count?: number;
  keep_top_k?: number;
  resolution?: number;
  deep_ticks?: number;
}

export function runWorldSweep(params: SweepParams = {}): Promise<CommandResult<SweepReport>> {
  return runCommand("run_world_sweep", params);
}
