/**
 * TypeScript mirrors of the API's pydantic models (api/settings.py,
 * api/commands.py, api/ws_events.py). Kept hand-written and minimal rather
 * than code-generated from `/schema` — the surface is small and stable, and
 * a hand-written contract makes drift visible in code review. If this ever
 * grows past a page, generate it from the OpenAPI document instead.
 */

/** api/settings.py: Settings.model_dump() shape, GET /settings. */
export interface SettingsTree {
  grid: { backend: "h3" | "isea" | "s2"; resolution: number };
  compute: {
    backend: "auto" | "numpy" | "cupy" | "jax" | "dask";
    gpus: string | number | number[];
    cpu_workers: string | number;
  };
  fidelity: { level: "fast" | "balanced" | "accurate" };
  planet: { axial_tilt_deg: number; rotation_rate: number; water_fraction: number };
  kernels: { flow_accumulation: "auto" | "python" | "c" };
  i18n: { locale: string };
  mods: { paths: string[] };
  api: { enabled: boolean; host: string; port: number };
}

/** One dotted-path option value, as returned by GET/PUT /settings/{path}. */
export interface SettingValue {
  path: string;
  value: unknown;
}

/** JSON Schema, as embedded in a command descriptor's `params` field. */
export type JsonSchema = Record<string, unknown>;

/** api/commands.py CommandRegistry.describe(): one entry of GET /commands. */
export interface CommandDescriptor {
  name: string;
  description: string;
  mutates: boolean;
  params: JsonSchema;
}

/** Envelope returned by POST /commands/{name}. */
export interface CommandResult<T = unknown> {
  command: string;
  result: T;
}

/** api/commands.py _describe_backend(): shape shared by get/set_compute_backend. */
export interface ComputeBackendInfo {
  backend: string;
  device: string;
  num_devices: number;
}

/** adapters/hardware.py probe_hardware command result. */
export interface HardwareProbe {
  capabilities: Record<string, unknown>;
  recommendation: Record<string, unknown>;
}

/** core/sim/world_sweep.py SweepCandidate, as embedded in a sweep report. */
export interface SweepCandidate {
  seed: number;
  score: number;
  details: Record<string, number>;
}

/** api/world_service.py deepen_seed(): the deep-run summary for one winner. */
export interface DeepenResult {
  seed: number;
  channel_count: number;
  biology_ticks: number;
  surviving_species: number;
  populations: Record<string, number>;
  telemetry: Record<string, unknown>;
}

/** api/world_service.py report_to_dict(): POST /commands/run_world_sweep result. */
export interface SweepReport {
  ranked: SweepCandidate[];
  selected: number[];
  deepened: Record<string, DeepenResult>;
}

/** GET /metrics. */
export interface Metrics {
  commands_executed: number;
  commands_denied: number;
  settings_changed: number;
  ws_subscribers: number;
  sim?: Record<string, unknown>;
}

// --- WS events (api/ws_events.py EVENT_MODELS) --------------------------

export interface HelloEvent {
  type: "hello";
  server: string;
  schema_url: string;
}

export interface SettingChangedEvent {
  type: "setting_changed";
  path: string;
  value: unknown;
}

export interface HeartbeatEvent {
  type: "heartbeat";
  tick: number;
}

export interface RunTelemetryEvent {
  type: "run_telemetry";
  ticks: number;
  ticks_per_second: number;
  mean_seconds_per_tick: number;
  wall_seconds: number;
}

export interface ComputeBackendChangedEvent {
  type: "compute_backend_changed";
  backend: string;
  device: string;
  num_devices: number;
}

export type WsEvent =
  | HelloEvent
  | SettingChangedEvent
  | HeartbeatEvent
  | RunTelemetryEvent
  | ComputeBackendChangedEvent;
