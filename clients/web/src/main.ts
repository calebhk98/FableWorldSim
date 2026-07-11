/**
 * App bootstrap: wires the API/WS clients, the Three.js globe, and the side
 * panel together. Kept deliberately thin — each piece it calls into already
 * knows how to do its one job; this module just sequences them.
 *
 * The globe now renders REAL data end-to-end: `ensureWorld` calls
 * `get_world`/`create_world` to get a live, persisted world; `loadWorld`
 * fetches its real `grid_geometry` centroids and boundary polygons, and
 * `renderField` fetches a real `query_field` layer (height/temperature/
 * precipitation); "Step" advances the world via `step_world` and re-renders
 * the current field. See `src/api/world.ts` for the commands and
 * `src/globe/cell-geometry.ts` / `src/globe/field-layer.ts` for how their
 * results become cell polygons and colors.
 */

import "./style.css";
import { getSettings } from "./api/settings";
import { getCommands } from "./api/discovery";
import { getMetrics } from "./api/metrics";
import { createWorld, getWorld, stepWorld, queryField, fetchGridGeometry } from "./api/world";
import type { WorldSummary } from "./api/world";
import { WorldSocket } from "./ws/client";
import { GlobeScene } from "./globe/scene";
import { cellPointsFromGeometry, type CellPoint } from "./globe/cell-geometry";
import { availableFieldDescriptors, buildFieldLayer, type FieldDescriptor } from "./globe/field-layer";
import { buildLegend, viridis } from "./globe/palette";
import { Panel } from "./ui/panel";
import { renderLegend } from "./ui/legend";

const METRICS_POLL_MS = 5000;
/** create_world's own default resolution (122 h3 cells) — cheap enough for an auto-create on load. */
const DEFAULT_RESOLUTION = 0;

/** Mutable state for the one live world this client drives; see refreshField/onCreate/onStep below. */
interface GlobeState {
  points: CellPoint[];
  fields: FieldDescriptor[];
  currentField: string;
}

async function main(): Promise<void> {
  const canvas = document.getElementById("globe-canvas") as HTMLCanvasElement;
  const panelRoot = document.getElementById("panel") as HTMLElement;

  const scene = new GlobeScene(canvas);
  const state: GlobeState = { points: [], fields: [], currentField: "height" };

  // Handlers close over `panel` before it's constructed (they only run
  // later, in response to user clicks, by which point it's assigned) — the
  // definite-assignment assertion below tells TS that's fine.
  let panel!: Panel;
  panel = new Panel(
    panelRoot,
    // Settings-form grid.* edits configure *future* worlds (via the World
    // panel's own resolution field / a fresh create_world call); they don't
    // retroactively resize the already-created live world, so there's
    // nothing to refresh here.
    () => {},
    {
      onCreate: (seed, resolution) => createLiveWorld(scene, panel, state, seed, resolution),
      onStep: (ticks) => stepLiveWorld(scene, panel, state, ticks),
      onFieldChange: (field) => switchField(scene, panel, state, field),
    },
  );

  const socket = new WorldSocket();
  socket.onStatusChange((connected) => panel.status.setWsConnected(connected));
  socket.onEvent((event) => panel.eventLog.push(event));
  socket.connect();

  const [settings, commands] = await Promise.all([getSettings(), getCommands()]);
  panel.settingsForm.setValues(settings);
  panel.commandsList.setCommands(commands);

  await ensureWorld(scene, panel, state);

  const pollMetrics = async () => {
    try {
      panel.status.setMetrics(await getMetrics());
    } catch {
      // Metrics are best-effort UI sugar; a transient failure just retries.
    }
  };
  void pollMetrics();
  window.setInterval(() => void pollMetrics(), METRICS_POLL_MS);
}

/** On load: reuse the server's live world if one already exists, else create a fresh one. */
async function ensureWorld(scene: GlobeScene, panel: Panel, state: GlobeState): Promise<void> {
  panel.worldPanel.setStatus("Checking for a live world…");
  const { result: existing } = await getWorld();
  const summary = existing.exists ? existing : await createDefaultWorld(panel);
  await loadWorld(scene, panel, state, summary);
}

async function createDefaultWorld(panel: Panel): Promise<WorldSummary> {
  panel.worldPanel.setStatus("No live world yet — creating one (seed 1)…");
  const settings = await getSettings();
  const { result } = await createWorld({
    seed: 1,
    resolution: DEFAULT_RESOLUTION,
    grid_backend: settings.grid.backend,
  });
  return result;
}

/** Explicit "Create world" button handler: always builds a fresh world, replacing the live one. */
async function createLiveWorld(
  scene: GlobeScene,
  panel: Panel,
  state: GlobeState,
  seed: number,
  resolution: number,
): Promise<void> {
  panel.worldPanel.setBusy(true);
  panel.worldPanel.setStatus(`Creating world (seed ${seed}, resolution ${resolution})…`);
  try {
    const settings = await getSettings();
    const { result } = await createWorld({ seed, resolution, grid_backend: settings.grid.backend });
    await loadWorld(scene, panel, state, result);
  } catch (error) {
    panel.worldPanel.setStatus(`Failed to create world: ${(error as Error).message}`);
  } finally {
    panel.worldPanel.setBusy(false);
  }
}

/** Fetch a fresh world's geometry once, pick a starting field, and render it. */
async function loadWorld(scene: GlobeScene, panel: Panel, state: GlobeState, summary: WorldSummary): Promise<void> {
  const { result: geometry } = await fetchGridGeometry();
  state.points = cellPointsFromGeometry(geometry);
  state.fields = availableFieldDescriptors(summary.fields_available ?? []);
  if (!state.fields.some((f) => f.name === state.currentField)) {
    state.currentField = state.fields[0]?.name ?? "height";
  }
  panel.worldPanel.setFieldOptions(state.fields, state.currentField);
  describeWorld(panel, summary);
  await renderField(scene, panel, state);
}

/** "Step" button handler: advance the live world, then refetch the current field. */
async function stepLiveWorld(scene: GlobeScene, panel: Panel, state: GlobeState, ticks: number): Promise<void> {
  panel.worldPanel.setBusy(true);
  panel.worldPanel.setStatus(`Stepping ${ticks} tick(s)…`);
  try {
    const { result } = await stepWorld(ticks);
    const { result: summary } = await getWorld();
    describeWorld(panel, summary);
    await renderField(scene, panel, state);
    const civPopTotal = Object.values(result.civ_population).reduce((a, b) => a + b, 0);
    panel.worldPanel.setStatus(
      `Advanced ${result.ticks_advanced} tick(s) to tick ${result.tick} — ` +
        `${result.surviving_species} surviving species, civ pop ${civPopTotal.toFixed(0)}.`,
    );
  } catch (error) {
    panel.worldPanel.setStatus(`Failed to step world: ${(error as Error).message}`);
  } finally {
    panel.worldPanel.setBusy(false);
  }
}

/** Field-switcher handler: refetch query_field for the newly picked field and re-render. */
async function switchField(scene: GlobeScene, panel: Panel, state: GlobeState, field: string): Promise<void> {
  state.currentField = field;
  panel.worldPanel.setBusy(true);
  try {
    await renderField(scene, panel, state);
  } finally {
    panel.worldPanel.setBusy(false);
  }
}

/** Fetch the current field's values for the current geometry and push both to the scene + legend. */
async function renderField(scene: GlobeScene, panel: Panel, state: GlobeState): Promise<void> {
  const descriptor = state.fields.find((f) => f.name === state.currentField) ?? state.fields[0];
  if (!descriptor || state.points.length === 0) return;
  const { result: values } = await queryField(descriptor.name);
  const layer = buildFieldLayer(descriptor, values, state.points);
  scene.setCells(state.points, layer);
  const legend = buildLegend(layer.title, layer.unit, layer.min, layer.max, viridis);
  renderLegend(panel.legendEl, legend);
}

function describeWorld(panel: Panel, summary: WorldSummary): void {
  panel.worldPanel.setSummary(
    `seed ${summary.seed} · tick ${summary.tick} · ${summary.cell_count} cells ` +
      `(${summary.grid_backend} res ${summary.grid_resolution}) · ${summary.planet_name}`,
  );
  panel.worldPanel.setStatus("Live world loaded.");
}

main().catch((error: unknown) => {
  const panelRoot = document.getElementById("panel");
  if (panelRoot) {
    panelRoot.textContent = `Failed to reach the API: ${(error as Error).message}. Is the server running (docker compose up)?`;
  }
  // eslint-disable-next-line no-console
  console.error(error);
});
