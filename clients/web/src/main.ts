/**
 * App bootstrap: wires the API/WS clients, the Three.js globe, and the side
 * panel together. Kept deliberately thin — each piece it calls into already
 * knows how to do its one job; this module just sequences them.
 */

import "./style.css";
import { getSettings } from "./api/settings";
import { getCommands } from "./api/discovery";
import { getMetrics } from "./api/metrics";
import { WorldSocket } from "./ws/client";
import { GlobeScene } from "./globe/scene";
import { placeholderSpherePoints } from "./globe/placeholder-sphere";
import { detectFieldCommand, syntheticDemoLayer } from "./globe/field-layer";
import { buildLegend, viridis } from "./globe/palette";
import { Panel } from "./ui/panel";
import { renderLegend } from "./ui/legend";
import type { CommandDescriptor } from "./api/types";

const METRICS_POLL_MS = 5000;

async function main(): Promise<void> {
  const canvas = document.getElementById("globe-canvas") as HTMLCanvasElement;
  const panelRoot = document.getElementById("panel") as HTMLElement;

  let knownCommands: CommandDescriptor[] = [];

  const scene = new GlobeScene(canvas);
  const panel = new Panel(panelRoot, () => void refreshGlobe(scene, panel, knownCommands));

  const socket = new WorldSocket();
  socket.onStatusChange((connected) => panel.status.setWsConnected(connected));
  socket.onEvent((event) => panel.eventLog.push(event));
  socket.connect();

  const [settings, commands] = await Promise.all([getSettings(), getCommands()]);
  knownCommands = commands;
  panel.settingsForm.setValues(settings);
  panel.commandsList.setCommands(commands);

  await refreshGlobe(scene, panel, knownCommands);

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

/**
 * Rebuild the globe's cell layer from the current grid.resolution setting.
 * Runs once at startup and again whenever the settings form changes a
 * grid.* option. See globe/field-layer.ts for why this is a demo layer.
 */
async function refreshGlobe(
  scene: GlobeScene,
  panel: Panel,
  commands: CommandDescriptor[],
): Promise<void> {
  const settings = await getSettings();
  const points = placeholderSpherePoints(settings.grid.resolution);

  const fieldCommand = detectFieldCommand(commands);
  if (fieldCommand) {
    // A server build has added a per-cell field command that this client's
    // capability detection recognizes, but no fetch/mapping is wired up yet
    // — the naming match alone doesn't tell us its param/return shape.
    // eslint-disable-next-line no-console
    console.warn(
      `Detected candidate field command "${fieldCommand.name}" but no live-data path is ` +
        "implemented yet; falling back to the synthetic demo layer. See globe/field-layer.ts.",
    );
  }
  const layer = syntheticDemoLayer(points);

  scene.setCells(points, layer);
  const legend = buildLegend(layer.title, layer.unit, layer.min, layer.max, viridis);
  renderLegend(panel.legendEl, legend, layer.isLiveServerData);
}

main().catch((error: unknown) => {
  const panelRoot = document.getElementById("panel");
  if (panelRoot) {
    panelRoot.textContent = `Failed to reach the API: ${(error as Error).message}. Is the server running (docker compose up)?`;
  }
  // eslint-disable-next-line no-console
  console.error(error);
});
