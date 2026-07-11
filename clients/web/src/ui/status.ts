/** Connection status + server counters (GET /metrics, WS hello/lifecycle). */

import type { Metrics } from "../api/types";

export class StatusSection {
  readonly el = document.createElement("section");
  private readonly wsBadge = document.createElement("span");
  private readonly metricsEl = document.createElement("dl");

  constructor(apiBaseUrl: string) {
    this.el.className = "panel-section";
    const heading = document.createElement("h2");
    heading.textContent = "FableWorldSim";
    this.el.appendChild(heading);

    const apiLine = document.createElement("p");
    apiLine.className = "muted";
    apiLine.textContent = `API: ${apiBaseUrl}`;
    this.el.appendChild(apiLine);

    const wsLine = document.createElement("p");
    this.wsBadge.className = "badge badge--pending";
    this.wsBadge.textContent = "connecting…";
    wsLine.append("WS: ", this.wsBadge);
    this.el.appendChild(wsLine);

    this.metricsEl.className = "metrics";
    this.el.appendChild(this.metricsEl);
  }

  setWsConnected(connected: boolean): void {
    this.wsBadge.textContent = connected ? "connected" : "reconnecting…";
    this.wsBadge.className = `badge ${connected ? "badge--ok" : "badge--pending"}`;
  }

  setMetrics(metrics: Metrics): void {
    this.metricsEl.innerHTML = "";
    const rows: [string, string | number][] = [
      ["commands executed", metrics.commands_executed],
      ["commands denied", metrics.commands_denied],
      ["settings changed", metrics.settings_changed],
      ["ws subscribers", metrics.ws_subscribers],
    ];
    for (const [label, value] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = String(value);
      this.metricsEl.append(dt, dd);
    }
  }
}
