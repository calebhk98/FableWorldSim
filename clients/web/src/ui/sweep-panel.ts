/**
 * Drives POST /commands/run_world_sweep — the one command that actually
 * runs a simulation — and renders its (non-spatial) results: ranked seeds
 * by habitability score, and the deep-run summaries for the kept winners.
 * This is real server data, unlike the globe's placeholder cell layer.
 */

import { runWorldSweep, type SweepParams } from "../api/commands";
import type { SweepReport } from "../api/types";

export class SweepPanel {
  readonly el = document.createElement("section");
  private readonly runButton = document.createElement("button");
  private readonly resultsEl = document.createElement("div");
  private readonly countInput = document.createElement("input");

  constructor() {
    this.el.className = "panel-section";
    const heading = document.createElement("h2");
    heading.textContent = "Run world sweep";
    this.el.appendChild(heading);

    const help = document.createElement("p");
    help.className = "muted";
    help.textContent = "Generates candidate worlds, scores each by habitability, deep-sims the best.";
    this.el.appendChild(help);

    this.countInput.type = "number";
    this.countInput.min = "1";
    this.countInput.max = "64";
    this.countInput.value = "6";
    const countLabel = document.createElement("label");
    countLabel.className = "field";
    countLabel.append("count", this.countInput);
    this.el.appendChild(countLabel);

    this.runButton.textContent = "Run sweep";
    this.runButton.addEventListener("click", () => void this.run());
    this.el.appendChild(this.runButton);

    this.resultsEl.className = "sweep-results";
    this.el.appendChild(this.resultsEl);
  }

  private async run(): Promise<void> {
    this.runButton.disabled = true;
    this.runButton.textContent = "Running…";
    const params: SweepParams = { count: Number(this.countInput.value) || 6 };
    try {
      const { result } = await runWorldSweep(params);
      this.renderResults(result);
    } catch (error) {
      this.resultsEl.textContent = `Sweep failed: ${(error as Error).message}`;
    } finally {
      this.runButton.disabled = false;
      this.runButton.textContent = "Run sweep";
    }
  }

  private renderResults(report: SweepReport): void {
    this.resultsEl.innerHTML = "";
    const table = document.createElement("table");
    const head = table.createTHead().insertRow();
    for (const column of ["seed", "score", "kept?"]) {
      const th = document.createElement("th");
      th.textContent = column;
      head.appendChild(th);
    }
    const body = table.createTBody();
    for (const candidate of report.ranked) {
      const row = body.insertRow();
      row.insertCell().textContent = String(candidate.seed);
      row.insertCell().textContent = candidate.score.toFixed(3);
      row.insertCell().textContent = report.selected.includes(candidate.seed) ? "yes" : "";
    }
    this.resultsEl.appendChild(table);

    for (const [seed, deep] of Object.entries(report.deepened)) {
      const card = document.createElement("div");
      card.className = "deep-card";
      const title = document.createElement("strong");
      title.textContent = `seed ${seed}`;
      const stats = document.createElement("p");
      stats.textContent =
        `river channels: ${deep.channel_count}, ` +
        `surviving species: ${deep.surviving_species}/${Object.keys(deep.populations).length}`;
      card.append(title, stats);
      this.resultsEl.appendChild(card);
    }
  }
}
