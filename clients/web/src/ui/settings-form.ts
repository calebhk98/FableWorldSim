/**
 * A minimal editor over the live settings tree (GET/PUT /settings/{path}).
 * Only the dials relevant to the globe (grid backend/resolution, compute
 * backend) get dedicated controls; the rest of the schema is enormous and
 * better served by a generic path/value editor, which this also provides.
 */

import { putSetting } from "../api/settings";
import { setComputeBackend } from "../api/commands";
import type { SettingsTree } from "../api/types";

export class SettingsForm {
  readonly el = document.createElement("section");
  private readonly gridBackendSelect = document.createElement("select");
  private readonly gridResolutionInput = document.createElement("input");
  private readonly computeBackendSelect = document.createElement("select");
  private readonly statusEl = document.createElement("p");

  constructor(private readonly onGridChanged: () => void) {
    this.el.className = "panel-section";
    const heading = document.createElement("h2");
    heading.textContent = "Settings";
    this.el.appendChild(heading);

    this.el.appendChild(
      this.field("grid.backend", this.gridBackendSelect, ["h3", "isea", "s2"]),
    );
    this.gridResolutionInput.type = "number";
    this.gridResolutionInput.min = "0";
    this.el.appendChild(this.field("grid.resolution", this.gridResolutionInput));

    this.el.appendChild(
      this.field("compute.backend", this.computeBackendSelect, ["auto", "numpy", "cupy", "jax"]),
    );

    this.statusEl.className = "muted";
    this.el.appendChild(this.statusEl);

    this.gridBackendSelect.addEventListener("change", () => {
      void this.apply("grid.backend", this.gridBackendSelect.value);
    });
    this.gridResolutionInput.addEventListener("change", () => {
      void this.apply("grid.resolution", Number(this.gridResolutionInput.value));
    });
    this.computeBackendSelect.addEventListener("change", () => {
      void this.applyComputeBackend(this.computeBackendSelect.value);
    });
  }

  private field(label: string, control: HTMLSelectElement | HTMLInputElement, options?: string[]): HTMLElement {
    const wrapper = document.createElement("label");
    wrapper.className = "field";
    const span = document.createElement("span");
    span.textContent = label;
    wrapper.appendChild(span);
    if (options && control instanceof HTMLSelectElement) {
      for (const option of options) {
        const opt = document.createElement("option");
        opt.value = option;
        opt.textContent = option;
        control.appendChild(opt);
      }
    }
    wrapper.appendChild(control);
    return wrapper;
  }

  /** Reflect the server's current values (e.g. right after GET /settings). */
  setValues(settings: SettingsTree): void {
    this.gridBackendSelect.value = settings.grid.backend;
    this.gridResolutionInput.value = String(settings.grid.resolution);
    this.computeBackendSelect.value = settings.compute.backend;
  }

  private async apply(path: string, value: unknown): Promise<void> {
    try {
      const result = await putSetting(path, value);
      this.statusEl.textContent = `${result.path} = ${JSON.stringify(result.value)}`;
      if (path.startsWith("grid.")) this.onGridChanged();
    } catch (error) {
      this.statusEl.textContent = `Failed to set ${path}: ${(error as Error).message}`;
    }
  }

  private async applyComputeBackend(backend: string): Promise<void> {
    try {
      const result = await setComputeBackend(backend as "auto" | "numpy" | "cupy" | "jax");
      this.statusEl.textContent = `compute backend -> ${result.result.backend} (${result.result.device})`;
    } catch (error) {
      this.statusEl.textContent = `Failed to hot-swap compute backend: ${(error as Error).message}`;
    }
  }
}
