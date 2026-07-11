/**
 * World lifecycle controls: create the live world (seed/resolution), pick
 * which real per-cell field colors the globe, and step the simulation
 * forward — the three actions behind `src/api/world.ts`'s
 * create_world/query_field/step_world. Kept dumb: it only renders inputs
 * and calls the handlers `main.ts` wires up; no fetch happens here.
 */

import type { FieldDescriptor } from "../globe/field-layer";

export interface WorldPanelHandlers {
  onCreate: (seed: number, resolution: number) => void | Promise<void>;
  onStep: (ticks: number) => void | Promise<void>;
  onFieldChange: (field: string) => void | Promise<void>;
}

export class WorldPanel {
  readonly el = document.createElement("section");
  private readonly seedInput = document.createElement("input");
  private readonly resolutionInput = document.createElement("input");
  private readonly createButton = document.createElement("button");
  private readonly fieldSelect = document.createElement("select");
  private readonly ticksInput = document.createElement("input");
  private readonly stepButton = document.createElement("button");
  private readonly summaryEl = document.createElement("p");
  private readonly statusEl = document.createElement("p");

  constructor(private readonly handlers: WorldPanelHandlers) {
    this.el.className = "panel-section";
    const heading = document.createElement("h2");
    heading.textContent = "World";
    this.el.appendChild(heading);

    this.summaryEl.className = "muted";
    this.summaryEl.textContent = "No world yet.";
    this.el.appendChild(this.summaryEl);

    this.seedInput.type = "number";
    this.seedInput.value = "1";
    this.resolutionInput.type = "number";
    this.resolutionInput.min = "0";
    this.resolutionInput.max = "4";
    this.resolutionInput.value = "0";
    this.createButton.textContent = "Create world";
    this.el.appendChild(this.row("seed", this.seedInput));
    this.el.appendChild(this.row("resolution", this.resolutionInput));
    this.el.appendChild(this.row("", this.createButton));

    this.el.appendChild(this.row("field", this.fieldSelect));

    this.ticksInput.type = "number";
    this.ticksInput.min = "1";
    this.ticksInput.value = "1";
    this.stepButton.textContent = "Step";
    this.el.appendChild(this.row("ticks", this.ticksInput));
    this.el.appendChild(this.row("", this.stepButton));

    this.statusEl.className = "muted";
    this.el.appendChild(this.statusEl);

    this.createButton.addEventListener("click", () => {
      void this.handlers.onCreate(Number(this.seedInput.value), Number(this.resolutionInput.value));
    });
    this.stepButton.addEventListener("click", () => {
      void this.handlers.onStep(Number(this.ticksInput.value));
    });
    this.fieldSelect.addEventListener("change", () => {
      void this.handlers.onFieldChange(this.fieldSelect.value);
    });
  }

  private row(label: string, control: HTMLElement): HTMLElement {
    const wrapper = document.createElement("label");
    wrapper.className = "field";
    if (label) {
      const span = document.createElement("span");
      span.textContent = label;
      wrapper.appendChild(span);
    }
    wrapper.appendChild(control);
    return wrapper;
  }

  /** Populate the field switcher from get_world's fields_available, keeping `selected` current. */
  setFieldOptions(fields: FieldDescriptor[], selected: string): void {
    this.fieldSelect.innerHTML = "";
    for (const field of fields) {
      const option = document.createElement("option");
      option.value = field.name;
      option.textContent = `${field.label} (${field.unit})`;
      this.fieldSelect.appendChild(option);
    }
    this.fieldSelect.value = selected;
  }

  /** One-line summary of the live world (tick, cell count, seed, ...). */
  setSummary(text: string): void {
    this.summaryEl.textContent = text;
  }

  setStatus(text: string): void {
    this.statusEl.textContent = text;
  }

  setBusy(busy: boolean): void {
    this.createButton.disabled = busy;
    this.stepButton.disabled = busy;
    this.fieldSelect.disabled = busy;
  }
}
