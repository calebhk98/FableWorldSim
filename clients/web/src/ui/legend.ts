/** Renders a `Legend` (structured palette data, see globe/palette.ts) as DOM. */

import type { Legend } from "../globe/palette";

export function renderLegend(container: HTMLElement, legend: Legend): void {
  container.innerHTML = "";

  const title = document.createElement("div");
  title.className = "legend__title";
  title.textContent = legend.title;
  container.appendChild(title);

  const bar = document.createElement("div");
  bar.className = "legend__bar";
  bar.style.background = `linear-gradient(to right, ${legend.stops.map((s) => s.color).join(", ")})`;
  container.appendChild(bar);

  const labels = document.createElement("div");
  labels.className = "legend__labels";
  for (const stop of legend.stops) {
    const label = document.createElement("span");
    label.textContent = stop.value.toFixed(2);
    labels.appendChild(label);
  }
  container.appendChild(labels);

  const unit = document.createElement("div");
  unit.className = "legend__unit";
  unit.textContent = legend.unit;
  container.appendChild(unit);

  const note = document.createElement("p");
  note.className = "legend__note";
  note.textContent =
    "Cells are rendered as true boundary polygons from the API's grid_geometry " +
    "(a centroid-fan triangulation). See README → Remaining limits.";
  container.appendChild(note);
}
