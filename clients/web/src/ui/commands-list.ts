/** Renders the self-described command registry (GET /commands) as a reference list. */

import type { CommandDescriptor } from "../api/types";

export class CommandsList {
  readonly el = document.createElement("section");
  private readonly listEl = document.createElement("ul");

  constructor() {
    this.el.className = "panel-section";
    const heading = document.createElement("h2");
    heading.textContent = "Commands (/commands)";
    this.el.appendChild(heading);
    this.listEl.className = "commands-list";
    this.el.appendChild(this.listEl);
  }

  setCommands(commands: CommandDescriptor[]): void {
    this.listEl.innerHTML = "";
    for (const command of commands) {
      const item = document.createElement("li");
      const badge = document.createElement("span");
      badge.className = `badge ${command.mutates ? "badge--write" : "badge--read"}`;
      badge.textContent = command.mutates ? "write" : "read";
      const name = document.createElement("strong");
      name.textContent = command.name;
      const desc = document.createElement("p");
      desc.textContent = command.description;
      item.append(name, " ", badge, desc);
      this.listEl.appendChild(item);
    }
  }
}
