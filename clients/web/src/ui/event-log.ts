/** A capped, scrolling log of every WS event received (see src/ws/client.ts). */

import type { WsEvent } from "../api/types";

const MAX_ENTRIES = 50;

export class EventLog {
  readonly el = document.createElement("section");
  private readonly listEl = document.createElement("ul");

  constructor() {
    this.el.className = "panel-section";
    const heading = document.createElement("h2");
    heading.textContent = "Live events (/ws)";
    this.el.appendChild(heading);
    this.listEl.className = "event-log";
    this.el.appendChild(this.listEl);
  }

  push(event: WsEvent): void {
    const item = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = new Date().toLocaleTimeString();
    const body = document.createElement("span");
    body.textContent = `${event.type} ${JSON.stringify(omitType(event))}`;
    item.append(time, " ", body);
    this.listEl.prepend(item);
    while (this.listEl.children.length > MAX_ENTRIES) {
      this.listEl.lastChild?.remove();
    }
  }
}

function omitType(event: WsEvent): Record<string, unknown> {
  const { type: _type, ...rest } = event;
  return rest;
}
