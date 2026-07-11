/** Composes the side panel from its focused sections. */

import { API_BASE_URL } from "../config";
import { StatusSection } from "./status";
import { SettingsForm } from "./settings-form";
import { CommandsList } from "./commands-list";
import { SweepPanel } from "./sweep-panel";
import { EventLog } from "./event-log";

export class Panel {
  readonly status: StatusSection;
  readonly settingsForm: SettingsForm;
  readonly commandsList = new CommandsList();
  readonly sweepPanel = new SweepPanel();
  readonly eventLog = new EventLog();
  readonly legendEl = document.createElement("section");

  constructor(root: HTMLElement, onGridChanged: () => void) {
    this.status = new StatusSection(API_BASE_URL);
    this.settingsForm = new SettingsForm(onGridChanged);
    this.legendEl.className = "panel-section legend";

    root.append(
      this.status.el,
      this.legendEl,
      this.settingsForm.el,
      this.sweepPanel.el,
      this.commandsList.el,
      this.eventLog.el,
    );
  }
}
