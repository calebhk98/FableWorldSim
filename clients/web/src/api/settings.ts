/**
 * The settings surface (api/app.py `_register_settings`): the same schema
 * backing the config/ folder and the /commands `set_setting`/`get_settings`
 * pair, exposed directly over REST so a form field can read/write one
 * dotted path without round-tripping the whole tree.
 */

import { getJson, putJson } from "./http";
import type { SettingsTree, SettingValue } from "./types";

/** GET /settings — the whole settings tree. */
export function getSettings(): Promise<SettingsTree> {
  return getJson("/settings");
}

/** GET /settings/paths — every dotted option path (for a generic editor). */
export function getSettingPaths(): Promise<string[]> {
  return getJson("/settings/paths");
}

/** GET /settings/{path} — one option's current value. */
export function getSetting(path: string): Promise<SettingValue> {
  return getJson(`/settings/${encodeURIComponent(path)}`);
}

/**
 * PUT /settings/{path} — change one option; the server validates against
 * the settings schema and broadcasts a `setting_changed` WS event, so every
 * other connected viewer (including this one, on its own WS feed) learns
 * about the change the same way.
 */
export function putSetting(path: string, value: unknown): Promise<SettingValue> {
  return putJson(`/settings/${encodeURIComponent(path)}`, { value });
}
