/**
 * Client-wide configuration: where the headless API lives.
 *
 * The server binds to `api.host`/`api.port` from its own settings (default
 * 127.0.0.1:8080, see `docker/compose.yaml` which publishes 8080); this
 * client is a separate process (Vite dev server / static build) so it needs
 * its own pointer to that address. Override with a `.env.local` containing
 * `VITE_FWS_API_URL=http://some-host:8080` when the API isn't on localhost.
 */

const DEFAULT_API_URL = "http://localhost:8080";

export const API_BASE_URL: string = (
  (import.meta.env.VITE_FWS_API_URL as string | undefined) ?? DEFAULT_API_URL
).replace(/\/+$/, "");

/** Derive the `ws://` (or `wss://` over https) URL for the `/ws` stream. */
export function wsUrl(): string {
  const url = new URL(`${API_BASE_URL}/ws`);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
