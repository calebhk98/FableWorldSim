/**
 * Typed client for the `/ws` event stream (api/app.py `_register_stream`).
 *
 * The server sends `hello` first, then fans out whatever it publishes on
 * `EventBus` (api/state.py): `setting_changed` and `compute_backend_changed`
 * are wired to real emitters today. `heartbeat` and `run_telemetry` are
 * declared in the event vocabulary (api/ws_events.py `EVENT_MODELS`) and are
 * handled below, but nothing in this API build currently publishes them
 * (`run_world_sweep` sets `state.sim_telemetry` directly without going
 * through `AppState.record_run_telemetry`, which is the only code path that
 * would emit `run_telemetry`) — see clients/web/README.md. This client
 * subscribes to the full vocabulary regardless, so it picks them up for
 * free the moment a server build wires an emitter.
 */

import { wsUrl } from "../config";
import type { WsEvent } from "../api/types";

export type WsListener = (event: WsEvent) => void;
export type WsStatusListener = (connected: boolean) => void;

const RECONNECT_DELAY_MS = 2000;

export class WorldSocket {
  private socket: WebSocket | null = null;
  private readonly listeners = new Set<WsListener>();
  private readonly statusListeners = new Set<WsStatusListener>();
  private closedByCaller = false;

  /** Open the connection (idempotent; call once from app bootstrap). */
  connect(): void {
    this.closedByCaller = false;
    this.open();
  }

  /** Stop reconnecting and close the live socket, if any. */
  disconnect(): void {
    this.closedByCaller = true;
    this.socket?.close();
  }

  /** Subscribe to every event; returns an unsubscribe function. */
  onEvent(listener: WsListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Subscribe to connect/disconnect transitions; returns an unsubscribe function. */
  onStatusChange(listener: WsStatusListener): () => void {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  private open(): void {
    const socket = new WebSocket(wsUrl());
    this.socket = socket;

    // `hello` is always the server's first message (api/app.py), so treating
    // any message as evidence of a live connection is simplest and correct.
    socket.onmessage = (message: MessageEvent<string>) => {
      const event = JSON.parse(message.data) as WsEvent;
      if (event.type === "hello") {
        for (const listener of this.statusListeners) listener(true);
      }
      for (const listener of this.listeners) listener(event);
    };

    // A dropped connection is expected (server restart, sleeping laptop);
    // reconnect on a fixed delay rather than surfacing an error state.
    socket.onclose = () => {
      for (const listener of this.statusListeners) listener(false);
      if (!this.closedByCaller) {
        window.setTimeout(() => this.open(), RECONNECT_DELAY_MS);
      }
    };
  }
}
