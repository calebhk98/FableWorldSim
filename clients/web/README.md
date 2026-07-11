# clients/web

Thin 3D-globe client (TypeScript + Three.js). Talks to the headless API only —
no simulation logic lives here (see `docs/ARCHITECTURE.md`'s hexagonal rule).

## Run it

```bash
# 1. Start the headless core (from the repository root):
docker compose -f docker/compose.yaml up --build      # publishes :8080

# 2. Start this client (from clients/web/):
npm install
npm run dev                                            # http://localhost:5173
```

The client's API base URL defaults to `http://localhost:8080`, matching the compose
publish port. Point it elsewhere with a `.env.local`:

```
VITE_FWS_API_URL=http://my-server:8080
```

`npm run build` type-checks (`tsc --noEmit`) and produces a static `dist/` bundle;
`npm run typecheck` runs just the type-check.

## What it consumes

Every network call in this client goes through `src/api/*` or `src/ws/client.ts`, wrapping
real, self-described endpoints (`api/app.py`):

| Surface | Endpoint | Used for |
|---|---|---|
| Discovery | `GET /schema` | (available; not yet rendered — see below) |
| Discovery | `GET /commands` | Command reference list in the side panel |
| Discovery | `GET /ws/schema` | (available; not yet rendered) |
| Settings | `GET /settings`, `PUT /settings/{path}` | The settings form (grid backend/resolution, compute backend) |
| Commands | `POST /commands/set_compute_backend` | Hot-swapping the compute backend from the UI |
| Commands | `POST /commands/run_world_sweep` | "Run world sweep" panel — the only command that actually simulates a world |
| Observability | `GET /metrics` | Server counters in the status section |
| Streaming | `WS /ws` | Live event log (`hello`, `setting_changed`, `compute_backend_changed`, …) |

`src/api/commands.ts` also wraps `ping`, `list_grid_backends`, `probe_hardware`, and
`get_run_telemetry`/`get_compute_backend` for completeness, even though the UI doesn't
surface all of them yet — they're one line away from a panel button.

## API gaps (found while building this client)

This API build has **no endpoint for per-cell grid geometry or a per-cell data layer** —
the two things a real "render the DGGS globe with a height/temperature overlay" feature
needs. Concretely:

1. **No cell geometry endpoint.** `ports/grid.py`'s `Grid` (cells, centroids, boundaries,
   neighbors) is never exposed over REST/commands. A client can't ask "what are this
   world's cells and where are they on the sphere."
2. **No per-cell field endpoint.** `run_world_sweep` (`api/commands.py`) is the only
   command that simulates a world, and it returns *aggregate* results only: a habitability
   score + component breakdown per candidate seed, and (for kept winners) a non-spatial
   deep-run summary — species population totals and a river-channel count, never a
   `cell_id -> value` mapping for height, temperature, or anything else.
3. **No persisted, steppable world.** Sweep worlds are generated and scored/deepened
   in-process, then discarded — there's no `create_world`/`get_world`/`step` command a
   client could call once, then re-query as it advances. The globe has nothing to poll.
4. **Two WS event types are declared but never emitted.** `api/ws_events.py`'s
   `EVENT_MODELS` includes `heartbeat` and `run_telemetry`, and `/ws/schema` describes
   both — but no code path in this build actually publishes them. `AppState.record_run_telemetry`
   (`api/state.py`) is the only emitter for `run_telemetry` and nothing calls it;
   `run_world_sweep`'s handler sets `state.sim_telemetry` directly instead. `heartbeat` has
   no emitter at all. This client still listens for both (see `src/ws/client.ts`), so it
   picks them up for free the day a server build wires an emitter.

Given the "talk only to the real API, don't invent endpoints" constraint, this client does
**not** fabricate calls to `/grid/cells` or similar. Instead:

- `src/globe/placeholder-sphere.ts` scatters a Fibonacci-sphere point cloud (a generic,
  backend-agnostic sphere-sampling method) whose *density* — not shape — responds to the
  live `grid.resolution` setting, clearly documented as **not** a reproduction of any DGGS
  backend's real tessellation.
- `src/globe/field-layer.ts` first asks the server's own `/commands` list (capability
  detection, matching names like `get_cell_field`) whether a per-cell field command exists;
  finding none, it falls back to a synthetic, clearly-labeled demo layer (a latitude-based
  gradient — generic textbook shading, not FableWorldSim's climate model) so the
  Viridis palette + legend machinery (`src/globe/palette.ts`, `src/ui/legend.ts`) has
  something real to render and can be verified end-to-end.
- The globe UI and the legend both visibly flag this as demo data (a "(demo)" suffix and an
  in-panel note), so it can never be mistaken for a simulated result.
- Everything *else* in the client — schema/command discovery, settings read/write, the
  compute-backend hot-swap, the world-sweep runner and its results table, the metrics
  poll, and the live WS event log — is wired to real endpoints and real data.

Closing gaps 1–3 (a `Grid`-backed cell-geometry query plus a per-cell field query, ideally
against a world the sweep can hand back a handle to) is the natural next step; the seam for
plugging live data into the globe is `src/globe/field-layer.ts`, and the point cloud itself
lives in `src/globe/placeholder-sphere.ts` — both are intentionally small and isolated so
swapping the placeholder for a real fetch doesn't touch rendering, palette, or UI code.

## Layout

```
src/
  config.ts              API base URL / WS URL
  api/
    http.ts              fetch wrapper (one error type, JSON in/out)
    types.ts              TS mirrors of the API's pydantic models
    discovery.ts           GET /schema, /commands, /ws/schema + capability lookup
    settings.ts             GET/PUT /settings
    commands.ts             POST /commands/{name} (generic + typed helpers)
    metrics.ts               GET /metrics
  ws/
    client.ts              typed WS client with auto-reconnect
  globe/
    scene.ts                Three.js scene/camera/controls/render loop
    placeholder-sphere.ts   DGGS-cell stand-in (see API gaps above)
    field-layer.ts           data-layer + capability-detected fallback
    palette.ts               Viridis (colorblind-safe, perceptually uniform) + legend data
  ui/
    panel.ts                composes the side panel
    status.ts                connection + /metrics
    settings-form.ts          grid/compute settings editor
    commands-list.ts          renders GET /commands
    sweep-panel.ts            run_world_sweep runner + results
    event-log.ts              live WS event feed
    legend.ts                 renders palette.ts's structured legend data
  main.ts                  bootstrap
  style.css
```
