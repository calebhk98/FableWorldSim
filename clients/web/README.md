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

## What it renders

On load, the client asks the server for its live world (`get_world`); if none exists
yet it creates one (`create_world`, seed 1, resolution 0 — the server's own cheapest
default, 122 cells). Either way it then:

1. Fetches `grid_geometry` — every cell's real lat/lng centroid — and places one point
   per cell on the globe (`src/globe/cell-geometry.ts`).
2. Fetches `query_field` for the selected field (height by default) and colors each
   point with the colorblind-safe, perceptually-uniform Viridis palette
   (`src/globe/field-layer.ts`, `src/globe/palette.ts`), with a structured legend
   (`src/ui/legend.ts`) that's always in sync with what's actually rendered.

The **World** panel lets you:

- **Create world** — pick a seed and grid resolution and call `create_world`, replacing
  the live world (a fresh set of cell geometry is fetched afterward).
- **field** — switch between whatever `get_world`'s `fields_available` reports (height,
  temperature, precipitation as of this build) by re-calling `query_field`.
- **Step** — advance the live world N orchestrator ticks via `step_world`, then refetch
  the current field so you can see the sim actually move (plus a one-line species/civ
  population readout from `step_world`'s own response).

There is no synthetic/demo fallback anymore: every point, color, and legend value on
the globe comes from a real API response. If the API is unreachable, or `create_world`
fails (e.g. an unsupported grid backend), the panel shows the real error instead of
silently substituting fake data — see `main.ts`'s error handling and `WorldPanel.setStatus`.

## What it consumes

Every network call in this client goes through `src/api/*` or `src/ws/client.ts`, wrapping
real, self-described endpoints (`api/app.py`):

| Surface | Endpoint | Used for |
|---|---|---|
| Discovery | `GET /schema` | (available; not yet rendered — see below) |
| Discovery | `GET /commands` | Command reference list in the side panel |
| Discovery | `GET /ws/schema` | (available; not yet rendered) |
| Settings | `GET /settings`, `PUT /settings/{path}` | The settings form (grid backend/resolution, compute backend) |
| World | `POST /commands/create_world` | Build/replace the live, steppable world (the "Create world" button, and the on-load auto-create) |
| World | `POST /commands/get_world` | Check for an already-live world on load; the World panel's summary line |
| World | `POST /commands/step_world` | The "Step" button — advances the live world and returns fresh species/civ stats |
| World | `POST /commands/query_field` | The globe's data layer — real per-cell height/temperature/precipitation |
| World | `POST /commands/grid_geometry` | Real per-cell centroids the globe places points at |
| World | `POST /commands/export_recipe` | Wrapped (`src/api/world.ts`) but not yet surfaced in the UI — one line away from a "copy recipe" button |
| Commands | `POST /commands/set_compute_backend` | Hot-swapping the compute backend from the UI |
| Commands | `POST /commands/run_world_sweep` | "Run world sweep" panel — scores/deepens candidate seeds (separate from the live world above) |
| Observability | `GET /metrics` | Server counters in the status section |
| Streaming | `WS /ws` | Live event log (`hello`, `setting_changed`, `compute_backend_changed`, …) |

`src/api/commands.ts` also wraps `ping`, `list_grid_backends`, `probe_hardware`, and
`get_run_telemetry`/`get_compute_backend` for completeness, even though the UI doesn't
surface all of them yet — they're one line away from a panel button.

## Why Three.js only, not deck.gl

The originating issue floated deck.gl for the data overlay; this build deliberately
does **not** add it. deck.gl earns its keep when you need GPU-instanced layers
(ScatterplotLayer, HexagonLayer, …) composited over a 2D basemap (Mapbox/MapLibre) or
alongside its own orthographic globe view — and often a second framework's worth of
dependency weight and its own render loop/camera model to get there. This client is
already a single Three.js scene with an orbit-controlled *3D* globe (`scene.ts`), and
the live world tops out at a few hundred thousand points (h3 resolution 4 ≈ 288k
cells) — well within what a plain `THREE.Points` buffer with vertex colors handles at
full frame rate with zero extra dependencies. Running deck.gl's WebGL2 context
side-by-side with Three's would mean reconciling two independent camera/projection
systems and render loops for a layer Three already draws directly against the same
sphere, camera, and controls as everything else on screen — added complexity with no
rendering capability this client is missing. If a future need arises for basemap-style
2D map layers (e.g. a flattened equirectangular view) alongside the 3D globe, that's
the point where deck.gl's tradeoffs would flip and it'd be worth reconsidering.

## Remaining limits

Closing the API's per-cell geometry/field/persisted-world gaps (issue #10) let this
client stop faking data, but a few honest limits remain:

1. **Centroid-only geometry, not polygons.** `ports/grid.py`'s `Grid` port exposes
   cell centroids, area, neighbors, and edge length — but no cell-boundary vertex
   accessor. `grid_geometry` therefore returns one `{lat, lng}` centroid per cell, not
   a polygon outline, so `cell-geometry.ts` places cells as points and `scene.ts`
   draws them as small billboarded discs (a soft radial-gradient sprite) rather than
   true cell-shaped tiles. Closing this needs the `Grid` port extended with a
   boundary-vertex method first.
2. **Borders and populations are summary-only.** `step_world` returns per-species
   population *totals* and per-civ population *totals* (`populations`,
   `civ_population` in `api/world_state.py`'s `step_persisted_world`) — there is no
   per-cell population or civilization-territory field in `query_field`'s
   `FIELD_NAMES` (`height`/`elevation`/`temperature`/`precipitation` only), so the
   globe cannot yet shade civilization borders or population density; the World
   panel's step readout is the only place those numbers surface, as plain text.
3. **No river/lake layer.** `create_persisted_world` explicitly skips river/lake
   routing (`api/world_state.py`'s docstring: "a deliberate scope cut, not an
   oversight" — neither biology nor civilization reads the river network), so there
   is no hydrology field to render even though `run_world_sweep`'s deep-run summaries
   mention a river-channel count elsewhere in the API.
4. **Two WS event types are declared but never emitted.** `api/ws_events.py`'s
   `EVENT_MODELS` includes `heartbeat` and `run_telemetry`, and `/ws/schema` describes
   both — but no code path in this build actually publishes them. This client still
   listens for both (see `src/ws/client.ts`), so it picks them up for free the day a
   server build wires an emitter.
5. **`export_recipe` is wrapped but not surfaced.** `src/api/world.ts` exposes it;
   no panel button calls it yet.

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
    world.ts                 create_world/get_world/step_world/query_field/grid_geometry/export_recipe
    metrics.ts               GET /metrics
  ws/
    client.ts              typed WS client with auto-reconnect
  globe/
    scene.ts                Three.js scene/camera/controls/render loop
    cell-geometry.ts         real cell centroids (grid_geometry) -> unit-sphere points
    field-layer.ts           real per-cell field (query_field) -> colored/legend-ready layer
    palette.ts               Viridis (colorblind-safe, perceptually uniform) + legend data
  ui/
    panel.ts                composes the side panel
    status.ts                connection + /metrics
    settings-form.ts          grid/compute settings editor
    world-panel.ts            create/step/field-switch controls for the live world
    commands-list.ts          renders GET /commands
    sweep-panel.ts            run_world_sweep runner + results
    event-log.ts              live WS event feed
    legend.ts                 renders palette.ts's structured legend data
  main.ts                  bootstrap
  style.css
```
