# FableWorldSim — Implementation Plan

## Context

FableWorldSim is a from-scratch, full-planet world simulator. The repository is currently
empty (no commits). The goal is a **3D globe** (not a flat map) whose surface is an
**equal-area discrete global grid** that can be displayed/backed by H3, S2, or ISEA/DGGRID
via a **single toggle**, and on top of which we simulate — in increasing layers —
topography, climate, hydrology, flora/fauna progression, and civilizations.

Two hard constraints shape every decision below:

1. **A sphere, never a square grid.** A rectangular/lat-lon square grid *cannot* tile a sphere
   without distortion (poles pinch, cells warp, edges don't wrap). The world surface is a true
   **Discrete Global Grid System** — cells that tessellate the sphere (hexes + 12 pentagons for
   H3, quad cells for S2, icosahedral triangles/hexes for ISEA). No layer ever assumes a flat
   square lattice; all neighbor/adjacency logic goes through the `Grid` port.
2. **Equal-surface correctness.** The user explicitly called out that "the sim will favor
   different cells based on size if done incorrectly." H3 and S2 are **not** equal-area;
   ISEA (Snyder equal-area, via DGGRID) **is**. So the grid must be an *abstraction* and all
   simulation math must be **area-weighted / per-unit-area**, making results invariant to
   which backend is toggled and to cell-size variation within a backend.
3. **Any planet, not just Earth-like.** Climate is parameterized by a **PlanetConfig**: axial
   tilt, **rotation rate (signed — prograde, retrograde/reverse, or zero) including tidally
   locked (rotation period = orbital period ⇒ permanent day & night sides, day = year)**,
   orbital energy — **either given directly as W/m², or derived from (stellar luminosity, orbital
   distance)** so a user can say "1.5 AU from a G-type star" and get the flux computed —
   radius/gravity, a **target water/ocean fraction** (a first-class dial —
   "90% ocean" ↔ "bone-dry like Mars" — reconciled with the height field by solving for the
   sea-level threshold that yields it; drives surface albedo, ocean heat-capacity buffering of
   day-night/seasonal swings, and the evaporation source term for precipitation), and an
   **atmosphere** (composition, surface pressure, greenhouse strength — Venus-thick ↔ Mars-thin ↔
   airless), and **satellites/moons** (zero, one, or many → **tidal range**, which defines an
   **intertidal habitat band** distinct from "coast" and drives tidal flooding/erosion). Coriolis,
   wind bands, heat transport, and tides are *derived from* these, so Earth, Mars, Venus, the Moon,
   and invented worlds all run the same code with different config. (Tidal mechanics are minimal in
   M1 — the intertidal zone exists as a habitat; erosion/flooding coupling is roadmap.)
4. **Accuracy-first, but tunable down for speed**, and must **scale from a 2010 laptop to a
   multi-GPU desktop/cluster** — with or without a GPU present.

### Decisions locked with the user (Q&A)

| Question | Answer | Consequence |
|---|---|---|
| Core language | "Doesn't matter — must run with/without GPU, doable by interns & basic AI" | **Python** core (numpy↔CuPy/JAX, same code CPU/GPU; best ecosystem for H3/S2/climate; most AI/intern-friendly). **TypeScript + Three.js/deck.gl** thin web client. |
| Rendering | Headless core + thin clients | Sim is a **headless library + API server**; all UIs (web, CLI, AI agent) are clients of the same API. No GUI-only actions. |
| First milestone | "Full minimal, without any future limitations" | M1 = a **thin vertical slice through every layer** (globe→topo→climate→creature→country), each layer minimal but with **full-shaped interfaces & data models** so nothing is a dead-end. |
| Scaffolding | Front-load it | Build CI/pre-commit discipline, hexagonal ports & adapters, and the config/API/i18n/modding skeleton **before** sim features. |

> The language choice is a *recommendation* derived from the stated criteria, not a user
> mandate. It is isolated behind ports (see below), so swapping the compute backend — or even
> rewriting a hot kernel in Rust/C++ later — is a contained change.

**Simulation time & horizon.** A tick maps to real simulated time (configurable; default on the
order of years so a run spans the intended **~1–5k-year, human-migration horizon**). At this
horizon M1 models **progression, not significant evolution** — trait *drift*/evolution is an
explicit long-horizon roadmap item. Climate and tectonics evolve on much slower clocks than
populations, so the orchestrator supports **multi-rate stepping** (fast biology, slow geology)
rather than one global timestep.

---

## Architecture: Hexagonal (Ports & Adapters)

The whole system is organized so that **any library can be swapped in ≤15 min** and the same
core runs on a laptop, a GPU box, or a cluster. Domain logic depends only on **ports**
(abstract interfaces); concrete libraries are **adapters** wired at startup from config.

```
                 ┌──────────────────────────────────────────────┐
   clients ──────┤  API layer (REST + WebSocket, self-describing) │
 (web / CLI /    └───────────────────────┬──────────────────────┘
  AI agent)                               │  commands + queries
                 ┌───────────────────────▼──────────────────────┐
                 │              DOMAIN CORE (pure Python)          │
                 │  world model · climate · hydrology · biology ·  │
                 │  civilization · time-stepping orchestrator      │
                 └───┬───────┬────────┬────────┬────────┬─────────┘
              PORTS: │Grid   │Compute │Storage │Content │Rng/Clock ... 
                 ┌───▼───┐┌──▼─────┐┌─▼──────┐┌▼───────┐┌─────────┐
        adapters │H3/S2/ ││numpy/  ││sqlite/ ││JSON/   ││seeded   │
                 │ISEA   ││CuPy/JAX││parquet ││TOML mod││PRNG     │
                 │DGGRID ││/Dask   ││/zarr   ││loader  ││         │
                 └───────┘└────────┘└────────┘└────────┘└─────────┘
```

### Ports (initial set)

- **`Grid`** — discrete global grid. `cells()`, `area(cell)`, `centroid(cell)`,
  `latlng(cell)`, `neighbors(cell)`, `edge_length(cell, neighbor)`, `resolution`,
  `parent/children` (for LOD). Adapters: `H3Grid`, `S2Grid`, `ISEAGrid` (DGGRID).
  **Toggle via config/API.** **H3** is the frictionless default (pure-Python wheels everywhere);
  **ISEA** is opt-in for users who want strict equal-area cells (see Platform & packaging).
  **Toggle semantics:** H3/S2/ISEA cells are *not* spatially congruent, so the backend is chosen
  at **world-creation time** (cheap: pick up front). Switching an *existing* world to another
  backend is an explicit **reprojection** operation (resample every field across tessellations) —
  supported but not free, and clearly distinguished in the API from the create-time choice.
  **Contract — variable degree:** neighbor count is *not* constant and is **topology-dependent**:
  H3 hexes have 6 but its 12 pentagons have 5; the recommended DGGRID **ISEA3H** topology is *also*
  hex + 12 pentagons (6/5, but **exactly** equal-area — pentagons are exactly 5/6 of a hex); S2
  quads have 4; DGGRID's ISEA4T *triangle* topology has 3. We standardize on **ISEA3H** for the
  ISEA adapter (equal-area, H3-like adjacency). Regardless, all flux/diffusion math (climate,
  hydrology, migration) must iterate `neighbors(cell)` and weight by `edge_length`/`area` — never
  hardcode a degree — or the backend toggle silently breaks at pentagon/triangle cells. A property
  test asserts no code path assumes a fixed degree.
  **Caveat — DGGRID is a native binary:** the ISEA adapter (via `dggrid4py`) shells out to the
  compiled C++ **DGGRID** program and parses its output files; it is *not* pip-installable pure
  Python (Windows build is notably fiddly; conda-forge has a prebuilt). H3 has healthy wheels
  (`h3` v4.x, actively maintained). **S2's Python story is weak — flag it:** `s2sphere` was
  archived in 2023 (last release 2017) and Google's official `s2geometry` bindings are documented
  as unstable, so **S2 is a display/interop toggle, not a recommended default**; H3 is the default
  and ISEA the accuracy opt-in.
- **`SubsurfaceGrid`** — a **volumetric** layer under the surface shell, for creatures/civs that
  live *inside* the world (dwarves, worms, Dune sandworms, hobbit-holes). The surface `Grid` is a 2D
  spherical shell; this adds **depth** — a graph of subsurface nodes (per surface cell × depth band,
  later cavern/tunnel segments) with **vertical + lateral adjacency distinct from surface
  `neighbors()`**, so "tunnel under the mountain range to the other side" is representable, not just
  "hop between adjacent mountain hexes." **M1 ships the port + a minimal impl** (a few depth bands
  per cell, vertical adjacency, a "diggability"/rock-type field). Full **cave-network topology,
  tunnel connectivity, and cross-massif digging** are roadmap — but the port exists now so it isn't a
  retrofit. Subterranean species and dwarf mining/migration route through *this* graph; surface and
  subsurface territory can overlap (humans above, dwarves below).
- **`Ocean`** — surface ocean state & circulation (see Ocean & seasonality section). Kept a distinct
  port from atmosphere `climate` so the wind→current→heat-advection coupling is explicit and the
  model can deepen (surface layer → full thermohaline) without touching consumers.
- **`ArrayBackend`** — per-cell field math via the **Python Array API standard**. Goal: the same
  domain code runs on `numpy` (CPU), `cupy`/`jax` (single/multi-GPU), or `dask`/`ray` (cluster),
  auto-detecting hardware and falling back to CPU. **Reality check:** the standard narrows the
  gap but does *not* make backends drop-in — CuPy tracks numpy closely, but **JAX is functional
  & immutable** (no in-place mutation; `jit` constrains control flow and shapes), so kernels need
  a JAX-friendly style. We treat "same code, every backend" as a **target enforced by a shared
  backend-conformance test suite**, not a freebie; hot kernels that can't be expressed uniformly
  get a per-backend adapter behind the same port method.
- **`Kernel`** — hot compute kernels (e.g. flow-accumulation) behind a stable signature, with
  interchangeable **pure-Python** and **out-of-Python (GPU/native)** adapters. This is the concrete
  proof of the cross-language 15-minute-swap claim (see Platform & packaging).
- **`Storage`** — world snapshots & time series (sqlite / parquet / zarr adapters). **Every
  snapshot carries a schema version**; a **migration chain** upgrades old saves as the model
  evolves across milestones, so worlds created in M1 still load later. A round-trip + migration
  test guards this from day one.
- **`Security`/`Auth`** — permission-scoped commands for the API (read vs. mutate), token auth for
  online mode, and the sandbox policy for mod code (below). Introduced now so "online + mutable API
  + mods" isn't an afterthought.
- **`ContentRegistry`** — moddable data (species, biomes, climate classes, tech, languages).
- **`GraphLayout`** — tech/language-tree layout, client-side; core just emits the graph. Note a
  tech tree is a **DAG** (multiple prerequisites), not a pure tree, so the default is **`elkjs`**
  (or Cytoscape.js+ELK); **`@dagrejs/dagre`** (the maintained fork — *not* the stale `dagre`) is
  the lightweight alternative. Plain **`d3-hierarchy`** only handles strict single-root trees, so
  it's suitable for language *family* trees but not the prerequisite DAG.
- **`Clock` / `Rng`** — deterministic, seedable time and randomness (reproducible sims).

**Rule:** domain core imports *ports only*. A grep-enforced check (see discipline) forbids
domain modules from importing adapter/library packages directly.

---

## The equal-area rule (single most important invariant)

Every quantity that could be biased by cell size is stored/computed as an **intensive**
value (per m²) or explicitly **area-weighted** when aggregated:

- Insolation = W/m²; population uses **density** (individuals/m²) internally, `count = density × area(cell)`.
- Flows between cells (heat, moisture, migration) are **fluxes across shared edge length**,
  not per-cell constants.
- Any sum over the globe multiplies by `area(cell)`.

A cross-backend **conservation test** (total energy/mass/population invariant under grid
toggle and resolution change, within tolerance) guards this in CI. This directly answers the
user's "favor cells by size" worry and is written **first** (red) for M1.

> **What this test proves (and doesn't):** it does *not* claim H3 or S2 are equal-area (they
> aren't). It verifies that our *simulation code correctly area-weights every aggregation* using
> each backend's own reported `area(cell)` — so a non-equal-area backend still gives unbiased
> global totals. Equal-area ISEA remains the accuracy default because uniform cells also reduce
> *local* per-cell discretization error, not just global sums.

---

## Ocean, seasonality & climate coupling

The climate is a **coupled atmosphere–ocean system**, not just per-cell energy balance — because the
requirement is real effects like *the Gulf Stream keeping Northern Europe warmer than its latitude
deserves*. The `Ocean` port is shaped for the full thing; **M1 implements the surface tier**:

- **Wind-driven surface ocean (M1):** wind stress → **Ekman transport**; the interior obeys the
  **Sverdrup balance**; the classic **Stommel model** yields **western-boundary intensification**
  (narrow, fast currents like the Gulf Stream/Kuroshio on western ocean margins). Surface currents are
  **routed around continents** (the land mask from hydrology) and **advect heat**, so ocean gyres
  redistribute warmth: "wind pushes surface water, water carries heat, land redirects it into boundary
  currents." Ocean heat capacity also buffers day-night/seasonal swings.
- **Radiative transfer + moisture + precipitation (M1):** a simple radiative-transfer step (shortwave
  in, longwave out, greenhouse from atmosphere composition) feeds temperature; **evaporation** (over
  ocean/lakes, temperature-driven) supplies atmospheric moisture that winds transport and drop as
  **precipitation** (orographic + convergence). This closes the wind↔ocean↔heat↔rain loop.
- **Cryosphere & ice–albedo feedback (M1):** **snow and ice cover** are modeled quantities — cells
  below freezing (seasonally) accumulate snow/ice, which has **high albedo** (reflects sunlight). This
  closes the critical **ice–albedo feedback loop**: more ice → more reflection → cooler → more ice (and
  the reverse), which is a large part of why poles are cold and why the climate has tipping behavior.
  Sea ice and land snow both count; they melt back with warming and season. Without this the poles and
  climate stability come out wrong.
- **Orographic / mountain-barrier effects (M1):** topography actively shapes weather, not just
  elevation-as-temperature. Mountains **lift air on the windward side** (cooling → **precipitation**)
  and leave a dry **rain shadow** on the lee side; tall ranges act as **barriers that deflect wind and
  block moisture** (the Himalaya/Tibet effect — blocking monsoon moisture, steering flow). This is
  derived from the height gradient along the wind vector, so it emerges wherever terrain and wind
  interact, on any planet.
- **Seasons (M1):** tilt + orbital position drive a **seasonal insolation cycle** → winter/summer
  temperature and precipitation. **Monsoons emerge** from seasonal land–sea heat contrast (land heats
  and cools faster than ocean) rather than being scripted.
- **Roadmap (port already shaped for it):** full **3D thermohaline circulation** with vertical layers,
  **salinity + density**, and **deep overturning** (the conveyor); **isostatic rebound & local
  subsidence** (crust rising/sinking under ice/sediment load, feeding back into topography/sea level);
  day-to-day **weather** (storms/fronts). **ENSO-like interannual oscillations** are expected to be
  **emergent** once equatorial ocean–atmosphere coupling has enough fidelity (delayed-feedback
  dynamics) — we add a diagnostic test that looks for such oscillations rather than hard-coding them.

Emergence is the design bet throughout: banding, monsoons, boundary currents, and (later) ENSO should
**fall out of the physics**, and the verification suite checks for them qualitatively.

---

## Tech stack (all behind ports)

| Concern | Default adapter | Alternatives (swappable) |
|---|---|---|
| Grid | **h3** (frictionless default) | **ISEA/DGGRID** (opt-in, strict equal-area), **s2geometry**/`s2sphere` |
| Array/compute | **numpy** | **cupy**, **jax**, **dask**, **ray**, **taichi** |
| Climate fields | xarray-style ndarray on the Grid | — |
| API server | **FastAPI** (REST + WS) + pydantic schemas | gRPC later |
| Config | **pydantic-settings** + TOML files | — |
| i18n | **Fluent** (`fluent.runtime`) locale files | gettext/ICU |
| Modding | entry-points + data files (TOML/JSON) | — |
| Frontend | **TypeScript + Three.js** (deck.gl for data overlays) | CesiumJS |
| Tests | **pytest** + hypothesis | — |
| Lint/format/type | **ruff** + **mypy** | — |
| Pre-commit | **pre-commit** framework | — |
| Duplicate detection | **jscpd** (polyglot) or `pylint --duplicate-code` | — |
| Packaging | **uv**/pip + Docker | — |

---

## Repository structure

```
fableworldsim/
├── core/                      # pure-Python domain, imports PORTS only
│   ├── grid/                  # Grid port + equal-area helpers
│   ├── topography/            # height field: load / generate / edit
│   ├── subsurface/            # volumetric depth layers / caverns (dwarves, worms)
│   ├── climate/               # radiative transfer, insolation, temp, wind, precip, seasons
│   ├── ocean/                 # surface currents (Ekman/Sverdrup/Stommel), heat advection
│   ├── hydrology/             # rivers, lakes, sea level, intertidal
│   ├── biology/               # traits, diets/food-web, populations, disease, migration
│   ├── civilization/          # influence/borders, settlements, language, tech, resources
│   ├── hazards/               # emergent + scripted disturbances (fire, volcano, impact)
│   ├── chronicle/             # event log / history; feeds the wiki
│   └── sim/                   # orchestrator, multi-rate time-stepping, fidelity knobs
├── ports/                     # abstract interfaces (Grid, ArrayBackend, Storage, ...)
├── adapters/                  # concrete libs: grid_h3, grid_s2, grid_isea, compute_numpy, ...
├── api/                       # FastAPI app: commands, queries, WS stream, OpenAPI/discovery
├── content/                   # base "mod": species, biomes, climate classes, tech, locales
│   └── locales/               # en, es, ja, ... (+ community Klingon)
├── clients/
│   ├── web/                   # TS + Three.js globe (thin client)
│   └── cli/                   # scriptable client over the API
├── config/                    # default.toml + overrides; every option settable here
├── tests/                     # unit + cross-backend conservation + property tests
├── tools/hooks/               # custom pre-commit checks (file length, port-boundary, nesting)
├── docker/                    # headless core image + compose
└── docs/
```

**File-length guard:** custom pre-commit hook fails any source file over a set limit
(~400 lines) to force decomposition.

---

## Engineering discipline (front-loaded, enforced on local commit)

`pre-commit` runs on every commit; CI re-runs the same set:

1. **Format + lint** — ruff. Never-nesting is enforced by **`PLR1702`** (too-many-nested-blocks,
   `max-nested-blocks` configurable, default 5) plus **`C901`** (McCabe complexity) — both ship in
   ruff today, so *no custom nesting hook is needed*. **Docstring/commenting style** (an explicit
   user requirement) is enforced via ruff's **`D` (pydocstyle) rules** + **`interrogate`** for
   docstring-coverage thresholds — previously missing from this list.
2. **Type-check** — mypy on core + ports (adapters typed against port signatures). *Expect missing
   third-party stubs:* `h3` has an open unresolved stubs issue, `s2sphere` is unmaintained, and the
   DGGRID wrapper is untyped — so we write thin `.pyi` **stub files** (PEP 561, in `adapters/stubs/`)
   for the surface we call rather than assuming type-checking reaches into those libraries.
3. **Tests (fast subset)** — pytest; full suite in CI. Supports **red→green** workflow:
   commit the failing test (red), then the implementation (green) as separate commits.
4. **File-length limit** — custom hook (`tools/hooks`).
5. **Duplicate-code detection** — jscpd across Python + TS; fails on copy-paste.
6. **Port-boundary check** — custom hook greps `core/` for forbidden adapter/library imports
   (enforces inverse dependencies / 15-min swap).
7. **Single-source-of-truth check** — names/enums (climate classes, biomes, etc.) defined once
   in `content/` registries; i18n keys reference those ids; a hook flags hard-coded duplicates.

Everything is configured in `.pre-commit-config.yaml` + `pyproject.toml` so it "checks
automatically on local commit," as requested.

---

## Config / API / Modding / i18n — the three control surfaces

Requirement: **every option changeable via in-sim UI, via API, or via the config folder.**
Achieved by a single settings model:

- One `pydantic-settings` schema is the source of truth for all options.
- **Config folder** (TOML) hydrates it at startup; **API** exposes get/set on the same schema;
  **UI** is just an API client. No option exists in only one surface.
- **API is self-describing**: `/schema` (OpenAPI) + a `/commands` discovery endpoint list every
  command, its params, and its effects — so an AI can enumerate capabilities, read the sim state,
  and mutate it with no mouse/keyboard. **Note:** OpenAPI 3.x covers only the **REST** half —
  it has no native WebSocket description. The WS streaming interface (message types, event
  schemas) gets its **own discovery endpoint** (`/ws/schema`, JSON-Schema per event; AsyncAPI doc
  optional), so "self-describing" holds for both halves rather than silently only for REST.
- **Modding**: base content is itself a mod loaded through `ContentRegistry`; user mods are
  data files (+ optional Python hook entry-points) with load order. Species, biomes, climate
  classes, tech nodes, and **languages** are all mod content.
- **i18n**: all user-facing strings are Fluent keys in `content/locales/<lang>.ftl`. A climate
  name (etc.) is defined **once** as a key and translated per locale — satisfying "state names
  1×." Adding Spanish/Japanese/Klingon = adding a locale file; selectable via config/API/UI.
- **Online/offline**: the core is a local server; "online" is the same server hosted remotely.
  No hard dependency on network.

---

## Platform, packaging & hardware scaling (decisions, not defaults-by-accident)

- **Windows is the primary platform; Linux/Ubuntu a supported target.** Stated explicitly because
  it collides with two native dependencies:
  - **DGGRID (ISEA)** is a compiled C++ binary. On Windows we either ship a prebuilt binary or
    build via the toolchain; to keep a **frictionless pure-Python default**, the out-of-the-box
    grid backend is **H3** (real wheels on Windows/Linux), with **ISEA/DGGRID opt-in** for users
    who want strict equal-area. The equal-area *correctness* (area-weighting) does not depend on
    which backend is default — see the conservation test.
  - **CuPy/JAX GPU** works on Windows but with more CUDA-toolkit/driver friction than Linux. The
    **default is CPU (numpy)**; GPU is auto-enabled when a working toolkit is detected, else it
    falls back — no hard GPU requirement anywhere.
- **Docker/WSL2 is an explicit option, not the assumed answer.** The headless core ships a Linux
  Docker image (clean home for DGGRID + CUDA). Running it on Windows via Docker Desktop/WSL2 is a
  *documented choice* for users who want the native bits handled for them — but native Windows
  (H3 + CPU, GPU optional) must also work without containers.
- **Hardware auto-scaling** is one **fidelity** knob (grid resolution + timestep + model detail)
  times a **backend** selection: `numpy` (2010 laptop, CPU) → `cupy`/`jax` (1–N GPUs, incl.
  2×3090/2×5090) → `dask`/`ray` (cluster). Auto-detect picks sane defaults; all overridable via
  config/API. A later **fidelity autotuner** sets resolution/timestep from detected hardware.
- **Use the whole machine — auto-detected.** At startup the sim **probes the host** (GPU count/VRAM,
  CPU core count, total RAM) and orchestrates to use *all* of it, with the *same code*:
  - **Multi-GPU** (e.g. **both 3090s**): fields are **sharded across GPUs** (device mesh via
    `jax`/`cupy`); the ArrayBackend hides which device holds which shard. One GPU, two, or four scale
    the same way.
  - **All CPU cores** (e.g. a **Ryzen 9**): the CPU backend parallelizes across every core
    (thread/process pool or `dask` local scheduler), not one core.
  - **Memory-aware**: `Storage` is **chunked/out-of-core**, so a **64 GB DDR5** box holds more
    resident and runs bigger worlds, while a **4 GB DDR4 laptop with no GPU** runs the *identical
    build* at lower fidelity by streaming tiles and falling back to numpy-on-CPU. Detected RAM caps
    resident resolution; OOM triggers graceful fidelity downgrade rather than a crash.
  - The user can pin any of this (force CPU, choose GPUs, cap RAM) via config/API; auto is just the
    default.

### The 15-minute-swap proof: flow-accumulation as an out-of-Python kernel

The "swap any library in ≤15 min, even across languages" claim is **proven, not asserted**, using the
hardest realistic case. River **flow-accumulation** (sequential graph traversal over the DGGS) is
implemented behind a `Kernel` port with **two interchangeable adapters**: (a) a reference **pure-Python/
numpy** implementation, and (b) an **out-of-Python kernel** — a GPU/native implementation (WGSL/`wgpu`
compute, or a small Rust/C extension) callable through the same port signature. A conformance test runs
*both* on identical inputs and asserts matching output; switching which one the sim uses is a one-line
config change. This simultaneously (i) demonstrates the port boundary genuinely permits non-Python
implementations, and (ii) gives the single most performance-critical kernel a fast path on capable
hardware while keeping the CPU-only laptop working on the reference path.

---

## Cross-cutting concerns (surfaced by review — designed in, not bolted on)

- **API & mod security.** "Online" + a fully mutation-capable API + mods with code hooks is a
  remote-code-execution-shaped surface. Mitigations, from M1: commands are **permission-scoped**
  (read vs. mutate) behind the `Security` port; online mode requires **auth tokens** + **rate
  limiting**; mods are **data-first** (TOML/JSON content needs no code) and any **Python hook runs
  sandboxed** with a restricted API (or is disabled by default and opt-in per trust level). Mod
  loading, malformed-mod handling, and a malicious-mod test are explicit verification items.
- **Concurrency / authority model.** The sim is a **single authoritative writer** with many read
  clients; mutations are **serialized commands** (queue/transaction), so multiple humans/AIs can
  observe freely and submit changes without corrupting state. Whether independent agents can each
  drive a different civilization concurrently is a defined roadmap extension, not an accidental
  behavior.
- **Capacity & performance budget (makes "auto-scale" falsifiable).** Name concrete targets per
  tier and benchmark them in CI: e.g. a **coarse grid (~low-res H3, ~10⁴–10⁵ cells)** must run on a
  no-GPU 2010-class laptop; **mid-res (~10⁶ cells)** on a single modern desktop; **high-res** on
  GPU (2×3090/2×5090 via multi-device sharding) → cluster (dask/ray). Fidelity (resolution ×
  timestep × model detail) is the dial; a **perf-regression test** guards each tier. Known hard
  kernels — **river flow-accumulation** (sequential graph traversal, doesn't trivially vectorize
  under JAX or shard cleanly across partitions) and **biology** (O(species × cells × neighbors)) —
  get dedicated per-backend implementations and profiling, not a naive "swap the backend" bet.
- **Multi-GPU vs. cluster are different regimes.** Single-node multi-GPU (device sharding) and
  multi-node cluster (dask/ray) are distinct scaling paths behind the same `ArrayBackend` port, not
  one interchangeable adapter — called out so they're budgeted separately.
- **Observability & degenerate-state guards.** Structured logging + metrics export from the
  headless core; per-cell inspector and time-series probes over the API; automatic detection of
  **runaway/degenerate states** (population explosion/collapse, energy non-conservation drift) that
  fail loudly. Prerequisite for the fidelity autotuner and for debugging emergent behavior.
- **Error handling & graceful degradation.** Defined behavior for DGGRID subprocess failure,
  malformed mod/config, OOM on large grids (auto-downgrade fidelity rather than crash), and
  corrupted/older saves (migrate or fail cleanly).

---

## Feature additions adopted from review (M1 = design in now; ⊕ = roadmap)

High-leverage capabilities that must be *designed into* the M1 data model even if shallow, because
they are expensive to retrofit:

- **Deterministic event log / chronicle** — append-only structured record of state-changing events
  (migration, border shift, river avulsion, extinction). Enables time-series probes, "history",
  and a **causal-trace** query ("why did this cell become desert?") — trace API stubbed in M1, full
  lineage ⊕. The chronicle *is* the project's history layer; retrofitting it means rebuilding lost history.
- **Reproducibility manifest ("world recipe")** — export seed + config + content/mod version pins +
  engine hash so anyone regenerates an identical world or shares an interesting one without shipping
  full snapshots. Nearly free given determinism is already required.
- **Golden-master / approval tests** — checked-in known-seed snapshots diffed with tolerance to
  catch *behavioral* drift in emergent output (distinct from unit tests).
- **Dimensional-analysis enforcement** — dev-mode unit-checked fields (pint-style) so W/m² can't be
  added to J, or counts to densities. Generalizes the equal-area guard to the whole unit-bug class.
  Pair with **strict internal-SI vs. display-unit separation** (the °F comfort-band example is a
  *display* conversion, never stored units).
- **Semantic query layer for AI clients** — a small cell-filter DSL / canned intents
  (`warmest_habitable_region`) above raw per-cell reads, so AI control is usable without pulling
  every cell. A **compact LLM affordance manifest** (name, one-line effect, risk tier, example) is
  generated from the same source as `/commands`.
- **AI-mutation guardrails** — magnitude caps (no "+1000 m sea level" in one call), **dry-run
  preview + rollback**, and an **audit trail** of which principal (human/AI/mod) changed what.
- **Schema-generated docs & settings UI** — extend the single-source-of-truth principle: auto-derive
  the settings reference docs *and* the in-app settings form from the one pydantic schema (kills
  triple-maintenance across config/API/UI).
- **Scenario/preset library** — tidally-locked ocean world, Mars-dry, high-tilt seasonal-extreme,
  fantasy default. Nearly free on top of `PlanetConfig`; doubles as test fixtures.
- **Colorblind-safe, perceptually-uniform overlay palettes**, with legends emitted as **structured
  API data** (screen-reader / non-visual-client friendly), not canvas-only.
- **Mod dependency/version resolution** — manifests declare `requires`/`conflicts`/`min_engine_version`;
  loader topologically sorts and fails fast. **Deterministic-mod contract:** mods route randomness/time
  through core `Rng`/`Clock` (lint-enforced) so the ecosystem can't break reproducibility. Data-only
  **hot-reload** at tick boundaries; Python-hook hot-reload ⊕ (needs the sandbox above).
- **Three independent version axes** — engine version, **snapshot-schema version** (migration chain),
  and **content/mod-schema version** — plus a **public API contract** on SemVer so AI/third-party
  integrations don't break each release.
- **Chunk-first / out-of-core storage, DGGS-aligned partitioning** — the `Storage` port (zarr) is
  lazy/tile-loaded by viewport+LOD, and Dask/Ray shards use H3/S2/ISEA's *own* hierarchical tiling as
  the partition key. This (not just the compute-backend swap) is what lets a high-res planet run on a
  weak machine; cheap to decide now, expensive to retrofit.
- **Real-Earth calibration harness + GIS import** — a lightweight **Köppen-classification** diff of
  the Earth preset against real climatology in M1 (the real "is the sim right?" check); **GeoTIFF**
  DEM import in M1; **NetCDF/GRIB reanalysis, shapefile coastlines, glTF globe export** ⊕. A
  **license/attribution manifest** travels with imported real-world data and community mods.
- **Interest-driven adaptive LOD** — the Grid port's `parent/children` gains a *refinement policy*
  hook (refine at coastlines/biome/border boundaries/steep gradients), designed in M1; incremental
  dirty-region recompute and a feedback-loop fidelity autotuner ⊕.
- **Holdridge life-zone biome classification** — adopt this established science-grounded scheme
  (temperature × precipitation × humidity → biome) for the biome registry instead of an ad-hoc one;
  fits "accuracy-first" and is mod-friendly.
- **Settlement entity from the start** — civilization models discrete **settlements**
  (`cell + population + civ ref`, with capital/city/town tiers and culture-flavored toponymy), not
  only territory. It's the anchor that trade routes, roads, diplomacy, and chronicle events all hang
  off — cheap to shape now, a rework if retrofitted. M1 ships the entity minimally; the systems on
  top (trade/roads/diplomacy) are ⊕.
- **Disturbances — scripted AND emergent (two triggers, one effect mechanism):**
  - *Scripted* — parameterized one-shot field injections (volcanic aerosol, meteor impact, plague seed,
    climate nudge) as API commands: a stress-scenario library for the conservation/property suite and an
    AI what-if vocabulary. In M1 **test scaffolding**.
  - *Emergent* — hazards that happen **on their own** as world texture: **wildfire** (dry + high fuel +
    ignition → burn → regrowth) is M1-feasible and shipped; **volcanism tied to plate boundaries** and
    **rare background impact risk** are roadmap (volcanism needs the tectonics generator). Both triggers
    route through the same field-perturbation code — the difference is *who fires it*.
- **Auto-generated, editable in-world wiki** — a browsable/queryable encyclopedia of **species, plants,
  civilizations, settlements, landmasses/regions**, generated from world state + the chronicle, with an
  **editable overlay** for human-authored lore. Localized via the same i18n keys; served over the API
  so AI clients can read it too. Minimal generated version in M1; rich cross-linking ⊕.

---

## Milestone 1 — Full minimal vertical slice (no dead-ends)

> **Sequencing within M1 (respects the data dependencies).** The full slice below is the M1 *goal*
> (nothing deferred out of it, per your "no future limitations" ask), but it ships as ordered,
> independently-green stages so commits stay small — and the order follows the **dependency DAG**,
> not the narrative one: **M1.0** grid + equal-area conservation test (numpy only) → **M1.1**
> topography → **M1.2 sea level + land/ocean mask** (from height field + target water fraction —
> needs *no* climate) → **M1.3** climate + surface ocean (consumes the land mask; Earth preset first,
> then the any-planet regime sweep) → **M1.4 rivers & lakes** (consume precipitation from climate) →
> **M1.5** biology → **M1.6** civilization → **M1.7** API + web + CLI + Docker, discipline
> (M1.x = green) throughout. **Note the split:** the old single "hydrology" layer is deliberately cut
> into the *coastline/sea-level* part (before climate, because ocean routing needs the land mask) and
> the *rivers* part (after climate, because rivers need rainfall). If you want the smallest possible
> first shippable, **M1.0 stands alone** and is genuinely demonstrable.

Each layer is implemented **minimally but with its full-shaped port/data-model**, so later
milestones deepen a layer without reworking interfaces.

1. **Grid + equal-area** — `Grid` port with H3 **and** ISEA adapters + the toggle; conservation
   test green. (S2 adapter can follow but the port is complete.)
2. **Topography** — height field on the grid with three sources behind one port: **load** a
   real DEM (Earth/Mars/Moon/Venus), **generate** procedurally (noise + plate-ish uplift),
   and **edit** (raise/lower cells via API). *Image→heightmap* is scoped as a later adapter
   (grayscale-import first; ML inference explicitly deferred, not designed out).
3. **Climate (simplified, physically-grounded, any-planet)** — driven by **PlanetConfig**
   (tilt, signed rotation rate incl. tidally-locked, orbital W/m², radius/gravity, atmosphere).
   Insolation per cell from tilt + orbital energy + latitude + albedo (for a **tidally-locked**
   world, insolation is fixed by longitude relative to the star, producing a permanent hot
   substellar point and a cold night side rather than latitude bands). Temperature via an
   energy-balance model with heat transport whose efficiency scales with **atmospheric
   pressure/composition** (thick CO₂ ⇒ strong greenhouse + even temps like Venus; thin/none ⇒
   huge day-night swings like Mars/Moon). **Wind** from pressure gradients + **Coriolis derived
   from the signed rotation rate** — fast spin ⇒ many narrow bands, slow ⇒ few wide cells,
   **retrograde ⇒ bands/gyres reverse**, zero/tidally-locked ⇒ substellar-to-antistellar
   circulation instead of zonal bands. **Precipitation** from evaporation over water + **orographic
   lift with lee-side rain shadow** + convergence, and tall ranges act as **wind/moisture barriers**
   (Himalaya/Tibet effect). **Snow/ice cover** is modeled with its **high-albedo feedback** (more ice
   → cooler → more ice). Coupled with the **surface `Ocean`** (Ekman/Sverdrup/Stommel boundary
   currents advecting heat — see Ocean & seasonality) so effects like the **Gulf Stream** emerge.
   Fidelity (resolution, timestep, model detail) is a config knob.
   **Scoping decision — seasons yes, day-to-day weather later:** M1 models a **seasonal cycle**
   (winter/summer temperature & precipitation from tilt+orbit, **emergent monsoons** from land–sea
   contrast) — seasons matter for a world sim. What M1 does *not* model is **day-to-day weather**
   (individual storms, fronts, short-timescale variability); that needs the time-dependent GCM and
   is **roadmap**. So M1 = seasonally-resolved climate, not steady-state annual averages, and not
   yet stochastic weather.
4. **Hydrology (runs in two parts, around climate).** Part A — **sea level + land/ocean mask**:
   solve the height-threshold that yields `PlanetConfig`'s target water fraction; this mask runs
   **before** climate because ocean routing and albedo need it (M1.2). Part B — **rivers & lakes**:
   downhill flow-accumulation over grid neighbors, lakes fill depressions; runs **after** climate
   (M1.4) because it consumes precipitation. Intertidal band derives from tidal range here.
5. **Biology (progression, not evolution)** — species = **data-driven trait distributions**,
   not single points. Each environmental trait (temperature, humidity, altitude/oxygen, temp
   *variance* tolerance, …) is a **comfort band** (optimum, e.g. a human's 68–72°F) nested inside
   a wider **tolerance band** (survivable-but-stressed, e.g. 40–80°F), with **intra-population
   variation** on top — the "multivariance of variances": the preferred point itself is a
   *distribution across individuals* (some humans skew warm, some cold), and the *width* of
   tolerance also varies. Cell suitability is a **graded** response (comfort ≈ full growth,
   tolerance edges ≈ stressed/declining, beyond ≈ die-off), integrated over that spread — not a
   hard in/out range.
   - **Habitat medium (first-class axis, not an extreme trait value):** every cell is tagged
     land / water / (coast) after hydrology, and every species declares a **medium** — terrestrial,
     **aquatic** (lives in water cells only), amphibious, aerial, or **subterranean**. A mermaid or
     seaweed is *suitable only over water cells*; a land creature is *suitable only over land*. This
     is a categorical gate applied **before** the comfort/tolerance bands, so aquatic/flying life
     isn't a dead-end bolted on later. **M1 ships one aquatic species (seaweed)** to exercise the
     water-habitat path end-to-end; deeper aquatic ecology is roadmap.
   - **Body-size crowding cap:** a per-species **max sustainable density** (independent of food) —
     grass and worms pack thousands per cell, a giant/cyclops/dragon physically cannot. Carrying
     capacity K = **min(crowding cap, food-limited capacity, environmental suitability)**, so even
     with infinite food a large-bodied species stays sparse.
   - **Habitat medium** includes an **intertidal** band (from tidal range) alongside land / water /
     amphibious / aerial / subterranean, so tide-pool/coastal-adapted life has a home.
   - **Specific diets, not generic "plants":** diet is a set of **specific food edges** (this species
     eats *these* taxa/food-categories) with preference weights — a horse grazes grass, not bamboo. A
     species starves amid food it can't eat. Generic "eats any plant" is expressible as a wide edge
     set for quick-and-dirty content, but specificity is the default the model supports.
   - **Food competition allocation:** when several consumers draw on the same food pool in a cell, the
     pool is split **proportionally to each consumer's demand × competitive ability** (consumption
     rate × population × preference). This naturally yields **competitive exclusion** — a much stronger
     competitor drives the weaker toward local extinction — without a special-case rule.
   - **Plants have habitat variability too:** plants are species with the same comfort/tolerance
     bands, so a **tree can't grow in desert or tundra**; vegetation type per cell is an *outcome* of
     plant suitability, which in turn sets the land-cover other species react to.
   - **Terrain / land-cover preference:** suitability includes a **biome/land-cover** axis, not just
     climate — **deer prefer woods, horses prefer plains** — so species sort into the landscapes that
     suit them. Hybrids/chimeras (e.g. **centaurs**) are just a trait mix (sapient + plains-preferring
     + terrestrial) — the framework doesn't special-case them.
   - **Food supply is actually checked (incl. plants):** the diet edges form a **food web**;
     food-limited capacity comes from **available prey/plant biomass in reachable cells** each step.
     Plants (autotrophs) are the base (biomass from insolation + water + suitability); herbivore K
     tracks plant biomass, carnivore K tracks prey biomass. Emit the **food web as a graph** for the
     client to render a **food-chain chart** (reuses `GraphLayout`).
   - **Predator–prey dynamics — dynamic or smoothed (toggle):** the default K-tracking makes
     populations *glide* toward current K (smoothed). Turning on an explicit **Lotka–Volterra lag term**
     produces real **boom/bust oscillation** (the snowshoe-hare/lynx cycle: predators overshoot, crash
     prey, then crash, then recover). Per-species/per-sim config chooses smoothed vs. oscillatory.
   - **Demography (age-structured, minimal in M1):** species carry **lifespan, maturity age, gestation,
     litter size, and a fertility window** (how long they stay reproductive). M1 derives the growth
     rate *r* from these (cohort-lite); full age-structured (Leslie-matrix) dynamics are an option/
     roadmap. This is what makes a 6-month-maturing species outbreed a 20-year one.
   - **Extinction, endangerment & viability:** populations can decline to **local extinction** and
     **global extinction** (recorded as first-class **chronicle** events). A **minimum viable
     population** gate applies: **sexual** species need ≥2 (and enough breeding stock/genetic
     viability) to recover — one lone individual is a dead lineage — whereas **asexual/cloning**
     species can rebound from a single survivor. Reproduction mode is a species trait.
   - **Disease (minimal, on by default):** a data-driven module with **density-dependent transmission**
     — bigger/denser settlements and herds get more spread — that can cause die-offs and feed back into
     population/extinction. Rich "syndrome" catalogs (vectors, symptoms, immunity) are roadmap; the data
     model is shaped now so it isn't a retrofit, and it's config/mod-toggleable.
   - **Wildfire & disturbance:** dry, high-fuel (dense vegetation) cells can **ignite** (lightning or
     spreading), burning biomass and resetting succession; vegetation **regrows** over following seasons
     (California-style fire→regrowth). This is one instance of the natural-hazard system (below).
   Plus the growth-rate mechanics above. **Populations** are densities with logistic growth toward K;
   **migration** = suitability-gradient **local diffusion** across the grid graph (with stochasticity)
   — populations drift toward better neighboring cells; subterranean species diffuse through the
   `SubsurfaceGrid` instead. *(Deliberate M1 simplification: gradual spread, not true **long-range
   seasonal migration** — penguins/birds jumping between two distant known regions — which is roadmap.)*
   A `sapient` flag lets a species found a civilization (layer 6). Ship a
   roster that **exercises every axis at least once** (per the "no dead-ends" rule): **land plants**
   (grass + a **tree/bush** — trees also feed civilization forestry, see layer 6), **an aquatic
   plant** (seaweed), **a non-sapient land animal** (lion), **an aerial animal** (a bird — proves
   the flying/medium gate, not just an extreme trait), **a fantastical creature** (dragon, flagged
   aerial), and **≥2 sapient races** including a **subterranean** one (e.g. humans + dwarves, dwarves
   exercising the underground medium and its link to mines) plus optionally elves. Realistic,
   fantastical, aquatic, aerial, subterranean, and civilization-founding species all share one
   data-driven framework.
6. **Civilization (minimal, multi-race)** — **any species flagged `sapient` can found a
   civilization**, so **multiple sapient races coexist** (humans, elves, dwarves). Mechanics are
   defined, not hand-waved:
   - **Model split (density fields + named-entity overlay):** the substrate is **continuous fields**
     on the grid — population density, **influence**, tech level — consistent with the rest of the sim.
     On top sits a thin **discrete-entity overlay** of named objects: **settlements** (capital/city/
     town tiers, with toponymy), and (roadmap) **rulers, treaties, wars**. Borders/tech/population are
     *fields*; diplomacy/succession operate on *entities*. This is how the "named-individual overlay"
     from the non-goals section concretely attaches to the density world.
   - **Borders via an influence field:** each **settlement emits influence** ∝ its population/power,
     **decaying with distance and terrain cost** (mountains/water are expensive to project across),
     diffused over the grid. A cell belongs to the civ with the **highest influence** there; frontiers
     are where two civs' influence is comparable.
   - **Frontier contests:** at a contested cell the holder is decided by **relative strength** with
     **stochasticity**, not a coin flip. **Strength is defined:** a civ's raw military strength =
     **mobilized population** (a fraction of population it can raise as fighters, itself a function of
     surplus/tech/policy) **× military-tech multiplier** (weapons/armor techs) **× equipment** (drawn
     from the resources it invests) — so "military factor" is a thing civs *build*, not a free constant.
   - **Supply lines / projection (no conquering across the world):** the strength a civ can actually
     bring to a *given* cell is its raw strength **× a projection factor that decays with distance and
     terrain cost from its nearest supplying settlement** — the same distance+terrain-cost model that
     defines influence. So reach is bounded: **Rome cannot conquer China or South Africa** because its
     projected strength there is ~0. Contest outcome compares *projected* strengths, not raw ones.
   - **Crossing water & air (boats/planes):** by default water is near-impassable terrain cost for land
     civs, so early civs are land-bound. **Naval tech lowers the water-crossing cost** (boats → sea
     projection/settlement), and **flight tech** later lowers it further (air). Flying/aquatic *creatures*
     already cross freely via their habitat medium; for *civs* it's tech-gated. M1 ships the tech-modified
     terrain-cost mechanism; discrete naval/air *units* are roadmap.
   - **Shared borders across domains:** different races can share borders, and because surface and
     `SubsurfaceGrid` territory are distinct domains, a **mountain-surface race and an under-mountain
     race can occupy the same column** without contesting — they only contest within the same domain.
   - **Tech progression is driven, not free:** a civ accumulates **research** from **population ×
     resource surplus × settlement density**, advancing through the **prerequisite DAG**. Per-civ/
     per-tech **propensity weights** and **environmental/resource gating** mean progress isn't uniform —
     no industrial revolution without accessible coal/iron, so a resource-poor or isolated civ (a
     "Rome without an industrial revolution") stalls where its context dictates.
   - **Resources mean something, and deplete realistically:** three structurally different types —
     **mines** (finite stock that **depletes and runs out**; dwarves get a subsurface affinity),
     **forestry** (renewable but **over-harvestable** faster than regrowth), and **farmland/forage**
     (**managed plant biomass** of specific crop species that renews while the underlying biology holds).
     Extraction has a **mechanical effect** (feeds population/settlement/tech growth) even before a
     market exists — trade/market/supply-demand is roadmap.
   - **Extraction feeds back into biology (automatic):** forestry/farmland draw on the *same* plant
     biomass the food web uses, so **clear-cutting a forest measurably starves the deer** that grazed it
     — resource use and ecology are one coupled system, not parallel bookkeeping.
   Ship **≥2 sapient races** so inter-civ contact/competition runs from the start. Emit each race's
   tech **DAG** for the client (`GraphLayout`/elkjs). Races differ only by data, so adding
   elves/dwarves/centaurs is content, not code.
7. **API + web client + CLI** — headless FastAPI server (REST+WS, self-describing) drives all
   of the above; a Three.js globe renders the grid, height, climate layers, populations, and
   borders; a CLI script proves API-only control. Docker image for the headless core.
8. **Discipline** — all pre-commit hooks + CI green from commit #1; conservation/property tests
   for the equal-area invariant.

---

## Roadmap beyond M1 (each deepens one layer, interfaces unchanged)

- Climate → full simplified GCM (multi-layer atmosphere, ocean currents, seasons, **day-to-day
  weather/storms**); eventually **two-way biosphere↔climate feedback** (vegetation/ocean biology →
  albedo/greenhouse), which M1's one-directional climate→biology coupling leaves open.
- Hydrology → erosion feedback into topography, aquifers, glaciers.
- Biology → richer food webs incl. **predator-prey boom/bust oscillation** (Lotka-Volterra lag on
  the K-tracking, not just smooth logistic convergence), **true long-range seasonal migration**
  (distinct mechanic from local diffusion), **disease** (data-driven "syndrome" content type, à la
  Dwarf Fortress), **keystone-species/extinction** recorded as first-class chronicle events,
  optional domestication; then **evolution** (trait drift) as the long-horizon goal.
- Civilization → **discrete settlements** (capital/city/town tiers with toponymy) beyond raw
  territory; **diplomacy as explicit, API-queryable state** (attitude with legible reasons,
  treaties, war) so an AI agent can reason about it; **trade/market economy** (supply-demand prices,
  routes pathed over the grid) and **roads/infrastructure**; **religion** as spread-able content
  (same shape as language); **dynastic succession / named rulers** (succession laws, legitimacy)
  feeding the chronicle; real language-family generation; full auto-scaling tech DAG; procedural
  fantasy-race rules via mod hooks.
- Topography → image/photo → heightmap ML adapter; plate-tectonics generator.
- Scaling → multi-GPU (cupy/jax device-sharding) and cluster (dask/ray) adapters hardened;
  incremental dirty-region recompute; feedback-loop fidelity autotuner.

**Reference implementations to build against (not reuse as-is):** WorldEngine (Python:
plates+erosion+rain-shadow+Holdridge, but lat-lon so not directly usable), PlaTec/ProcGenesis
(tectonics), Vulgarlang (language generation), NASA ROCKE-3D (exoplanet GCM + biosphere coupling),
Dwarf Fortress *Legends* (chronicle) & Azgaar's generator (settlements/toponymy/cultures).

**Deliberate non-goals (so they aren't mistaken for omissions):** individual-agent mood/mental-break
cascades (Rimworld-style) conflict with the density-based population model and are *not* adopted at
population scale — at most an optional "named individual" overlay tied to rulers; audio/ambience is
out of scope for an accuracy-first headless-first sim.

---

## Verification

- **Unit + property tests** (pytest + hypothesis) per layer; run locally via pre-commit's fast
  subset and fully in CI.
- **Equal-area conservation test** (the keystone): total insolation/mass/population is invariant
  (within tolerance) when toggling H3↔ISEA and when changing resolution. Must be green before
  any sim layer builds on the grid.
- **Determinism test (two separate claims — don't conflate):** (a) **within a fixed backend +
  seed + config ⇒ bit-identical** world (seeded `Rng`/`Clock`); (b) **across backends/resolutions
  ⇒ field-equivalent within tolerance**, *not* bit-exact — numpy/CuPy/JAX/dask use different
  reduction orders, so cross-backend equality is a tolerance test, never `==`.
- **Any-planet climate test (qualitative regimes, not GCM-exact):** a simplified energy-balance
  model won't reproduce a full GCM numerically, so these assert *directional/structural* outcomes
  with generous tolerance, not exact temperatures — Earth-like tilt/spin ⇒ Hadley/Ferrel/Polar
  banding; **retrograde spin ⇒ reversed circulation** (cf. MPI "Project Retrograde", *Earth System
  Dynamics* 2018); **tidally locked ⇒ substellar hot-spot + cold night side**, no zonal bands (cf.
  published TRAPPIST-1 EBMs using substellar-longitude coordinates); **thick vs thin atmosphere ⇒
  small vs large day-night swing**. A separate **Earth-calibration test** compares the Earth preset
  against real climatology (temperature/precipitation zones) within a stated band. Guards that
  rotation/tilt/atmosphere are truly parameters, not Earth constants.
- **API contract test**: the self-describing `/commands` + OpenAPI schema matches implemented
  handlers; a headless run driven **only** by the CLI/API client (no GUI) reproduces a scripted
  scenario — proving AI/headless control.
- **Coupled ocean/climate emergence checks (qualitative):** a **western-boundary current** forms on
  the correct ocean margin and warms the poleward-adjacent land (a **Gulf-Stream / N.-Europe-warmer-
  than-its-latitude** signature); **seasonal monsoon** reversal emerges over a land–sea-contrast
  region; a **rain shadow** appears on the lee of a mountain range with a wet windward side; an
  **ice–albedo feedback** test shows cooling the planet expands ice which further lowers temperature
  (and reverses on warming); an **ENSO-diagnostic** probe looks for interannual oscillation
  (informational in M1, since full ENSO needs roadmap ocean depth).
- **Hardware-utilization + kernel-swap checks:** the **`Kernel` conformance test** runs the
  pure-Python and out-of-Python flow-accumulation adapters on identical inputs and asserts matching
  output (proving the cross-language swap); a **multi-device test** runs the sim sharded across ≥2
  GPUs and confirms results match the single-device/CPU run within tolerance; auto-detection is
  verified to pick GPU-sharded on a multi-GPU host and numpy-CPU on a no-GPU host from the *same build*.
- **Ecology dynamics checks:** predator–prey in **oscillatory** mode shows lag-driven boom/bust (vs.
  smooth glide in **smoothed** mode); a species driven below its **minimum viable population** goes
  **extinct** and logs a chronicle event; **clear-cutting** a forest cell measurably reduces the
  dependent herbivore there (extraction↔biology feedback); competitive exclusion resolves two
  consumers on one food pool.
- **Subsurface check:** a `SubsurfaceGrid` path connects two surface cells on opposite sides of a
  massif that are *not* surface-adjacent (tunneling is representable), and subterranean migration uses
  it rather than surface `neighbors()`.
- **End-to-end smoke**: `docker compose up` the headless core, load Earth topography, step the
  sim N ticks (across seasons), and confirm the web client renders plausible seasonal temperature
  bands, ocean currents, rivers flowing downhill to the sea, and a population migrating toward
  higher-suitability cells.
- **Save round-trip + migration test**: snapshot → load reproduces state; an old-schema fixture
  upgrades cleanly through the migration chain.
- **Security tests**: mutate-command requires permission/token; malformed and **malicious mod**
  fixtures are rejected/sandboxed (no code execution outside the mod API); API rejects unauthorized
  or malformed input (light fuzzing, given the API is AI-driven).
- **Performance-regression tests**: each named hardware tier's target (cells × species × timestep)
  runs within a wall-clock/memory budget; flow-accumulation and biology kernels are profiled.
- **Discipline gates**: pre-commit (lint, types, docstrings, file length, duplicate detection,
  port-boundary) and CI — **including a Windows CI job from commit #1** (Windows is the primary
  platform) plus Linux — must pass on every commit.
