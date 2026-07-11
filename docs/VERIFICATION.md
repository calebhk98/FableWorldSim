# FableWorldSim — Verification status

This document maps every item on the plan's **Verification** checklist
(`docs/PLAN.md` → *Verification*) to the tests that cover it, and states plainly
what is **not** covered yet and why. It is the ledger for the "prove the sim does
what the design claims" milestone.

Legend:

- ✅ **Covered** — a real test exercises the behavior end-to-end.
- 🟡 **Covered, bounded by design** — the tractable part is tested; the rest needs
  a capability that is explicitly roadmap (linked below), not an oversight.
- 🔭 **Deferred (capability is roadmap)** — nothing to test yet without building a
  feature the plan schedules *after* M1. Listed so it is not mistaken for an omission.

Run the whole suite with `pytest`. Heavy full-sim / cross-backend / end-to-end
tests carry the `slow` marker; the pre-commit fast subset runs `-m "not slow"` and
CI runs everything.

---

## Checklist coverage

### Unit + property tests per layer ✅
Per-layer `pytest` + `hypothesis` suites across `tests/` (grid, climate, ocean,
hydrology, biology, civilization, storage, API). The **fast subset** runs on every
commit via pre-commit (`pytest -m "not slow"`); the **full suite** runs in CI.

### Equal-area conservation (the keystone) ✅
`tests/test_conservation.py`, `tests/test_area_weighted.py`, `tests/test_reproject.py`
— totals invariant under H3↔S2 and across resolutions within tolerance.

### Determinism — two separate claims ✅
- **(a) within backend+seed ⇒ bit-identical:** per-layer
  (`tests/test_rng_clock.py`, `tests/test_topography.py::test_procedural_terrain_is_deterministic_per_seed`,
  `tests/test_bio_engine.py::test_same_seed_replays_identically`,
  `tests/test_civ_engine.py::test_same_seed_replays_identically`) **and now at the
  whole-orchestrator level:**
  `tests/test_determinism.py::test_orchestrator_run_is_bit_identical_per_seed`.
- **(b) across backends/resolutions ⇒ field-equivalent within tolerance, never `==`:**
  `tests/test_determinism.py::test_cross_grid_backend_fields_are_equivalent_within_tolerance`
  (h3 res1 vs res2, and h3 vs s2 when `s2sphere` is present) plus
  `tests/test_array_backend.py` (numpy/CuPy/JAX conformance) and `tests/test_reproject.py`.

### Any-planet climate regimes (qualitative) ✅
- Three-band structure (Hadley/Ferrel/Polar):
  `tests/test_climate.py::test_earth_has_the_three_zonal_wind_bands`.
- Retrograde ⇒ reversed circulation: `test_retrograde_spin_reverses_the_bands`.
- Tidally locked ⇒ substellar hot-spot, no bands:
  `test_tidally_locked_world_has_an_eye_not_bands`, `test_slow_rotation_kills_the_bands`.
- Thick vs thin atmosphere ⇒ day-night swing, **isolated to pressure alone**:
  `test_atmosphere_thickness_alone_sets_the_day_night_swing`
  (the older `test_thin_air_swings_harder_than_thick_air` compares whole presets).
- **Earth-calibration** vs real climatology within a stated band:
  `tests/test_earth_calibration.py` (zonal temperature monotone tropical→polar;
  global mean in an Earth-like band; tropics wetter than poles; polar ice without a
  snowball). Bands are directional/structural with generous tolerance — an
  energy-balance model is not a GCM.

### API contract ✅
- `/commands` + OpenAPI stay in sync with the registered handlers:
  `tests/test_api_contract.py::test_every_command_is_discoverable_and_self_consistent`,
  `::test_openapi_schema_documents_command_execution`,
  `::test_self_described_read_commands_are_actually_executable`.
- Malformed/unauthorized input rejected (light Hypothesis fuzz — never a 500):
  `::test_set_setting_command_never_500s_on_garbage` and the three sibling
  `*_never_500s_*` tests. **These found and drove the fix of a real bug**: the
  `/commands/{name}` handler now mirrors `/settings/{path}` (unknown key → 404,
  bad value → 422) instead of leaking a 500 (`api/app.py::_execute_command`).
- Scenario replay purely through the API is exercised via `TestClient` in
  `tests/test_api.py`; a standalone **CLI** client replaying a saved scenario is 🔭
  roadmap (`clients/cli/` is a stub).

### Coupled ocean/climate emergence (qualitative) 🟡
- **Western-boundary / Gulf-Stream** warming:
  `tests/test_ocean_emergence.py::test_western_boundary_current_warms_the_poleward_coast`
  (boundary-intensified poleward current + positive downwind heat flux).
- **Seasonal monsoon** wind reversal:
  `::test_seasonal_monsoon_reverses_the_coastal_wind`.
- **Rain shadow:** `tests/test_climate.py::test_orographic_shadow_on_real_grid`.
- **Ice–albedo feedback** (cooling expands ice, lowers temperature, reverses on
  warming): `tests/test_climate.py::test_ice_albedo_feedback_amplifies_cooling`.
- **ENSO interannual oscillation:** 🔭 the M1 climate is season-resolved but has no
  year-over-year ocean-atmosphere memory, so there is no oscillator to probe yet
  (roadmap: ocean depth / interannual state). Informational-only in the plan.

### Hardware-utilization + kernel-swap 🟡
- **Kernel conformance** (pure-Python vs native C flow-accumulation match):
  `tests/test_kernels_native.py::test_native_kernel_matches_reference_exactly`,
  `tests/test_kernels.py`.
- **Auto-detection** picks a backend per host: `tests/test_hardware.py`,
  `tests/test_array_backend.py::test_auto_detection_always_yields_a_backend`.
- **Multi-GPU sharding matches CPU:** 🔭 no device-sharding adapter exists yet
  (multi-GPU/cluster is a separately-budgeted roadmap regime); there is nothing to
  shard-vs-compare until it is built.

### Ecology dynamics ✅
- **Oscillatory boom/bust vs smoothed glide** (now a genuine multi-species engine
  run, not just the single-species map):
  `tests/test_bio_dynamics.py::test_oscillatory_mode_glides_less_than_smoothed_mode_flattens`,
  and a clean **predator-lags-prey** phase offset in
  `::test_oscillatory_predator_population_peaks_lag_prey_population_peaks`.
- **Extinction below MVP + chronicle event:**
  `tests/test_bio_engine.py::test_a_species_below_viability_goes_extinct_with_a_chronicle_event`.
- **Clear-cutting starves the dependent herbivore:**
  `tests/test_bio_engine.py::test_clear_cutting_the_plants_starves_the_herbivore`,
  `tests/test_bio_foodweb.py::test_clear_cutting_starves_the_dependent_herbivore`,
  and the civ side `tests/test_civ_economy.py::test_forestry_can_outrun_regrowth_and_crash_the_shared_biomass`.
  (A single wired extraction→biology feedback loop across both engines on one
  orchestrator is roadmap; the two halves are each verified.)
- **Competitive exclusion:**
  `tests/test_bio_foodweb.py::test_competitive_exclusion_favours_the_specialist`.

### Subsurface ✅
`tests/test_bio_engine.py::test_subterranean_species_migrate_through_the_volume_graph_not_the_surface`
— a `SubsurfaceGrid` tunnel connects two cells that are *not* surface-adjacent and
migration uses the volume graph, not surface `neighbors()`. (The shipped
`DepthBandSubsurface` mirrors the surface graph for now; real cross-massif tunnel
topology is roadmap behind the tested port.)

### End-to-end smoke 🟡
`tests/test_end_to_end.py` runs the **headless** core: load Earth topography → run
the seasonal climate → confirm seasonal temperature bands
(`test_earth_climate_shows_seasonal_temperature_bands`), rivers flowing downhill to
the sea (`test_rivers_flow_downhill_to_the_sea`), and a population migrating toward
higher-suitability cells (`test_population_migrates_toward_higher_suitability`).
The `docker compose up` + **web-client render** wrapper is 🔭 deferred — `clients/web`
and `clients/cli` are stubs; that wrapper renders these same headless outputs.

### Save round-trip + migration ✅
`tests/test_storage.py` — snapshot→load reproduces state; an old-schema fixture
upgrades through the migration chain; future-schema snapshots are rejected.

### Security 🟡
- **Write tier requires authorization:** `tests/test_access.py` (viewer 403s on
  writes, admin allowed, denials counted).
- **Malformed mods / API input rejected:** `tests/test_mod_manifest.py`,
  `tests/test_content_registry.py`, and the API fuzz tests above.
- **Real token authentication** and **mod code sandboxing** are 🔭 deferred: M1
  identity is a header placeholder (`api/app.py::principal_from`) and mods are
  data-only (no code-execution surface to sandbox yet).

### Performance-regression ✅ (M1 tiers)
`tests/test_capacity.py` — the coarse-tier diffusion budget, plus new profiling
budgets for the two hot kernels the plan names:
`test_flow_accumulation_kernel_fits_the_budget` (200k-cell drainage network) and
`test_biology_step_fits_the_budget` (cells × species × ticks). Mid/high GPU-tier
budgets await the multi-GPU capability (🔭 roadmap).

### Discipline gates ✅
Pre-commit: ruff (lint+format), mypy (strict), interrogate (docstrings),
`file-length`, `port-boundary`, `single-source-of-truth`, `jscpd` (duplicate
detection), and a **fast test subset** (`pytest -m "not slow"`).
CI (`.github/workflows/ci.yml`) runs on **`windows-latest` + `ubuntu-latest`** for
every push/PR and now mirrors the full gate set — including `single-source` and
(Linux) `jscpd` — plus the complete `pytest` suite.

---

## Deliberately deferred (capability is roadmap, not a missing test)

These are 🔭 above, collected so they are not read as gaps in verification. Each
needs a feature the plan schedules after M1 (`docs/PLAN.md` → *Roadmap beyond M1*):

| Item | Blocked on |
|---|---|
| ENSO interannual oscillation probe | interannual ocean-atmosphere state (deeper ocean tier) |
| Multi-GPU sharded run matches CPU; mid/high perf tiers | device-sharding array backend |
| CLI-driven scenario replay == web client | the scriptable `clients/cli/` client |
| `docker compose up` + web-client render smoke | the `clients/web/` 3D globe client |
| Real token authentication enforced | auth beyond the M1 header placeholder |
| Malicious-mod sandboxing (no code exec outside the mod API) | Python-hook mods (M1 mods are data-only) |
| Single wired extraction→ecology loop on one orchestrator | biology↔civilization biomass back-sync |
| Real cross-massif tunnel topology | subsurface cave-network generator |

---

## Post-checklist enhancements

Capabilities added after the M1 verification pass, each with tests:

- **Rivers productionized** — `core/hydrology/rivers.py` builds the steepest-descent
  (D8-style) flow graph from a DEM, accumulates drainage through an injected kernel,
  classifies channels past a threshold, and sizes width/depth via Leopold & Maddock
  hydraulic geometry. `tests/test_rivers.py`. (Previously the routing lived only in
  the end-to-end test; lakes/depression-filling remain roadmap.)
- **Sim-speed telemetry** — `Orchestrator` times each run against an injectable
  monotonic clock (ticks/sec, seconds/tick, per-process wall time); surfaced on
  `/metrics`, the `get_run_telemetry` command, and a `run_telemetry` WS event.
  `tests/test_telemetry.py`.
- **Migration realism** — oxygen axis (barometric from altitude), seasonal-extreme
  axes (coldest-season / range from `ClimateState.seasons`), and a bounded long-range
  seasonal-migration pull distinct from local diffusion. `tests/test_bio_migration_realism.py`.
  (Full two-region path-memory migration remains roadmap.)
- **Batch/overnight world sweep** — `core/sim/world_sweep.py` (pure engine +
  habitability scorer) and `api/world_service.py` (concrete generate→score→select→deepen
  over the real stack), exposed as the `run_world_sweep` command: generate N worlds
  from seeds, keep the top K by habitability, deep-sim the winners (higher fidelity +
  rivers + biology, telemetry captured). `tests/test_world_sweep.py`,
  `tests/test_world_service.py`. (An unattended CLI wrapper for true overnight batches
  awaits the `clients/cli/` client.)
