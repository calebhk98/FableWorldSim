# FableWorldSim

A headless, API-first **3D-globe world simulator**. The planet surface is a true
**equal-area discrete global grid** (H3 / S2 / ISEA behind one toggle), and on top of it the
simulator builds up, layer by layer: topography → climate & oceans → hydrology → biology
(realistic *and* fantastical creatures) → multi-race civilizations. Accuracy-first, but tunable
down for speed, and designed to scale from a no-GPU laptop to a multi-GPU workstation or cluster
using the **same code**.

> **Status: scaffolding.** This repository currently contains the project structure, tooling, and
> documentation only — no simulation logic yet. The full design lives in [`docs/PLAN.md`](docs/PLAN.md).

## What it is (design goals)

- **A sphere, never a flat square grid** — cells tessellate the globe; all math is area-weighted so
  the sim never favors cells by size.
- **Any planet** — tilt, rotation (incl. retrograde / tidally-locked), atmosphere, water fraction,
  and moons are configuration; Earth, Mars, Venus, or invented worlds run the same code.
- **Headless & API-first** — the core is a library + server with a self-describing API, so it runs
  with no GUI and can be driven by a human, a script, or an AI.
- **Hexagonal (ports & adapters)** — the domain core depends on *ports* only; any library can be
  swapped in minutes. Enforced automatically on commit.
- **Moddable, localizable, online or offline** — content and languages are data; nothing is hard-coded.

## Repository layout

| Path | Purpose |
|---|---|
| `core/` | Pure-Python domain logic (imports `ports` only) |
| `ports/` | Abstract interfaces (Grid, ArrayBackend, Kernel, Storage, …) |
| `adapters/` | Concrete implementations of the ports (grid backends, compute, storage) |
| `api/` | Headless REST + WebSocket server (self-describing) |
| `content/` | Base "mod": species, biomes, tech, and `locales/` for i18n |
| `clients/` | Thin clients — `web/` (3D globe) and `cli/` (scriptable) |
| `config/` | Default configuration (also settable via API and UI) |
| `tests/` | Unit, property, and cross-backend conservation tests |
| `tools/hooks/` | Custom commit-time discipline checks |
| `docs/` | Architecture, contributing guide, and the full plan |

## Getting started (development)

```bash
uv venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
pre-commit install                        # discipline gates run on every commit
pytest
```

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for the TDD (red→green) workflow and coding
conventions, and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the ports & adapters design.
