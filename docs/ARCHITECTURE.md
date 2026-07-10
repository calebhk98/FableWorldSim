# Architecture

FableWorldSim is built as a **hexagonal (ports & adapters)** system so that any library can be
swapped in minutes and the same code runs on a laptop, a GPU workstation, or a cluster. This
document is the durable summary; the full design and rationale live in [`PLAN.md`](PLAN.md).

## The rule

> The domain **core** (`core/`) depends on **ports** (`ports/`) only — never on a concrete library
> or adapter. Concrete libraries live in `adapters/` and are wired at startup from configuration.

This is enforced automatically: `tools/hooks/check_port_boundary.py` fails any commit where a
`core/` module imports an adapter or third-party package directly.

```
 clients (web / CLI / AI)  ─▶  api/  ─▶  core/  ─▶  ports/  ◀─  adapters/  ◀─  libraries
```

## Ports (planned)

| Port | Responsibility | Example adapters |
|---|---|---|
| `Grid` | Equal-area discrete global grid; cells, area, neighbors, edge length | H3, ISEA/DGGRID, S2 |
| `SubsurfaceGrid` | Volumetric depth layers under the surface (dwarves, worms) | depth-band, cavern-network |
| `ArrayBackend` | Per-cell field math (CPU/GPU/cluster), auto-detected | numpy, cupy, jax, dask |
| `Kernel` | Hot compute kernels behind a stable signature | pure-Python, GPU/native |
| `Ocean` | Surface currents & heat advection (deepens to thermohaline) | surface-Ekman |
| `Storage` | Versioned world snapshots & time series (migration chain) | sqlite, parquet, zarr |
| `ContentRegistry` | Moddable data: species, biomes, tech, languages | data-file loader |
| `GraphLayout` | Tech/language-tree (DAG) layout, client-side | elkjs, dagre |
| `Security`/`Auth` | Permission-scoped commands, tokens, mod sandbox policy | — |
| `Clock` / `Rng` | Deterministic, seedable time and randomness | seeded PRNG |

## Two invariants that shape everything

1. **Equal-area / area-weighting.** Every quantity that could be biased by cell size is stored per
   unit area or explicitly area-weighted when aggregated. A cross-backend conservation test guards
   this: global totals stay invariant when the grid backend or resolution changes.
2. **Variable neighbor degree.** Neighbor count is not constant (hexagons 6, pentagons 5, quads 4,
   triangles 3). All flux/diffusion code iterates `neighbors(cell)` and weights by edge length —
   never hardcodes a degree.

## Three control surfaces, one source of truth

All options live in a single settings schema. The **config folder**, the **API**, and the **UI**
all read and write that same schema, so no option exists in only one place.

## Hardware scaling

`ArrayBackend` auto-detects GPUs, CPU cores, and RAM and uses all of them with the same code:
fields shard across multiple GPUs, the CPU backend spans all cores, and chunked/out-of-core storage
lets a small machine run the identical build at lower fidelity by streaming tiles.
