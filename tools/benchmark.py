"""Performance profiler: where does the simulation spend its time?

The capacity tests (``tests/test_capacity.py``) are pass/fail budget
guards — "not catastrophically slow".  This tool answers the other
question: *what should we optimize first?*  It times the hot kernels the
design calls out — per-cell diffusion (climate transport), sequential
flow-accumulation (rivers), and the area-weighted reduction (every layer
aggregates every tick, and the one place sharding is wired in) — then
prints a table ranked by wall time with each component's share of the
total, plus a machine-readable JSON dump for tracking regressions over
time.

Run it::

    python -m tools.benchmark                 # coarse tier, auto backend
    python -m tools.benchmark --resolution 4  # finer grid
    python -m tools.benchmark --backend numpy --json bench.json

Timings are best-of-N (``--repeat``) so backend JIT warmup and one-off
noise do not dominate.  It is a measurement tool, not a test: no budgets
are asserted here.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from adapters.compute_registry import create_array_backend
from adapters.grid_registry import create_grid
from adapters.hardware import probe_host, recommend
from adapters.kernels_python import NO_RECEIVER, flow_accumulation
from core.grid.area_weighted import area_weighted_total
from core.grid.flux import diffusion_step

if TYPE_CHECKING:
    from ports.array_backend import ArrayBackend

_DIFFUSIVITY_M2_S = 1.0e7
_DEFAULT_RESOLUTION = 3
_DEFAULT_REPEAT = 3
_FLOW_CELLS = 200_000
_QUICK_FLOW_CELLS = 20_000
_MILLION = 1.0e6
_THOUSAND = 1.0e3


@dataclass(frozen=True)
class BenchResult:
    """One component's measured cost."""

    component: str
    scale: str
    count: int
    wall_seconds: float

    @property
    def throughput(self) -> float:
        """Return items processed per second (0 when the timing was zero)."""
        return self.count / self.wall_seconds if self.wall_seconds > 0.0 else 0.0


def _time(fn: Callable[[], object], repeat: int) -> float:
    """Return the best (minimum) wall time of ``fn`` over ``repeat`` runs."""
    best = float("inf")
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def _bench_diffusion(resolution: int, repeat: int) -> BenchResult:
    """Time one climate-transport diffusion step over an H3 grid."""
    grid = create_grid("h3", resolution=resolution)
    values = {cell: float(i % 2) for i, cell in enumerate(grid.cells())}
    min_area = min(grid.area_m2(cell) for cell in values)
    dt_s = 0.05 * min_area / _DIFFUSIVITY_M2_S
    wall = _time(lambda: diffusion_step(values, grid, _DIFFUSIVITY_M2_S, dt_s), repeat)
    return BenchResult(
        "diffusion_step (climate transport)", f"h3 res{resolution}", grid.cell_count, wall
    )


def _bench_flow(cells: int, repeat: int) -> BenchResult:
    """Time flow-accumulation over a single-outlet drainage chain."""
    receivers = [*range(1, cells), NO_RECEIVER]
    areas = [1.0] * cells
    elevations = [float(cells - i) for i in range(cells)]
    wall = _time(lambda: flow_accumulation(receivers, areas, elevations), repeat)
    return BenchResult("flow_accumulation (rivers, sequential)", "drainage chain", cells, wall)


def _bench_reduction(cells: int, backend: ArrayBackend | None, repeat: int) -> BenchResult:
    """Time an area-weighted planet total (pure-Python or sharded backend)."""
    values = [float(i % 97) for i in range(cells)]
    areas = [1_000.0 + (i % 7) for i in range(cells)]
    wall = _time(lambda: area_weighted_total(values, areas, backend=backend), repeat)
    if backend is None:
        scale = "pure-python"
    else:
        scale = f"{backend.name} x{backend.num_devices}dev"
    return BenchResult("area_weighted_total (reduction)", scale, cells, wall)


def run_suite(resolution: int, backend_name: str, repeat: int, quick: bool) -> list[BenchResult]:
    """Run every component benchmark and return the results unsorted."""
    grid = create_grid("h3", resolution=resolution)
    cells = grid.cell_count
    flow_cells = _QUICK_FLOW_CELLS if quick else _FLOW_CELLS
    backend = create_array_backend(backend_name)
    return [
        _bench_diffusion(resolution, repeat),
        _bench_flow(flow_cells, repeat),
        _bench_reduction(cells, None, repeat),
        _bench_reduction(cells, backend, repeat),
    ]


def _host_label() -> dict[str, object]:
    """Return a small host/recommendation summary for the report header."""
    caps = probe_host()
    rec = recommend(caps)
    return {
        "cpu_cores": caps.cpu_cores,
        "ram_gib": round(caps.ram_gib, 1),
        "gpu_count": caps.gpu_count,
        "recommended_backend": rec.compute_backend,
        "recommended_fidelity": rec.fidelity_level,
        "recommended_resolution": rec.grid_resolution,
        "recommended_device_count": rec.device_count,
    }


def render_table(results: list[BenchResult], backend: ArrayBackend, host: dict[str, object]) -> str:
    """Return a human-readable table ranked by wall time (worst first)."""
    ranked = sorted(results, key=lambda r: r.wall_seconds, reverse=True)
    total = sum(r.wall_seconds for r in ranked) or 1.0
    lines = [
        "FableWorldSim performance profile",
        (
            f"host: {host['cpu_cores']} cores, {host['ram_gib']} GiB RAM, "
            f"{host['gpu_count']} GPU(s)   backend: {backend.name} ({backend.device}), "
            f"{backend.num_devices} device(s)"
        ),
        "",
        f"{'component':<40}{'scale':<16}{'count':>9}{'wall_s':>10}{'throughput':>16}{'share':>8}",
        "-" * 99,
    ]
    for r in ranked:
        lines.append(
            f"{_fit(r.component, 40)}{_fit(r.scale, 16)}{r.count:>9}{r.wall_seconds:>9.4f}s"
            f"{_human(r.throughput) + '/s':>16}{100.0 * r.wall_seconds / total:>7.1f}%"
        )
    return "\n".join(lines)


def _fit(text: str, width: int) -> str:
    """Return ``text`` left-justified to ``width``, truncated if too long."""
    if len(text) >= width:
        return text[: width - 1] + " "
    return text.ljust(width)


def _human(value: float) -> str:
    """Return a compact human count (1.2K, 3.4M) for throughput."""
    if value >= _MILLION:
        return f"{value / _MILLION:.1f}M"
    if value >= _THOUSAND:
        return f"{value / _THOUSAND:.1f}K"
    return f"{value:.0f}"


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the benchmark tool."""
    parser = argparse.ArgumentParser(description="Profile FableWorldSim hot kernels.")
    parser.add_argument("--resolution", type=int, default=_DEFAULT_RESOLUTION)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--repeat", type=int, default=_DEFAULT_REPEAT)
    parser.add_argument("--quick", action="store_true", help="smaller sizes for a fast check")
    parser.add_argument("--json", metavar="PATH", help="also write results as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the suite, print the table, and optionally dump JSON."""
    args = _build_parser().parse_args(argv)
    host = _host_label()
    backend = create_array_backend(args.backend)
    results = run_suite(args.resolution, args.backend, args.repeat, args.quick)
    print(render_table(results, backend, host))
    if args.json:
        payload = {
            "host": host,
            "backend": {"name": backend.name, "device": backend.device},
            "results": [asdict(r) | {"throughput": r.throughput} for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
