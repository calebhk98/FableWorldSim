"""Performance-regression guard: named hardware tiers must stay in budget.

The doc names concrete targets per hardware tier (cells x species x
timestep); these tests pin them so a regression that makes a tier
unusable fails CI, and profile the two hot kernels the design calls out
(flow-accumulation and the biology step). Budgets are generous ceilings
(CI machines vary wildly), not performance goals — a green here means
"not catastrophically slow", not "fast".

The multi-GPU / cluster tiers are deliberately absent: that sharding
capability is roadmap (see docs/VERIFICATION.md), so there is nothing to
benchmark yet without faking a device fleet.
"""

from __future__ import annotations

import time

import pytest

from adapters.grid_registry import create_grid
from adapters.kernels_python import NO_RECEIVER, flow_accumulation
from adapters.rng_seeded import SeededRng
from core.biology.engine import step_biology
from core.biology.state import WorldBiologyState
from core.grid.flux import diffusion_step
from core.sim.constants import SECONDS_PER_YEAR
from tests.bio_helpers import make_organism, uniform_context
from tests.civ_helpers import FakeGrid

_COARSE_RESOLUTION = 2  # h3 res 2 = 5,882 cells (the ~10^4 coarse tier)
_MIN_COARSE_CELLS = 1_000
_MAX_COARSE_CELLS = 100_000
_BUDGET_SECONDS = 30.0
_DIFFUSIVITY = 1.0e7

# Mid-tier flow-accumulation profiling: a long descending drainage chain so
# every cell drains through the whole network (worst-case accumulation depth).
_FLOW_CELLS = 200_000
_FLOW_BUDGET_SECONDS = 30.0

# Biology-step profiling: cells x species x one tick on the coarse tier.
_BIO_ROWS = 30
_BIO_COLS = 30  # 900 cells
_BIO_SPECIES = 4
_BIO_TICKS = 5
_BIO_BUDGET_SECONDS = 30.0


def test_coarse_tier_diffusion_step_fits_the_laptop_budget() -> None:
    pytest.importorskip("h3")
    grid = create_grid("h3", resolution=_COARSE_RESOLUTION)
    assert _MIN_COARSE_CELLS <= grid.cell_count <= _MAX_COARSE_CELLS

    values = {cell: float(i % 2) for i, cell in enumerate(grid.cells())}
    min_area = min(grid.area_m2(cell) for cell in values)
    dt_s = 0.05 * min_area / _DIFFUSIVITY

    started = time.perf_counter()
    diffusion_step(values, grid, _DIFFUSIVITY, dt_s)
    elapsed = time.perf_counter() - started
    assert elapsed < _BUDGET_SECONDS, f"coarse-tier step took {elapsed:.1f}s"


@pytest.mark.slow
def test_flow_accumulation_kernel_fits_the_budget() -> None:
    """Profile flow-accumulation on a large single-outlet drainage network."""
    receivers = [*range(1, _FLOW_CELLS), NO_RECEIVER]
    areas = [1.0] * _FLOW_CELLS
    elevations = [float(_FLOW_CELLS - i) for i in range(_FLOW_CELLS)]

    started = time.perf_counter()
    drainage = flow_accumulation(receivers, areas, elevations)
    elapsed = time.perf_counter() - started

    # The chain outlet must have gathered the whole network's area.
    assert drainage[-1] == pytest.approx(float(_FLOW_CELLS))
    assert elapsed < _FLOW_BUDGET_SECONDS, (
        f"flow accumulation over {_FLOW_CELLS} cells took {elapsed:.1f}s"
    )


@pytest.mark.slow
def test_biology_step_fits_the_budget() -> None:
    """Profile a multi-species biology tick over the coarse-tier cell count."""
    grid = FakeGrid(rows=_BIO_ROWS, cols=_BIO_COLS)
    plants = make_organism("grass", crowding_cap_per_m2=5.0, reproduction="asexual")
    grazers = [
        make_organism(f"grazer{i}", diet={"grass": 1.0}, crowding_cap_per_m2=1.0e-3)
        for i in range(_BIO_SPECIES - 1)
    ]
    organisms = [plants, *grazers]
    ctx = uniform_context(grid, organisms)  # type: ignore[arg-type]
    land = list(grid.cells())
    state = WorldBiologyState(
        surface_populations={org.species_id: dict.fromkeys(land, 1.0) for org in organisms},
        subsurface_populations={},
    )

    started = time.perf_counter()
    for _ in range(_BIO_TICKS):
        state = step_biology(state, ctx, SeededRng(7), SECONDS_PER_YEAR)
    elapsed = time.perf_counter() - started

    assert elapsed < _BIO_BUDGET_SECONDS, (
        f"{_BIO_TICKS} biology ticks over {len(land)} cells x "
        f"{_BIO_SPECIES} species took {elapsed:.1f}s"
    )
