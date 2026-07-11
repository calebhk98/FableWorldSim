"""Determinism at the whole-orchestrator level, and across grid backends.

The checklist names two *separate* determinism claims, and this module
keeps them in two separate tests so neither can be quietly weakened into
the other:

1. **Same backend + seed + config => bit-identical.**  Existing tests
   (e.g. ``tests/test_bio_engine.py::test_same_seed_replays_identically``)
   only prove this one layer's ``step`` function is a pure function of its
   inputs.  ``test_orchestrator_run_is_bit_identical_per_seed`` elevates
   the claim to the ``Orchestrator`` itself, across several ticks, with a
   freshly constructed process + RNG each run (the correct replay
   pattern: a ``BiologyProcess`` forks its own RNG stream in
   ``__init__``, so replaying means rebuilding it, not reusing a
   part-consumed one).  This is checked with exact ``==``.

2. **Across grid backends/resolutions => field-equivalent within
   tolerance, never bit-identical.**  H3 and S2 cells are not equal-area
   and different resolutions discretize the same continuous fields
   differently, so per-cell values (and even area-weighted reductions)
   will *never* match exactly — only the underlying physical field they
   approximate should agree, and only up to the discretization error.
   ``test_cross_grid_backend_fields_are_equivalent_within_tolerance``
   reduces a climate world to a few area-weighted global scalars (the
   same reduction ``tests/test_golden_master.py`` uses) and compares
   them with ``pytest.approx(rel=...)``, never ``==``.

   The primary comparison is h3 resolution 1 vs resolution 2, not the
   coarser resolution 0 vs 1: resolution 0 has only 122 cells (including
   the 12 icosahedral pentagons that dominate a grid that coarse), and
   observed global mean precipitation swings ~38% between res 0 and res
   1 -- too close to "wildly different" to make a stable, non-flaky
   tolerance test.  Resolutions 1 and 2 (842 vs 5882 cells) are still a
   meaningfully different discretization but agree far more tightly, so
   the tolerance below has real headroom instead of being propped open
   to force a pass.  A secondary h3-vs-s2 cross-backend comparison runs
   too, at cell counts of similar order (h3 res 1, ~842 cells, vs s2
   level 4, ~1536 cells); if ``s2sphere`` is not installed, that sub-check
   is skipped in place (module import guarded below) and the
   cross-resolution comparison remains the primary assertion.
"""

from __future__ import annotations

import pytest

pytest.importorskip("h3")

from adapters.grid_registry import create_grid
from adapters.rng_seeded import SeededRng
from core.biology.engine import BiologyProcess
from core.biology.state import WorldBiologyState
from core.climate.model import simulate_climate
from core.grid.area_weighted import area_fraction, area_weighted_mean
from core.hydrology.sea_mask import build_sea_mask
from core.sim.constants import SECONDS_PER_YEAR
from core.sim.orchestrator import Orchestrator
from core.sim.presets import earth
from core.topography.procedural import ProceduralTopography
from tests.bio_helpers import make_organism, uniform_context
from tests.civ_helpers import FakeGrid

try:
    import s2sphere  # noqa: F401

    _S2_AVAILABLE = True
except ImportError:
    _S2_AVAILABLE = False

# --- (a) orchestrator-level bit-identical replay -----------------------------

_BIO_SEED = 17
_BIO_TICKS = 5


def _biology_organisms() -> tuple[object, object]:
    grass = make_organism("grass", crowding_cap_per_m2=5.0, reproduction="asexual")
    herbivore = make_organism(
        "herbivore", diet={"grass": 1.0}, body_mass_kg=50.0, crowding_cap_per_m2=1.0e-3
    )
    return grass, herbivore


def _run_biology_orchestrator(seed: int, ticks: int) -> WorldBiologyState:
    """Build a fresh multi-species world, orchestrator, process and RNG, and run.

    Everything is rebuilt from scratch (not reused across calls) so that
    replaying with the same seed exercises the real "start a new run"
    path rather than resuming a process whose forked RNG stream has
    already been partly consumed.
    """
    grass, herbivore = _biology_organisms()
    grid = FakeGrid(rows=3, cols=3)
    ctx = uniform_context(grid, [grass, herbivore])  # type: ignore[list-item]
    land = list(grid.cells())
    state = WorldBiologyState(
        surface_populations={
            "grass": dict.fromkeys(land, 2.5),
            "herbivore": dict.fromkeys(land, 5.0e-4),
        },
        subsurface_populations={},
    )
    orchestrator: Orchestrator[WorldBiologyState] = Orchestrator(tick_seconds=SECONDS_PER_YEAR)
    orchestrator.register(BiologyProcess(ctx, SeededRng(seed)))
    return orchestrator.run(state, ticks=ticks)


def test_orchestrator_run_is_bit_identical_per_seed() -> None:
    """Same backend + seed + config replayed through the Orchestrator matches exactly."""
    first = _run_biology_orchestrator(_BIO_SEED, _BIO_TICKS)
    second = _run_biology_orchestrator(_BIO_SEED, _BIO_TICKS)
    assert first == second
    # Sanity: the run actually did something non-trivial over 5 ticks, so an
    # exact match is a meaningful claim rather than two empty states.
    assert first.tick == _BIO_TICKS
    assert sum(first.surface_populations["herbivore"].values()) > 0.0


# --- (b) cross-backend / cross-resolution field-equivalence -----------------

_CLIMATE_SEED = 42
_SEASON_COUNT = 2

# Generous headroom over what was actually observed (h3 res1 vs res2):
# ocean fraction differed ~0.03%, mean temperature ~0.35%, mean precipitation
# ~7.7%.  Precipitation is the noisiest reduction (most spatially variable
# field), hence the wider tolerance.
_OCEAN_REL_TOL = 0.05
_TEMPERATURE_REL_TOL = 0.05
_PRECIPITATION_REL_TOL = 0.20


def _climate_summary(backend: str, resolution: int) -> dict[str, float]:
    """Build a known-seed climate world on ``backend`` and reduce it to globals."""
    planet = earth()
    grid = create_grid(backend, resolution=resolution, radius_m=planet.radius_m)
    heights = ProceduralTopography(SeededRng(_CLIMATE_SEED)).heights(grid)
    mask = build_sea_mask(grid, heights, planet.ocean_fraction, planet.tidal_range_m)
    state = simulate_climate(grid, planet, heights, mask, season_count=_SEASON_COUNT)

    cells = sorted(state.annual_mean_temperature_k)
    areas = [grid.area_m2(c) for c in cells]
    return {
        "ocean_area_fraction": area_fraction([mask.ocean[c] for c in cells], areas),
        "mean_temperature_k": area_weighted_mean(
            [state.annual_mean_temperature_k[c] for c in cells], areas
        ),
        "mean_precipitation_mm_yr": area_weighted_mean(
            [state.annual_precipitation_mm_yr[c] for c in cells], areas
        ),
    }


def _assert_field_equivalent(actual: dict[str, float], expected: dict[str, float]) -> None:
    """Assert two global-scalar summaries agree within tolerance, never exactly."""
    assert actual["ocean_area_fraction"] == pytest.approx(
        expected["ocean_area_fraction"], rel=_OCEAN_REL_TOL
    )
    assert actual["mean_temperature_k"] == pytest.approx(
        expected["mean_temperature_k"], rel=_TEMPERATURE_REL_TOL
    )
    assert actual["mean_precipitation_mm_yr"] == pytest.approx(
        expected["mean_precipitation_mm_yr"], rel=_PRECIPITATION_REL_TOL
    )


@pytest.mark.slow
def test_cross_grid_backend_fields_are_equivalent_within_tolerance() -> None:
    """Global climate scalars agree across resolutions/backends, but never bit-exactly."""
    res1 = _climate_summary("h3", 1)
    res2 = _climate_summary("h3", 2)

    # Never bit-identical -- different discretizations reduce differently.
    assert res1 != res2
    _assert_field_equivalent(res1, res2)

    if _S2_AVAILABLE:
        # h3 res 1 (~842 cells) vs s2 level 4 (~1536 cells): similar order of
        # cell count, so this isolates the backend swap from a resolution swap.
        s2_summary = _climate_summary("s2", 4)
        assert res1 != s2_summary
        _assert_field_equivalent(res1, s2_summary)
    # else: s2sphere is not installed in this environment. The cross-resolution
    # comparison above already exercises the "field-equivalent within
    # tolerance, never exact" claim, so this sub-check is simply skipped
    # rather than failing the whole test.
