"""Multi-species ecology dynamics: oscillatory boom/bust vs the smoothed glide.

``tests/test_bio_population.py`` proves the single-species growth *map*
overshoots when ``oscillatory=True`` (the Ricker map) and glides otherwise
(Beverton-Holt). That is a one-species, one-cell claim about
:func:`core.biology.population.logistic_step`. What it does not cover is
whether the *full* multi-species pipeline — feed, offtake, grow, migrate,
run through :func:`core.biology.engine.step_biology` tick after tick —
actually produces the checklist's headline ecology claim: a predator-prey
pair in oscillatory mode shows lag-driven boom/bust (the snowshoe-hare/lynx
cycle), while the same food chain in smoothed mode glides to equilibrium.

The food chain is grass (autotroph, always smoothed) -> prey (herbivore,
diet=grass) -> predator (diet=prey). Only the prey and predator's
``oscillatory`` flag toggles between the two runs; the prey's life-history
knobs (short maturity, big litter) are tuned so its feeding-scaled growth
exponent overshoots capacity under the Ricker map. What emerged when this
was run and observed (see the module-level comment further down) was not
just "more amplitude" but a textbook lag: every prey population peak is
followed, one tick later, by a predator population peak — never
simultaneous, never earlier.
"""

from __future__ import annotations

from adapters.rng_seeded import SeededRng
from core.biology.engine import step_biology
from core.biology.organism import Organism
from core.biology.state import WorldBiologyState
from core.sim.constants import SECONDS_PER_YEAR
from tests.bio_helpers import make_organism, uniform_context
from tests.civ_helpers import FakeGrid

_YEAR = SECONDS_PER_YEAR
_TICKS = 45
_SEED = 7
_TRANSIENT = 10
"""Ticks discarded before peak-finding, so the initial ramp-up isn't mistaken for a cycle."""


def _build_food_chain(oscillatory: bool) -> tuple[Organism, Organism, Organism]:
    """Return grass -> prey -> predator, with prey/predator's growth map toggled.

    Grass stays smoothed (an ever-present resource, not itself the subject of
    the claim). Prey gets a short maturity and a big litter so its
    feeding-scaled intrinsic rate overshoots capacity when ``oscillatory``
    selects the overcompensating Ricker map; the predator inherits the same
    toggle so a boom/bust upstream in prey can cascade into one in predator.
    """
    grass = make_organism(
        "grass", crowding_cap_per_m2=5.0, reproduction="asexual", oscillatory=False
    )
    prey = make_organism(
        "prey",
        diet={"grass": 1.0},
        body_mass_kg=1.0,
        crowding_cap_per_m2=0.05,
        lifespan_years=5.0,
        maturity_years=1.0,
        litter_size=25.0,
        reproduction="asexual",
        oscillatory=oscillatory,
    )
    predator = make_organism(
        "predator",
        diet={"prey": 1.0},
        body_mass_kg=10.0,
        crowding_cap_per_m2=0.005,
        lifespan_years=8.0,
        maturity_years=1.5,
        litter_size=8.0,
        reproduction="asexual",
        oscillatory=oscillatory,
    )
    return grass, prey, predator


def _run_totals(oscillatory: bool, ticks: int = _TICKS) -> tuple[list[float], list[float]]:
    """Step the food chain for ``ticks`` years; return (prey, predator) total series.

    A fresh :class:`SeededRng` is built from the same seed for every call, so
    the oscillatory and smoothed runs start from identical randomness — only
    the growth map differs.
    """
    grass, prey, predator = _build_food_chain(oscillatory)
    grid = FakeGrid(rows=2, cols=2)
    ctx = uniform_context(grid, [grass, prey, predator])
    land = list(grid.cells())
    state = WorldBiologyState(
        surface_populations={
            "grass": dict.fromkeys(land, 2.5),
            "prey": dict.fromkeys(land, 0.02),
            "predator": dict.fromkeys(land, 0.002),
        },
        subsurface_populations={},
    )
    rng = SeededRng(_SEED)
    prey_totals: list[float] = []
    predator_totals: list[float] = []
    for _ in range(ticks):
        state = step_biology(state, ctx, rng, _YEAR)
        prey_totals.append(sum(state.surface_populations["prey"].values()))
        predator_totals.append(sum(state.surface_populations["predator"].values()))
    return prey_totals, predator_totals


def _late_window_amplitude(series: list[float], window: int = 20) -> float:
    """Return max - min over the series' last ``window`` ticks."""
    tail = series[-window:]
    return max(tail) - min(tail)


def _local_maxima(series: list[float], start: int = _TRANSIENT) -> list[int]:
    """Return indices of strict local maxima, skipping the initial transient."""
    return [
        i
        for i in range(max(1, start), len(series) - 1)
        if series[i] > series[i - 1] and series[i] > series[i + 1]
    ]


def _peak_lags(prey: list[float], predator: list[float], start: int = _TRANSIENT) -> list[int]:
    """Return, for each prey peak, ticks to the next predator peak after it."""
    prey_peaks = _local_maxima(prey, start)
    predator_peaks = _local_maxima(predator, start)
    lags = []
    for prey_peak in prey_peaks:
        after = [p for p in predator_peaks if p > prey_peak]
        if after:
            lags.append(min(after) - prey_peak)
    return lags


def test_oscillatory_mode_glides_less_than_smoothed_mode_flattens() -> None:
    """The smoothed run's late-time wobble is negligible next to oscillatory's.

    Observed with seed 7 on a 2x2 grid over 45 ticks: oscillatory prey settle
    into a late-window amplitude around 0.49 (individuals/m2 x cell count)
    and predator around 9e-4, while the smoothed run's late-window amplitude
    is of order 1e-6 (prey) and exactly flat (predator) — numerical residue
    around a fixed point, not a cycle. The margin below (10x) is far inside
    that gap.

    Predator's amplitude (and the module docstring's headline numbers) moved
    from an earlier tuning after issue #15 added a food-limited term to
    carrying capacity (``K = min(crowding cap x suitability, food cap)``):
    predator's own capacity now tracks prey biomass instead of being a fixed
    crowding x suitability product, which damps its oscillation somewhat
    without changing the qualitative claim this test makes.
    """
    oscillatory_prey, oscillatory_predator = _run_totals(oscillatory=True)
    smoothed_prey, smoothed_predator = _run_totals(oscillatory=False)

    oscillatory_prey_amp = _late_window_amplitude(oscillatory_prey)
    smoothed_prey_amp = _late_window_amplitude(smoothed_prey)
    oscillatory_predator_amp = _late_window_amplitude(oscillatory_predator)
    smoothed_predator_amp = _late_window_amplitude(smoothed_predator)

    # The smoothed run should have all but flatlined by the tail window.
    assert smoothed_prey_amp < 0.001
    assert smoothed_predator_amp < 0.0001

    # The oscillatory run should still be swinging by a wide, unmistakable margin.
    assert oscillatory_prey_amp > 0.1
    assert oscillatory_predator_amp > 0.0005

    # And the gap between the two regimes should be large, not knife-edge.
    assert oscillatory_prey_amp > 10.0 * max(smoothed_prey_amp, 1e-9)
    assert oscillatory_predator_amp > 10.0 * max(smoothed_predator_amp, 1e-9)


def test_oscillatory_predator_population_peaks_lag_prey_population_peaks() -> None:
    """Every prey boom is followed, one tick later, by a predator boom.

    This is the sharper claim the checklist wants: not just "more noise" but
    a genuine lag-driven cycle, the game-scale echo of the snowshoe-hare/lynx
    dynamic. Peak-finding (rather than cross-correlation) is used because the
    tuned regime here settles into an exact period-2 cycle, which aliases
    cross-correlation at every odd lag; forward peak-to-peak distance has no
    such ambiguity. Observed with seed 7: 17 prey peaks after the transient,
    every one followed by a predator peak exactly 1 tick later — never 0,
    never 2 or more.
    """
    prey, predator = _run_totals(oscillatory=True)
    lags = _peak_lags(prey, predator)

    # There should be a real, healthy number of cycles to measure, not one lucky peak.
    assert len(lags) >= 8
    # Every prey peak's nearest following predator peak should lag it — never coincide.
    assert all(lag > 0 for lag in lags)
    # And the lag should be small and essentially constant: a tight cycle, not noise.
    assert all(lag <= 2 for lag in lags)
    assert lags.count(1) == len(lags)
