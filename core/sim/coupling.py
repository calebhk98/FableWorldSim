"""Cross-layer coupling: one shared plant-biomass field for biology and civ.

Civilization forestry/farmland and the biology food web must draw on the
*same* standing plant biomass -- clear-cutting a forest has to starve the
deer that grazed it, not a parallel bookkeeping copy that regrows on its
own schedule.  Each coupled tick does three things in order:

1. Read civilization's view of "how much plant is standing here" straight
   off biology's own autotroph populations (:func:`plant_biomass_field`).
2. Let civilization extract from that live field (forestry, farmland);
   nothing regrows it in between, so every kg/m2 gone is genuine drawdown.
3. Fold that drawdown back into the same autotroph populations
   (:func:`apply_biomass_drawdown`) before biology's own feed/grow/migrate
   pass runs, so the herbivores that eat there see exactly what forestry
   left behind.

This is the seam ``core/biology/state.py`` calls out and
``core/civilization/economy.py`` was written to expect, wired up: neither
layer imports the other's engine internals, they just agree on one field
of standing biomass per cell.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from core.biology.engine import step_biology
from core.biology.state import apply_biomass_drawdown, plant_biomass_field
from core.civilization.engine import step_civilization

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core.biology.context import BiologyContext
    from core.biology.organism import Organism
    from core.biology.state import WorldBiologyState
    from core.chronicle.log import Chronicle
    from core.civilization.context import CivContext
    from core.civilization.state import WorldCivState
    from ports.grid import CellId
    from ports.rng import Rng

PROCESS_NAME = "biology_civilization"


@dataclass(frozen=True)
class WorldState:
    """The whole cross-layer world: biology and civilization, stepped together."""

    biology: WorldBiologyState
    civ: WorldCivState


@dataclass(frozen=True)
class CoupledContext:
    """Per-run invariants the coupled step shares: both layers' own contexts.

    ``organisms`` is the full organism roster (not just autotrophs): it is
    handed straight to :func:`plant_biomass_field` and
    :func:`apply_biomass_drawdown`, which each filter to autotrophs
    themselves.
    """

    bio_ctx: BiologyContext
    civ_ctx: CivContext
    organisms: Sequence[Organism]


def _drawdown(before: dict[CellId, float], after: dict[CellId, float]) -> dict[CellId, float]:
    """Return how much biomass extraction removed from each cell.

    Civilization never regrows the field itself while coupled -- biology's
    own growth pass is the only regrowth model in play -- so every kg/m2
    missing between the live snapshot and the post-extraction field is
    extraction's doing, never negative.
    """
    return {cell: max(0.0, value - after.get(cell, 0.0)) for cell, value in before.items()}


def step_coupled(  # noqa: PLR0913 - one param per coupled-layer input
    state: WorldState,
    ctx: CoupledContext,
    civ_rng: Rng,
    bio_rng: Rng,
    dt_s: float,
    chronicle: Chronicle | None = None,
) -> WorldState:
    """Advance civilization's extraction and biology's food web as one loop.

    Civilization reads and draws down the exact biomass biology is
    currently carrying; the drawdown is folded back into the same
    autotroph populations before biology feeds its herbivores this same
    tick, so a forest clear-cut this year measurably starves the grazers
    that depended on it this year, not several ticks later.
    """
    shared_biomass = plant_biomass_field(state.biology, ctx.organisms)
    civ_out = step_civilization(
        replace(state.civ, plant_biomass_kg_m2=shared_biomass),
        ctx.civ_ctx,
        civ_rng,
        dt_s,
        chronicle,
        biomass_override=shared_biomass,
    )
    removed = _drawdown(shared_biomass, dict(civ_out.plant_biomass_kg_m2))
    grazed = apply_biomass_drawdown(state.biology, ctx.organisms, removed)
    bio_out = step_biology(grazed, ctx.bio_ctx, bio_rng, dt_s, chronicle)
    return WorldState(biology=bio_out, civ=civ_out)


class CoupledProcess:
    """Orchestrator process advancing biology and civilization together.

    Forks its own persistent "civilization" and "biology" RNG streams once
    at construction (mirroring :class:`~core.biology.engine.BiologyProcess`
    and :class:`~core.civilization.engine.CivilizationProcess`), so replaying
    a run from the same seed replays identically.
    """

    def __init__(
        self,
        ctx: CoupledContext,
        rng: Rng,
        chronicle: Chronicle | None = None,
    ) -> None:
        """Bind the coupled step to both layers' contexts and forked RNG streams."""
        self._ctx = ctx
        self._civ_rng = rng.fork("civilization")
        self._bio_rng = rng.fork("biology")
        self._chronicle = chronicle

    @property
    def name(self) -> str:
        """Return the process name shown by the orchestrator."""
        return PROCESS_NAME

    def step(self, state: WorldState, dt_s: float) -> WorldState:
        """Advance both layers by one orchestrator step."""
        return step_coupled(state, self._ctx, self._civ_rng, self._bio_rng, dt_s, self._chronicle)
