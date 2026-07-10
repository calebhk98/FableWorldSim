"""Degenerate-state guards: runaway or non-conserving sims fail loudly.

Emergence is the design bet, so the difference between "interesting
dynamics" and "the model broke" must be mechanical: conservation drift
and explosive growth/collapse raise instead of silently corrupting a
long run.  These checks are prerequisites for the fidelity autotuner and
for debugging emergent behavior.
"""

from __future__ import annotations

import math

DEFAULT_CONSERVATION_REL_TOL = 1e-6
DEFAULT_MAX_GROWTH_FACTOR = 10.0


class DegenerateStateError(RuntimeError):
    """A simulation quantity left the physically plausible envelope."""


def assert_conserved(
    before: float,
    after: float,
    *,
    rel_tol: float = DEFAULT_CONSERVATION_REL_TOL,
    quantity: str = "total",
) -> None:
    """Raise when a conserved global total drifted beyond tolerance.

    Use around any step that must conserve (energy, mass, population
    without births/deaths).  A zero 'before' with a nonzero 'after' is
    always a violation.
    """
    if before == 0.0:
        if after != 0.0:
            msg = f"{quantity} appeared from nothing: 0 -> {after}"
            raise DegenerateStateError(msg)
        return
    drift = abs(after - before) / abs(before)
    if not math.isfinite(after) or drift > rel_tol:
        msg = (
            f"{quantity} not conserved: {before} -> {after} "
            f"(relative drift {drift:.3e} > {rel_tol:.1e})"
        )
        raise DegenerateStateError(msg)


def check_runaway(
    previous: float,
    current: float,
    *,
    max_growth_factor: float = DEFAULT_MAX_GROWTH_FACTOR,
    quantity: str = "value",
) -> None:
    """Raise on explosive growth or collapse between two checkpoints.

    Flags population explosions (growth beyond ``max_growth_factor`` per
    checkpoint), collapses (shrinking by more than the same factor), and
    non-finite values.  Checkpoints should be far enough apart that real
    dynamics stay inside the envelope.
    """
    if not math.isfinite(current):
        msg = f"{quantity} became non-finite: {current}"
        raise DegenerateStateError(msg)
    if previous <= 0.0:
        return
    ratio = current / previous
    if ratio > max_growth_factor:
        msg = f"{quantity} exploded: {previous} -> {current} (x{ratio:.1f})"
        raise DegenerateStateError(msg)
    if current > 0.0 and ratio < 1.0 / max_growth_factor:
        msg = f"{quantity} collapsed: {previous} -> {current} (x{ratio:.4f})"
        raise DegenerateStateError(msg)
