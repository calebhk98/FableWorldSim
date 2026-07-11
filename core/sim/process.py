"""A reusable orchestrator-process wrapper for simulation layers.

Every layer (climate, biology, civilization) exposes its step as a free
``step_<layer>(state, ctx, rng, dt_s, chronicle)`` function and needs the
same thin adapter to satisfy the orchestrator's :class:`Process` protocol:
hold the per-run context, fork a dedicated RNG stream so the layer's
randomness never perturbs its neighbours, and delegate each step.  This
generic binds those once instead of each layer re-writing the boilerplate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from core.chronicle.log import Chronicle
    from ports.rng import Rng

StateT = TypeVar("StateT")
CtxT = TypeVar("CtxT")


class BoundProcess(Generic[StateT, CtxT]):
    """Binds a layer's step function to its context and a forked RNG stream."""

    def __init__(
        self,
        name: str,
        ctx: CtxT,
        rng: Rng,
        chronicle: Chronicle | None,
        runner: Callable[[StateT, CtxT, Rng, float, Chronicle | None], StateT],
    ) -> None:
        """Fork a dedicated RNG stream and remember how to step the layer."""
        self._name = name
        self._ctx = ctx
        self._rng = rng.fork(name)
        self._chronicle = chronicle
        self._runner = runner

    @property
    def name(self) -> str:
        """Return the process name shown by the orchestrator."""
        return self._name

    def step(self, state: StateT, dt_s: float) -> StateT:
        """Advance the layer by one orchestrator step."""
        return self._runner(state, self._ctx, self._rng, dt_s, self._chronicle)
