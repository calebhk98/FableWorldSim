"""Shared application state: live settings + the event bus.

The bus is a fan-out of asyncio queues: every WS connection subscribes,
every state change publishes.  Publishing is non-blocking; a slow
consumer only backs up its own queue.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from adapters.access_roles import RoleBasedAccess
from adapters.compute_registry import create_array_backend
from api.settings import with_setting
from api.ws_events import ComputeBackendChangedEvent, RunTelemetryEvent
from ports.array_backend import ComputeBackendUnavailableError

if TYPE_CHECKING:
    from pydantic import BaseModel

    from api.settings import Settings
    from core.sim.orchestrator import RunTelemetry
    from ports.access import Access
    from ports.array_backend import ArrayBackend


class EventBus:
    """Fan-out pub/sub for server events."""

    def __init__(self) -> None:
        """Start with no subscribers."""
        self._queues: list[asyncio.Queue[BaseModel]] = []

    def subscribe(self) -> asyncio.Queue[BaseModel]:
        """Return a fresh queue that will receive all future events."""
        queue: asyncio.Queue[BaseModel] = asyncio.Queue()
        self._queues.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[BaseModel]) -> None:
        """Stop delivering events to ``queue``."""
        if queue in self._queues:
            self._queues.remove(queue)

    def publish(self, event: BaseModel) -> None:
        """Deliver ``event`` to every subscriber without blocking."""
        for queue in self._queues:
            queue.put_nowait(event)

    @property
    def subscriber_count(self) -> int:
        """Return how many queues are currently subscribed."""
        return len(self._queues)


class AppState:
    """Mutable per-server state shared by REST, commands, and WS.

    The sim is a single authoritative writer with many read-only
    viewers: every mutation goes through ``write_lock``, serializing
    changes no matter how many connections submit them.  ``metrics``
    holds monotonically increasing counters for the /metrics endpoint.
    """

    def __init__(
        self,
        settings: Settings,
        bus: EventBus,
        access: Access | None = None,
    ) -> None:
        """Bind live settings, the event bus, and the access policy."""
        self.settings = settings
        self.bus = bus
        self.access = access if access is not None else RoleBasedAccess()
        self.write_lock = threading.RLock()
        self.metrics: dict[str, int] = {
            "commands_executed": 0,
            "commands_denied": 0,
            "settings_changed": 0,
        }
        self.sim_telemetry: dict[str, object] | None = None
        """Timing of the most recent simulation run (None until one runs)."""
        self._compute_backend: ArrayBackend | None = None

    @property
    def compute_backend(self) -> ArrayBackend:
        """Return the live compute backend, built from settings on first use.

        This is the single handle the sim's backend-aware reductions read,
        so hot-swapping it (see :meth:`set_compute_backend`) redirects
        their math to a new device on the next call.
        """
        with self.write_lock:
            if self._compute_backend is None:
                self._compute_backend = create_array_backend(self.settings.compute.backend)
            return self._compute_backend

    def set_compute_backend(self, name: str) -> ArrayBackend:
        """Hot-swap the compute backend at runtime; return the new backend.

        Safe to call while the sim runs: the swap takes the single-writer
        ``write_lock``, so it lands between ticks, never mid-tick, and the
        next reduction picks up the new device.  The new backend is built
        (and thus proven available) *before* anything is committed, so
        asking for a GPU backend on a CPU-only host raises
        :class:`ComputeBackendUnavailableError` and leaves the running
        backend and the ``compute.backend`` setting untouched.
        """
        with self.write_lock:
            validated = with_setting(self.settings, "compute.backend", name)
            try:
                candidate = create_array_backend(name)
            except ValueError as exc:  # unknown/unimplemented toggle (e.g. dask)
                raise ComputeBackendUnavailableError(str(exc)) from exc
            self.settings = validated
            self._compute_backend = candidate
        self.bus.publish(
            ComputeBackendChangedEvent(
                backend=candidate.name,
                device=candidate.device,
                num_devices=candidate.num_devices,
            )
        )
        return candidate

    def record_run_telemetry(self, telemetry: RunTelemetry) -> None:
        """Store the latest run's timing and broadcast it to WS subscribers."""
        self.sim_telemetry = telemetry.to_metrics()
        self.bus.publish(
            RunTelemetryEvent(
                ticks=telemetry.ticks,
                ticks_per_second=telemetry.ticks_per_second,
                mean_seconds_per_tick=telemetry.mean_seconds_per_tick,
                wall_seconds=telemetry.wall_seconds,
            )
        )
