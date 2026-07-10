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

if TYPE_CHECKING:
    from pydantic import BaseModel

    from api.settings import Settings
    from ports.access import Access


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
