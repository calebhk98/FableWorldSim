"""Shared application state: live settings + the event bus.

The bus is a fan-out of asyncio queues: every WS connection subscribes,
every state change publishes.  Publishing is non-blocking; a slow
consumer only backs up its own queue.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from api.settings import Settings


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
    """Mutable per-server state shared by REST, commands, and WS."""

    def __init__(self, settings: Settings, bus: EventBus) -> None:
        """Bind the live settings object and the event bus."""
        self.settings = settings
        self.bus = bus
