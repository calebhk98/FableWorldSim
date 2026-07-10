"""WebSocket event schemas — the documented streaming interface.

OpenAPI only covers the REST half, so every WS event type is a pydantic
model here and ``/ws/schema`` serves their JSON Schemas (one per event
type).  Clients discover the streaming vocabulary the same way they
discover REST: by asking the server.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class HelloEvent(BaseModel):
    """First event on every WS connection: identifies the server."""

    type: Literal["hello"] = "hello"
    server: str = "fableworldsim"
    schema_url: str = "/ws/schema"


class SettingChangedEvent(BaseModel):
    """An option changed (via API/UI); carries the new validated value."""

    type: Literal["setting_changed"] = "setting_changed"
    path: str
    value: Any = None


class HeartbeatEvent(BaseModel):
    """Periodic liveness signal; carries the current simulation tick."""

    type: Literal["heartbeat"] = "heartbeat"
    tick: int = 0


EVENT_MODELS: dict[str, type[BaseModel]] = {
    "hello": HelloEvent,
    "setting_changed": SettingChangedEvent,
    "heartbeat": HeartbeatEvent,
}
"""Every streamable event type, keyed by its ``type`` discriminator."""


def ws_schema() -> dict[str, dict[str, Any]]:
    """Return a JSON Schema per WS event type (the /ws/schema payload)."""
    return {name: model.model_json_schema() for name, model in EVENT_MODELS.items()}
