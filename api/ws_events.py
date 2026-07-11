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


class RunTelemetryEvent(BaseModel):
    """Wall-clock speed of the most recent simulation run."""

    type: Literal["run_telemetry"] = "run_telemetry"
    ticks: int = 0
    ticks_per_second: float = 0.0
    mean_seconds_per_tick: float = 0.0
    wall_seconds: float = 0.0


class ComputeBackendChangedEvent(BaseModel):
    """The live compute backend was hot-swapped (e.g. CPU numpy <-> GPU jax).

    Broadcast so every viewer learns the sim's math moved devices; carries
    the resolved backend, where its arrays live, and how many devices it
    shards across.
    """

    type: Literal["compute_backend_changed"] = "compute_backend_changed"
    backend: str = "numpy"
    device: str = "cpu"
    num_devices: int = 1


EVENT_MODELS: dict[str, type[BaseModel]] = {
    "hello": HelloEvent,
    "setting_changed": SettingChangedEvent,
    "heartbeat": HeartbeatEvent,
    "run_telemetry": RunTelemetryEvent,
    "compute_backend_changed": ComputeBackendChangedEvent,
}
"""Every streamable event type, keyed by its ``type`` discriminator."""


def ws_schema() -> dict[str, dict[str, Any]]:
    """Return a JSON Schema per WS event type (the /ws/schema payload)."""
    return {name: model.model_json_schema() for name, model in EVENT_MODELS.items()}
