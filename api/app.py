"""The FastAPI application: REST + WebSocket, fully self-describing.

Discovery surfaces:

- ``GET /schema``    — the OpenAPI document (REST half).
- ``GET /commands``  — every command with params JSON Schema.
- ``GET /ws/schema`` — JSON Schema per WS event type (OpenAPI cannot
  describe WebSockets, so the streaming half documents itself here).

Settings are readable/writable over the same schema the config folder
hydrates, so config file, API, and UI are one option surface.  The core
stays a local server; "online" is this same server hosted remotely.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from api.commands import (
    CommandNotFoundError,
    CommandRegistry,
    apply_setting,
    build_default_registry,
)
from api.settings import Settings, get_setting, load_settings, setting_paths
from api.state import AppState, EventBus
from api.ws_events import HelloEvent, ws_schema


class SetValueBody(BaseModel):
    """Body for PUT /settings/{path}."""

    value: Any = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API server around a settings instance.

    Passing ``settings=None`` hydrates from the ``config/`` folder, the
    normal startup path.
    """
    state = AppState(
        settings=settings if settings is not None else load_settings(),
        bus=EventBus(),
    )
    app = FastAPI(
        title="FableWorldSim",
        description="Headless world-simulator API; see /commands and /ws/schema.",
        version="0.0.0",
    )
    _register_discovery(app, state, build_default_registry())
    _register_settings(app, state)
    _register_stream(app, state)
    app.state.sim = state
    return app


def _register_discovery(app: FastAPI, state: AppState, registry: CommandRegistry) -> None:
    """Mount the self-description + command-execution endpoints."""

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        """Return the OpenAPI document (alias of /openapi.json)."""
        return app.openapi()

    @app.get("/commands")
    def commands() -> list[dict[str, Any]]:
        """List every command with its parameter JSON Schema."""
        return registry.describe()

    @app.post("/commands/{name}")
    def run_command(name: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate params against the command's model and execute it."""
        try:
            command = registry.get(name)
        except CommandNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            params = command.params.model_validate(body or {})
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return {"command": name, "result": command.handler(state, params)}


def _register_settings(app: FastAPI, state: AppState) -> None:
    """Mount settings read/write endpoints over the shared schema."""

    @app.get("/settings")
    def read_settings() -> dict[str, Any]:
        """Return the whole settings tree."""
        return state.settings.model_dump()

    @app.get("/settings/paths")
    def read_setting_paths() -> list[str]:
        """Return every dotted option path."""
        return list(setting_paths(state.settings))

    @app.get("/settings/{path}")
    def read_setting(path: str) -> dict[str, Any]:
        """Return one option's value by dotted path."""
        try:
            return {"path": path, "value": get_setting(state.settings, path)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/settings/{path}")
    def write_setting(path: str, body: SetValueBody) -> dict[str, Any]:
        """Change one option; validates and broadcasts like set_setting."""
        try:
            return {"path": path, "value": apply_setting(state, path, body.value)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _register_stream(app: FastAPI, state: AppState) -> None:
    """Mount the WS event stream and its schema endpoint."""

    @app.get("/ws/schema")
    def websocket_schema() -> dict[str, dict[str, Any]]:
        """Return a JSON Schema per WS event type."""
        return ws_schema()

    @app.websocket("/ws")
    async def websocket_stream(websocket: WebSocket) -> None:
        """Stream server events (see /ws/schema for the vocabulary)."""
        await websocket.accept()
        await websocket.send_json(HelloEvent().model_dump())
        queue = state.bus.subscribe()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event.model_dump())
        except WebSocketDisconnect:
            pass
        finally:
            state.bus.unsubscribe(queue)
