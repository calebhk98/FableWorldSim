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

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from api.commands import (
    Command,
    CommandNotFoundError,
    CommandRegistry,
    apply_setting,
    build_default_registry,
)
from api.settings import Settings, get_setting, load_settings, setting_paths
from api.state import AppState, EventBus
from api.ws_events import HelloEvent, ws_schema
from ports.access import Access, Principal

_LOGGER = logging.getLogger("fableworldsim.api")

_DEFAULT_PRINCIPAL_ID = "local"
_DEFAULT_ROLES = ("admin",)


class SetValueBody(BaseModel):
    """Body for PUT /settings/{path}."""

    value: Any = None


def principal_from(request: Request) -> Principal:
    """Return the requesting principal.

    M1 placeholder identity for local play: connections are the local
    admin unless they present ``x-fws-principal`` / ``x-fws-roles``
    headers (which viewers and scripts use to run in the read tier).
    Real login + throttling replaces this extraction when online play
    lands; everything downstream (tiers, Access checks) stays as-is.
    """
    principal_id = request.headers.get("x-fws-principal", _DEFAULT_PRINCIPAL_ID)
    raw_roles = request.headers.get("x-fws-roles")
    roles = (
        _DEFAULT_ROLES
        if raw_roles is None
        else tuple(role.strip() for role in raw_roles.split(",") if role.strip())
    )
    return Principal(principal_id=principal_id, roles=roles)


def create_app(settings: Settings | None = None, access: Access | None = None) -> FastAPI:
    """Build the API server around a settings instance.

    Passing ``settings=None`` hydrates from the ``config/`` folder, the
    normal startup path; ``access`` defaults to role-based two-tier
    access (read for everyone, write for editor/admin).
    """
    state = AppState(
        settings=settings if settings is not None else load_settings(),
        bus=EventBus(),
        access=access,
    )
    app = FastAPI(
        title="FableWorldSim",
        description="Headless world-simulator API; see /commands and /ws/schema.",
        version="0.0.0",
    )
    _register_discovery(app, state, build_default_registry())
    _register_settings(app, state)
    _register_stream(app, state)
    _register_observability(app, state)
    app.state.sim = state
    return app


def _deny(state: AppState, principal: Principal, resource: str) -> HTTPException:
    """Record and build the 403 for an access-tier denial."""
    state.metrics["commands_denied"] += 1
    _LOGGER.warning("access denied: principal=%s resource=%s", principal.principal_id, resource)
    return HTTPException(
        status_code=403,
        detail=f"principal {principal.principal_id!r} may not write {resource!r}",
    )


def _execute_command(command: Command, state: AppState, params: BaseModel) -> object:
    """Run a command's handler, translating late execution errors.

    A handler can still reject validated params at execution time (e.g.
    set_setting on an unknown path or an invalid value). Mirror the
    /settings/{path} endpoint: unknown key -> 404, bad value -> 422,
    never a raw 500 — the API is AI-driven, so malformed input is normal.
    Write-tier handlers run under the single-writer lock.
    """
    try:
        if command.mutates:
            with state.write_lock:
                return command.handler(state, params)
        return command.handler(state, params)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


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
    def run_command(
        name: str, request: Request, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate params, enforce the access tier, and execute.

        Read-tier commands run concurrently; write-tier commands are
        checked against the Access policy and serialized through the
        single-writer lock.
        """
        try:
            command = registry.get(name)
        except CommandNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        principal = principal_from(request)
        resource = f"command.{name}"
        if command.mutates and not state.access.can_write(principal, resource):
            raise _deny(state, principal, resource)
        if not command.mutates and not state.access.can_read(principal, resource):
            raise _deny(state, principal, resource)
        try:
            params = command.params.model_validate(body or {})
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        _LOGGER.info(
            "command: name=%s principal=%s mutates=%s",
            name,
            principal.principal_id,
            command.mutates,
        )
        result = _execute_command(command, state, params)
        state.metrics["commands_executed"] += 1
        return {"command": name, "result": result}


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
    def write_setting(path: str, body: SetValueBody, request: Request) -> dict[str, Any]:
        """Change one option; validates and broadcasts like set_setting."""
        principal = principal_from(request)
        if not state.access.can_write(principal, "settings"):
            raise _deny(state, principal, "settings")
        try:
            return {"path": path, "value": apply_setting(state, path, body.value)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _register_observability(app: FastAPI, state: AppState) -> None:
    """Mount the metrics endpoint (structured logs go to stdlib logging)."""

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        """Return server counters plus the latest run's speed telemetry."""
        report: dict[str, object] = {
            **state.metrics,
            "ws_subscribers": state.bus.subscriber_count,
        }
        if state.sim_telemetry is not None:
            report["sim"] = state.sim_telemetry
        return report


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
