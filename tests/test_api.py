"""Tests for the self-describing API (REST + WS discovery surfaces)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings

_DEFAULT_RESOLUTION = 3
_NEW_RESOLUTION = 5
_HTTP_OK = 200
_HTTP_NOT_FOUND = 404
_HTTP_UNPROCESSABLE = 422


def _client() -> TestClient:
    return TestClient(create_app(Settings()))


def test_openapi_schema_is_served() -> None:
    response = _client().get("/schema")
    assert response.status_code == _HTTP_OK
    assert response.json()["info"]["title"] == "FableWorldSim"


def test_commands_are_discoverable_with_schemas() -> None:
    body = _client().get("/commands").json()
    by_name = {entry["name"]: entry for entry in body}
    assert "ping" in by_name
    assert "set_setting" in by_name
    set_params = by_name["set_setting"]["params"]["properties"]
    assert "path" in set_params
    assert by_name["set_setting"]["description"]


def test_commands_execute_and_validate() -> None:
    client = _client()
    assert client.post("/commands/ping").json()["result"] == {"pong": True}
    backends = client.post("/commands/list_grid_backends").json()["result"]
    assert {"h3", "s2", "isea"} <= set(backends)
    assert client.post("/commands/warp").status_code == _HTTP_NOT_FOUND
    bad = client.post("/commands/set_setting", json={"value": 1})
    assert bad.status_code == _HTTP_UNPROCESSABLE


def test_settings_readable_and_writable_over_the_api() -> None:
    client = _client()
    assert client.get("/settings").json()["grid"]["backend"] == "h3"
    assert "grid.backend" in client.get("/settings/paths").json()
    assert client.get("/settings/grid.resolution").json()["value"] == _DEFAULT_RESOLUTION
    updated = client.put("/settings/grid.resolution", json={"value": _NEW_RESOLUTION})
    assert updated.json()["value"] == _NEW_RESOLUTION
    assert client.get("/settings/grid.resolution").json()["value"] == _NEW_RESOLUTION
    assert client.get("/settings/grid.nope").status_code == _HTTP_NOT_FOUND
    rejected = client.put("/settings/grid.backend", json={"value": "square"})
    assert rejected.status_code == _HTTP_UNPROCESSABLE


def test_ws_schema_documents_every_event_type() -> None:
    schemas = _client().get("/ws/schema").json()
    for expected in ("hello", "setting_changed", "heartbeat"):
        assert expected in schemas
        assert "properties" in schemas[expected]


def test_ws_streams_setting_changes() -> None:
    client = _client()
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        client.put("/settings/fidelity.level", json={"value": "fast"})
        event = ws.receive_json()
        assert event["type"] == "setting_changed"
        assert event["path"] == "fidelity.level"
        assert event["value"] == "fast"


def test_command_and_rest_share_one_write_path() -> None:
    client = _client()
    client.post(
        "/commands/set_setting",
        json={"path": "compute.backend", "value": "numpy"},
    )
    assert client.get("/settings/compute.backend").json()["value"] == "numpy"
