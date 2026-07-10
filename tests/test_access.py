"""Tests for the two-tier access split and the single-writer path."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from adapters.access_roles import RoleBasedAccess
from api.app import create_app
from api.settings import Settings
from ports.access import Principal

_HTTP_OK = 200
_HTTP_FORBIDDEN = 403
_VIEWER = {"x-fws-principal": "spectator", "x-fws-roles": "viewer"}


def _client() -> TestClient:
    return TestClient(create_app(Settings()))


def test_policy_splits_read_and_write_tiers() -> None:
    access = RoleBasedAccess()
    viewer = Principal("spectator", ("viewer",))
    editor = Principal("player", ("editor",))
    assert access.can_read(viewer, "command.get_settings")
    assert not access.can_write(viewer, "command.set_setting")
    assert access.can_write(editor, "command.set_setting")


def test_commands_declare_their_tier() -> None:
    tiers = {c["name"]: c["mutates"] for c in _client().get("/commands").json()}
    assert tiers["set_setting"] is True
    assert tiers["get_settings"] is False
    assert tiers["probe_hardware"] is False


def test_viewers_read_but_cannot_write() -> None:
    client = _client()
    assert client.get("/settings", headers=_VIEWER).status_code == _HTTP_OK
    assert client.post("/commands/ping", headers=_VIEWER).status_code == _HTTP_OK
    denied_cmd = client.post(
        "/commands/set_setting",
        json={"path": "fidelity.level", "value": "fast"},
        headers=_VIEWER,
    )
    assert denied_cmd.status_code == _HTTP_FORBIDDEN
    denied_put = client.put("/settings/fidelity.level", json={"value": "fast"}, headers=_VIEWER)
    assert denied_put.status_code == _HTTP_FORBIDDEN
    assert client.get("/settings/fidelity.level").json()["value"] == "balanced"


def test_local_default_holds_the_write_tier() -> None:
    client = _client()
    ok = client.post("/commands/set_setting", json={"path": "fidelity.level", "value": "fast"})
    assert ok.status_code == _HTTP_OK


def test_metrics_count_executions_and_denials() -> None:
    client = _client()
    client.post("/commands/ping")
    client.post(
        "/commands/set_setting",
        json={"path": "fidelity.level", "value": "fast"},
        headers=_VIEWER,
    )
    metrics = client.get("/metrics").json()
    assert metrics["commands_executed"] >= 1
    assert metrics["commands_denied"] >= 1
    assert "ws_subscribers" in metrics


def test_probe_hardware_command_reports_the_host() -> None:
    result = _client().post("/commands/probe_hardware").json()["result"]
    assert result["capabilities"]["cpu_cores"] >= 1
    assert result["recommendation"]["fidelity_level"] in {"fast", "balanced", "accurate"}
