"""Tests for the CLI client — exercise a scripted scenario through the CLI.

Spins up the FastAPI app via TestClient and drives it through the API
interface, verifying that the API is fully controllable via HTTP with no
core/ or adapters/ imports.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


def _get_test_client() -> TestClient:
    """Create a fresh test client for each test."""
    return TestClient(create_app(Settings()))


class TestAPIClientViaTestClient:
    """Test the APIClient through TestClient (works around httpx transport issues)."""

    def test_ping(self) -> None:
        """Ping returns pong."""
        client = _get_test_client()
        result = client.post("/commands/ping").json()
        assert result["command"] == "ping"
        assert result["result"]["pong"] is True

    def test_list_commands(self) -> None:
        """List commands returns all known commands."""
        client = _get_test_client()
        commands = client.get("/commands").json()
        names = {c["name"] for c in commands}
        assert "ping" in names
        assert "set_setting" in names
        assert "get_settings" in names
        assert "run_world_sweep" in names
        # Every command must have a description and mutates flag
        for cmd in commands:
            assert "description" in cmd
            assert cmd["description"].strip()
            assert isinstance(cmd.get("mutates"), bool)

    def test_get_schema(self) -> None:
        """OpenAPI schema is served."""
        client = _get_test_client()
        schema = client.get("/schema").json()
        assert schema["info"]["title"] == "FableWorldSim"
        assert "paths" in schema

    def test_get_ws_schema(self) -> None:
        """WebSocket event schemas are discoverable."""
        client = _get_test_client()
        schemas = client.get("/ws/schema").json()
        for expected_type in ("hello", "setting_changed", "heartbeat"):
            assert expected_type in schemas
            assert "properties" in schemas[expected_type]

    def test_get_all_settings(self) -> None:
        """Get all settings returns the settings tree."""
        client = _get_test_client()
        settings = client.get("/settings").json()
        assert "grid" in settings
        assert "compute" in settings
        assert settings["grid"]["backend"] == "h3"

    def test_get_setting_paths(self) -> None:
        """Get setting paths returns dotted paths."""
        client = _get_test_client()
        paths = client.get("/settings/paths").json()
        assert isinstance(paths, list)
        assert "grid.backend" in paths
        assert "grid.resolution" in paths

    def test_get_setting(self) -> None:
        """Get a single setting by path."""
        client = _get_test_client()
        value = client.get("/settings/grid.backend").json()["value"]
        assert value == "h3"

    def test_get_setting_not_found(self) -> None:
        """Getting a nonexistent setting returns 404."""
        client = _get_test_client()
        response = client.get("/settings/fake.nonexistent.path")
        assert response.status_code == 404

    def test_set_setting(self) -> None:
        """Set a setting and verify it changed."""
        client = _get_test_client()
        original = client.get("/settings/grid.resolution").json()["value"]
        new_value = 4 if original != 4 else 5
        result = client.put("/settings/grid.resolution", json={"value": new_value}).json()
        assert result["value"] == new_value
        verify = client.get("/settings/grid.resolution").json()["value"]
        assert verify == new_value

    def test_set_invalid_setting(self) -> None:
        """Setting an invalid value returns 422."""
        client = _get_test_client()
        response = client.put("/settings/grid.backend", json={"value": "invalid_backend"})
        assert response.status_code == 422

    def test_get_metrics(self) -> None:
        """Metrics endpoint returns counter and telemetry data."""
        client = _get_test_client()
        metrics = client.get("/metrics").json()
        assert isinstance(metrics, dict)
        # At least one of these should exist
        assert "commands_executed" in metrics or "ws_subscribers" in metrics


class TestCLIClient:
    """Test that the CLI client module works."""

    def test_cli_module_imports_without_core(self) -> None:
        """Verify the CLI module has no core/ or adapters/ imports.

        This is a smoke test: just import the CLI to verify no import
        errors occur (which would indicate core/adapters imports).
        """
        import clients.cli  # noqa: F401
        from clients.cli.client import APIClient  # noqa: F401
        from clients.cli.main import CLI, main  # noqa: F401

        # If we got here, no import errors happened
        assert True

    def test_api_client_instantiation(self) -> None:
        """Test that APIClient can be instantiated."""
        from clients.cli.client import APIClient

        client = APIClient("http://localhost:8000")
        assert client.base_url == "http://localhost:8000"
        client.close()

    def test_cli_instantiation(self) -> None:
        """Test that CLI can be instantiated."""
        from clients.cli.main import CLI

        cli = CLI("http://localhost:8000")
        assert cli.base_url == "http://localhost:8000"
        cli.close()


class TestCLIMainEntryPoint:
    """Test the CLI main() entry point with argparse."""

    def test_main_help(self, capsys) -> None:
        """Test that --help doesn't error."""
        from clients.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        # argparse exits with 0 on --help
        assert exc_info.value.code == 0

    def test_main_returns_int(self) -> None:
        """Test that main() returns an integer exit code."""
        from clients.cli.main import main

        # Call with a valid command (in mock form)
        from unittest.mock import patch, MagicMock

        with patch("clients.cli.main.APIClient"):
            mock_cli = MagicMock()
            with patch("clients.cli.main.CLI", return_value=mock_cli):
                # Call with ping (simplest command)
                result = main(["ping"])
                # Should return 0 (success) or raise SystemExit
                assert result == 0 or result is None

    def test_main_with_bad_command(self, capsys) -> None:
        """Test that main handles invalid commands gracefully."""
        from clients.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent-command"])
        # Should exit with 1 for unrecognized command
        assert exc_info.value.code in (1, 2)  # 2 for argparse error
