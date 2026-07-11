"""Pure HTTP/WS API client for FableWorldSim.

All communication goes over HTTP REST + optional WebSocket. No direct
imports from core/ or adapters/. This module provides low-level API
access; see main.py for the CLI wrapper.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class APIClient:
    """Synchronous HTTP client for the FableWorldSim API."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize the client pointing to a running server.

        Args:
            base_url: URL of the FastAPI server (default: localhost:8000).
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url)

    def close(self) -> None:
        """Close the HTTP client connection."""
        self.client.close()

    def __enter__(self) -> APIClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    # -----------------------------------------------------------------------
    # Discovery: schema and commands
    # -----------------------------------------------------------------------

    def get_schema(self) -> dict[str, Any]:
        """Fetch the OpenAPI schema (alias of /openapi.json)."""
        response = self.client.get("/schema")
        response.raise_for_status()
        return response.json()

    def list_commands(self) -> list[dict[str, Any]]:
        """Fetch every command with its parameter JSON Schema."""
        response = self.client.get("/commands")
        response.raise_for_status()
        return response.json()

    def get_ws_schema(self) -> dict[str, Any]:
        """Fetch WebSocket event schemas (one per event type)."""
        response = self.client.get("/ws/schema")
        response.raise_for_status()
        return response.json()

    # -----------------------------------------------------------------------
    # Commands: execution (all read and write via same endpoint)
    # -----------------------------------------------------------------------

    def execute_command(self, name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a command by name with optional params.

        Args:
            name: Command name (e.g. 'ping', 'set_setting').
            params: Optional dict of command parameters.

        Returns:
            Response dict with 'command' and 'result' keys.

        Raises:
            httpx.HTTPStatusError: If the command fails (404, 422, etc).
        """
        response = self.client.post(f"/commands/{name}", json=params or {})
        response.raise_for_status()
        return response.json()

    # -----------------------------------------------------------------------
    # Settings: get and set individual options
    # -----------------------------------------------------------------------

    def get_all_settings(self) -> dict[str, Any]:
        """Fetch the whole settings tree."""
        response = self.client.get("/settings")
        response.raise_for_status()
        return response.json()

    def get_setting_paths(self) -> list[str]:
        """Fetch every dotted option path."""
        response = self.client.get("/settings/paths")
        response.raise_for_status()
        return response.json()

    def get_setting(self, path: str) -> Any:
        """Fetch one option's value by dotted path.

        Args:
            path: Dotted path (e.g., 'grid.backend').

        Returns:
            The setting value.

        Raises:
            httpx.HTTPStatusError: If path not found (404) or invalid (422).
        """
        response = self.client.get(f"/settings/{path}")
        response.raise_for_status()
        return response.json()["value"]

    def set_setting(self, path: str, value: Any) -> Any:
        """Change one option and validate it.

        Args:
            path: Dotted path (e.g., 'grid.resolution').
            value: New value (validated by schema).

        Returns:
            The validated new value.

        Raises:
            httpx.HTTPStatusError: If path not found (404) or value invalid (422).
        """
        response = self.client.put(f"/settings/{path}", json={"value": value})
        response.raise_for_status()
        return response.json()["value"]

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Fetch server counters and telemetry."""
        response = self.client.get("/metrics")
        response.raise_for_status()
        return response.json()

    # -----------------------------------------------------------------------
    # WebSocket (optional, for live streaming)
    # -----------------------------------------------------------------------

    def subscribe_to_events(self) -> httpx.WebSocketClientProtocol:
        """Connect to the WebSocket event stream.

        Returns:
            A WebSocket context manager. Use with:
                with client.subscribe_to_events() as ws:
                    event = ws.receive_json()

        Example:
            with client.subscribe_to_events() as ws:
                hello = ws.receive_json()  # HelloEvent
                print(hello["type"])
        """
        return self.client.stream("GET", "/ws")
