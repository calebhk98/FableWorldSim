"""FableWorldSim command-line client — pure HTTP API over the FastAPI server."""

from clients.cli.client import APIClient
from clients.cli.main import CLI, main

__all__ = ["CLI", "APIClient", "main"]
