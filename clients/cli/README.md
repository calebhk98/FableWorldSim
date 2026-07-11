# CLI Client for FableWorldSim

Scriptable command-line client over the REST API — proves the sim is fully controllable
with no GUI (and thus AI-controllable). This is a **pure HTTP API client** with no direct
imports from `core/` or `adapters/`; all interaction goes via the FastAPI server.

## Requirements

- Python 3.11+
- `httpx` (installed as part of the dev extras)
- A running FableWorldSim FastAPI server (e.g., `uvicorn api.app:create_app --reload`)

## Installation

The CLI is part of the FableWorldSim package:

```bash
pip install -e ".[dev]"
```

## Usage

All commands communicate with a running FastAPI server at `http://localhost:8000` by default.

### Liveness

```bash
# Ping the server
python -m clients.cli.main ping
```

### Configuration

```bash
# Get all settings
python -m clients.cli.main config get all

# Get a specific setting
python -m clients.cli.main config get grid.backend
python -m clients.cli.main config get grid.resolution

# List all available setting paths
python -m clients.cli.main config get paths

# Set a setting
python -m clients.cli.main config set grid.resolution 4
python -m clients.cli.main config set fidelity.level balanced
```

### World Generation & Simulation

```bash
# Run a world sweep: generate worlds, rank by habitability, deep-sim the winners
python -m clients.cli.main world-sweep \
  --base-seed 1 \
  --count 6 \
  --keep-top-k 2 \
  --resolution 1 \
  --deep-ticks 5
```

### Commands Discovery

```bash
# List all available commands
python -m clients.cli.main commands list

# Describe a specific command's parameters
python -m clients.cli.main commands describe set_setting
python -m clients.cli.main commands describe run_world_sweep
```

### Hardware & Diagnostics

```bash
# Probe host hardware and get compute recommendations
python -m clients.cli.main hardware-probe

# Fetch server metrics (command counts, telemetry)
python -m clients.cli.main metrics
```

### JSON Output

Add `--json` to any command for raw JSON output instead of formatted text:

```bash
python -m clients.cli.main --json config get grid.backend
python -m clients.cli.main --json commands list
python -m clients.cli.main --json world-sweep --count 3
```

### Custom Server URL

Use `--url` to point at a different server:

```bash
python -m clients.cli.main --url http://example.com:8000 ping
```

## Architecture

### `client.py`

Low-level `APIClient` class wrapping `httpx`:

- `execute_command(name, params)` — call any registered command
- `get_setting(path)`, `set_setting(path, value)` — config
- `get_schema()`, `list_commands()`, `get_ws_schema()` — discovery
- `get_metrics()` — diagnostics
- `subscribe_to_events()` — optional WebSocket streaming

All methods raise `httpx.HTTPStatusError` on non-2xx responses.

### `main.py`

High-level `CLI` class and argparse entry point:

- Parses subcommands: `ping`, `config`, `world-sweep`, `commands`, `hardware-probe`, `metrics`
- Formats output (pretty JSON or human-readable text)
- Handles errors gracefully

No direct simulation imports; the CLI is entirely API-driven.

## Testing

Run the test suite:

```bash
cd /home/user/wt/10_cli
python -m pytest tests/test_cli_client.py -v
```

Tests use `fastapi.testclient.TestClient` to spin up a test server and drive
scripted scenarios through the CLI, verifying end-to-end functionality.

## Example: Scripted World Generation

```python
#!/usr/bin/env python3
"""Generate and analyze worlds via the CLI client."""

from clients.cli.client import APIClient

with APIClient("http://localhost:8000") as client:
    # Run a tiny sweep
    result = client.execute_command(
        "run_world_sweep",
        {
            "base_seed": 42,
            "count": 3,
            "keep_top_k": 1,
            "resolution": 1,
            "deep_ticks": 1,
        }
    )
    
    # Inspect results
    sweep = result["result"]
    print(f"Generated {len(sweep['ranked'])} worlds")
    for entry in sweep["ranked"]:
        print(f"  Seed {entry['seed']}: habitability={entry['score']:.4f}")
    
    # Deep results for the winner
    if sweep["selected"]:
        winner_seed = sweep["selected"][0]
        deep_result = sweep["deepened"][str(winner_seed)]
        print(f"\nWinner (seed {winner_seed}):")
        print(f"  Rivers: {deep_result['channel_count']}")
        print(f"  Populations: {deep_result['populations']}")
```

## Implementation Notes

1. **No core imports**: The CLI is a thin HTTP wrapper. It never imports `core/`, `adapters/`,
   or `ports/`. This keeps it lightweight and AI-controllable (can be driven by another agent).

2. **API-first design**: The sim's entire state and behavior is expressed through the REST API.
   If you can do it via the API, you can do it via the CLI.

3. **Self-describing API**: Commands and settings are discovered at runtime via `/commands` and
   `/settings/paths`. The CLI can adapt to API changes without code updates.

4. **Error handling**: HTTP errors (404, 422, etc.) map cleanly to user-facing messages. The
   CLI never returns a raw server traceback.
