"""Scriptable command-line interface for FableWorldSim.

This is a PURE HTTP API client — no direct imports from core/ or adapters/.
All interaction with the sim goes over the REST API (plus optional WebSocket).

Typical usage:
    fws-cli ping
    fws-cli world create --seed 42
    fws-cli config get grid.backend
    fws-cli config set grid.resolution 4
    fws-cli world step --ticks 10
    fws-cli commands list
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from clients.cli.client import APIClient
from clients.cli.output import fatal, print_json
from clients.cli.world_cli import add_world_subparser, dispatch_world


class CLI:
    """Main CLI router and subcommand handlers."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize with a server URL."""
        self.base_url = base_url
        self.client = APIClient(base_url)

    def close(self) -> None:
        """Close the client."""
        self.client.close()

    # -----------------------------------------------------------------------
    # Liveness
    # -----------------------------------------------------------------------

    def ping(self, args: argparse.Namespace) -> None:
        """Ping the server for liveness."""
        try:
            result = self.client.execute_command("ping")
            if args.json:
                print_json(result)
            else:
                print("PONG" if result["result"]["pong"] else "No response")
        except Exception as e:
            fatal(f"ping failed: {e}")

    # -----------------------------------------------------------------------
    # World operations (via commands)
    # -----------------------------------------------------------------------

    def world_sweep(self, args: argparse.Namespace) -> None:
        """Generate and rank worlds via world sweep."""
        try:
            params = {
                "base_seed": args.base_seed,
                "count": args.count,
                "keep_top_k": args.keep_top_k,
                "resolution": args.resolution,
                "deep_ticks": args.deep_ticks,
            }
            result = self.client.execute_command("run_world_sweep", params)
            if args.json:
                print_json(result)
            else:
                sweep_result = result["result"]
                ranked = sweep_result["ranked"]
                selected = sweep_result["selected"]
                print(
                    f"Sweep complete: {len(ranked)} worlds ranked, "
                    f"{len(selected)} selected for deep-sim"
                )
                for entry in ranked[:5]:
                    print(f"  Seed {entry['seed']}: score={entry['score']:.4f}")
                print("\nDeepened seeds:", selected)
                if selected and str(selected[0]) in sweep_result["deepened"]:
                    deep = sweep_result["deepened"][str(selected[0])]
                    print(
                        f"  Top winner: {deep.get('channel_count', 'N/A')} rivers, "
                        f"{deep.get('surviving_species', 0)} species survived"
                    )
        except Exception as e:
            fatal(f"world sweep failed: {e}")

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    def config_get(self, args: argparse.Namespace) -> None:
        """Get a setting by dotted path or list all."""
        try:
            if args.path == "all":
                settings = self.client.get_all_settings()
                if args.json:
                    print_json(settings)
                else:
                    self._print_settings_tree(settings)
            elif args.path == "paths":
                paths = self.client.get_setting_paths()
                if args.json:
                    print_json(paths)
                else:
                    for path in paths:
                        print(path)
            else:
                value = self.client.get_setting(args.path)
                if args.json:
                    print_json({"path": args.path, "value": value})
                else:
                    print(f"{args.path} = {value}")
        except Exception as e:
            fatal(f"config get failed: {e}")

    def config_set(self, args: argparse.Namespace) -> None:
        """Set a setting by dotted path."""
        try:
            # Try to parse value as JSON first, then treat as string
            try:
                value = json.loads(args.value)
            except (json.JSONDecodeError, ValueError):
                value = args.value

            new_value = self.client.set_setting(args.path, value)
            if args.json:
                print_json({"path": args.path, "value": new_value})
            else:
                print(f"{args.path} = {new_value}")
        except Exception as e:
            fatal(f"config set failed: {e}")

    @staticmethod
    def _print_settings_tree(obj: dict[str, Any], prefix: str = "", indent: str = "  ") -> None:
        """Recursively print a settings tree."""
        for key, value in sorted(obj.items()):
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                print(f"{path}/")
                CLI._print_settings_tree(value, path, indent)
            else:
                print(f"{path} = {value}")

    # -----------------------------------------------------------------------
    # Commands discovery
    # -----------------------------------------------------------------------

    def commands_list(self, args: argparse.Namespace) -> None:
        """List all available commands."""
        try:
            commands = self.client.list_commands()
            if args.json:
                print_json(commands)
            else:
                for cmd in sorted(commands, key=lambda c: c["name"]):
                    mutates = "(write)" if cmd.get("mutates") else "(read)"
                    print(f"{cmd['name']:<30} {mutates:<8} {cmd['description']}")
        except Exception as e:
            fatal(f"commands list failed: {e}")

    def commands_describe(self, args: argparse.Namespace) -> None:
        """Describe one command's schema."""
        try:
            commands = self.client.list_commands()
            cmd = next((c for c in commands if c["name"] == args.name), None)
            if cmd is None:
                fatal(f"unknown command: {args.name}")
            if args.json:
                print_json(cmd)
            else:
                print(f"Command: {cmd['name']}")
                print(f"Description: {cmd['description']}")
                print(f"Mutates: {cmd.get('mutates', False)}")
                print("Parameters:")
                params_schema = cmd.get("params", {})
                props = params_schema.get("properties", {})
                if props:
                    for param_name, param_schema in props.items():
                        desc = param_schema.get("description", "")
                        default = param_schema.get("default", "")
                        type_info = param_schema.get("type", "")
                        print(f"  {param_name} ({type_info}): {desc}")
                        if default:
                            print(f"    default: {default}")
                else:
                    print("  (none)")
        except Exception as e:
            fatal(f"commands describe failed: {e}")

    # -----------------------------------------------------------------------
    # Hardware and diagnostics
    # -----------------------------------------------------------------------

    def hardware_probe(self, args: argparse.Namespace) -> None:
        """Probe host hardware and get compute recommendations."""
        try:
            result = self.client.execute_command("probe_hardware")
            if args.json:
                print_json(result)
            else:
                hw = result["result"]
                caps = hw.get("capabilities", {})
                rec = hw.get("recommendation", {})
                print("Host capabilities:")
                for key, val in caps.items():
                    print(f"  {key}: {val}")
                print("\nRecommendation:")
                for key, val in rec.items():
                    print(f"  {key}: {val}")
        except Exception as e:
            fatal(f"hardware probe failed: {e}")

    def metrics(self, args: argparse.Namespace) -> None:
        """Fetch server metrics and telemetry."""
        try:
            data = self.client.get_metrics()
            if args.json:
                print_json(data)
            else:
                for key, val in sorted(data.items()):
                    if isinstance(val, dict):
                        print(f"{key}:")
                        for k, v in val.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"{key}: {val}")
        except Exception as e:
            fatal(f"metrics fetch failed: {e}")


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915 - argparse dispatcher
    """Main entry point: parse args and dispatch."""
    parser = argparse.ArgumentParser(
        description="Scriptable CLI for FableWorldSim (HTTP API client, no GUI)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="FastAPI server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Liveness
    subparsers.add_parser("ping", help="Ping the server for liveness")

    # World operations
    sweep = subparsers.add_parser("world-sweep", help="Generate and rank worlds")
    sweep.add_argument("--base-seed", type=int, default=1, help="First seed to sweep from")
    sweep.add_argument("--count", type=int, default=6, help="How many worlds to generate")
    sweep.add_argument("--keep-top-k", type=int, default=2, help="How many top worlds to deep-sim")
    sweep.add_argument(
        "--resolution", type=int, default=1, help="Grid resolution (coarser = faster)"
    )
    sweep.add_argument("--deep-ticks", type=int, default=5, help="Biology ticks for deep-sim")

    # The live, steppable world: create/get/step/query-field/geometry/export
    add_world_subparser(subparsers)

    # Configuration
    config = subparsers.add_parser("config", help="Get/set configuration")
    config_subs = config.add_subparsers(dest="config_cmd")
    get_cmd = config_subs.add_parser("get", help="Get a setting or list all")
    get_cmd.add_argument(
        "path",
        nargs="?",
        default="all",
        help="Dotted path (e.g. grid.backend), 'all', or 'paths'",
    )
    set_cmd = config_subs.add_parser("set", help="Set a setting")
    set_cmd.add_argument("path", help="Dotted path (e.g. grid.resolution)")
    set_cmd.add_argument("value", help="New value (parsed as JSON if possible)")

    # Commands discovery
    commands = subparsers.add_parser("commands", help="List and describe commands")
    commands_subs = commands.add_subparsers(dest="commands_cmd")
    commands_subs.add_parser("list", help="List all available commands")
    describe_cmd = commands_subs.add_parser("describe", help="Describe one command")
    describe_cmd.add_argument("name", help="Command name")

    # Hardware and diagnostics
    subparsers.add_parser("hardware-probe", help="Probe host hardware and get recommendations")
    subparsers.add_parser("metrics", help="Fetch server metrics and telemetry")

    args = parser.parse_args(argv)

    cli = CLI(args.url)
    try:
        if args.command == "ping":
            cli.ping(args)
        elif args.command == "world-sweep":
            cli.world_sweep(args)
        elif args.command == "world":
            if getattr(args, "world_cmd", None) is None:
                parser.print_help()
                return 1
            return dispatch_world(cli, args)
        elif args.command == "config":
            if args.config_cmd == "get":
                cli.config_get(args)
            elif args.config_cmd == "set":
                cli.config_set(args)
            else:
                parser.print_help()
                return 1
        elif args.command == "commands":
            if args.commands_cmd == "list":
                cli.commands_list(args)
            elif args.commands_cmd == "describe":
                cli.commands_describe(args)
            else:
                parser.print_help()
                return 1
        elif args.command == "hardware-probe":
            cli.hardware_probe(args)
        elif args.command == "metrics":
            cli.metrics(args)
        else:
            parser.print_help()
            return 1
        return 0
    finally:
        cli.close()


if __name__ == "__main__":
    sys.exit(main())
