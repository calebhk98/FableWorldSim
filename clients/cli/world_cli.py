"""``world`` subcommand: create/get/step the live world over HTTP.

Also reads its geometry, its per-cell fields, and its reproducibility
recipe. Split out of :mod:`clients.cli.main` to keep both files under the
line-length cap. ``add_world_subparser`` wires the argparse subcommand
tree; ``dispatch_world`` runs the requested one against a :class:`CLI`
instance (see ``clients.cli.main.CLI``).
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from clients.cli.output import fatal, print_json

if TYPE_CHECKING:
    from clients.cli.main import CLI


def add_world_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``world`` subcommand (create/get/step/query-field/geometry/export)."""
    world = subparsers.add_parser("world", help="Create, inspect, and step the live world")
    world_subs = world.add_subparsers(dest="world_cmd")

    create = world_subs.add_parser("create", help="Build a world and make it the live world")
    create.add_argument("--seed", type=int, required=True, help="Random seed for terrain")
    create.add_argument("--preset", default="earth", help="Scenario preset (default: earth)")
    create.add_argument(
        "--resolution", type=int, default=0, help="Grid resolution (0 = coarsest/fastest)"
    )
    create.add_argument("--season-count", type=int, default=2, help="Number of seasons")
    create.add_argument("--grid-backend", default="h3", help="Grid backend: h3, s2, or isea")

    world_subs.add_parser("get", help="Show a summary of the live world")

    step = world_subs.add_parser("step", help="Advance the live world by N ticks")
    step.add_argument("--ticks", type=int, default=1, help="Orchestrator ticks to advance by")

    query = world_subs.add_parser("query-field", help="Fetch cell_id -> value for one field")
    query.add_argument("name", help="height, elevation, temperature, or precipitation")

    world_subs.add_parser("geometry", help="Fetch cell_id -> {lat, lng} for every cell")
    world_subs.add_parser("export", help="Fetch the world's reproducibility recipe")


def _world_create(cli: CLI, args: argparse.Namespace) -> None:
    """Handle ``world create``."""
    result = cli.client.create_world(
        args.seed,
        preset_name=args.preset,
        resolution=args.resolution,
        season_count=args.season_count,
        grid_backend=args.grid_backend,
    )
    if args.json:
        print_json(result)
    else:
        summary = result["result"]
        print(
            f"Created world: seed={summary['seed']} planet={summary['planet_name']} "
            f"cells={summary['cell_count']} tick={summary['tick']}"
        )


def _world_get(cli: CLI, args: argparse.Namespace) -> None:
    """Handle ``world get``."""
    result = cli.client.get_world()
    if args.json:
        print_json(result)
    else:
        summary = result["result"]
        if not summary.get("exists"):
            print("No world created yet.")
        else:
            print(
                f"World: seed={summary['seed']} planet={summary['planet_name']} "
                f"tick={summary['tick']} cells={summary['cell_count']} "
                f"backend={summary['grid_backend']}"
            )


def _world_step(cli: CLI, args: argparse.Namespace) -> None:
    """Handle ``world step``."""
    result = cli.client.step_world(args.ticks)
    if args.json:
        print_json(result)
    else:
        stats = result["result"]
        print(
            f"Stepped to tick {stats['tick']} (+{stats['ticks_advanced']}): "
            f"{stats['surviving_species']} species surviving"
        )


def _world_query_field(cli: CLI, args: argparse.Namespace) -> None:
    """Handle ``world query-field``."""
    result = cli.client.query_field(args.name)
    if args.json:
        print_json(result)
    else:
        values = result["result"]
        print(f"{args.name}: {len(values)} cells")
        for cell_id, value in list(values.items())[:5]:
            print(f"  {cell_id}: {value}")


def _world_geometry(cli: CLI, args: argparse.Namespace) -> None:
    """Handle ``world geometry``."""
    result = cli.client.grid_geometry()
    if args.json:
        print_json(result)
    else:
        cells = result["result"]
        print(f"geometry: {len(cells)} cells")
        for cell_id, point in list(cells.items())[:5]:
            print(f"  {cell_id}: lat={point['lat']:.3f} lng={point['lng']:.3f}")


def _world_export(cli: CLI, args: argparse.Namespace) -> None:
    """Handle ``world export``."""
    result = cli.client.export_recipe()
    print_json(result if args.json else result["result"])


_HANDLERS = {
    "create": _world_create,
    "get": _world_get,
    "step": _world_step,
    "query-field": _world_query_field,
    "geometry": _world_geometry,
    "export": _world_export,
}


def dispatch_world(cli: CLI, args: argparse.Namespace) -> int:
    """Run the requested ``world`` subcommand; return the process exit code."""
    handler = _HANDLERS.get(args.world_cmd)
    if handler is None:
        return 1
    try:
        handler(cli, args)
    except Exception as exc:  # CLI boundary: report cleanly, don't traceback
        fatal(f"world {args.world_cmd} failed: {exc}")
    return 0
