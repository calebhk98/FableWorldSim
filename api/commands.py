"""Self-describing command registry — the /commands surface.

Every command declares a name, a human description, and a pydantic
params model; ``/commands`` serves the full list with JSON Schemas so
the web UI, a CLI script, or any other client can discover what is
possible and call it directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from adapters.grid_registry import available_backends
from adapters.hardware import probe_host, recommend
from api.settings import get_setting, setting_paths, with_setting
from api.state import AppState
from api.world_service import report_to_dict, run_world_sweep
from api.ws_events import SettingChangedEvent
from core.sim.world_sweep import SweepReport

CommandHandler = Callable[[AppState, BaseModel], object]
"""Executes a command against the app state; the return must be JSON-able."""


class CommandNotFoundError(LookupError):
    """Raised when no command is registered under a name."""


@dataclass(frozen=True)
class Command:
    """One invocable command with its self-description.

    ``mutates`` is the access tier: False = read tier (any connection),
    True = write tier (checked against the Access policy and serialized
    through the single-writer lock).
    """

    name: str
    description: str
    params: type[BaseModel]
    handler: CommandHandler
    mutates: bool = False


class CommandRegistry:
    """Name -> command lookup with a describe() surface."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        """Register (or replace) a command by name."""
        self._commands[command.name] = command

    def get(self, name: str) -> Command:
        """Return one command; raises CommandNotFoundError."""
        command = self._commands.get(name)
        if command is None:
            known = ", ".join(sorted(self._commands))
            msg = f"unknown command {name!r}; known: {known}"
            raise CommandNotFoundError(msg)
        return command

    def describe(self) -> list[dict[str, Any]]:
        """Return every command's name, description, and params schema."""
        return [
            {
                "name": command.name,
                "description": command.description,
                "mutates": command.mutates,
                "params": command.params.model_json_schema(),
            }
            for command in sorted(self._commands.values(), key=lambda c: c.name)
        ]


class NoParams(BaseModel):
    """Empty params model for commands that take no arguments."""


class SetSettingParams(BaseModel):
    """Arguments for set_setting."""

    path: str = Field(description="Dotted option path, e.g. 'grid.backend'.")
    value: Any = Field(default=None, description="New value; validated by the schema.")


class SetComputeBackendParams(BaseModel):
    """Arguments for set_compute_backend."""

    backend: Literal["auto", "numpy", "cupy", "jax"] = Field(
        description="Compute backend to swap to live: auto | numpy | cupy | jax."
    )


class SweepParams(BaseModel):
    """Arguments for run_world_sweep."""

    base_seed: int = Field(
        default=1, description="First seed; the sweep uses count consecutive seeds."
    )
    count: int = Field(default=6, ge=1, le=64, description="How many worlds to generate and score.")
    keep_top_k: int = Field(default=2, ge=1, le=64, description="How many top worlds to deep-sim.")
    resolution: int = Field(
        default=1, ge=0, le=4, description="Grid resolution (coarser = faster)."
    )
    deep_ticks: int = Field(
        default=5, ge=0, le=200, description="Biology ticks for each deep-sim run."
    )


def apply_setting(state: AppState, path: str, value: object) -> object:
    """Set one option, publish the change event, return the new value.

    The single write path shared by the set_setting command and the REST
    settings endpoint, so every surface behaves identically — and every
    write serializes through the single-writer lock.

    ``compute.backend`` is special-cased through the atomic hot-swap so
    the live backend handle and the stored setting can never disagree
    (an unavailable backend rejects the whole change).
    """
    if path == "compute.backend":
        backend = state.set_compute_backend(str(value))
        with state.write_lock:
            state.metrics["settings_changed"] += 1
        state.bus.publish(SettingChangedEvent(path=path, value=backend.name))
        return backend.name
    with state.write_lock:
        state.settings = with_setting(state.settings, path, value)
        new_value = get_setting(state.settings, path)
        state.metrics["settings_changed"] += 1
    state.bus.publish(SettingChangedEvent(path=path, value=new_value))
    return new_value


def _ping(state: AppState, params: BaseModel) -> object:
    """Liveness check."""
    return {"pong": True}


def _get_settings(state: AppState, params: BaseModel) -> object:
    """Return every option's current value."""
    return state.settings.model_dump()


def _list_setting_paths(state: AppState, params: BaseModel) -> object:
    """Return every dotted option path."""
    return list(setting_paths(state.settings))


def _set_setting(state: AppState, params: BaseModel) -> object:
    """Change one option (validated) and broadcast the change."""
    if not isinstance(params, SetSettingParams):
        msg = "set_setting invoked with the wrong params model"
        raise TypeError(msg)
    return {"path": params.path, "value": apply_setting(state, params.path, params.value)}


def _list_grid_backends(state: AppState, params: BaseModel) -> object:
    """Return the known grid backend toggle names."""
    return list(available_backends())


def _describe_backend(state: AppState) -> dict[str, object]:
    """Return the live compute backend's name, device, and device count."""
    backend = state.compute_backend
    return {
        "backend": backend.name,
        "device": backend.device,
        "num_devices": backend.num_devices,
    }


def _get_compute_backend(state: AppState, params: BaseModel) -> object:
    """Return the compute backend the sim is currently running on."""
    return _describe_backend(state)


def _set_compute_backend(state: AppState, params: BaseModel) -> object:
    """Hot-swap the compute backend (CPU<->GPU) while the sim runs."""
    if not isinstance(params, SetComputeBackendParams):
        msg = "set_compute_backend invoked with the wrong params model"
        raise TypeError(msg)
    state.set_compute_backend(params.backend)
    return _describe_backend(state)


def _get_run_telemetry(state: AppState, params: BaseModel) -> object:
    """Return the most recent simulation run's speed telemetry (or empty)."""
    return state.sim_telemetry or {}


def _surface_top_telemetry(state: AppState, report: SweepReport) -> None:
    """Publish the best deepened world's telemetry on the /metrics surface."""
    if not report.selected:
        return
    top = report.deepened.get(report.selected[0].seed)
    if isinstance(top, dict) and isinstance(top.get("telemetry"), dict):
        state.sim_telemetry = top["telemetry"]


def _run_world_sweep(state: AppState, params: BaseModel) -> object:
    """Generate many worlds, keep the best, and deep-sim the winners."""
    if not isinstance(params, SweepParams):
        msg = "run_world_sweep invoked with the wrong params model"
        raise TypeError(msg)
    report = run_world_sweep(
        base_seed=params.base_seed,
        count=params.count,
        keep_top_k=params.keep_top_k,
        resolution=params.resolution,
        deep_ticks=params.deep_ticks,
    )
    _surface_top_telemetry(state, report)
    return report_to_dict(report)


def _probe_hardware(state: AppState, params: BaseModel) -> object:
    """Probe the host and return capabilities plus the auto profile."""
    caps = probe_host()
    return {"capabilities": asdict(caps), "recommendation": asdict(recommend(caps))}


def build_default_registry() -> CommandRegistry:
    """Return the registry of built-in commands."""
    registry = CommandRegistry()
    registry.register(Command("ping", "Liveness check; returns pong.", NoParams, _ping))
    registry.register(
        Command(
            "get_settings",
            "Return every option's current value (the whole settings tree).",
            NoParams,
            _get_settings,
        )
    )
    registry.register(
        Command(
            "list_setting_paths",
            "Return every dotted option path usable with set_setting.",
            NoParams,
            _list_setting_paths,
        )
    )
    registry.register(
        Command(
            "set_setting",
            "Change one option by dotted path; the new value is validated "
            "against the settings schema and broadcast to WS subscribers.",
            SetSettingParams,
            _set_setting,
            mutates=True,
        )
    )
    registry.register(
        Command(
            "list_grid_backends",
            "Return the grid backend toggle names known to this build.",
            NoParams,
            _list_grid_backends,
        )
    )
    registry.register(
        Command(
            "probe_hardware",
            "Probe host hardware (CPU cores, RAM, GPUs) and return the "
            "recommended auto-scaling profile.",
            NoParams,
            _probe_hardware,
        )
    )
    registry.register(
        Command(
            "get_run_telemetry",
            "Return the most recent simulation run's speed telemetry "
            "(ticks/sec, seconds/tick, per-process wall time); empty before any run.",
            NoParams,
            _get_run_telemetry,
        )
    )
    registry.register(
        Command(
            "get_compute_backend",
            "Return the compute backend the sim is running on right now "
            "(name, device cpu/gpu, and how many devices it shards across).",
            NoParams,
            _get_compute_backend,
        )
    )
    registry.register(
        Command(
            "set_compute_backend",
            "Hot-swap the compute backend (e.g. CPU numpy <-> GPU jax) while "
            "the sim runs. The swap lands between ticks via the single-writer "
            "lock; requesting a backend this host cannot provide is rejected "
            "and leaves the running backend unchanged.",
            SetComputeBackendParams,
            _set_compute_backend,
            mutates=True,
        )
    )
    registry.register(
        Command(
            "run_world_sweep",
            "Generate `count` worlds from consecutive seeds, fast-score each by "
            "habitability, keep the top `keep_top_k`, and deep-sim the winners "
            "(higher fidelity + rivers + biology). Returns the ranked report.",
            SweepParams,
            _run_world_sweep,
            mutates=True,
        )
    )
    return registry
