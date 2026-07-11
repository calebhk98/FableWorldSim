"""Emergent wildfire: dry, high-fuel cells ignite, burn, and regrow.

Wildfire is the biology layer's emergent hazard, routed through the shared
field-perturbation mechanism (:mod:`core.hazards.perturb`) exactly like a
scripted disturbance — the only difference is that biology, not an API
command, pulls the trigger.  A cell with enough standing vegetation (fuel)
in dry conditions may ignite stochastically; the fire's radial footprint
removes a fraction of every plant's biomass, and ordinary logistic regrowth
brings the vegetation back over following steps (the California
fire-to-regrowth cycle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.hazards.perturb import Disturbance, apply_disturbance, radial_deltas

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ports.grid import Grid
    from ports.rng import Rng

_BIOMASS_FIELD_PREFIX = "biomass:"


@dataclass(frozen=True)
class FireParams:
    """Ignition and burn tuning for the wildfire pass."""

    base_rate_per_year: float = 0.02
    fuel_threshold_kg_m2: float = 1.0
    dryness_threshold: float = 0.5
    radius_m: float = 300_000.0
    burn_fraction: float = 0.7


@dataclass(frozen=True)
class IgnitionField:
    """The per-cell inputs the ignition test reads."""

    fuel_by_cell: Mapping[str, float]
    dryness_by_cell: Mapping[str, float]
    land_cells: Iterable[str]


def ignite(
    field: IgnitionField,
    params: FireParams,
    dt_years: float,
    rng: Rng,
) -> list[str]:
    """Return the cells that ignite this step (dry, high-fuel, stochastic)."""
    ignited: list[str] = []
    for cell in field.land_cells:
        fuel = field.fuel_by_cell.get(cell, 0.0)
        dryness = field.dryness_by_cell.get(cell, 0.0)
        if fuel < params.fuel_threshold_kg_m2 or dryness < params.dryness_threshold:
            continue
        fuel_ratio = min(1.0, fuel / params.fuel_threshold_kg_m2)
        probability = params.base_rate_per_year * dryness * fuel_ratio * dt_years
        if rng.random() < probability:
            ignited.append(cell)
    return ignited


def burn(
    plant_fields: Mapping[str, Mapping[str, float]],
    grid: Grid,
    ignited: Iterable[str],
    params: FireParams,
) -> dict[str, dict[str, float]]:
    """Return plant-biomass fields after burning every ignited footprint.

    Each fire is a :class:`~core.hazards.perturb.Disturbance` whose deltas
    subtract a share of standing biomass over a radial footprint, applied
    through :func:`~core.hazards.perturb.apply_disturbance` — the same path
    a scripted ash cloud takes.
    """
    fields: dict[str, dict[str, float]] = {
        f"{_BIOMASS_FIELD_PREFIX}{sid}": dict(field) for sid, field in plant_fields.items()
    }
    for cell in ignited:
        footprint = radial_deltas(grid, cell, params.radius_m, 1.0)
        disturbance = Disturbance(
            kind="wildfire",
            trigger="emergent:wildfire",
            field_deltas=_burn_deltas(fields, footprint, params.burn_fraction),
        )
        fields = apply_disturbance(fields, disturbance)
    return {sid: fields[f"{_BIOMASS_FIELD_PREFIX}{sid}"] for sid in plant_fields}


def _burn_deltas(
    fields: Mapping[str, Mapping[str, float]],
    footprint: Mapping[str, float],
    burn_fraction: float,
) -> dict[str, dict[str, float]]:
    """Build negative per-cell biomass deltas for one fire's footprint."""
    deltas: dict[str, dict[str, float]] = {}
    for name, field in fields.items():
        deltas[name] = {
            cell: -field[cell] * burn_fraction * falloff
            for cell, falloff in footprint.items()
            if cell in field
        }
    return deltas
