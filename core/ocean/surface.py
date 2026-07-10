"""Surface ocean (M1 tier): wind-driven currents advecting heat.

Wind stress deflected by the signed rotation (Ekman-style) drives
surface drift; currents route around the land mask; ocean cells whose
westward neighbor is land get boundary intensification (the Stommel
western-boundary result: Gulf Stream/Kuroshio analogues), and sea
surface temperature advects along the flow — so warm water rides
poleward on western margins and warms downwind coasts.

Deeper tiers (salinity/density, thermohaline overturning) replace this
behind the same :class:`ports.ocean.Ocean` port.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.grid.vectors import rotate, unit_direction
from core.sim.planet_config import PlanetConfig
from ports.ocean import Ocean, OceanForcing, OceanSurface

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.grid.vectors import Vector2
    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid

_DRIFT_PER_STRESS = 1.0
_EKMAN_DEFLECTION_RAD = math.pi / 4.0
_WESTERN_BOUNDARY_FACTOR = 3.0
_ADVECTION_PASSES = 2
_ADVECTION_BLEND = 0.4
_MIXED_LAYER_HEAT_CAPACITY_J_M2_K = 4.2e8
_ANOMALY_FLUX_W_M2_K = 15.0
_SEA_ICE_FREEZING_K = 271.35
_CALM_CURRENT_M_S = 1e-9


class SurfaceOcean(Ocean):
    """The M1 wind-driven surface tier."""

    def __init__(
        self,
        grid: Grid,
        planet: PlanetConfig,
        sea_mask: SeaMask,
        initial_sst_k: Mapping[CellId, float],
    ) -> None:
        """Initialize SST over ocean cells (land cells are absent)."""
        self._grid = grid
        self._planet = planet
        self._mask = sea_mask
        self._sst = {cell: temp for cell, temp in initial_sst_k.items() if sea_mask.ocean[cell]}
        self._current: dict[CellId, Vector2] = dict.fromkeys(self._sst, (0.0, 0.0))

    def surface(self) -> OceanSurface:
        """Return the current surface fields."""
        return OceanSurface(
            sea_surface_temp_k=dict(self._sst),
            current_east_m_s={c: v[0] for c, v in self._current.items()},
            current_north_m_s={c: v[1] for c, v in self._current.items()},
            sea_ice_fraction={
                c: 1.0 if t < _SEA_ICE_FREEZING_K else 0.0 for c, t in self._sst.items()
            },
        )

    def step(self, forcing: OceanForcing, dt_s: float) -> None:
        """Advance currents from wind stress, advect SST, absorb heat."""
        self._update_currents(forcing)
        self._advect_sst()
        for cell, flux in forcing.surface_heat_flux_w_m2.items():
            if cell in self._sst:
                self._sst[cell] += flux * dt_s / _MIXED_LAYER_HEAT_CAPACITY_J_M2_K

    def _update_currents(self, forcing: OceanForcing) -> None:
        """Turn wind stress into deflected, boundary-intensified drift."""
        for cell in self._current:
            stress = (
                forcing.wind_stress_east_pa.get(cell, 0.0),
                forcing.wind_stress_north_pa.get(cell, 0.0),
            )
            coriolis = self._planet.coriolis_parameter(self._grid.centroid(cell).lat_deg)
            deflection = -_EKMAN_DEFLECTION_RAD * _sign(coriolis)
            east, north = rotate(stress, deflection)
            east *= _DRIFT_PER_STRESS
            north *= _DRIFT_PER_STRESS
            if self._has_land_to_west(cell):
                north *= _WESTERN_BOUNDARY_FACTOR
            self._current[cell] = (east, north)

    def _has_land_to_west(self, cell: CellId) -> bool:
        """Return whether the most-westward neighbor is land."""
        origin = self._grid.centroid(cell)
        best: tuple[float, CellId] | None = None
        for neighbor in self._grid.neighbors(cell):
            direction = unit_direction(origin, self._grid.centroid(neighbor), self._grid.radius_m)
            westness = -direction[0]
            if best is None or westness > best[0]:
                best = (westness, neighbor)
        return best is not None and best[0] > 0.0 and self._mask.is_land(best[1])

    def _advect_sst(self) -> None:
        """Pull SST from up-current: heat rides the flow."""
        for _ in range(_ADVECTION_PASSES):
            advected = {}
            for cell, temp in self._sst.items():
                source = self._upcurrent_neighbor(cell)
                if source is None:
                    advected[cell] = temp
                else:
                    advected[cell] = (1.0 - _ADVECTION_BLEND) * temp + _ADVECTION_BLEND * self._sst[
                        source
                    ]
            self._sst = advected

    def _upcurrent_neighbor(self, cell: CellId) -> CellId | None:
        """Return the ocean neighbor the current arrives from."""
        east, north = self._current[cell]
        if math.hypot(east, north) < _CALM_CURRENT_M_S:
            return None
        origin = self._grid.centroid(cell)
        best: tuple[float, CellId] | None = None
        for neighbor in self._grid.neighbors(cell):
            if neighbor not in self._sst:
                continue
            direction = unit_direction(origin, self._grid.centroid(neighbor), self._grid.radius_m)
            upstream = -(direction[0] * east + direction[1] * north)
            if best is None or upstream > best[0]:
                best = (upstream, neighbor)
        if best is None or best[0] <= 0.0:
            return None
        return best[1]

    def heat_transport_to_atmosphere_w_m2(self) -> Mapping[CellId, float]:
        """Return flux from each cell's SST anomaly vs its latitude band.

        Positive where currents delivered anomalously warm water (the
        Gulf-Stream effect on downwind coasts), negative over cold
        upwelling-like anomalies.
        """
        bands: dict[int, list[float]] = {}
        for cell, temp in self._sst.items():
            bands.setdefault(self._band(cell), []).append(temp)
        means = {band: sum(v) / len(v) for band, v in bands.items()}
        return {
            cell: _ANOMALY_FLUX_W_M2_K * (temp - means[self._band(cell)])
            for cell, temp in self._sst.items()
        }

    def _band(self, cell: CellId) -> int:
        """Return the 10-degree latitude band index of a cell."""
        return int(self._grid.centroid(cell).lat_deg // 10)


def _sign(value: float) -> float:
    """Return -1, 0, or +1."""
    if value > 0:
        return 1.0
    return -1.0 if value < 0 else 0.0
