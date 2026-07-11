"""Surface ocean (M1 tier): wind-driven currents advecting heat.

Three linked pieces of physics, all derived from the wind field and
``PlanetConfig`` rather than scripted:

* **Ekman drift** — wind stress deflected 45 deg by the signed
  Coriolis parameter drives direct, ageostrophic surface flow.
* **Sverdrup interior balance** — the linear vorticity balance
  ``beta * v = curl_z(tau) / (rho * H)`` gives the interior's
  north-south geostrophic drift straight from the wind-stress curl
  and beta (``core.grid.vectors.gradient``'s directional-derivative
  averaging applied to each stress component, then combined into the
  curl) with no elliptic PDE solve.
* **Stommel western-boundary layer** — a basin bounded by land cannot
  carry net meridional transport across a latitude circle, so
  whatever the wide interior carries must return through a narrow
  friction layer; concentrating the same transport into the boundary
  cells' much smaller area gives the classic narrow, fast current
  (Gulf Stream/Kuroshio analogues) with the intensification factor
  set by the interior/boundary area ratio, not a fixed constant.

Currents route around the land mask, and sea surface temperature
advects along the flow, so warm water rides poleward on western
margins and warms downwind coasts.

Approximations, named honestly: this is a single-layer barotropic
model with a fixed effective wind-driven layer depth, not a
stratified ocean; the boundary-layer closure treats each latitude
band as one basin (two basins at the same latitude, e.g. Atlantic and
Pacific, would incorrectly balance against each other); beta is
floored near the poles to keep the Sverdrup division finite; and
current speeds are capped at a plausible surface-current bound since
the linear model has no nonlinear/frictional saturation. Deeper tiers
(salinity/density, thermohaline overturning) replace this behind the
same :class:`ports.ocean.Ocean` port.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.grid.vectors import local_offset_m, rotate, unit_direction
from core.sim.planet_config import PlanetConfig
from ports.ocean import Ocean, OceanForcing, OceanSurface

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.grid.vectors import Vector2
    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId, Grid

_DRIFT_PER_STRESS = 1.0
_EKMAN_DEFLECTION_RAD = math.pi / 4.0
_SEAWATER_DENSITY_KG_M3 = 1025.0
_SVERDRUP_LAYER_DEPTH_M = 300.0
_MAX_BETA_LAT_DEG = 80.0
_MAX_CURRENT_SPEED_M_S = 2.5
_BOUNDARY_LAYER_BAND_DEG = 5.0
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
        """Sverdrup interior transport, closed by a western boundary layer."""
        interior = self._interior_currents(forcing)
        self._current = self._apply_western_boundary_layer(interior)

    def _interior_currents(self, forcing: OceanForcing) -> dict[CellId, Vector2]:
        """Return each cell's Ekman drift plus Sverdrup interior term."""
        tau_east = forcing.wind_stress_east_pa
        tau_north = forcing.wind_stress_north_pa
        currents: dict[CellId, Vector2] = {}
        for cell in self._current:
            stress = (tau_east.get(cell, 0.0), tau_north.get(cell, 0.0))
            coriolis = self._planet.coriolis_parameter(self._grid.centroid(cell).lat_deg)
            deflection = -_EKMAN_DEFLECTION_RAD * _sign(coriolis)
            east, north = rotate(stress, deflection)
            east *= _DRIFT_PER_STRESS
            north *= _DRIFT_PER_STRESS
            north += self._sverdrup_north_speed(cell, tau_east, tau_north)
            currents[cell] = (_clamp_speed(east), _clamp_speed(north))
        return currents

    def _beta(self, cell: CellId) -> float:
        """Return d(f)/dy, the Coriolis parameter's meridional gradient.

        Latitude is clamped away from the poles (where ``cos(lat)``
        vanishes) so the Sverdrup division below stays finite; this
        under-states beta very near the poles rather than blowing up.
        """
        lat = self._grid.centroid(cell).lat_deg
        clamped_lat = max(-_MAX_BETA_LAT_DEG, min(_MAX_BETA_LAT_DEG, lat))
        return (
            2.0
            * self._planet.rotation_rate_rad_s
            * math.cos(math.radians(clamped_lat))
            / self._planet.radius_m
        )

    def _stress_curl(
        self,
        cell: CellId,
        tau_east: Mapping[CellId, float],
        tau_north: Mapping[CellId, float],
    ) -> float:
        """Return d(tau_north)/d(east) - d(tau_east)/d(north) at a cell.

        Same directional-derivative averaging as
        ``core.grid.vectors.gradient`` (degree-agnostic; no fixed
        neighbor count) applied to each stress component and combined
        into the vertical curl. Land neighbors (absent from the
        forcing maps) are skipped, the way heat advection already
        skips them.
        """
        origin = self._grid.centroid(cell)
        here_east = tau_east.get(cell, 0.0)
        here_north = tau_north.get(cell, 0.0)
        d_north_d_east = 0.0
        d_east_d_north = 0.0
        count = 0
        for neighbor in self._grid.neighbors(cell):
            if neighbor not in tau_east:
                continue
            offset_e, offset_n = local_offset_m(
                origin, self._grid.centroid(neighbor), self._grid.radius_m
            )
            distance = math.hypot(offset_e, offset_n)
            if distance == 0.0:
                continue
            slope_north = (tau_north.get(neighbor, 0.0) - here_north) / distance
            slope_east = (tau_east[neighbor] - here_east) / distance
            d_north_d_east += slope_north * (offset_e / distance)
            d_east_d_north += slope_east * (offset_n / distance)
            count += 1
        if count == 0:
            return 0.0
        scale = 2.0 / count
        return scale * d_north_d_east - scale * d_east_d_north

    def _sverdrup_north_speed(
        self,
        cell: CellId,
        tau_east: Mapping[CellId, float],
        tau_north: Mapping[CellId, float],
    ) -> float:
        """Return the interior meridional velocity from the Sverdrup relation.

        ``beta * v = curl_z(tau) / (rho * H)``: the linear vorticity
        balance, giving the interior's geostrophic north-south drift
        directly from the curl and beta. The zonal return flow mass
        conservation demands is supplied by
        ``_apply_western_boundary_layer``, not here.
        """
        beta = self._beta(cell)
        if beta == 0.0:
            return 0.0
        curl = self._stress_curl(cell, tau_east, tau_north)
        return curl / (_SEAWATER_DENSITY_KG_M3 * _SVERDRUP_LAYER_DEPTH_M * beta)

    def _apply_western_boundary_layer(
        self, interior: dict[CellId, Vector2]
    ) -> dict[CellId, Vector2]:
        """Close the interior Sverdrup transport through a narrow western layer.

        A basin bounded by land cannot carry net meridional transport
        across a latitude circle: whatever the wide interior carries
        north or south must return through the boundary layer. Summed
        per latitude band, the interior's area-weighted transport
        gets concentrated into the boundary cells' much smaller area —
        the Stommel result (a narrow, fast current) with the emergent
        intensification factor set by that area ratio, not a fixed
        constant. Each band is treated as one basin (a simplification:
        two basins sharing a latitude, e.g. Atlantic and Pacific,
        would incorrectly balance together).
        """
        bands: dict[int, list[CellId]] = {}
        for cell in interior:
            bands.setdefault(self._band(cell, _BOUNDARY_LAYER_BAND_DEG), []).append(cell)
        boundary = {cell for cell in interior if self._has_land_to_west(cell)}
        result = dict(interior)
        for cells in bands.values():
            boundary_cells = [c for c in cells if c in boundary]
            if not boundary_cells:
                continue
            interior_cells = [c for c in cells if c not in boundary]
            interior_transport = sum(interior[c][1] * self._grid.area_m2(c) for c in interior_cells)
            boundary_area = sum(self._grid.area_m2(c) for c in boundary_cells)
            if boundary_area == 0.0:
                continue
            closure_north = -interior_transport / boundary_area
            for cell in boundary_cells:
                east, north = interior[cell]
                result[cell] = (east, _clamp_speed(north + closure_north))
        return result

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

    def _band(self, cell: CellId, width_deg: float = 10.0) -> int:
        """Return the latitude band index of a cell at the given bin width."""
        return int(self._grid.centroid(cell).lat_deg // width_deg)


def _sign(value: float) -> float:
    """Return -1, 0, or +1."""
    if value > 0:
        return 1.0
    return -1.0 if value < 0 else 0.0


def _clamp_speed(value: float) -> float:
    """Bound a current component to a plausible surface-current speed.

    The linear model has no nonlinear/frictional saturation, so an
    extreme wind-stress curl or a very narrow boundary layer could
    otherwise predict implausibly fast water; real western boundary
    currents rarely exceed a few m/s at the surface, so this caps
    there.
    """
    if value > _MAX_CURRENT_SPEED_M_S:
        return _MAX_CURRENT_SPEED_M_S
    if value < -_MAX_CURRENT_SPEED_M_S:
        return -_MAX_CURRENT_SPEED_M_S
    return value
