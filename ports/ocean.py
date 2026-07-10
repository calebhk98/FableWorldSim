"""Ocean port: surface ocean state and circulation.

Kept distinct from the atmosphere/climate layer so the wind -> current ->
heat-advection coupling is explicit and the model can deepen (surface
layer now, full 3D thermohaline circulation with salinity/density and
deep overturning later) without touching consumers.

M1 implements the surface tier behind this port: wind stress -> Ekman
transport, Sverdrup interior balance, and the Stommel model's
western-boundary intensification (Gulf Stream/Kuroshio analogues),
with currents routed around continents and advecting heat.  Ocean heat
capacity also buffers day-night and seasonal swings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports.grid import CellId


@dataclass(frozen=True)
class OceanSurface:
    """Per-cell surface-ocean fields (intensive; land cells absent)."""

    sea_surface_temp_k: Mapping[CellId, float]
    current_east_m_s: Mapping[CellId, float]
    current_north_m_s: Mapping[CellId, float]
    sea_ice_fraction: Mapping[CellId, float]


@dataclass(frozen=True)
class OceanForcing:
    """Atmosphere-side inputs driving the ocean for one step."""

    wind_stress_east_pa: Mapping[CellId, float]
    wind_stress_north_pa: Mapping[CellId, float]
    surface_heat_flux_w_m2: Mapping[CellId, float]
    ocean_mask: Mapping[CellId, bool]


class Ocean(ABC):
    """Surface ocean state and its response to atmospheric forcing."""

    @abstractmethod
    def surface(self) -> OceanSurface:
        """Return the current surface-ocean fields."""

    @abstractmethod
    def step(self, forcing: OceanForcing, dt_s: float) -> None:
        """Advance the ocean by ``dt_s`` seconds under ``forcing``.

        M1: Ekman/Sverdrup/Stommel surface circulation with heat
        advection around the land mask.  Deeper tiers (thermohaline,
        salinity, overturning) slot in behind this same call.
        """

    @abstractmethod
    def heat_transport_to_atmosphere_w_m2(self) -> Mapping[CellId, float]:
        """Return per-cell ocean-to-atmosphere heat flux (W/m^2)."""
