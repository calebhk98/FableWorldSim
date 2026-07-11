"""PlanetConfig: the single parameterization every world runs from.

Earth, Mars, Venus, a tidally locked exoplanet, or an invented world all
run the same simulation code; only this config differs.  Coriolis, wind
bands, heat transport, day length, and tides are *derived* from these
fields, never hard-coded.

Key conventions:

* ``rotation_period_s`` is **signed**: positive is prograde, negative is
  retrograde (Venus-style reverse spin), and ``math.inf`` means a
  non-rotating body.  Tidal locking is rotation period equal to orbital
  period (permanent day and night sides; the solar day is infinite).
* Orbital energy is given **either** directly as ``insolation_wm2``
  **or** derived from ``(stellar_luminosity_w, orbital_distance_m)`` —
  "1.5 AU from a G-type star" — via ``L / (4 * pi * d**2)``.
* ``ocean_fraction`` is a first-class dial ("90% ocean" to bone-dry); the
  topography layer reconciles it with the height field by solving for the
  sea level that yields it (``core.ocean.sea_level``).
* Satellites drive ``tidal_range_m``, which defines the intertidal
  habitat band (distinct from "coast"); erosion/flooding coupling is
  roadmap, the habitat band is M1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.sim.constants import (
    EARTH_SURFACE_PRESSURE_PA,
    LUNAR_EQUILIBRIUM_TIDE_M,
    MOON_MASS_KG,
    MOON_SEMI_MAJOR_AXIS_M,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_TIDAL_LOCK_REL_TOL = 1e-6
_LUNAR_TIDAL_FORCING = MOON_MASS_KG / MOON_SEMI_MAJOR_AXIS_M**3

_GHG_ABSORPTION_COEFFICIENT: dict[str, float] = {
    "CO2": 1.0,
    "H2O": 8.0,
    "CH4": 30.0,
    "N2O": 200.0,
    "SO2": 120.0,
    "O3": 150.0,
}
"""Per-mole-fraction IR absorption strength, relative to CO2 = 1.

Only species with a permanent or collision-induced dipole trap outgoing
longwave radiation (water vapor, methane, nitrous oxide, sulfur dioxide,
ozone); homonuclear background gases (N2, O2, Ar, H2, He, Ne, ...) are
absent here and contribute exactly zero, however abundant -- "thick
nitrogen" alone buys no warming. Coefficients are order-of-magnitude
band-model ratios (roughly following real per-molecule IR opacity), not
line-by-line spectroscopy; that is the right fidelity for M1's grey-gas
energy balance (see ``core/climate/temperature.py``)."""

_GREENHOUSE_CEILING_K = 510.0
"""Saturation ceiling for greenhouse warming (kelvin), approached but
never reached as optical depth grows without bound -- calibrated so a
crushing, all-CO2, ~90-bar atmosphere (Venus) saturates close to its
real ~505 K greenhouse warming."""


def _coerce_satellites(sats: object) -> tuple[Satellite, ...]:
    """Convert satellites field: list of dicts or Satellite -> tuple of Satellite.

    Raises ValueError if invalid.
    """
    if not isinstance(sats, (list, tuple)):
        msg = f"satellites must be list/tuple, got {type(sats).__name__}"
        raise ValueError(msg)
    sat_list = []
    for sat in sats:
        if isinstance(sat, dict):
            try:
                sat_list.append(Satellite(**sat))
            except (TypeError, ValueError) as exc:
                msg = f"invalid satellite specification: {exc}"
                raise ValueError(msg) from exc
        elif isinstance(sat, Satellite):
            sat_list.append(sat)
        else:
            msg = f"satellite must be dict/Satellite, got {type(sat).__name__}"
            raise ValueError(msg)
    return tuple(sat_list)


@dataclass(frozen=True)
class Atmosphere:
    """Bulk atmosphere: composition and surface pressure.

    Greenhouse strength is *derived* from them (see
    ``greenhouse_offset_k`` below), never a hand-supplied literal -- so
    Venus-thick, Mars-thin, and airless all fall out of composition +
    pressure, not per-planet constants. ``composition`` maps species
    name to mole fraction.
    """

    surface_pressure_pa: float
    composition: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate pressure."""
        if self.surface_pressure_pa < 0:
            msg = f"surface pressure must be >= 0, got {self.surface_pressure_pa}"
            raise ValueError(msg)

    @classmethod
    def airless(cls) -> Atmosphere:
        """Return a no-atmosphere config (Moon-like)."""
        return cls(surface_pressure_pa=0.0)

    @property
    def is_airless(self) -> bool:
        """Return whether there is effectively no atmosphere."""
        return self.surface_pressure_pa == 0.0

    @property
    def greenhouse_optical_depth(self) -> float:
        """Return the dimensionless grey-gas IR optical depth.

        Each greenhouse-active species' column amount scales with its
        partial pressure -- mole fraction times surface pressure,
        normalized to Earth's -- weighted by its relative absorption
        strength (``_GHG_ABSORPTION_COEFFICIENT``); the contributions
        sum linearly. Zero for a vacuum or a purely non-absorbing mix
        (pure N2, say), however thick.
        """
        if self.surface_pressure_pa <= 0.0:
            return 0.0
        pressure_ratio = self.surface_pressure_pa / EARTH_SURFACE_PRESSURE_PA
        ghg_mole_fraction = sum(
            self.composition.get(species, 0.0) * coefficient
            for species, coefficient in _GHG_ABSORPTION_COEFFICIENT.items()
        )
        return pressure_ratio * ghg_mole_fraction

    @property
    def greenhouse_saturation(self) -> float:
        """Return how saturated the greenhouse effect is, in [0, 1).

        A smooth, monotone function of optical depth that approaches
        (never reaches) 1: a wisp of absorber lets nearly all of it
        through (small fraction), a crushing column is nearly opaque
        (fraction near 1). This is also used to boost lateral heat
        transport efficiency (see ``core.climate.temperature``): a
        radiatively opaque atmosphere mixes heat better, not just a
        physically thick one.
        """
        return -math.expm1(-self.greenhouse_optical_depth)

    @property
    def greenhouse_offset_k(self) -> float:
        """Return the greenhouse warming over the airless equilibrium, in K.

        ``_GREENHOUSE_CEILING_K * greenhouse_saturation``: a trace of
        CO2 over a thin column (Mars) buys a few kelvin, Earth's dilute
        water vapor + CO2 over a full atmosphere buys tens, and Venus's
        crushing, all-CO2 blanket saturates near the ceiling -- derived
        from composition and pressure, never a per-planet literal.
        """
        return _GREENHOUSE_CEILING_K * self.greenhouse_saturation


@dataclass(frozen=True)
class Satellite:
    """A moon: enough to derive its tidal forcing on the primary."""

    name: str
    mass_kg: float
    semi_major_axis_m: float

    def __post_init__(self) -> None:
        """Validate mass and orbital distance."""
        if self.mass_kg <= 0:
            msg = f"satellite mass must be > 0, got {self.mass_kg}"
            raise ValueError(msg)
        if self.semi_major_axis_m <= 0:
            msg = f"satellite distance must be > 0, got {self.semi_major_axis_m}"
            raise ValueError(msg)

    @property
    def tidal_forcing(self) -> float:
        """Return the tide-raising strength, proportional to mass/distance^3."""
        return self.mass_kg / self.semi_major_axis_m**3


@dataclass(frozen=True)
class PlanetConfig:
    """Full parameterization of a world (see module docstring)."""

    name: str
    radius_m: float
    surface_gravity_m_s2: float
    axial_tilt_deg: float
    rotation_period_s: float
    orbital_period_s: float
    ocean_fraction: float
    atmosphere: Atmosphere
    satellites: tuple[Satellite, ...] = ()
    insolation_wm2: float | None = None
    stellar_luminosity_w: float | None = None
    orbital_distance_m: float | None = None

    def __post_init__(self) -> None:
        """Validate geometry, spin, and the orbital-energy specification."""
        if self.radius_m <= 0:
            msg = f"radius must be > 0, got {self.radius_m}"
            raise ValueError(msg)
        if self.surface_gravity_m_s2 <= 0:
            msg = f"gravity must be > 0, got {self.surface_gravity_m_s2}"
            raise ValueError(msg)
        if self.rotation_period_s == 0:
            msg = "rotation_period_s of 0 is undefined; use math.inf for none"
            raise ValueError(msg)
        if self.orbital_period_s <= 0 or math.isinf(self.orbital_period_s):
            msg = f"orbital period must be finite and > 0, got {self.orbital_period_s}"
            raise ValueError(msg)
        if not 0.0 <= self.ocean_fraction <= 1.0:
            msg = f"ocean fraction must be in [0, 1], got {self.ocean_fraction}"
            raise ValueError(msg)
        self._validate_energy_spec()

    def _validate_energy_spec(self) -> None:
        """Require exactly one way of specifying orbital energy."""
        direct = self.insolation_wm2 is not None
        derived = self.stellar_luminosity_w is not None and self.orbital_distance_m is not None
        partial = (self.stellar_luminosity_w is None) != (self.orbital_distance_m is None)
        if partial:
            msg = "stellar_luminosity_w and orbital_distance_m must be given together"
            raise ValueError(msg)
        if direct == derived:
            msg = (
                "give orbital energy exactly one way: insolation_wm2, or "
                "stellar_luminosity_w with orbital_distance_m"
            )
            raise ValueError(msg)

    @property
    def solar_constant_wm2(self) -> float:
        """Return top-of-atmosphere flux: direct, or L / (4 * pi * d^2)."""
        if self.insolation_wm2 is not None:
            return self.insolation_wm2
        luminosity = self.stellar_luminosity_w
        distance = self.orbital_distance_m
        if luminosity is None or distance is None:
            msg = "orbital energy spec missing despite validation"
            raise RuntimeError(msg)
        return luminosity / (4.0 * math.pi * distance**2)

    @property
    def rotation_rate_rad_s(self) -> float:
        """Return the signed spin rate (0.0 for a non-rotating body)."""
        if math.isinf(self.rotation_period_s):
            return 0.0
        return 2.0 * math.pi / self.rotation_period_s

    @property
    def is_retrograde(self) -> bool:
        """Return whether the planet spins opposite its orbital motion."""
        return self.rotation_period_s < 0

    @property
    def is_tidally_locked(self) -> bool:
        """Return whether rotation and orbital period match (same sense)."""
        return math.isclose(
            self.rotation_period_s,
            self.orbital_period_s,
            rel_tol=_TIDAL_LOCK_REL_TOL,
        )

    @property
    def solar_day_s(self) -> float:
        """Return the sun-to-sun day length in seconds.

        Derived from the synodic relation ``1/day = 1/rot - 1/orb``.
        Infinite when tidally locked (permanent day and night sides: the
        day *is* the year); equal to the orbital period for a
        non-rotating body.
        """
        if self.is_tidally_locked:
            return math.inf
        rotation_rate = 0.0 if math.isinf(self.rotation_period_s) else 1.0 / self.rotation_period_s
        synodic_rate = rotation_rate - 1.0 / self.orbital_period_s
        return abs(1.0 / synodic_rate)

    def coriolis_parameter(self, lat_deg: float) -> float:
        """Return the Coriolis parameter ``f = 2 * omega * sin(lat)`` in 1/s."""
        return 2.0 * self.rotation_rate_rad_s * math.sin(math.radians(lat_deg))

    @property
    def tidal_range_m(self) -> float:
        """Return the first-order tidal range from all satellites.

        Sums each moon's ``mass / distance**3`` forcing and scales it
        against the Earth-Moon equilibrium tide.  This defines the width
        of the intertidal habitat band in M1; tidal flooding/erosion
        coupling is roadmap.
        """
        forcing = sum(moon.tidal_forcing for moon in self.satellites)
        return LUNAR_EQUILIBRIUM_TIDE_M * forcing / _LUNAR_TIDAL_FORCING

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PlanetConfig:
        """Construct PlanetConfig from a dict, coercing nested dicts to dataclasses.

        Converts nested ``atmosphere`` and ``satellites`` dicts to their
        respective dataclass types. Raises ``ValueError`` if invalid.

        This is the robust path for round-tripping: pass the result of
        ``dataclasses.asdict(preset)`` here to reconstruct the config.

        Args:
            data: A dict with ``atmosphere`` and ``satellites`` optionally
                as dicts instead of objects.

        Returns:
            A validated PlanetConfig instance.

        Raises:
            ValueError: If the dict is structurally invalid.
        """
        params: dict[str, Any] = dict(data)

        # Coerce atmosphere: dict -> Atmosphere, or validate it's an Atmosphere
        if "atmosphere" in params:
            atm = params["atmosphere"]
            if isinstance(atm, dict):
                try:
                    params["atmosphere"] = Atmosphere(**atm)
                except (TypeError, ValueError) as exc:
                    msg = f"invalid atmosphere specification: {exc}"
                    raise ValueError(msg) from exc
            elif not isinstance(atm, Atmosphere):
                msg = f"atmosphere must be a dict or Atmosphere, got {type(atm).__name__}"
                raise ValueError(msg)

        # Coerce satellites: list of dicts -> tuple of Satellite
        if "satellites" in params:
            params["satellites"] = _coerce_satellites(params["satellites"])

        # Construct PlanetConfig, converting TypeError to ValueError.
        try:
            return cls(**params)
        except TypeError as exc:
            msg = f"missing or invalid field in planet config: {exc}"
            raise ValueError(msg) from exc
