"""Dev-mode dimensional analysis: the whole unit-bug class, guarded.

Generalizes the equal-area rule: W/m^2 cannot be added to J, counts
cannot be added to densities.  Internally everything is strict SI;
display units (the Fahrenheit comfort band) are *conversions at the
edge*, never stored (see :func:`kelvin_to_celsius` and friends).

Fields stay plain floats in hot paths; wrap values in :class:`Quantity`
inside tests and dev assertions to catch unit bugs mechanically.
"""

from __future__ import annotations

from dataclasses import dataclass

_ZERO_CELSIUS_K = 273.15


class UnitMismatchError(TypeError):
    """Raised when quantities of different dimensions are combined."""


@dataclass(frozen=True)
class Unit:
    """A physical unit as base-dimension exponents (m, kg, s, K, count)."""

    dims: tuple[tuple[str, int], ...]

    def __mul__(self, other: Unit) -> Unit:
        """Return the product unit (exponents add)."""
        return Unit(_combine(self.dims, other.dims, 1))

    def __truediv__(self, other: Unit) -> Unit:
        """Return the quotient unit (exponents subtract)."""
        return Unit(_combine(self.dims, other.dims, -1))


def _combine(
    a: tuple[tuple[str, int], ...], b: tuple[tuple[str, int], ...], sign: int
) -> tuple[tuple[str, int], ...]:
    """Merge two exponent maps (sign=+1 multiply, -1 divide)."""
    merged = dict(a)
    for dim, exp in b:
        merged[dim] = merged.get(dim, 0) + sign * exp
    return tuple(sorted((d, e) for d, e in merged.items() if e != 0))


DIMENSIONLESS = Unit(())
METER = Unit((("m", 1),))
SQUARE_METER = Unit((("m", 2),))
SECOND = Unit((("s", 1),))
KILOGRAM = Unit((("kg", 1),))
KELVIN = Unit((("K", 1),))
COUNT = Unit((("count", 1),))
WATT = Unit((("kg", 1), ("m", 2), ("s", -3)))
JOULE = Unit((("kg", 1), ("m", 2), ("s", -2)))
WATT_PER_M2 = WATT / SQUARE_METER
DENSITY_PER_M2 = COUNT / SQUARE_METER


@dataclass(frozen=True)
class Quantity:
    """A value bound to its unit; arithmetic enforces dimensions."""

    value: float
    unit: Unit

    def __add__(self, other: Quantity) -> Quantity:
        """Add same-unit quantities; anything else is a unit bug."""
        if self.unit != other.unit:
            msg = f"cannot add {self.unit.dims} to {other.unit.dims}"
            raise UnitMismatchError(msg)
        return Quantity(self.value + other.value, self.unit)

    def __sub__(self, other: Quantity) -> Quantity:
        """Subtract same-unit quantities."""
        if self.unit != other.unit:
            msg = f"cannot subtract {other.unit.dims} from {self.unit.dims}"
            raise UnitMismatchError(msg)
        return Quantity(self.value - other.value, self.unit)

    def __mul__(self, other: Quantity) -> Quantity:
        """Multiply quantities (units compose)."""
        return Quantity(self.value * other.value, self.unit * other.unit)

    def __truediv__(self, other: Quantity) -> Quantity:
        """Divide quantities (units compose)."""
        return Quantity(self.value / other.value, self.unit / other.unit)


def kelvin_to_celsius(kelvin: float) -> float:
    """Display conversion only — state is always stored in kelvin."""
    return kelvin - _ZERO_CELSIUS_K


def kelvin_to_fahrenheit(kelvin: float) -> float:
    """Display conversion only — state is always stored in kelvin."""
    return kelvin_to_celsius(kelvin) * 9.0 / 5.0 + 32.0
