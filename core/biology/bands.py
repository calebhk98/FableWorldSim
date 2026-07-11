"""Graded comfort/tolerance response with intra-population variation.

Each environmental trait is a **comfort band** (optimum, full growth)
nested inside a wider **tolerance band** (survivable but stressed), and
the preferred point itself varies across individuals — the design's
"multivariance of variances": some individuals of a species skew warm,
some cold.  Cell suitability for a trait is therefore a **graded** 0..1
response integrated over that spread, never a hard in/out range: inside
comfort it is ~1, toward the tolerance edges it declines, and beyond
tolerance it is 0.
"""

from __future__ import annotations

from dataclasses import dataclass

# A 3-point symmetric quadrature (in standard-deviation units) that
# approximates averaging a trait response over the spread of individual
# optima; the weights sum to 1 so a zero-spread band is unchanged.
_SPREAD_OFFSETS = (-1.0, 0.0, 1.0)
_SPREAD_WEIGHTS = (0.25, 0.5, 0.25)


@dataclass(frozen=True)
class EnvBand:
    """One environmental trait's comfort band nested in a tolerance band.

    ``spread`` is the standard deviation of the preferred point across
    individuals; a larger spread softens the band edges, because some
    members of the population always thrive a little past the mean's
    comfort range.
    """

    comfort_min: float
    comfort_max: float
    tolerance_min: float
    tolerance_max: float
    spread: float = 0.0

    def __post_init__(self) -> None:
        """Reject inverted or non-nested bands at load time."""
        ordered = self.tolerance_min <= self.comfort_min <= self.comfort_max <= self.tolerance_max
        if not ordered:
            msg = "band needs tolerance_min <= comfort_min <= comfort_max <= tolerance_max"
            raise ValueError(msg)
        if self.spread < 0.0:
            msg = "band spread must be >= 0"
            raise ValueError(msg)

    def response(self, value: float) -> float:
        """Return the graded 0..1 suitability of ``value`` for this trait."""
        if self.spread <= 0.0:
            return self._trapezoid(value)
        total = 0.0
        for offset, weight in zip(_SPREAD_OFFSETS, _SPREAD_WEIGHTS, strict=True):
            total += weight * self._trapezoid(value - offset * self.spread)
        return total

    def _trapezoid(self, value: float) -> float:
        """Piecewise-linear membership: 1 across comfort, 0 past tolerance."""
        if value <= self.tolerance_min or value >= self.tolerance_max:
            return 0.0
        if self.comfort_min <= value <= self.comfort_max:
            return 1.0
        if value < self.comfort_min:
            return _ramp(value, self.tolerance_min, self.comfort_min)
        return _ramp(value, self.tolerance_max, self.comfort_max)


def _ramp(value: float, zero_at: float, one_at: float) -> float:
    """Return the linear interpolation that is 0 at ``zero_at``, 1 at ``one_at``."""
    if one_at == zero_at:
        return 1.0
    return (value - zero_at) / (one_at - zero_at)
