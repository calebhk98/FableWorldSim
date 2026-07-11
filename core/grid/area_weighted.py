"""Area-weighted aggregation: the equal-surface correctness rule.

H3 and S2 cells are not equal-area, so any unweighted per-cell average
would silently favor bigger cells and change results when the grid
backend is toggled.  Every aggregation in the simulation goes through
these helpers (or follows the same rule), making results invariant to
the backend and to cell-size variation within a backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.grid.field_reduce import area_weighted_mean_on, area_weighted_total_on

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ports.array_backend import ArrayBackend


def _check_lengths(values: Sequence[float], areas_m2: Sequence[float]) -> None:
    """Validate that values and areas describe the same cells."""
    if len(values) != len(areas_m2):
        msg = f"got {len(values)} values but {len(areas_m2)} areas"
        raise ValueError(msg)
    if not values:
        msg = "cannot aggregate zero cells"
        raise ValueError(msg)
    if any(area <= 0 for area in areas_m2):
        msg = "every cell area must be > 0"
        raise ValueError(msg)


def area_weighted_mean(
    values: Sequence[float],
    areas_m2: Sequence[float],
    backend: ArrayBackend | None = None,
) -> float:
    """Return the mean of an intensive quantity, weighted by cell area.

    Splitting any cell into smaller cells with the same value leaves the
    result unchanged — the invariance that makes the grid toggle safe.

    Passing a ``backend`` runs the reduction through that ArrayBackend,
    sharded across its devices; the result matches this reference within
    the backend's float precision.  ``None`` (the default) uses the exact
    pure-Python path.
    """
    _check_lengths(values, areas_m2)
    if backend is not None:
        return area_weighted_mean_on(backend, values, areas_m2)
    total_area = sum(areas_m2)
    weighted = sum(value * area for value, area in zip(values, areas_m2, strict=True))
    return weighted / total_area


def area_weighted_total(
    densities_per_m2: Sequence[float],
    areas_m2: Sequence[float],
    backend: ArrayBackend | None = None,
) -> float:
    """Return the planet total of a per-unit-area density (e.g. biomass/m^2).

    Passing a ``backend`` runs the reduction sharded across its devices
    (identical math, within float precision); ``None`` uses the exact
    pure-Python path.
    """
    _check_lengths(densities_per_m2, areas_m2)
    if backend is not None:
        return area_weighted_total_on(backend, densities_per_m2, areas_m2)
    return sum(density * area for density, area in zip(densities_per_m2, areas_m2, strict=True))


def area_fraction(flags: Sequence[bool], areas_m2: Sequence[float]) -> float:
    """Return the fraction of total surface area where ``flags`` is true."""
    _check_lengths([1.0 if flag else 0.0 for flag in flags], areas_m2)
    total_area = sum(areas_m2)
    flagged = sum(area for flag, area in zip(flags, areas_m2, strict=True) if flag)
    return flagged / total_area
