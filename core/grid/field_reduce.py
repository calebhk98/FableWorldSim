"""Backend-accelerated, shardable area-weighted reductions.

The pure-Python helpers in :mod:`core.grid.area_weighted` are the
reference implementation; these run the identical math through an
:class:`~ports.array_backend.ArrayBackend` so a planet-wide reduction can
execute distributed on a device mesh.  Each field is sharded across every
device (:meth:`ArrayBackend.shard`); the zero-padded capacity cells carry
zero area, so they drop out of the weighted sum and the distributed
result matches the reference within backend float precision.  One GPU,
eight, or twenty-seven run the same call — the backend hides placement.

Callers reach these through the optional ``backend`` argument on the
public :mod:`core.grid.area_weighted` functions; passing a sharded backend
is the one-line switch from the CPU reference to the distributed path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ports.array_backend import ArrayBackend


def area_weighted_mean_on(
    backend: ArrayBackend,
    values: Sequence[float],
    areas_m2: Sequence[float],
) -> float:
    """Return the area-weighted mean, sharded across ``backend``'s devices."""
    xp = backend.namespace()
    sharded_values = backend.shard(backend.asarray(list(values)))
    sharded_areas = backend.shard(backend.asarray(list(areas_m2)))
    return float(xp.sum(sharded_values * sharded_areas) / xp.sum(sharded_areas))


def area_weighted_total_on(
    backend: ArrayBackend,
    densities_per_m2: Sequence[float],
    areas_m2: Sequence[float],
) -> float:
    """Return the planet total of a per-area density, sharded across devices."""
    xp = backend.namespace()
    sharded_densities = backend.shard(backend.asarray(list(densities_per_m2)))
    sharded_areas = backend.shard(backend.asarray(list(areas_m2)))
    return float(xp.sum(sharded_densities * sharded_areas))
