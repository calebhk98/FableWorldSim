"""Pure-math tests for the ISEA adapter's equal-area geometry helpers.

These exercise the spherical-polygon-area and great-circle-distance
computations the real :class:`adapters.grid_isea.IseaGrid` relies on for
its equal-area invariant, without needing the external DGGRID binary —
only the ``dggrid4py`` package (for the module import) has to be
installed. Real end-to-end ISEA coverage (which does need the binary)
lives in ``test_grid_backends.py`` and ``test_conservation.py``.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("dggrid4py")

from adapters.grid_isea import _haversine_m, _ring_area_m2
from ports.grid import LatLon

_RADIUS_M = 6_371_008.8


def test_ring_area_matches_exact_graticule_cell_area() -> None:
    """A lon/lat rectangle's computed area matches the closed-form formula.

    For a cell bounded by two meridians and two parallels the exact area
    is ``R^2 * dlon_rad * (sin(lat2) - sin(lat1))``; this is the textbook
    check for any spherical-polygon-area implementation.
    """
    lon1, lon2, lat1, lat2 = 10.0, 40.0, 5.0, 25.0
    ring = [(lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2), (lon1, lat1)]
    area = _ring_area_m2(ring, _RADIUS_M)
    expected = (
        _RADIUS_M**2
        * math.radians(lon2 - lon1)
        * (math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)))
    )
    assert area == pytest.approx(expected, rel=1e-12)


def test_ring_area_is_additive_across_a_shared_edge() -> None:
    """Splitting a cell in two and summing the halves reproduces the whole area.

    This is the invariant area-weighted aggregation depends on (splitting
    a cell into smaller cells must not change a conserved total, per
    ``core/grid/area_weighted.py``), checked directly against the ISEA
    adapter's own area helper rather than assumed.
    """
    lon1, lon_mid, lon2, lat1, lat2 = 10.0, 25.0, 40.0, 5.0, 25.0
    whole = [(lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2), (lon1, lat1)]
    left = [(lon1, lat1), (lon_mid, lat1), (lon_mid, lat2), (lon1, lat2), (lon1, lat1)]
    right = [(lon_mid, lat1), (lon2, lat1), (lon2, lat2), (lon_mid, lat2), (lon_mid, lat1)]

    whole_area = _ring_area_m2(whole, _RADIUS_M)
    split_area = _ring_area_m2(left, _RADIUS_M) + _ring_area_m2(right, _RADIUS_M)
    assert split_area == pytest.approx(whole_area, rel=1e-9)


def test_ring_area_scales_with_radius_squared() -> None:
    """Doubling the planet radius quadruples a cell's area (area ~ r^2)."""
    ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    small = _ring_area_m2(ring, _RADIUS_M)
    large = _ring_area_m2(ring, 2.0 * _RADIUS_M)
    assert large == pytest.approx(4.0 * small, rel=1e-12)


def test_haversine_quarter_great_circle() -> None:
    """A 90-degree arc along the equator is a quarter of the circumference."""
    start = LatLon(lat_deg=0.0, lon_deg=0.0)
    end = LatLon(lat_deg=0.0, lon_deg=90.0)
    distance_m = _haversine_m(start, end, _RADIUS_M)
    assert distance_m == pytest.approx(math.pi / 2 * _RADIUS_M, rel=1e-9)
