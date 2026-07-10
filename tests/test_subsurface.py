"""Tests for the volumetric subsurface graph (depth bands under cells)."""

from __future__ import annotations

import pytest

from core.subsurface.depth_bands import BandSpec, DepthBandSubsurface
from tests.test_grid_port import FakeGrid

_BANDS = (
    BandSpec(thickness_m=10.0, rock_type="soil", diggability=0.9),
    BandSpec(thickness_m=100.0, rock_type="limestone", diggability=0.5),
    BandSpec(thickness_m=1_000.0, rock_type="basalt", diggability=0.1),
)
_MIDDLE_BAND = 1
_VERTICAL_AT_ENDS = 1
_VERTICAL_IN_MIDDLE = 2


def _volume() -> DepthBandSubsurface:
    """Return three depth bands under the four-cell fake grid."""
    return DepthBandSubsurface(FakeGrid(0, 1_000.0), _BANDS)


def test_every_cell_gets_every_band() -> None:
    """Node count is cells x bands, each mapping back to its cell/band."""
    volume = _volume()
    nodes = list(volume.nodes())
    assert len(nodes) == volume.surface_grid.cell_count * volume.band_count
    node = volume.node_at("c1", _MIDDLE_BAND)
    assert volume.surface_cell(node) == "c1"
    assert volume.depth_band(node) == _MIDDLE_BAND


def test_vertical_adjacency_is_distinct_from_lateral() -> None:
    """Top/bottom bands see one vertical neighbor, middles see two."""
    volume = _volume()
    top = volume.node_at("c0", 0)
    middle = volume.node_at("c0", 1)
    bottom = volume.node_at("c0", 2)
    assert len(volume.vertical_neighbors(top)) == _VERTICAL_AT_ENDS
    assert len(volume.vertical_neighbors(middle)) == _VERTICAL_IN_MIDDLE
    assert len(volume.vertical_neighbors(bottom)) == _VERTICAL_AT_ENDS
    assert volume.node_at("c0", 1) in volume.vertical_neighbors(top)


def test_lateral_adjacency_follows_surface_neighbors() -> None:
    """Lateral neighbors stay in the same band, under adjacent cells."""
    volume = _volume()
    node = volume.node_at("c0", _MIDDLE_BAND)
    lateral = volume.lateral_neighbors(node)
    assert len(lateral) == len(volume.surface_grid.neighbors("c0"))
    assert all(volume.depth_band(n) == _MIDDLE_BAND for n in lateral)
    assert set(volume.neighbors(node)) == set(volume.vertical_neighbors(node)) | set(lateral)


def test_rock_properties_and_volume() -> None:
    """Diggability/rock type come from the band; volume = area x depth."""
    volume = _volume()
    node = volume.node_at("c2", 2)
    assert volume.rock_type(node) == "basalt"
    assert volume.diggability(node) == pytest.approx(0.1)
    expected = volume.surface_grid.area_m2("c2") * 1_000.0
    assert volume.volume_m3(node) == pytest.approx(expected)


def test_validation_rejects_bad_bands() -> None:
    """Empty band lists and invalid band specs raise."""
    with pytest.raises(ValueError, match="at least one"):
        DepthBandSubsurface(FakeGrid(0, 1_000.0), ())
    with pytest.raises(ValueError, match="thickness"):
        BandSpec(thickness_m=0.0, rock_type="soil", diggability=0.5)
    with pytest.raises(ValueError, match="diggability"):
        BandSpec(thickness_m=1.0, rock_type="soil", diggability=2.0)
    with pytest.raises(ValueError, match="band"):
        _volume().node_at("c0", 99)
