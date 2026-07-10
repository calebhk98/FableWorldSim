"""Topography port: one interface, three height-field sources.

The height field (meters relative to the datum; sea level is solved
separately from the target water fraction) can come from **loading** a
real DEM (Earth/Mars/Moon/Venus), **generating** procedurally, or
**editing** an existing field.  All three live behind this port so the
rest of the sim never knows where terrain came from.

Image->heightmap (grayscale import first, ML inference later) is a
future adapter behind this same port — deferred, not designed out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports.grid import CellId, Grid


class TopographySource(ABC):
    """Produces a height field over a grid."""

    @abstractmethod
    def heights(self, grid: Grid) -> Mapping[CellId, float]:
        """Return elevation in meters (datum-relative) for every cell."""
