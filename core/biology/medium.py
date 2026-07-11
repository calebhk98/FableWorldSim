"""Habitat medium: the categorical gate applied before the trait bands.

After hydrology every cell is land, water, or intertidal, and every
organism declares the **medium** it lives in — terrestrial, aquatic,
amphibious, aerial, intertidal, or subterranean.  A seaweed is suitable
only over water, a land animal only over land: a hard yes/no gate applied
*before* the graded comfort/tolerance bands, so aquatic and flying life
are first-class rather than an extreme trait value bolted on later.

Subterranean organisms live in the ``SubsurfaceGrid`` domain and are gated
there, so on the surface shell this gate rejects them everywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.hydrology.sea_mask import SeaMask
    from ports.grid import CellId

TERRESTRIAL = "terrestrial"
AQUATIC = "aquatic"
AMPHIBIOUS = "amphibious"
AERIAL = "aerial"
INTERTIDAL = "intertidal"
SUBTERRANEAN = "subterranean"

MEDIA = (TERRESTRIAL, AQUATIC, AMPHIBIOUS, AERIAL, INTERTIDAL, SUBTERRANEAN)
"""Every declarable medium; a species' medium must be one of these."""

SURFACE_MEDIA = (TERRESTRIAL, AQUATIC, AMPHIBIOUS, AERIAL, INTERTIDAL)
"""Media that occupy the surface grid; subterranean uses the volume graph."""

# Which surface classifications each medium may occupy.  Aerial life flies
# over land and water alike, which is what distinguishes it from a
# land-locked terrestrial species in migration even where the gate agrees.
_LAND_MEDIA = frozenset({TERRESTRIAL, AMPHIBIOUS, AERIAL})
_WATER_MEDIA = frozenset({AQUATIC, AMPHIBIOUS, AERIAL})


def is_surface_medium(medium: str) -> bool:
    """Return whether the medium lives on the surface grid (not the volume)."""
    return medium in SURFACE_MEDIA


def medium_allows(medium: str, cell: CellId, sea_mask: SeaMask) -> bool:
    """Return whether a surface ``medium`` may occupy a surface ``cell``.

    Subterranean always returns ``False`` here; those organisms are gated
    on the ``SubsurfaceGrid`` instead.
    """
    if medium == INTERTIDAL:
        return sea_mask.intertidal[cell]
    if sea_mask.ocean[cell]:
        return medium in _WATER_MEDIA
    return medium in _LAND_MEDIA
