"""Height field: load real DEMs, generate procedurally, or edit.

Three sources behind the TopographySource port:
- load: LatLonGridDem, GeoTiffDem (adapters)
- generate: ProceduralTopography
- edit: EditableTopography (wraps any source with editable deltas)
"""

from core.topography.editable import EditableTopography
from core.topography.procedural import ProceduralTopography

__all__ = ["EditableTopography", "ProceduralTopography"]
