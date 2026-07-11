"""Editable topography source: wraps a base source and applies raise/lower edits.

Provides a TopographySource that accumulates terrain edits with full guardrails:
per-call magnitude cap, preview-before-apply mode, and optional chronicle audit
trail.  Fits the same port as load (DEM) and generate (procedural).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.topography.edit import (
    EDIT_EVENT_KIND,
    MAX_DELTA_PER_CALL_M,
    EditRejectedError,
    EditRequest,
    _validate,
)
from ports.topography import TopographySource

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.chronicle.log import Chronicle
    from ports.grid import CellId, Grid


class EditableTopography(TopographySource):
    """A TopographySource that wraps a base source and applies accumulated edits.

    Combines the functionality of a base TopographySource with the ability to
    raise or lower individual cells via the edit port.  All edits are validated
    against the magnitude cap, previewable before applying, and optionally
    chronicled for audit trail.
    """

    def __init__(
        self,
        base: TopographySource,
        chronicle: Chronicle | None = None,
    ) -> None:
        """Wrap a base source with edit capability.

        Args:
            base: The underlying TopographySource (e.g., ProceduralTopography,
                LatLonGridDem).
            chronicle: Optional chronicle to log edits to. If None, edits are
                still applied but not recorded in any audit trail.
        """
        self._base = base
        self._chronicle = chronicle
        self._edits: dict[CellId, float] = {}

    def heights(self, grid: Grid) -> Mapping[CellId, float]:
        """Return elevations with all accumulated edits applied.

        The result combines the base source's heights with every delta
        from every successfully applied edit.
        """
        base_heights = self._base.heights(grid)
        if not self._edits:
            return base_heights
        return {cell: base_heights[cell] + self._edits.get(cell, 0.0) for cell in grid.cells()}

    def preview_edit(
        self,
        heights: Mapping[CellId, float],
        request: EditRequest,
        max_delta_m: float = MAX_DELTA_PER_CALL_M,
    ) -> Mapping[str, object]:
        """Show what an edit would do without applying it.

        Returns a dict with:
          - rejected: None if valid, error message if not
          - cell_count: number of cells to change
          - max_abs_delta_m: largest per-cell magnitude
          - total_delta_m: sum of all changes
          - changes: {cell_id: (before, after), ...} if valid

        Args:
            heights: Current height field (typically from heights(grid)).
            request: The proposed edit.
            max_delta_m: Magnitude guardrail (default 500 m).

        Returns:
            Dictionary with preview details (never raises, always reports
            rejection reason if the edit is invalid).
        """
        rejected = _validate(heights, request, max_delta_m)
        if rejected is not None:
            return {"rejected": rejected}
        changes = {
            cell: (heights[cell], heights[cell] + delta) for cell, delta in request.deltas_m.items()
        }
        return {
            "rejected": None,
            "cell_count": len(changes),
            "max_abs_delta_m": max(abs(d) for d in request.deltas_m.values()),
            "total_delta_m": sum(request.deltas_m.values()),
            "changes": changes,
        }

    def apply_edit(
        self,
        heights: Mapping[CellId, float],
        request: EditRequest,
        tick: int,
        max_delta_m: float = MAX_DELTA_PER_CALL_M,
    ) -> None:
        """Apply an edit to the accumulated deltas, with validation and logging.

        Modifies this source's internal state (accumulates the delta).
        If the edit is rejected (exceeds guardrails), EditRejectedError is
        raised and nothing is changed.

        Args:
            heights: Current height field (for validation only).
            request: The proposed edit.
            tick: Simulation tick (for chronicle logging).
            max_delta_m: Magnitude guardrail (default 500 m).

        Raises:
            EditRejectedError: If the edit violates a guardrail.
        """
        rejected = _validate(heights, request, max_delta_m)
        if rejected is not None:
            raise EditRejectedError(rejected)
        for cell, delta in request.deltas_m.items():
            self._edits[cell] = self._edits.get(cell, 0.0) + delta
        if self._chronicle is not None:
            self._chronicle.append(
                tick=tick,
                kind=EDIT_EVENT_KIND,
                subject=request.editor,
                payload={
                    "cells": len(request.deltas_m),
                    "max_abs_delta_m": max(abs(d) for d in request.deltas_m.values()),
                    "reason": request.reason,
                },
            )
