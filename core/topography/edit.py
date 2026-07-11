"""Height-field editing: guardrails, preview, and a change log.

The third topography source: raise/lower cells via the API.  Every edit
is bounded (no "+1000 m sea level" in one call), previewable before
applying, and chronicled with who made it (player, script, or mod) so a
bad call can be traced and undone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.chronicle.log import Chronicle
    from ports.grid import CellId

MAX_DELTA_PER_CALL_M = 500.0
"""Default guardrail: largest per-cell change one call may make."""

EDIT_EVENT_KIND = "topography.edit"
"""Chronicle event kind recorded for every applied edit."""


class EditRejectedError(ValueError):
    """Raised when an edit violates a guardrail (nothing is applied)."""


@dataclass(frozen=True)
class EditRequest:
    """A proposed set of per-cell elevation changes, with provenance."""

    deltas_m: Mapping[CellId, float]
    editor: str
    reason: str = ""


@dataclass(frozen=True)
class EditPreview:
    """What an edit would do, without doing it."""

    cell_count: int
    max_abs_delta_m: float
    total_delta_m: float
    rejected: str | None = None
    changes: Mapping[CellId, tuple[float, float]] = field(default_factory=dict)
    """Per-cell (before, after) elevations."""


def _validate(
    heights: Mapping[CellId, float],
    request: EditRequest,
    max_delta_m: float,
) -> str | None:
    """Return the rejection reason for a request, or None if it is fine."""
    if not request.deltas_m:
        return "edit contains no cells"
    unknown = [cell for cell in request.deltas_m if cell not in heights]
    if unknown:
        return f"edit targets unknown cells: {unknown[:3]}"
    worst = max(abs(d) for d in request.deltas_m.values())
    if worst > max_delta_m:
        return (
            f"per-cell change {worst:.0f} m exceeds the {max_delta_m:.0f} m "
            "guardrail; split the edit into smaller calls"
        )
    return None


def preview_edit(
    heights: Mapping[CellId, float],
    request: EditRequest,
    max_delta_m: float = MAX_DELTA_PER_CALL_M,
) -> EditPreview:
    """Return what the edit would change (the preview-before-apply mode)."""
    rejected = _validate(heights, request, max_delta_m)
    if rejected is not None:
        return EditPreview(0, 0.0, 0.0, rejected=rejected)
    changes = {
        cell: (heights[cell], heights[cell] + delta) for cell, delta in request.deltas_m.items()
    }
    return EditPreview(
        cell_count=len(changes),
        max_abs_delta_m=max(abs(d) for d in request.deltas_m.values()),
        total_delta_m=sum(request.deltas_m.values()),
        changes=changes,
    )


def apply_edit(
    heights: Mapping[CellId, float],
    request: EditRequest,
    chronicle: Chronicle,
    tick: int,
    max_delta_m: float = MAX_DELTA_PER_CALL_M,
) -> dict[CellId, float]:
    """Return the edited height field, recording the change log entry.

    Raises :class:`EditRejectedError` (applying nothing) when a
    guardrail trips; otherwise appends one chronicle event per call
    naming the editor, so every terrain change is traceable.
    """
    rejected = _validate(heights, request, max_delta_m)
    if rejected is not None:
        raise EditRejectedError(rejected)
    edited = dict(heights)
    for cell, delta in request.deltas_m.items():
        edited[cell] += delta
    log_edit(chronicle, tick, request)
    return edited


def log_edit(chronicle: Chronicle, tick: int, request: EditRequest) -> None:
    """Append one chronicle event describing an applied terrain edit.

    Shared by :func:`apply_edit` and the ``EditableTopography`` source so
    every terrain change is traced the same way (editor, cell count, peak
    delta, reason) from a single place.
    """
    chronicle.append(
        tick=tick,
        kind=EDIT_EVENT_KIND,
        subject=request.editor,
        payload={
            "cells": len(request.deltas_m),
            "max_abs_delta_m": max(abs(d) for d in request.deltas_m.values()),
            "reason": request.reason,
        },
    )
