"""The chronicle: an append-only record of state-changing events.

Every migration, border shift, river avulsion, extinction, or manual
edit appends a structured event.  The chronicle *is* the project's
history layer — time-series probes, the in-world wiki, and the causal
trace ("why did this cell become desert?") all read from it, and
retrofitting it later would mean rebuilding lost history.

M1 ships append + query + a direct-chain causal trace; full multi-parent
lineage is roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


@dataclass(frozen=True)
class ChronicleEvent:
    """One state-changing event.

    ``subject`` names what changed (a cell id, settlement id, species
    id...); ``caused_by`` is the sequence number of the event that
    triggered this one, forming the causal chain.
    """

    seq: int
    tick: int
    kind: str
    subject: str
    payload: Mapping[str, object] = field(default_factory=dict)
    caused_by: int | None = None


class Chronicle:
    """Append-only event log with query and causal tracing."""

    def __init__(self) -> None:
        """Start with an empty history."""
        self._events: list[ChronicleEvent] = []

    def append(
        self,
        tick: int,
        kind: str,
        subject: str,
        payload: Mapping[str, object] | None = None,
        caused_by: int | None = None,
    ) -> ChronicleEvent:
        """Record an event and return it (seq is assigned here)."""
        if caused_by is not None and not 0 <= caused_by < len(self._events):
            msg = f"caused_by {caused_by} references a nonexistent event"
            raise ValueError(msg)
        event = ChronicleEvent(
            seq=len(self._events),
            tick=tick,
            kind=kind,
            subject=subject,
            payload=dict(payload or {}),
            caused_by=caused_by,
        )
        self._events.append(event)
        return event

    def __len__(self) -> int:
        """Return how many events have been recorded."""
        return len(self._events)

    def events(
        self,
        kind: str | None = None,
        subject: str | None = None,
        since_tick: int = 0,
    ) -> Iterator[ChronicleEvent]:
        """Iterate events, optionally filtered by kind/subject/time."""
        for event in self._events:
            if kind is not None and event.kind != kind:
                continue
            if subject is not None and event.subject != subject:
                continue
            if event.tick < since_tick:
                continue
            yield event

    def get(self, seq: int) -> ChronicleEvent:
        """Return one event by sequence number."""
        if not 0 <= seq < len(self._events):
            msg = f"no chronicle event with seq {seq}"
            raise KeyError(msg)
        return self._events[seq]

    def trace(self, seq: int) -> tuple[ChronicleEvent, ...]:
        """Return the causal chain ending at ``seq`` (oldest first).

        M1 walks the direct ``caused_by`` chain; full multi-parent
        lineage ("all contributing causes") is roadmap.
        """
        chain: list[ChronicleEvent] = []
        current: int | None = seq
        while current is not None:
            event = self.get(current)
            chain.append(event)
            current = event.caused_by
        return tuple(reversed(chain))
