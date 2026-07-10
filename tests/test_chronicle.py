"""Tests for the chronicle (append-only history + causal trace)."""

from __future__ import annotations

import pytest

from core.chronicle.log import Chronicle


def _sample() -> Chronicle:
    chronicle = Chronicle()
    drought = chronicle.append(10, "climate.drought", "cell-42")
    dieoff = chronicle.append(11, "biology.dieoff", "grass", {"loss": 0.8}, caused_by=drought.seq)
    chronicle.append(12, "biology.migration", "deer", caused_by=dieoff.seq)
    chronicle.append(12, "civ.border_shift", "kingdom-a")
    return chronicle


def test_events_are_appended_in_order_with_sequence_numbers() -> None:
    chronicle = _sample()
    assert len(chronicle) == 4
    assert [e.seq for e in chronicle.events()] == [0, 1, 2, 3]


def test_query_filters_by_kind_subject_and_time() -> None:
    chronicle = _sample()
    assert [e.kind for e in chronicle.events(kind="biology.dieoff")] == ["biology.dieoff"]
    assert [e.subject for e in chronicle.events(subject="deer")] == ["deer"]
    assert all(e.tick >= 12 for e in chronicle.events(since_tick=12))


def test_causal_trace_walks_the_chain_oldest_first() -> None:
    chronicle = _sample()
    chain = chronicle.trace(2)
    assert [e.kind for e in chain] == [
        "climate.drought",
        "biology.dieoff",
        "biology.migration",
    ]


def test_invalid_references_fail_loudly() -> None:
    chronicle = _sample()
    with pytest.raises(KeyError, match="seq"):
        chronicle.get(99)
    with pytest.raises(ValueError, match="caused_by"):
        chronicle.append(13, "x", "y", caused_by=99)
