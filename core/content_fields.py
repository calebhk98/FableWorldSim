"""Typed extraction helpers for civilization content tables.

TOML content arrives as ``Mapping[str, object]``; these helpers cast fields
to the expected shapes.  Required fields raise ``KeyError`` so each loader
can wrap the error once, naming the offending content item.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def read_float(data: Mapping[str, object], key: str, default: float | None = None) -> float:
    """Return a numeric field as a float, falling back to the default when given."""
    if key not in data:
        if default is None:
            raise KeyError(key)
        return default
    return float(str(data[key]))


def read_int(data: Mapping[str, object], key: str, default: int | None = None) -> int:
    """Return an integer field, falling back to the default when given."""
    if key not in data:
        if default is None:
            raise KeyError(key)
        return default
    return int(str(data[key]))


def read_bool(data: Mapping[str, object], key: str, *, default: bool | None = None) -> bool:
    """Return a boolean field, falling back to the default when given."""
    if key not in data:
        if default is None:
            raise KeyError(key)
        return default
    return data[key] is True


def read_str(data: Mapping[str, object], key: str, default: str | None = None) -> str:
    """Return a string field, falling back to the default when given."""
    if key not in data:
        if default is None:
            raise KeyError(key)
        return default
    return str(data[key])


def read_str_tuple(data: Mapping[str, object], key: str) -> tuple[str, ...]:
    """Return a list-of-strings field as a tuple; an absent field is empty."""
    raw = data.get(key, ())
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        msg = f"field {key!r} must be a list of strings"
        raise ValueError(msg)
    return tuple(str(part) for part in raw)


def read_weight_table(data: Mapping[str, object], key: str) -> dict[str, float]:
    """Return a TOML table of numeric weights; an absent field is empty."""
    raw = data.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        msg = f"field {key!r} must be a table of numbers"
        raise ValueError(msg)
    return {str(name): float(str(value)) for name, value in raw.items()}
