"""Localizer port: every user-facing string is a locale key.

Domain names (climate classes, biomes, species...) are defined once as
content ids; display text lives in ``content/locales/<lang>.ftl`` and is
looked up through this port.  Adding a language = adding a locale file.
The default adapter is Fluent; gettext/ICU could slot in behind the same
interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class Localizer(ABC):
    """Translates message keys into localized strings."""

    @abstractmethod
    def available_locales(self) -> tuple[str, ...]:
        """Return the sorted locale codes with loaded resources."""

    @abstractmethod
    def translate(
        self,
        key: str,
        args: Mapping[str, object] | None = None,
        locale: str | None = None,
    ) -> str:
        """Return the localized text for ``key``.

        Falls back to the default locale when the requested locale lacks
        the key, and to the key itself when no locale has it (a visible,
        greppable marker rather than a crash).
        """
