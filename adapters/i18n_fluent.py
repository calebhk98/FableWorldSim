"""Fluent i18n adapter: one FluentBundle per ``<lang>.ftl`` locale file.

Layout: ``content/locales/en.ftl``, ``es.ftl``, ``ja.ftl``, ... — adding
a language is adding a file.  Typed via ``adapters/stubs/fluent/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fluent.runtime import FluentBundle, FluentResource

from ports.i18n import Localizer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class FluentLocalizer(Localizer):
    """Localizer over a directory of per-locale Fluent files."""

    def __init__(self, locales_dir: Path, default_locale: str = "en") -> None:
        """Load every ``*.ftl`` in ``locales_dir`` eagerly."""
        self._bundles: dict[str, FluentBundle] = {}
        for path in sorted(locales_dir.glob("*.ftl")):
            bundle = FluentBundle([path.stem], use_isolating=False)
            bundle.add_resource(FluentResource(path.read_text(encoding="utf-8")))
            self._bundles[path.stem] = bundle
        if default_locale not in self._bundles:
            found = ", ".join(sorted(self._bundles)) or "none"
            msg = f"default locale {default_locale!r} not found in {locales_dir} (found: {found})"
            raise ValueError(msg)
        self._default_locale = default_locale

    @property
    def default_locale(self) -> str:
        """Return the fallback locale code."""
        return self._default_locale

    def available_locales(self) -> tuple[str, ...]:
        """Return the sorted locale codes with loaded resources."""
        return tuple(sorted(self._bundles))

    def translate(
        self,
        key: str,
        args: Mapping[str, object] | None = None,
        locale: str | None = None,
    ) -> str:
        """Return localized text, falling back default-locale -> key."""
        for code in (locale or self._default_locale, self._default_locale):
            bundle = self._bundles.get(code)
            if bundle is not None and bundle.has_message(key):
                return self._format(bundle, key, args)
        return key

    def _format(self, bundle: FluentBundle, key: str, args: Mapping[str, object] | None) -> str:
        """Format one message pattern, ignoring recoverable errors."""
        message = bundle.get_message(key)
        value, _errors = bundle.format_pattern(message.value, dict(args or {}))
        return str(value)


def create(locales_dir: Path, default_locale: str = "en") -> Localizer:
    """Create a :class:`FluentLocalizer`; wiring entry point."""
    return FluentLocalizer(locales_dir, default_locale)
