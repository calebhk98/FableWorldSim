"""Single-source-of-truth check (pre-commit hook).

Names/enums (climate classes, biomes, species, ...) are defined once in
``content/`` registries; display text lives in locale files that
reference those ids.  This hook enforces both directions:

1. Every content item ``content/<kind>/<id>.toml`` must have a Fluent
   key ``<kind>-<id>`` in the reference locale (``en.ftl``), so ids are
   translated rather than shown raw.
2. No content item id may appear as a quoted string literal in domain
   code (``core/`` by default): the domain looks ids up through the
   ContentRegistry port; hard-coding an id duplicates the registry.

Pre-commit passes staged file names as positional arguments; they are
accepted and ignored because both rules are global properties of the
tree, not per-file ones.

Usage:
    python tools/hooks/check_single_source.py [files...]
        [--content-dir content] [--locales-dir content/locales]
        [--reference-locale en] [--code-dir core]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_KEY_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\s*=")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI options (positional file args are accepted and ignored)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="ignored; hook checks globally")
    parser.add_argument("--content-dir", default="content", type=Path)
    parser.add_argument("--locales-dir", default=None, type=Path)
    parser.add_argument("--reference-locale", default="en")
    parser.add_argument(
        "--code-dir",
        action="append",
        type=Path,
        default=None,
        help="directory forbidden from hard-coding content ids (repeatable)",
    )
    return parser.parse_args(argv)


def content_items(content_dir: Path) -> list[tuple[str, str]]:
    """Return (kind, item_id) for every content item file."""
    if not content_dir.is_dir():
        return []
    items: list[tuple[str, str]] = []
    for kind_dir in sorted(p for p in content_dir.iterdir() if p.is_dir()):
        if kind_dir.name == "locales":
            continue
        items.extend((kind_dir.name, f.stem) for f in sorted(kind_dir.glob("*.toml")))
    return items


def fluent_keys(ftl_path: Path) -> set[str]:
    """Return the message keys defined in one Fluent file."""
    if not ftl_path.is_file():
        return set()
    keys: set[str] = set()
    for line in ftl_path.read_text(encoding="utf-8").splitlines():
        match = _KEY_PATTERN.match(line)
        if match:
            keys.add(match.group(1))
    return keys


def missing_locale_keys(items: list[tuple[str, str]], keys: set[str]) -> list[str]:
    """Return the required ``<kind>-<id>`` keys absent from the locale."""
    return [f"{kind}-{item}" for kind, item in items if f"{kind}-{item}" not in keys]


def hardcoded_ids(items: list[tuple[str, str]], code_dirs: list[Path]) -> list[str]:
    """Return 'file: id' entries where domain code quotes a content id."""
    findings: list[str] = []
    for code_dir in code_dirs:
        for py_file in sorted(code_dir.rglob("*.py")):
            text = py_file.read_text(encoding="utf-8")
            findings.extend(
                f"{py_file}: {item!r}"
                for _, item in items
                if f'"{item}"' in text or f"'{item}'" in text
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run both rules; return a non-zero exit code on violations."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    locales_dir = args.locales_dir or args.content_dir / "locales"
    code_dirs = args.code_dir or [Path("core")]

    items = content_items(args.content_dir)
    problems: list[str] = []
    reference = locales_dir / f"{args.reference_locale}.ftl"
    problems.extend(
        f"content id has no {args.reference_locale} locale key: {key} (add to {reference})"
        for key in missing_locale_keys(items, fluent_keys(reference))
    )
    problems.extend(
        f"content id hard-coded in domain code (use ContentRegistry): {finding}"
        for finding in hardcoded_ids(items, code_dirs)
    )
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
