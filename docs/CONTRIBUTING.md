# Contributing

## Test-driven development: red → green

Work in small, verifiable steps. Each behavior is added as two commits:

1. **Red** — commit a failing test that specifies the behavior (`test: <behavior> (red)`).
2. **Green** — commit the implementation that makes it pass (`feat: <behavior> (green)`).

Keep the two commits separate so the history shows the test genuinely drove the code. Refactors
come after green, with tests staying green throughout.

## Coding conventions (enforced on commit)

These run via `pre-commit` locally and again in CI — identical gates on both:

- **Never nesting.** Prefer early returns; maximum nesting depth is capped (ruff `PLR1702`) and
  cyclomatic complexity is bounded (ruff `C901`). Keep functions short and flat.
- **Docstrings / good comments.** Public modules, classes, and functions carry docstrings
  (ruff `D`, Google style); coverage is gated by `interrogate`. Comment the *why*, not the *what*.
- **File length.** No source file exceeds 400 lines (`tools/hooks/check_file_length.py`); split
  instead of growing a file.
- **No duplication.** Copy-paste is rejected by `jscpd`. State each name/enum once (e.g. climate
  classes live in one registry and are referenced everywhere, including i18n keys).
- **Ports only in the core.** `core/` may not import adapters or third-party libraries directly
  (`tools/hooks/check_port_boundary.py`). Depend on a port; wire the concrete library at startup.
- **Types.** `core/`, `ports/`, `adapters/`, and `api/` are type-checked with `mypy --strict`.

## Swappable libraries (the 15-minute rule)

Any library must be replaceable with at most ~15 minutes of work. If you reach for a concrete
library inside the core, stop and add (or extend) a port instead. New adapters must pass the same
conformance tests as the existing ones for that port.

## Determinism

Randomness and time go through the `Rng` and `Clock` ports only — never `random`, `time`, or
wall-clock calls. A fixed seed + config must reproduce a bit-identical world on the same backend.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
pre-commit install
pytest
```
