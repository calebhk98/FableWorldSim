"""Open access adapter: the single-player default (everything allowed).

Online play and mod sandboxing swap in a real policy behind the same
port; nothing in the core changes.
"""

from __future__ import annotations

from ports.access import Access, Principal


class OpenAccess(Access):
    """Allow-all policy for local single-player worlds."""

    def can_read(self, principal: Principal, resource: str) -> bool:
        """Every principal may read everything."""
        return True

    def can_write(self, principal: Principal, resource: str) -> bool:
        """Every principal may change everything."""
        return True


def create() -> Access:
    """Create an :class:`OpenAccess`; wiring entry point."""
    return OpenAccess()
