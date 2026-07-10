"""Role-based access adapter: read tier for all, write tier by role.

M1's two-tier split: commands that only read world state versus commands
that change it.  Any connected principal may read; only principals
holding an editor/admin role may write.  Online login and request
throttling build on this same port when multiplayer lands.
"""

from __future__ import annotations

from ports.access import Access, Principal

WRITE_ROLES = frozenset({"editor", "admin"})
"""Roles granted the mutating command tier."""


class RoleBasedAccess(Access):
    """Everyone reads; editor/admin write."""

    def can_read(self, principal: Principal, resource: str) -> bool:
        """Return True: the read tier is open to every connection."""
        return True

    def can_write(self, principal: Principal, resource: str) -> bool:
        """Return whether the principal holds a write-tier role."""
        return bool(WRITE_ROLES.intersection(principal.roles))


def create() -> Access:
    """Create a :class:`RoleBasedAccess`; wiring entry point."""
    return RoleBasedAccess()
