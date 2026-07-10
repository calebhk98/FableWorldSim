"""Access port: who may read vs. change world state.

Introduced now — before online play and mods exist — so access control
is not an afterthought.  Mod code runs outside the core and reaches
world state only through interfaces guarded by this port; details live
in the platform/scaling document.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    """An actor (player, mod, service) whose access is being checked."""

    principal_id: str
    roles: tuple[str, ...] = ()


class Access(ABC):
    """Policy deciding reads and writes per principal and resource.

    Resources are dotted paths (e.g. ``"world.topography"``,
    ``"world.civilization.borders"``) so policies can scope by subtree.
    """

    @abstractmethod
    def can_read(self, principal: Principal, resource: str) -> bool:
        """Return whether ``principal`` may read ``resource``."""

    @abstractmethod
    def can_write(self, principal: Principal, resource: str) -> bool:
        """Return whether ``principal`` may change ``resource``."""
