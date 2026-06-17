from __future__ import annotations

from typing import Protocol


class Policy(Protocol):
    """Select legal option indexes for a non-initial observation."""

    def select(self, obs: object) -> list[int]:
        """Return option indexes accepted by the simulator."""
