from __future__ import annotations

import random


class RandomPolicy:
    """Baseline policy that samples a legal action uniformly."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, obs: object) -> list[int]:
        select = getattr(obs, "select")
        if select is None:
            raise ValueError("RandomPolicy cannot choose the initial deck.")

        count = getattr(select, "maxCount")
        options = list(range(len(getattr(select, "option"))))
        return self._rng.sample(options, count)
