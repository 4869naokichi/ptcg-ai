from __future__ import annotations

from ptcg_ai.game.constants import (
    BASIC_GRASS_ENERGY,
    CRUSTLE,
    DWEBBLE,
    MAXIMUM_BELT,
)


CRUSTLE_DECK: list[int] = [
    MAXIMUM_BELT,
    *([DWEBBLE] * 4),
    *([CRUSTLE] * 4),
    *([BASIC_GRASS_ENERGY] * 51),
]
