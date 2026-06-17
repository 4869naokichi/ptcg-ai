from __future__ import annotations

from ptcg_ai.game.constants import (
    BASIC_GRASS_ENERGY,
    BOSS_ORDERS,
    BUDDY_BUDDY_POFFIN,
    CARMINE,
    CRUSTLE,
    DWEBBLE,
    LILLIES_DETERMINATION,
    MAXIMUM_BELT,
    NIGHT_STRETCHER,
    SWITCH,
    ULTRA_BALL,
    WAITRESS,
)


# Crustle is an anti-ex wall (Mysterious Rock Inn prevents all damage from ex
# Pokémon) that swings for 120 with Superb Scissors (1 Grass + 2 colorless).
# The old list was 51 energy and 9 other cards, so it could never find a second
# Pokémon. This build keeps the simple Dwebble -> Crustle plan the agent already
# pilots well, and spends the freed slots on consistency the rule policy can use:
# Pokémon search, draw, energy acceleration (Waitress), gusting, and mobility.
CRUSTLE_DECK: list[int] = [
    # Pokémon line (8)
    *([DWEBBLE] * 4),
    *([CRUSTLE] * 4),
    # Pokémon search / setup (8)
    *([BUDDY_BUDDY_POFFIN] * 4),
    *([ULTRA_BALL] * 4),
    # Draw + energy acceleration + disruption (14)
    *([LILLIES_DETERMINATION] * 4),
    *([WAITRESS] * 4),
    *([BOSS_ORDERS] * 3),
    *([CARMINE] * 3),
    # Mobility / recovery (5)
    *([SWITCH] * 3),
    *([NIGHT_STRETCHER] * 2),
    # ACE SPEC tool (1) — boosts damage against ex
    MAXIMUM_BELT,
    # Energy (24)
    *([BASIC_GRASS_ENERGY] * 24),
]
