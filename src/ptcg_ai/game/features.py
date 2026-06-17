from __future__ import annotations


def your_player(obs: object) -> object | None:
    current = getattr(obs, "current", None)
    if current is None:
        return None
    return getattr(current, "players")[getattr(current, "yourIndex")]


def opponent_player(obs: object) -> object | None:
    current = getattr(obs, "current", None)
    if current is None:
        return None
    return getattr(current, "players")[1 - getattr(current, "yourIndex")]


def active_pokemon(player: object | None) -> object | None:
    if player is None:
        return None
    active = getattr(player, "active", [])
    if not active:
        return None
    return active[0]


def remaining_hp_ratio(pokemon: object | None) -> float:
    if pokemon is None:
        return 0.0
    hp = getattr(pokemon, "hp", 0) or 0
    max_hp = getattr(pokemon, "maxHp", 0) or 0
    if max_hp <= 0:
        return 0.0
    return hp / max_hp
