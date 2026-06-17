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


def bench_pokemon(player: object | None) -> list[object]:
    if player is None:
        return []
    return list(getattr(player, "bench", []) or [])


def deck_count(player: object | None) -> int:
    if player is None:
        return 0
    return int(getattr(player, "deckCount", 0) or 0)


def prize_count(player: object | None) -> int:
    if player is None:
        return 0
    return len(getattr(player, "prize", []) or [])


def card_ids(cards: object | None) -> list[int]:
    if cards is None:
        return []
    ids = []
    for card in cards:
        if card is None:
            continue
        card_id = getattr(card, "id", None)
        if card_id is not None:
            ids.append(int(card_id))
    return ids


def count_card_id(cards: object | None, card_id: int) -> int:
    return card_ids(cards).count(card_id)


def attached_energy_count(pokemon: object | None) -> int:
    if pokemon is None:
        return 0
    return len(getattr(pokemon, "energyCards", []) or [])


def remaining_hp_ratio(pokemon: object | None) -> float:
    if pokemon is None:
        return 0.0
    hp = getattr(pokemon, "hp", 0) or 0
    max_hp = getattr(pokemon, "maxHp", 0) or 0
    if max_hp <= 0:
        return 0.0
    return hp / max_hp
