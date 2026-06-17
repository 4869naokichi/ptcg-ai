from __future__ import annotations

from functools import lru_cache

from cg.api import all_attack, all_card_data


@lru_cache(maxsize=1)
def cards_by_id() -> dict[int, object]:
    return {card.cardId: card for card in all_card_data()}


@lru_cache(maxsize=1)
def attacks_by_id() -> dict[int, object]:
    return {attack.attackId: attack for attack in all_attack()}


def card_by_id(card_id: int | None) -> object | None:
    if card_id is None:
        return None
    return cards_by_id().get(card_id)


def attack_by_id(attack_id: int | None) -> object | None:
    if attack_id is None:
        return None
    return attacks_by_id().get(attack_id)
