from __future__ import annotations

from dataclasses import dataclass, field

from ptcg_ai.game.constants import (
    BASIC_WATER_ENERGY,
    CYRANO,
    KYOGRE,
    LILLIES_DETERMINATION,
    MAXIMUM_BELT,
    MEGA_ABOMASNOW_EX,
    MEGA_SIGNAL,
    SNOVER,
    WAITRESS,
)


@dataclass(frozen=True)
class ScoringWeights:
    """Rule weights kept separate so later tuning can swap them out."""

    key_cards: dict[int, float] = field(
        default_factory=lambda: {
            MEGA_ABOMASNOW_EX: 520.0,
            KYOGRE: 430.0,
            SNOVER: 390.0,
            MEGA_SIGNAL: 330.0,
            WAITRESS: 310.0,
            LILLIES_DETERMINATION: 280.0,
            CYRANO: 230.0,
            MAXIMUM_BELT: 210.0,
            BASIC_WATER_ENERGY: 160.0,
        }
    )
    play_card: float = 220.0
    attach_energy: float = 260.0
    evolve: float = 460.0
    attack: float = 520.0
    ability: float = 320.0
    retreat: float = 90.0
    end_turn: float = 0.0


def card_priority(card_id: int | None, weights: ScoringWeights) -> float:
    if card_id is None:
        return 0.0
    return weights.key_cards.get(card_id, 40.0)


def discard_priority(card_id: int | None, weights: ScoringWeights) -> float:
    """Higher means more acceptable to discard."""

    if card_id is None:
        return 0.0
    if card_id == BASIC_WATER_ENERGY:
        return 180.0
    return 120.0 - card_priority(card_id, weights)
