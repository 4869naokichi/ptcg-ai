from __future__ import annotations

from collections.abc import Iterable

from cg.api import AreaType, OptionType, SelectContext

from ptcg_ai.agent.policy import Policy
from ptcg_ai.agent.scoring import (
    ScoringWeights,
    card_priority,
    discard_priority,
)
from ptcg_ai.game.card_db import attack_by_id
from ptcg_ai.game.constants import (
    BASIC_WATER_ENERGY,
    KYOGRE,
    MEGA_ABOMASNOW_EX,
    SNOVER,
)


def _enum_value(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is(value: object, enum_member: object) -> bool:
    return _enum_value(value) == int(enum_member)


class RuleBasedPolicy(Policy):
    """Small score-based policy for the current water deck."""

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def select(self, obs: object) -> list[int]:
        select = getattr(obs, "select")
        if select is None:
            raise ValueError("RuleBasedPolicy cannot choose the initial deck.")

        options = list(getattr(select, "option"))
        if not options or getattr(select, "maxCount") == 0:
            return []

        scores = [self.score_option(obs, option) for option in options]
        target_count = self._target_count(select, scores)
        if target_count <= 0:
            return []

        ranked_indexes = sorted(
            range(len(options)),
            key=lambda index: (scores[index], -index),
            reverse=True,
        )
        return ranked_indexes[:target_count]

    def score_option(self, obs: object, option: object) -> float:
        option_type = getattr(option, "type", None)
        context = getattr(getattr(obs, "select"), "context", None)

        if _is(option_type, OptionType.YES):
            return self._score_yes(context)
        if _is(option_type, OptionType.NO):
            return self._score_no(context)
        if _is(option_type, OptionType.NUMBER):
            return float(getattr(option, "number", 0) or 0)
        if _is(option_type, OptionType.PLAY):
            return self.weights.play_card + self._score_hand_card(obs, option)
        if _is(option_type, OptionType.ATTACH):
            return self.weights.attach_energy + self._score_attach(obs, option)
        if _is(option_type, OptionType.EVOLVE):
            return self.weights.evolve + self._score_evolve(obs, option)
        if _is(option_type, OptionType.ABILITY):
            return self.weights.ability + self._score_located_card(obs, option)
        if _is(option_type, OptionType.ATTACK):
            return self.weights.attack + self._score_attack(option)
        if _is(option_type, OptionType.RETREAT):
            return self._score_retreat(obs)
        if _is(option_type, OptionType.END):
            return self.weights.end_turn
        if _is(option_type, OptionType.DISCARD):
            return 30.0
        if _is(option_type, OptionType.CARD):
            return self._score_card_selection(obs, option, context)
        if _is(option_type, OptionType.TOOL_CARD):
            return self._score_card_selection(obs, option, context)
        if _is(option_type, OptionType.ENERGY_CARD):
            return self._score_card_selection(obs, option, context)
        if _is(option_type, OptionType.ENERGY):
            return 120.0 + float(getattr(option, "count", 0) or 0)
        if _is(option_type, OptionType.SKILL):
            return card_priority(getattr(option, "cardId", None), self.weights)
        if _is(option_type, OptionType.SPECIAL_CONDITION):
            return 10.0

        return 0.0

    def _target_count(self, select: object, scores: list[float]) -> int:
        min_count = int(getattr(select, "minCount"))
        max_count = int(getattr(select, "maxCount"))
        if min_count == max_count:
            return min_count

        context = getattr(select, "context", None)
        if _is(context, SelectContext.SETUP_BENCH_POKEMON):
            return min(max_count, len([score for score in scores if score > 0.0]))

        if min_count == 0:
            positive_count = len([score for score in scores if score > 0.0])
            return min(max_count, positive_count)

        return max_count

    def _score_yes(self, context: object) -> float:
        if _is(context, SelectContext.IS_FIRST):
            return -20.0
        if _is(context, SelectContext.MULLIGAN):
            return 100.0
        if _is(context, SelectContext.ACTIVATE):
            return 95.0
        if _is(context, SelectContext.COIN_HEAD):
            return 50.0
        return 20.0

    def _score_no(self, context: object) -> float:
        if _is(context, SelectContext.IS_FIRST):
            return 80.0
        if _is(context, SelectContext.MULLIGAN):
            return -50.0
        if _is(context, SelectContext.ACTIVATE):
            return -10.0
        return 0.0

    def _score_attack(self, option: object) -> float:
        attack_id = getattr(option, "attackId", None)
        if attack_id == 1046:
            return 380.0
        if attack_id == 1042:
            return 260.0

        attack = attack_by_id(attack_id)
        if attack is None:
            return 0.0
        return float(getattr(attack, "damage", 0) or 0)

    def _score_hand_card(self, obs: object, option: object) -> float:
        card_id = self._hand_card_id(obs, getattr(option, "index", None))
        return card_priority(card_id, self.weights)

    def _score_attach(self, obs: object, option: object) -> float:
        source_id = self._card_id_from_area(
            obs,
            getattr(option, "area", None),
            getattr(option, "index", None),
            getattr(option, "playerIndex", None),
        )
        target_id = self._card_id_from_area(
            obs,
            getattr(option, "inPlayArea", None),
            getattr(option, "inPlayIndex", None),
            getattr(option, "playerIndex", None),
        )
        score = card_priority(target_id, self.weights)
        if source_id == BASIC_WATER_ENERGY:
            score += 160.0
        return score

    def _score_evolve(self, obs: object, option: object) -> float:
        evolved_id = self._card_id_from_area(
            obs,
            getattr(option, "area", None),
            getattr(option, "index", None),
            getattr(option, "playerIndex", None),
        )
        target_id = self._card_id_from_area(
            obs,
            getattr(option, "inPlayArea", None),
            getattr(option, "inPlayIndex", None),
            getattr(option, "playerIndex", None),
        )
        return card_priority(evolved_id, self.weights) + card_priority(target_id, self.weights)

    def _score_retreat(self, obs: object) -> float:
        active = self._your_active(obs)
        if active is None:
            return 0.0

        hp = getattr(active, "hp", 0) or 0
        max_hp = getattr(active, "maxHp", 1) or 1
        if hp * 2 <= max_hp:
            return self.weights.retreat + 120.0
        return self.weights.retreat

    def _score_located_card(self, obs: object, option: object) -> float:
        card_id = self._card_id_from_area(
            obs,
            getattr(option, "area", None),
            getattr(option, "index", None),
            getattr(option, "playerIndex", None),
        )
        return card_priority(card_id, self.weights)

    def _score_card_selection(self, obs: object, option: object, context: object) -> float:
        card_id = self._option_card_id(obs, option)

        if _is(context, SelectContext.SETUP_ACTIVE_POKEMON):
            if card_id == KYOGRE:
                return 1000.0
            if card_id == SNOVER:
                return 760.0
            return card_priority(card_id, self.weights)

        if _is(context, SelectContext.SETUP_BENCH_POKEMON):
            if card_id == SNOVER:
                return 900.0
            if card_id == KYOGRE:
                return 700.0
            return card_priority(card_id, self.weights)

        if _is(context, SelectContext.EVOLVES_TO):
            if card_id == MEGA_ABOMASNOW_EX:
                return 1000.0
            return card_priority(card_id, self.weights)

        if _is(context, SelectContext.EVOLVES_FROM):
            if card_id == SNOVER:
                return 920.0
            return card_priority(card_id, self.weights)

        if _is(context, SelectContext.TO_HAND):
            return 340.0 + card_priority(card_id, self.weights)

        if _is(context, SelectContext.TO_FIELD):
            return 250.0 + card_priority(card_id, self.weights)

        if _is(context, SelectContext.TO_BENCH):
            return 240.0 + card_priority(card_id, self.weights)

        if _is(context, SelectContext.ATTACH_TO):
            return 220.0 + card_priority(card_id, self.weights)

        if _is(context, SelectContext.DISCARD):
            return discard_priority(card_id, self.weights)

        return card_priority(card_id, self.weights)

    def _option_card_id(self, obs: object, option: object) -> int | None:
        explicit_id = getattr(option, "cardId", None)
        if explicit_id is not None:
            return explicit_id

        return self._card_id_from_area(
            obs,
            getattr(option, "area", None),
            getattr(option, "index", None),
            getattr(option, "playerIndex", None),
        )

    def _hand_card_id(self, obs: object, index: int | None) -> int | None:
        if index is None:
            return None
        current = getattr(obs, "current", None)
        if current is None:
            return None
        your_index = getattr(current, "yourIndex")
        player = getattr(current, "players")[your_index]
        hand = getattr(player, "hand", None)
        if hand is None or index >= len(hand):
            return None
        return getattr(hand[index], "id", None)

    def _card_id_from_area(
        self,
        obs: object,
        area: object,
        index: int | None,
        player_index: int | None,
    ) -> int | None:
        if index is None:
            return None

        select = getattr(obs, "select", None)
        if select is not None and _is(area, AreaType.DECK):
            deck = getattr(select, "deck", None)
            return self._id_at(deck, index)

        current = getattr(obs, "current", None)
        if current is None:
            return None

        if player_index is None:
            player_index = getattr(current, "yourIndex")

        players = getattr(current, "players")
        if player_index >= len(players):
            return None
        player = players[player_index]

        if _is(area, AreaType.HAND):
            return self._id_at(getattr(player, "hand", None), index)
        if _is(area, AreaType.DISCARD):
            return self._id_at(getattr(player, "discard", None), index)
        if _is(area, AreaType.ACTIVE):
            return self._id_at(getattr(player, "active", None), index)
        if _is(area, AreaType.BENCH):
            return self._id_at(getattr(player, "bench", None), index)
        if _is(area, AreaType.PRIZE):
            return self._id_at(getattr(player, "prize", None), index)
        if _is(area, AreaType.LOOKING):
            return self._id_at(getattr(current, "looking", None), index)
        if _is(area, AreaType.STADIUM):
            return self._id_at(getattr(current, "stadium", None), index)

        return None

    def _your_active(self, obs: object) -> object | None:
        current = getattr(obs, "current", None)
        if current is None:
            return None
        player = getattr(current, "players")[getattr(current, "yourIndex")]
        active = getattr(player, "active", [])
        if not active:
            return None
        return active[0]

    def _id_at(self, cards: Iterable[object] | None, index: int) -> int | None:
        if cards is None:
            return None
        cards_list = list(cards)
        if index >= len(cards_list):
            return None
        card = cards_list[index]
        if card is None:
            return None
        return getattr(card, "id", None)
