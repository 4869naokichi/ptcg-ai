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
    BASIC_GRASS_ENERGY,
    BASIC_WATER_ENERGY,
    BOSS_ORDERS,
    BUDDY_BUDDY_POFFIN,
    CRUSTLE,
    DWEBBLE,
    KYOGRE,
    LILLIES_DETERMINATION,
    MEGA_ABOMASNOW_EX,
    SNOVER,
    ULTRA_BALL,
)
from ptcg_ai.game.features import (
    active_pokemon,
    attached_energy_count,
    bench_pokemon,
    count_card_id,
    deck_count,
    opponent_player,
    remaining_hp_ratio,
    your_player,
)

# Superb Scissors (Crustle's only attack) costs 1 Grass + 2 colorless. Energy
# beyond this on the Crustle line is wasted, so the policy stops attaching there.
CRUSTLE_ATTACK_COST = 3


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
    """Small score-based policy with deck and resource awareness."""

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
            return self._score_yes(obs, context)
        if _is(option_type, OptionType.NO):
            return self._score_no(obs, context)
        if _is(option_type, OptionType.NUMBER):
            return self._score_number(obs, option, context)
        if _is(option_type, OptionType.PLAY):
            return self.weights.play_card + self._score_hand_card(obs, option)
        if _is(option_type, OptionType.ATTACH):
            return self.weights.attach_energy + self._score_attach(obs, option)
        if _is(option_type, OptionType.EVOLVE):
            return self.weights.evolve + self._score_evolve(obs, option)
        if _is(option_type, OptionType.ABILITY):
            return self.weights.ability + self._score_located_card(obs, option)
        if _is(option_type, OptionType.ATTACK):
            return self.weights.attack + self._score_attack(obs, option)
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

    def _score_yes(self, obs: object, context: object) -> float:
        if _is(context, SelectContext.IS_FIRST):
            return -20.0
        if _is(context, SelectContext.MULLIGAN):
            return 100.0
        if _is(context, SelectContext.ACTIVATE):
            effect_id = self._effect_card_id(obs)
            if effect_id == LILLIES_DETERMINATION and self._your_deck_count(obs) <= 10:
                return -350.0
            return 95.0
        if _is(context, SelectContext.COIN_HEAD):
            return 50.0
        return 20.0

    def _score_no(self, obs: object, context: object) -> float:
        if _is(context, SelectContext.IS_FIRST):
            return 80.0
        if _is(context, SelectContext.MULLIGAN):
            return -50.0
        if _is(context, SelectContext.ACTIVATE):
            effect_id = self._effect_card_id(obs)
            if effect_id == LILLIES_DETERMINATION and self._your_deck_count(obs) <= 10:
                return 160.0
            return -10.0
        return 0.0

    def _score_number(self, obs: object, option: object, context: object) -> float:
        number = int(getattr(option, "number", 0) or 0)
        if _is(context, SelectContext.DRAW_COUNT):
            safe_draw = max(0, self._your_deck_count(obs) - 2)
            excess = max(0, number - safe_draw)
            return (number * 20.0) - (excess * 500.0)
        return float(number)

    def _score_attack(self, obs: object, option: object) -> float:
        attack_id = getattr(option, "attackId", None)
        opponent_active = self._opponent_active(obs)
        opponent_active_id = getattr(opponent_active, "id", None)
        opponent_hp = getattr(opponent_active, "hp", 0) or 0

        if attack_id == 478:
            return 420.0
        if attack_id == 479:
            score = 300.0
            if opponent_active_id == CRUSTLE:
                score += 180.0
            if 0 < opponent_hp <= 120:
                score += 120.0
            return score
        if attack_id == 1046:
            score = 380.0
            if opponent_active_id == CRUSTLE:
                score -= 1_200.0
            deck_left = self._your_deck_count(obs)
            if deck_left <= 6:
                score -= 1_000.0
            elif deck_left <= 10:
                score -= 520.0
            elif deck_left <= 14:
                score -= 220.0
            return score
        if attack_id == 1042:
            water_in_discard = self._your_discard_count(obs, BASIC_WATER_ENERGY)
            score = 160.0 + (water_in_discard * 35.0)
            if self._your_deck_count(obs) <= 12 and water_in_discard > 0:
                score += 260.0 + (water_in_discard * 20.0)
            if 0 < opponent_hp <= water_in_discard * 20:
                score += 180.0
            return score

        attack = attack_by_id(attack_id)
        if attack is None:
            return 0.0
        score = float(getattr(attack, "damage", 0) or 0)
        if attack_id == 1047 and opponent_active_id == CRUSTLE:
            score -= 1_000.0
        if 0 < opponent_hp <= score:
            score += 120.0
        return score

    def _score_hand_card(self, obs: object, option: object) -> float:
        card_id = self._hand_card_id(obs, getattr(option, "index", None))
        score = card_priority(card_id, self.weights)
        if card_id == LILLIES_DETERMINATION and self._your_deck_count(obs) <= 10:
            score -= 800.0
        if card_id == MEGA_ABOMASNOW_EX and self._opponent_has_active(obs, CRUSTLE):
            score -= 450.0

        # Keep a bench so a knocked-out active doesn't end the game, and develop
        # it early with search items.
        bench = self._your_bench_count(obs)
        if card_id == DWEBBLE and bench < 3:
            score += 240.0
        if card_id in {BUDDY_BUDDY_POFFIN, ULTRA_BALL} and bench < 2:
            score += 280.0
        return score

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
        if source_id == BASIC_GRASS_ENERGY:
            score += 150.0
        if target_id in {DWEBBLE, CRUSTLE}:
            target = self._pokemon_from_area(
                obs,
                getattr(option, "inPlayArea", None),
                getattr(option, "inPlayIndex", None),
                getattr(option, "playerIndex", None),
            )
            if attached_energy_count(target) >= CRUSTLE_ATTACK_COST:
                # Already enough to attack; piling on more energy is wasted.
                return -200.0
            score += 220.0
        if self._opponent_has_active(obs, CRUSTLE):
            if target_id == KYOGRE:
                score += 320.0
            if target_id in {SNOVER, MEGA_ABOMASNOW_EX}:
                score -= 260.0
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
        score = card_priority(evolved_id, self.weights) + card_priority(target_id, self.weights)
        if evolved_id == CRUSTLE and target_id == DWEBBLE:
            score += 500.0
        if evolved_id == MEGA_ABOMASNOW_EX and self._opponent_has_active(obs, CRUSTLE):
            score -= 420.0
        return score

    def _score_retreat(self, obs: object) -> float:
        active = self._your_active(obs)
        if active is None:
            return 0.0

        if getattr(active, "id", None) == MEGA_ABOMASNOW_EX and self._opponent_has_active(obs, CRUSTLE):
            if self._your_bench_has(obs, KYOGRE) or self._your_bench_has(obs, CRUSTLE):
                return self.weights.retreat + 520.0

        if remaining_hp_ratio(active) <= 0.5:
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
            if card_id == DWEBBLE:
                return 1120.0
            if card_id == KYOGRE:
                return 1000.0
            if card_id == SNOVER:
                return 760.0
            return card_priority(card_id, self.weights)

        if _is(context, SelectContext.SETUP_BENCH_POKEMON):
            if card_id == DWEBBLE:
                return 980.0
            if card_id == SNOVER:
                return 900.0
            if card_id == KYOGRE:
                return 700.0
            return card_priority(card_id, self.weights)

        if _is(context, SelectContext.EVOLVES_TO):
            if card_id == CRUSTLE:
                return 1120.0
            if card_id == MEGA_ABOMASNOW_EX:
                if self._opponent_has_active(obs, CRUSTLE):
                    return 120.0
                return 1000.0
            return card_priority(card_id, self.weights)

        if _is(context, SelectContext.EVOLVES_FROM):
            if card_id == DWEBBLE:
                return 1100.0
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

        if _is(context, SelectContext.TO_DECK) or _is(context, SelectContext.TO_DECK_BOTTOM):
            if _is(getattr(option, "area", None), AreaType.DISCARD):
                return 200.0 + card_priority(card_id, self.weights)
            return discard_priority(card_id, self.weights)

        if _is(context, SelectContext.NOT_MOVE):
            return card_priority(card_id, self.weights)

        return card_priority(card_id, self.weights)

    def _effect_card_id(self, obs: object) -> int | None:
        select = getattr(obs, "select", None)
        if select is None:
            return None
        effect = getattr(select, "effect", None)
        return getattr(effect, "id", None)

    def _your_deck_count(self, obs: object) -> int:
        return deck_count(your_player(obs))

    def _your_discard_count(self, obs: object, card_id: int) -> int:
        player = your_player(obs)
        if player is None:
            return 0
        return count_card_id(getattr(player, "discard", None), card_id)

    def _opponent_active(self, obs: object) -> object | None:
        return active_pokemon(opponent_player(obs))

    def _opponent_has_active(self, obs: object, card_id: int) -> bool:
        active = self._opponent_active(obs)
        return getattr(active, "id", None) == card_id

    def _your_bench_has(self, obs: object, card_id: int) -> bool:
        return any(
            getattr(pokemon, "id", None) == card_id
            for pokemon in bench_pokemon(your_player(obs))
        )

    def _your_bench_count(self, obs: object) -> int:
        return len(bench_pokemon(your_player(obs)))

    def _pokemon_from_area(
        self,
        obs: object,
        area: object,
        index: int | None,
        player_index: int | None,
    ) -> object | None:
        if index is None:
            return None
        current = getattr(obs, "current", None)
        if current is None:
            return None
        if player_index is None:
            player_index = getattr(current, "yourIndex")
        players = getattr(current, "players")
        if player_index >= len(players):
            return None
        player = players[player_index]
        if _is(area, AreaType.ACTIVE):
            return self._object_at(getattr(player, "active", None), index)
        if _is(area, AreaType.BENCH):
            return self._object_at(getattr(player, "bench", None), index)
        return None

    def _object_at(self, cards: Iterable[object] | None, index: int) -> object | None:
        if cards is None:
            return None
        cards_list = list(cards)
        if index >= len(cards_list):
            return None
        return cards_list[index]

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
