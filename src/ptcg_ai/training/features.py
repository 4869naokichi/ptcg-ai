from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from cg.api import AreaType, CardType, OptionType, SelectContext

from ptcg_ai.game.card_db import attack_by_id, card_by_id
from ptcg_ai.game.constants import (
    BASIC_GRASS_ENERGY,
    BASIC_WATER_ENERGY,
    CRUSTLE,
    DWEBBLE,
    KYOGRE,
    LILLIES_DETERMINATION,
    MAXIMUM_BELT,
    MEGA_ABOMASNOW_EX,
    MEGA_SIGNAL,
    SNOVER,
    WAITRESS,
)
from ptcg_ai.game.features import (
    active_pokemon,
    attached_energy_count,
    bench_pokemon,
    count_card_id,
    deck_count,
    opponent_player,
    prize_count,
    remaining_hp_ratio,
    your_player,
)


CARD_FLAGS = {
    BASIC_GRASS_ENERGY: "grass_energy",
    BASIC_WATER_ENERGY: "water_energy",
    DWEBBLE: "dwebble",
    CRUSTLE: "crustle",
    KYOGRE: "kyogre",
    SNOVER: "snover",
    MEGA_ABOMASNOW_EX: "mega_abomasnow",
    MEGA_SIGNAL: "mega_signal",
    MAXIMUM_BELT: "maximum_belt",
    LILLIES_DETERMINATION: "lillies_determination",
    WAITRESS: "waitress",
}

ATTACK_FLAGS = {
    478: "dwebble_ascension",
    479: "crustle_scissors",
    1042: "kyogre_riptide",
    1046: "abomasnow_hammer_lanche",
    1047: "abomasnow_frost_barrier",
}

BASE_FEATURE_NAMES = (
    "bias",
    "min_count",
    "max_count",
    "option_count",
    "your_deck_count",
    "opponent_deck_count",
    "deck_count_delta",
    "your_prize_count",
    "opponent_prize_count",
    "prize_count_delta",
    "your_hand_count",
    "opponent_hand_count",
    "your_bench_count",
    "opponent_bench_count",
    "your_active_hp_ratio",
    "opponent_active_hp_ratio",
    "your_active_energy_count",
    "opponent_active_energy_count",
    "your_discard_grass_energy",
    "your_discard_water_energy",
    "low_deck_6",
    "low_deck_10",
    "low_deck_14",
    "opponent_active_crustle",
    "opponent_active_mega_abomasnow",
    "card_known",
    "card_is_pokemon",
    "card_is_item",
    "card_is_tool",
    "card_is_supporter",
    "card_is_stadium",
    "card_is_basic_energy",
    "card_is_basic_pokemon",
    "card_is_stage1",
    "card_is_ex",
    "card_is_mega_ex",
    "card_hp",
    "card_retreat",
    "target_known",
    "target_is_pokemon",
    "target_is_basic_pokemon",
    "target_is_stage1",
    "target_is_ex",
    "target_is_mega_ex",
    "target_hp",
    "target_retreat",
    "option_number",
    "option_energy_count",
    "attack_damage",
    "attack_cost",
    "attack_can_ko",
)

OPTION_FEATURE_NAMES = tuple(f"option_{member.name.lower()}" for member in OptionType)
CONTEXT_FEATURE_NAMES = tuple(f"context_{member.name.lower()}" for member in SelectContext)
CARD_FLAG_FEATURE_NAMES = tuple(f"card_{name}" for name in CARD_FLAGS.values())
TARGET_FLAG_FEATURE_NAMES = tuple(f"target_{name}" for name in CARD_FLAGS.values())
ACTIVE_FLAG_FEATURE_NAMES = (
    "your_active_dwebble",
    "your_active_crustle",
    "your_active_kyogre",
    "your_active_snover",
    "your_active_mega_abomasnow",
    "opponent_active_dwebble",
    "opponent_active_crustle_specific",
    "opponent_active_kyogre",
    "opponent_active_snover",
    "opponent_active_mega_abomasnow_specific",
)
ATTACK_FLAG_FEATURE_NAMES = tuple(f"attack_{name}" for name in ATTACK_FLAGS.values())

FEATURE_NAMES = (
    BASE_FEATURE_NAMES
    + OPTION_FEATURE_NAMES
    + CONTEXT_FEATURE_NAMES
    + CARD_FLAG_FEATURE_NAMES
    + TARGET_FLAG_FEATURE_NAMES
    + ACTIVE_FLAG_FEATURE_NAMES
    + ATTACK_FLAG_FEATURE_NAMES
)


def option_features(obs: object, option: object) -> dict[str, float]:
    select = getattr(obs, "select", None)
    current = getattr(obs, "current", None)
    your = your_player(obs)
    opponent = opponent_player(obs)
    your_active = active_pokemon(your)
    opponent_active = active_pokemon(opponent)
    your_deck = deck_count(your)
    opponent_deck = deck_count(opponent)
    your_prize = prize_count(your)
    opponent_prize = prize_count(opponent)

    features: dict[str, float] = {
        "bias": 1.0,
        "min_count": _scale_count(getattr(select, "minCount", 0), 6),
        "max_count": _scale_count(getattr(select, "maxCount", 0), 6),
        "option_count": _scale_count(len(getattr(select, "option", []) or []), 20),
        "your_deck_count": _scale_count(your_deck, 60),
        "opponent_deck_count": _scale_count(opponent_deck, 60),
        "deck_count_delta": (your_deck - opponent_deck) / 60.0,
        "your_prize_count": _scale_count(your_prize, 6),
        "opponent_prize_count": _scale_count(opponent_prize, 6),
        "prize_count_delta": (opponent_prize - your_prize) / 6.0,
        "your_hand_count": _scale_count(getattr(your, "handCount", 0), 20),
        "opponent_hand_count": _scale_count(getattr(opponent, "handCount", 0), 20),
        "your_bench_count": _scale_count(len(bench_pokemon(your)), 5),
        "opponent_bench_count": _scale_count(len(bench_pokemon(opponent)), 5),
        "your_active_hp_ratio": remaining_hp_ratio(your_active),
        "opponent_active_hp_ratio": remaining_hp_ratio(opponent_active),
        "your_active_energy_count": _scale_count(attached_energy_count(your_active), 6),
        "opponent_active_energy_count": _scale_count(attached_energy_count(opponent_active), 6),
        "your_discard_grass_energy": _scale_count(count_card_id(getattr(your, "discard", None), BASIC_GRASS_ENERGY), 20),
        "your_discard_water_energy": _scale_count(count_card_id(getattr(your, "discard", None), BASIC_WATER_ENERGY), 20),
        "low_deck_6": float(your_deck <= 6),
        "low_deck_10": float(your_deck <= 10),
        "low_deck_14": float(your_deck <= 14),
        "opponent_active_crustle": float(getattr(opponent_active, "id", None) == CRUSTLE),
        "opponent_active_mega_abomasnow": float(getattr(opponent_active, "id", None) == MEGA_ABOMASNOW_EX),
        "option_number": _scale_count(getattr(option, "number", 0), 10),
        "option_energy_count": _scale_count(getattr(option, "count", 0), 6),
    }

    _set_one_hot(features, "option", getattr(option, "type", None), OptionType)
    _set_one_hot(features, "context", getattr(select, "context", None), SelectContext)
    _add_active_flags(features, "your_active", getattr(your_active, "id", None))
    _add_active_flags(features, "opponent_active", getattr(opponent_active, "id", None))

    card_id = option_card_id(obs, option)
    target_id = target_card_id(obs, option)
    _add_card_features(features, "card", card_id)
    _add_card_features(features, "target", target_id)
    _add_attack_features(features, getattr(option, "attackId", None), opponent_active)

    if current is None:
        return _sparse(features)
    return _sparse(features)


def option_card_id(obs: object, option: object) -> int | None:
    explicit_id = getattr(option, "cardId", None)
    if explicit_id is not None:
        return int(explicit_id)

    option_type = getattr(option, "type", None)
    if _is(option_type, OptionType.TOOL_CARD):
        pokemon = _object_from_area(
            obs,
            getattr(option, "area", None),
            getattr(option, "index", None),
            getattr(option, "playerIndex", None),
        )
        return _id_at(getattr(pokemon, "tools", None), getattr(option, "toolIndex", None))
    if _is(option_type, OptionType.ENERGY_CARD):
        pokemon = _object_from_area(
            obs,
            getattr(option, "area", None),
            getattr(option, "index", None),
            getattr(option, "playerIndex", None),
        )
        return _id_at(getattr(pokemon, "energyCards", None), getattr(option, "energyIndex", None))

    return _card_id_from_area(
        obs,
        getattr(option, "area", None),
        getattr(option, "index", None),
        getattr(option, "playerIndex", None),
    )


def target_card_id(obs: object, option: object) -> int | None:
    return _card_id_from_area(
        obs,
        getattr(option, "inPlayArea", None),
        getattr(option, "inPlayIndex", None),
        getattr(option, "playerIndex", None),
    )


def vectorize(features: dict[str, float], feature_names: Iterable[str] = FEATURE_NAMES) -> np.ndarray:
    return np.array([float(features.get(name, 0.0)) for name in feature_names], dtype=np.float32)


def _add_card_features(features: dict[str, float], prefix: str, card_id: int | None) -> None:
    if card_id is None:
        return

    card = card_by_id(card_id)
    features[f"{prefix}_known"] = 1.0 if card is not None else 0.0
    for known_id, flag_name in CARD_FLAGS.items():
        features[f"{prefix}_{flag_name}"] = float(card_id == known_id)

    if card is None:
        return

    card_type = getattr(card, "cardType", None)
    features[f"{prefix}_is_pokemon"] = float(_is(card_type, CardType.POKEMON))
    features[f"{prefix}_is_item"] = float(_is(card_type, CardType.ITEM))
    features[f"{prefix}_is_tool"] = float(_is(card_type, CardType.TOOL))
    features[f"{prefix}_is_supporter"] = float(_is(card_type, CardType.SUPPORTER))
    features[f"{prefix}_is_stadium"] = float(_is(card_type, CardType.STADIUM))
    features[f"{prefix}_is_basic_energy"] = float(_is(card_type, CardType.BASIC_ENERGY))
    features[f"{prefix}_is_basic_pokemon"] = float(getattr(card, "basic", False))
    features[f"{prefix}_is_stage1"] = float(getattr(card, "stage1", False))
    features[f"{prefix}_is_ex"] = float(getattr(card, "ex", False))
    features[f"{prefix}_is_mega_ex"] = float(getattr(card, "megaEx", False))
    features[f"{prefix}_hp"] = _scale_count(getattr(card, "hp", 0), 350)
    features[f"{prefix}_retreat"] = _scale_count(getattr(card, "retreatCost", 0), 4)


def _add_attack_features(features: dict[str, float], attack_id: int | None, opponent_active: object | None) -> None:
    if attack_id is None:
        return

    for known_id, flag_name in ATTACK_FLAGS.items():
        features[f"attack_{flag_name}"] = float(int(attack_id) == known_id)

    attack = attack_by_id(int(attack_id))
    if attack is None:
        return

    damage = float(getattr(attack, "damage", 0) or 0)
    opponent_hp = float(getattr(opponent_active, "hp", 0) or 0)
    features["attack_damage"] = min(damage, 300.0) / 300.0
    features["attack_cost"] = _scale_count(len(getattr(attack, "energies", []) or []), 5)
    features["attack_can_ko"] = float(opponent_hp > 0 and damage >= opponent_hp)


def _add_active_flags(features: dict[str, float], prefix: str, card_id: int | None) -> None:
    flag_ids = {
        DWEBBLE: "dwebble",
        CRUSTLE: "crustle" if prefix == "your_active" else "crustle_specific",
        KYOGRE: "kyogre",
        SNOVER: "snover",
        MEGA_ABOMASNOW_EX: "mega_abomasnow" if prefix == "your_active" else "mega_abomasnow_specific",
    }
    for known_id, flag_name in flag_ids.items():
        features[f"{prefix}_{flag_name}"] = float(card_id == known_id)


def _card_id_from_area(
    obs: object,
    area: object,
    index: int | None,
    player_index: int | None,
) -> int | None:
    obj = _object_from_area(obs, area, index, player_index)
    return getattr(obj, "id", None)


def _object_from_area(
    obs: object,
    area: object,
    index: int | None,
    player_index: int | None,
) -> object | None:
    if index is None:
        return None

    select = getattr(obs, "select", None)
    if select is not None and _is(area, AreaType.DECK):
        return _at(getattr(select, "deck", None), index)

    current = getattr(obs, "current", None)
    if current is None:
        return None

    if player_index is None:
        player_index = getattr(current, "yourIndex")

    players = getattr(current, "players", [])
    if player_index >= len(players):
        return None
    player = players[player_index]

    if _is(area, AreaType.HAND):
        return _at(getattr(player, "hand", None), index)
    if _is(area, AreaType.DISCARD):
        return _at(getattr(player, "discard", None), index)
    if _is(area, AreaType.ACTIVE):
        return _at(getattr(player, "active", None), index)
    if _is(area, AreaType.BENCH):
        return _at(getattr(player, "bench", None), index)
    if _is(area, AreaType.PRIZE):
        return _at(getattr(player, "prize", None), index)
    if _is(area, AreaType.LOOKING):
        return _at(getattr(current, "looking", None), index)
    if _is(area, AreaType.STADIUM):
        return _at(getattr(current, "stadium", None), index)

    return None


def _id_at(cards: object | None, index: int | None) -> int | None:
    item = _at(cards, index)
    return getattr(item, "id", None)


def _at(items: object | None, index: int | None) -> object | None:
    if items is None or index is None:
        return None
    items_list = list(items)
    if index >= len(items_list):
        return None
    return items_list[index]


def _set_one_hot(features: dict[str, float], prefix: str, value: object, enum_type: type) -> None:
    for member in enum_type:
        features[f"{prefix}_{member.name.lower()}"] = float(_is(value, member))


def _is(value: object, enum_member: object) -> bool:
    if value is None:
        return False
    try:
        return int(value) == int(enum_member)
    except (TypeError, ValueError):
        return False


def _scale_count(value: object, denominator: float) -> float:
    try:
        numeric = float(value or 0)
    except (TypeError, ValueError):
        numeric = 0.0
    if denominator <= 0:
        return 0.0
    return numeric / denominator


def _sparse(features: dict[str, float]) -> dict[str, float]:
    return {
        name: value
        for name, value in features.items()
        if abs(float(value)) > 1e-8
    }
