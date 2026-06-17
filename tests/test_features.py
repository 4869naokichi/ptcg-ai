from __future__ import annotations

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_ai.game.features import (
    active_pokemon,
    card_ids,
    count_card_id,
    deck_count,
    remaining_hp_ratio,
)


class FeaturesTest(unittest.TestCase):
    def test_active_pokemon_returns_first_active(self) -> None:
        pokemon = SimpleNamespace(id=721)
        player = SimpleNamespace(active=[pokemon])
        self.assertIs(active_pokemon(player), pokemon)

    def test_remaining_hp_ratio_handles_missing_max_hp(self) -> None:
        pokemon = SimpleNamespace(hp=10, maxHp=0)
        self.assertEqual(remaining_hp_ratio(pokemon), 0.0)

    def test_remaining_hp_ratio(self) -> None:
        pokemon = SimpleNamespace(hp=75, maxHp=150)
        self.assertEqual(remaining_hp_ratio(pokemon), 0.5)

    def test_deck_count_handles_missing_player(self) -> None:
        self.assertEqual(deck_count(None), 0)

    def test_card_ids_skips_hidden_cards(self) -> None:
        cards = [SimpleNamespace(id=344), None, SimpleNamespace(id=345)]
        self.assertEqual(card_ids(cards), [344, 345])

    def test_count_card_id(self) -> None:
        cards = [SimpleNamespace(id=1), SimpleNamespace(id=344), SimpleNamespace(id=1)]
        self.assertEqual(count_card_id(cards, 1), 2)


if __name__ == "__main__":
    unittest.main()
