from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from ptcg_ai.decks import ABOMASNOW_DECK
from ptcg_ai.game.constants import BASIC_WATER_ENERGY


class DeckTest(unittest.TestCase):
    def test_deck_has_60_cards(self) -> None:
        self.assertEqual(len(ABOMASNOW_DECK), 60)

    def test_non_basic_energy_cards_have_at_most_four_copies(self) -> None:
        counts = Counter(ABOMASNOW_DECK)
        for card_id, count in counts.items():
            if card_id == BASIC_WATER_ENERGY:
                continue
            self.assertLessEqual(count, 4, card_id)


if __name__ == "__main__":
    unittest.main()
