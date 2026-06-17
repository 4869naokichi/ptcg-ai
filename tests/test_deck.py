from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from ptcg_ai.decks import ABOMASNOW_DECK, CRUSTLE_DECK, DEFAULT_DECK
from ptcg_ai.game.constants import BASIC_GRASS_ENERGY, BASIC_WATER_ENERGY


class DeckTest(unittest.TestCase):
    def test_deck_has_60_cards(self) -> None:
        self.assertEqual(len(ABOMASNOW_DECK), 60)
        self.assertEqual(len(CRUSTLE_DECK), 60)
        self.assertEqual(len(DEFAULT_DECK), 60)

    def test_non_basic_energy_cards_have_at_most_four_copies(self) -> None:
        for deck in [ABOMASNOW_DECK, CRUSTLE_DECK, DEFAULT_DECK]:
            counts = Counter(deck)
            for card_id, count in counts.items():
                if card_id in {BASIC_GRASS_ENERGY, BASIC_WATER_ENERGY}:
                    continue
                self.assertLessEqual(count, 4, card_id)

    def test_submission_deck_matches_default_deck(self) -> None:
        deck_path = ROOT / "submission" / "deck.csv"
        submission_deck = [
            int(line.strip())
            for line in deck_path.read_text().splitlines()
            if line.strip()
        ]

        self.assertEqual(submission_deck, DEFAULT_DECK)


if __name__ == "__main__":
    unittest.main()
