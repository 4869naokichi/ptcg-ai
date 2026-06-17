from __future__ import annotations

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_ai.game.features import active_pokemon, remaining_hp_ratio


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


if __name__ == "__main__":
    unittest.main()
