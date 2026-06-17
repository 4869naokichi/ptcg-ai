from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from cg.api import OptionType, SelectContext
from ptcg_ai.agent.rule_based import RuleBasedPolicy


class RuleBasedPolicyTest(unittest.TestCase):
    def test_prefers_going_second(self) -> None:
        policy = RuleBasedPolicy()
        obs = SimpleNamespace(
            select=SimpleNamespace(
                minCount=1,
                maxCount=1,
                context=SelectContext.IS_FIRST,
                option=[
                    SimpleNamespace(type=OptionType.YES),
                    SimpleNamespace(type=OptionType.NO),
                ],
            ),
            current=SimpleNamespace(),
        )

        self.assertEqual(policy.select(obs), [1])

    def test_prefers_higher_priority_card_for_setup_active(self) -> None:
        policy = RuleBasedPolicy()
        obs = SimpleNamespace(
            select=SimpleNamespace(
                minCount=1,
                maxCount=1,
                context=SelectContext.SETUP_ACTIVE_POKEMON,
                option=[
                    SimpleNamespace(type=OptionType.CARD, cardId=722),
                    SimpleNamespace(type=OptionType.CARD, cardId=721),
                ],
            ),
            current=SimpleNamespace(),
        )

        self.assertEqual(policy.select(obs), [1])

    def test_prefers_dwebble_for_setup_active(self) -> None:
        policy = RuleBasedPolicy()
        obs = SimpleNamespace(
            select=SimpleNamespace(
                minCount=1,
                maxCount=1,
                context=SelectContext.SETUP_ACTIVE_POKEMON,
                option=[
                    SimpleNamespace(type=OptionType.CARD, cardId=721),
                    SimpleNamespace(type=OptionType.CARD, cardId=344),
                ],
            ),
            current=SimpleNamespace(),
        )

        self.assertEqual(policy.select(obs), [1])

    def test_optional_negative_selection_can_skip(self) -> None:
        policy = RuleBasedPolicy()
        obs = SimpleNamespace(
            select=SimpleNamespace(
                minCount=0,
                maxCount=1,
                context=SelectContext.ACTIVATE,
                option=[SimpleNamespace(type=OptionType.NO)],
            ),
            current=SimpleNamespace(),
        )

        self.assertEqual(policy.select(obs), [])

    def test_avoids_large_draw_count_when_deck_is_low(self) -> None:
        policy = RuleBasedPolicy()
        obs = SimpleNamespace(
            select=SimpleNamespace(
                minCount=1,
                maxCount=1,
                context=SelectContext.DRAW_COUNT,
                option=[
                    SimpleNamespace(type=OptionType.NUMBER, number=6),
                    SimpleNamespace(type=OptionType.NUMBER, number=2),
                ],
            ),
            current=SimpleNamespace(
                yourIndex=0,
                players=[
                    SimpleNamespace(deckCount=5),
                    SimpleNamespace(deckCount=40),
                ],
            ),
        )

        self.assertEqual(policy.select(obs), [1])

    def test_avoids_mega_abomasnow_mill_attack_into_crustle(self) -> None:
        policy = RuleBasedPolicy()
        obs = SimpleNamespace(
            select=SimpleNamespace(
                minCount=1,
                maxCount=1,
                context=SelectContext.ATTACK,
                option=[
                    SimpleNamespace(type=OptionType.ATTACK, attackId=1046),
                    SimpleNamespace(type=OptionType.ATTACK, attackId=1047),
                ],
            ),
            current=SimpleNamespace(
                yourIndex=0,
                players=[
                    SimpleNamespace(deckCount=20, discard=[]),
                    SimpleNamespace(active=[SimpleNamespace(id=345, hp=150)]),
                ],
            ),
        )

        self.assertEqual(policy.select(obs), [1])


if __name__ == "__main__":
    unittest.main()
