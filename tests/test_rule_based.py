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


if __name__ == "__main__":
    unittest.main()
