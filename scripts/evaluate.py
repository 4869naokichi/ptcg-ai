from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from ptcg_ai.agent.random_policy import RandomPolicy
from ptcg_ai.agent.rule_based import RuleBasedPolicy
from ptcg_ai.decks import ABOMASNOW_DECK
from ptcg_ai.evaluation.self_play import evaluate_self_play


def build_policy(name: str):
    if name == "random":
        return RandomPolicy()
    if name == "rule":
        return RuleBasedPolicy()
    raise ValueError(f"unknown policy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--player0", choices=["rule", "random"], default="rule")
    parser.add_argument("--player1", choices=["rule", "random"], default="random")
    args = parser.parse_args()

    result = evaluate_self_play(
        deck=ABOMASNOW_DECK,
        policy0=build_policy(args.player0),
        policy1=build_policy(args.player1),
        games=args.games,
        max_steps=args.max_steps,
    )
    print(f"games={result.games}")
    print(f"player0_wins={result.player0_wins}")
    print(f"player1_wins={result.player1_wins}")
    print(f"draws={result.draws}")
    print(f"errors={result.errors}")
    print(f"average_steps={result.average_steps:.2f}")
    print(f"player0_win_rate={result.player0_win_rate:.3f}")


if __name__ == "__main__":
    main()
