from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from ptcg_ai.agent.random_policy import RandomPolicy
from ptcg_ai.agent.rule_based import RuleBasedPolicy
from ptcg_ai.decks import ABOMASNOW_DECK, CRUSTLE_DECK, DEFAULT_DECK
from ptcg_ai.training.logs import append_jsonl_many
from ptcg_ai.training.self_play import collect_logged_game


def build_policy(name: str, seed: int | None = None):
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "rule":
        return RuleBasedPolicy()
    raise ValueError(f"unknown policy: {name}")


def build_deck(name: str) -> list[int]:
    if name == "default":
        return list(DEFAULT_DECK)
    if name == "crustle":
        return list(CRUSTLE_DECK)
    if name == "abomasnow":
        return list(ABOMASNOW_DECK)
    raise ValueError(f"unknown deck: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--player0", choices=["rule", "random"], default="rule")
    parser.add_argument("--player1", choices=["rule", "random"], default="rule")
    parser.add_argument("--deck0", choices=["default", "crustle", "abomasnow"], default="default")
    parser.add_argument("--deck1", choices=["default", "crustle", "abomasnow"], default="default")
    parser.add_argument("--seed", type=int, default=4869)
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "training" / "self_play.jsonl",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.append:
        args.output.unlink()

    wins = [0, 0, 0]
    errors = 0
    decisions = 0
    total_steps = 0
    deck0 = build_deck(args.deck0)
    deck1 = build_deck(args.deck1)

    for game_id in range(args.games):
        result = collect_logged_game(
            deck0=deck0,
            deck1=deck1,
            policy0=build_policy(args.player0, seed=args.seed + (game_id * 2)),
            policy1=build_policy(args.player1, seed=args.seed + (game_id * 2) + 1),
            game_id=game_id,
            max_steps=args.max_steps,
            policy0_name=args.player0,
            policy1_name=args.player1,
        )
        append_jsonl_many(args.output, result.records)
        decisions += len([record for record in result.records if record.get("recordType") == "decision"])
        total_steps += result.steps
        if result.error:
            errors += 1
        elif result.winner in (0, 1, 2):
            wins[result.winner] += 1

    print(f"wrote {args.output}")
    print(f"games={args.games}")
    print(f"decisions={decisions}")
    print(f"player0_wins={wins[0]}")
    print(f"player1_wins={wins[1]}")
    print(f"draws={wins[2]}")
    print(f"errors={errors}")
    print(f"average_steps={total_steps / args.games if args.games else 0.0:.2f}")


if __name__ == "__main__":
    main()
