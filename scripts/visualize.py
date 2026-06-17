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
from ptcg_ai.visualization.capture import capture_game
from ptcg_ai.visualization.render import write_visualizer_html


def build_policy(name: str):
    if name == "random":
        return RandomPolicy()
    if name == "rule":
        return RuleBasedPolicy()
    raise ValueError(f"unknown policy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--player0", choices=["rule", "random"], default="rule")
    parser.add_argument("--player1", choices=["rule", "random"], default="random")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "visualizer" / "game.html",
        help="Path to write the HTML replay.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ROOT / "data" / "card_images",
        help="Optional directory containing card images named like 721.jpg.",
    )
    args = parser.parse_args()

    result = capture_game(
        deck0=ABOMASNOW_DECK,
        deck1=ABOMASNOW_DECK,
        policy0=build_policy(args.player0),
        policy1=build_policy(args.player1),
        max_steps=args.max_steps,
    )
    write_visualizer_html(
        snapshots=result.snapshots,
        output_path=args.output,
        title=f"{args.player0} vs {args.player1}",
        metadata={
            "steps": result.steps,
            "winner": result.winner,
            "error": result.error,
            "player0": args.player0,
            "player1": args.player1,
        },
        image_dir=args.image_dir,
    )

    print(f"wrote {args.output}")
    print(f"snapshots={len(result.snapshots)} steps={result.steps} winner={result.winner} error={result.error}")


if __name__ == "__main__":
    main()
