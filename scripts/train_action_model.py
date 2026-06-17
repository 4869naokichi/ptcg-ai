from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from ptcg_ai.training.linear_model import train_from_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "training" / "self_play.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "models" / "action_model.npz",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--outcome-weighting", choices=["imitation", "winner"], default="imitation")
    args = parser.parse_args()

    stats = train_from_jsonl(
        input_path=args.input,
        output_path=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        outcome_weighting=args.outcome_weighting,
    )
    print(f"wrote {stats.output_path}")
    print(f"decisions={stats.decisions}")
    print(f"pairs={stats.pairs}")
    print(f"epochs={stats.epochs}")
    print(f"loss={stats.loss:.6f}")
    print(f"pairwise_accuracy={stats.accuracy:.3f}")


if __name__ == "__main__":
    main()
