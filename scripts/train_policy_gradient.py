from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from ptcg_ai.training.mlp_model import train_policy_gradient_from_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="REINFORCE training for the MLP action policy.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "training" / "self_play.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "models" / "mlp_policy.npz",
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=4869)
    args = parser.parse_args()

    stats = train_policy_gradient_from_jsonl(
        input_path=args.input,
        output_path=args.output,
        hidden=args.hidden,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
    )
    print(f"wrote {stats.output_path}")
    print(f"decisions={stats.decisions}")
    print(f"samples={stats.samples}")
    print(f"epochs={stats.epochs}")
    print(f"loss={stats.loss:.6f}")


if __name__ == "__main__":
    main()
