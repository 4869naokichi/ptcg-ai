from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from ptcg_ai.agent.learned_policy import LearnedPolicy
from ptcg_ai.agent.policy import Policy
from ptcg_ai.agent.rule_based import RuleBasedPolicy
from ptcg_ai.decks import ABOMASNOW_DECK, CRUSTLE_DECK, DEFAULT_DECK
from ptcg_ai.evaluation.self_play import evaluate_self_play
from ptcg_ai.training.linear_model import train_from_jsonl
from ptcg_ai.training.logs import append_jsonl_many
from ptcg_ai.training.self_play import collect_logged_game


def build_deck(name: str) -> list[int]:
    if name == "default":
        return list(DEFAULT_DECK)
    if name == "crustle":
        return list(CRUSTLE_DECK)
    if name == "abomasnow":
        return list(ABOMASNOW_DECK)
    raise ValueError(f"unknown deck: {name}")


def build_best_policy(best_model: Path) -> Policy:
    """The current best policy: learned if a model exists, else rule-based bootstrap."""
    if best_model.exists():
        return LearnedPolicy(model_path=best_model, fallback=RuleBasedPolicy())
    return RuleBasedPolicy()


def collect_round(
    deck: list[int],
    policy: Policy,
    games: int,
    log_path: Path,
    max_steps: int,
) -> tuple[int, int]:
    """Self-play the current policy against itself and log every decision."""
    if log_path.exists():
        log_path.unlink()

    decisions = 0
    errors = 0
    for game_id in range(games):
        result = collect_logged_game(
            deck0=deck,
            deck1=deck,
            policy0=policy,
            policy1=policy,
            game_id=game_id,
            max_steps=max_steps,
            policy0_name="best",
            policy1_name="best",
        )
        append_jsonl_many(log_path, result.records)
        decisions += len([r for r in result.records if r.get("recordType") == "decision"])
        if result.error:
            errors += 1
    return decisions, errors


def head_to_head(
    deck: list[int],
    candidate: Policy,
    opponent: Policy,
    games: int,
    max_steps: int,
) -> tuple[float, int, int, int]:
    """Play both seats equally so the first/second-player bias cancels out."""
    half = games // 2
    as_p0 = evaluate_self_play(deck, candidate, opponent, half, max_steps=max_steps)
    as_p1 = evaluate_self_play(deck, opponent, candidate, games - half, max_steps=max_steps)

    candidate_wins = as_p0.player0_wins + as_p1.player1_wins
    opponent_wins = as_p0.player1_wins + as_p1.player0_wins
    decided = candidate_wins + opponent_wins
    win_rate = candidate_wins / decided if decided else 0.0
    return win_rate, candidate_wins, opponent_wins, decided


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward-weighted policy iteration loop.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--games", type=int, default=400, help="self-play games per round")
    parser.add_argument("--eval-games", type=int, default=200, help="head-to-head games for gating")
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--deck", choices=["default", "crustle", "abomasnow"], default="default")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.02, help="win-rate margin required to promote")
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="seed the best model (e.g. the imitation-trained model)",
    )
    parser.add_argument("--workdir", type=Path, default=ROOT / "outputs" / "policy_iteration")
    parser.add_argument(
        "--best-model",
        type=Path,
        default=ROOT / "outputs" / "models" / "policy_iteration_best.npz",
    )
    args = parser.parse_args()

    deck = build_deck(args.deck)
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.best_model.parent.mkdir(parents=True, exist_ok=True)

    if args.init_model is not None and args.init_model.exists() and not args.best_model.exists():
        shutil.copyfile(args.init_model, args.best_model)
        print(f"seeded best model from {args.init_model}")

    for round_index in range(args.rounds):
        print(f"=== round {round_index} ===")
        best_policy = build_best_policy(args.best_model)
        log_path = args.workdir / f"round_{round_index}.jsonl"
        decisions, collect_errors = collect_round(
            deck=deck,
            policy=best_policy,
            games=args.games,
            log_path=log_path,
            max_steps=args.max_steps,
        )
        print(f"collected games={args.games} decisions={decisions} errors={collect_errors}")

        candidate_model = args.workdir / f"round_{round_index}.npz"
        stats = train_from_jsonl(
            input_path=log_path,
            output_path=candidate_model,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            outcome_weighting="advantage",
            temperature=args.temperature,
        )
        print(f"trained pairs={stats.pairs} loss={stats.loss:.6f} pairwise_accuracy={stats.accuracy:.3f}")

        candidate_policy = LearnedPolicy(model_path=candidate_model, fallback=RuleBasedPolicy())
        win_rate, candidate_wins, opponent_wins, decided = head_to_head(
            deck=deck,
            candidate=candidate_policy,
            opponent=best_policy,
            games=args.eval_games,
            max_steps=args.max_steps,
        )
        print(
            f"candidate vs best: win_rate={win_rate:.3f} "
            f"({candidate_wins}/{decided}, opponent={opponent_wins})"
        )

        if win_rate >= 0.5 + args.margin:
            shutil.copyfile(candidate_model, args.best_model)
            print(f"PROMOTED candidate -> {args.best_model}")
        else:
            print("kept previous best (candidate did not clear the margin)")

    print(f"final best model: {args.best_model}")


if __name__ == "__main__":
    main()
