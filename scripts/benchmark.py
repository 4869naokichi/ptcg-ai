from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start, visualize_data

from ptcg_ai.agent.learned_policy import LearnedPolicy, MLPPolicy
from ptcg_ai.agent.rule_based import RuleBasedPolicy
from ptcg_ai.decks import CRUSTLE_DECK
from ptcg_ai.visualization.render import write_visualizer_html

BENCHMARKS_DIR = ROOT / "benchmarks"


def extract_agent_code(ipynb_path: Path) -> str:
    """Pull the `%%writefile main.py` agent cell out of a sample-kernel notebook."""
    notebook = json.loads(ipynb_path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if "def agent(" in source:
            lines = source.splitlines()
            if lines and lines[0].startswith("%%"):
                lines = lines[1:]
            return "\n".join(lines)
    raise ValueError(f"no agent cell found in {ipynb_path}")


def reconstruct_deck(code: str) -> list[int]:
    """Rebuild the opponent deck from the documented `Name = id  # xN` constants."""
    deck: list[int] = []
    for match in re.finditer(r"=\s*(\d+)\s*#\s*[x×X]\s*(\d+)", code):
        card_id, count = int(match.group(1)), int(match.group(2))
        deck.extend([card_id] * count)
    return deck


class OpponentAgent:
    """Wraps a sample-kernel agent function, resetting its module globals per game."""

    def __init__(self, ipynb_path: Path) -> None:
        self.name = ipynb_path.parent.name
        self.code = extract_agent_code(ipynb_path)
        self.deck = reconstruct_deck(self.code)
        if len(self.deck) != 60:
            raise ValueError(f"{ipynb_path.name}: reconstructed deck has {len(self.deck)} cards")
        self._compiled = compile(self.code, str(ipynb_path), "exec")
        self._namespace: dict = {}

    def reset(self) -> None:
        # Sample agents keep per-game state in module globals; a fresh exec clears it.
        # They read deck.csv at import, so make sure one exists in the cwd.
        deck_csv = ROOT / "deck.csv"
        wrote = False
        if not deck_csv.exists():
            deck_csv.write_text("\n".join(str(c) for c in self.deck) + "\n", encoding="utf-8")
            wrote = True
        try:
            self._namespace = {"__name__": "opponent_agent"}
            exec(self._compiled, self._namespace)
        finally:
            if wrote:
                deck_csv.unlink()

    def select(self, obs_dict: dict) -> list[int]:
        return self._namespace["agent"](obs_dict)


def build_our_policy(name: str, model: Path):
    if name == "rule":
        return RuleBasedPolicy()
    if name == "learned":
        return LearnedPolicy(model_path=model, fallback=RuleBasedPolicy())
    if name == "mlp":
        return MLPPolicy(model_path=model, fallback=RuleBasedPolicy())
    raise ValueError(f"unknown policy: {name}")


def play_match(our_policy, our_deck, opponent: OpponentAgent, our_seat: int, max_steps: int):
    opponent.reset()
    deck0 = our_deck if our_seat == 0 else opponent.deck
    deck1 = opponent.deck if our_seat == 0 else our_deck

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return "error"
    try:
        for _ in range(max_steps):
            obs = to_observation_class(obs_dict)
            current = obs.current
            if current is not None and current.result != -1:
                if current.result == 2:
                    return "draw"
                return "win" if current.result == our_seat else "loss"
            if current is None or obs.select is None:
                return "error"
            if current.yourIndex == our_seat:
                selection = our_policy.select(obs)
            else:
                selection = opponent.select(obs_dict)
            obs_dict = battle_select(selection)
        return "draw"
    except Exception:
        return "error"
    finally:
        battle_finish()


def visualize_match(our_policy, our_deck, opponent: OpponentAgent, our_seat: int, output: Path,
                    image_dir: Path, max_steps: int) -> None:
    """Play one game vs an opponent and write an HTML replay for the visualizer."""
    opponent.reset()
    deck0 = our_deck if our_seat == 0 else opponent.deck
    deck1 = opponent.deck if our_seat == 0 else our_deck

    obs_dict, start = battle_start(list(deck0), list(deck1))
    winner, error = -1, None
    if obs_dict is None:
        print(f"battle_start failed: {start.errorPlayer}/{start.errorType}")
        return
    try:
        for _ in range(max_steps):
            obs = to_observation_class(obs_dict)
            current = obs.current
            if current is not None and current.result != -1:
                winner = current.result
                break
            if current is None or obs.select is None:
                error = "abnormal state"
                break
            if current.yourIndex == our_seat:
                selection = our_policy.select(obs)
            else:
                selection = opponent.select(obs_dict)
            obs_dict = battle_select(selection)
        snapshots = json.loads(visualize_data())
    finally:
        if obs_dict is not None:
            battle_finish()

    our_label = f"ours(seat{our_seat})"
    opp_label = opponent.name
    write_visualizer_html(
        snapshots=snapshots,
        output_path=output,
        title=f"{our_label} vs {opp_label}",
        metadata={"winner": winner, "error": error, "ourSeat": our_seat, "opponent": opp_label},
        image_dir=image_dir,
    )
    outcome = "win" if winner == our_seat else "loss" if winner in (0, 1) else "draw/none"
    print(f"wrote {output} (snapshots={len(snapshots)} winner=player{winner} -> ours={outcome})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark our agent vs sample-kernel opponents.")
    parser.add_argument("--policy", choices=["rule", "learned", "mlp"], default="rule")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "models" / "action_model.npz")
    parser.add_argument("--games", type=int, default=100, help="games per opponent (split across both seats)")
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--opponents", nargs="*", default=None, help="opponent dir names under benchmarks/")
    parser.add_argument("--visualize-vs", default=None, help="opponent dir name to capture one game as an HTML replay")
    parser.add_argument("--seat", type=int, choices=[0, 1], default=0, help="our seat when visualizing")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "visualizer" / "game.html")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data" / "card_images")
    args = parser.parse_args()

    our_policy = build_our_policy(args.policy, args.model)
    our_deck = list(CRUSTLE_DECK)

    if args.visualize_vs:
        matches = list(BENCHMARKS_DIR.glob(f"{args.visualize_vs}/*.ipynb"))
        if not matches:
            print(f"no opponent notebook found for {args.visualize_vs}")
            return
        opponent = OpponentAgent(matches[0])
        visualize_match(our_policy, our_deck, opponent, args.seat, args.output, args.image_dir, args.max_steps)
        return

    notebooks = []
    for path in sorted(BENCHMARKS_DIR.glob("*/*.ipynb")):
        if args.opponents and path.parent.name not in args.opponents:
            continue
        notebooks.append(path)

    if not notebooks:
        print("no opponent notebooks found under benchmarks/")
        return

    total = {"win": 0, "loss": 0, "draw": 0, "error": 0}
    for path in notebooks:
        try:
            opponent = OpponentAgent(path)
        except Exception as exc:
            print(f"{path.parent.name}: SKIP ({exc})")
            continue

        counts = {"win": 0, "loss": 0, "draw": 0, "error": 0}
        for game in range(args.games):
            seat = game % 2
            counts[play_match(our_policy, our_deck, opponent, seat, args.max_steps)] += 1
        for key in total:
            total[key] += counts[key]
        decided = counts["win"] + counts["loss"]
        win_rate = counts["win"] / decided if decided else 0.0
        print(
            f"{path.parent.name:<52} win={counts['win']:3d} loss={counts['loss']:3d} "
            f"draw={counts['draw']:2d} err={counts['error']:2d}  winrate={win_rate:.2f}"
        )

    decided = total["win"] + total["loss"]
    print("-" * 90)
    print(
        f"{'TOTAL':<52} win={total['win']:3d} loss={total['loss']:3d} "
        f"draw={total['draw']:2d} err={total['error']:2d}  winrate={total['win']/decided if decided else 0:.2f}"
    )


if __name__ == "__main__":
    main()
