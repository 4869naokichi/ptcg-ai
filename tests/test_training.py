from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from cg.api import OptionType, SelectContext
from ptcg_ai.training.features import FEATURE_NAMES, option_features, vectorize
from ptcg_ai.training.linear_model import (
    LinearActionModel,
    _decision_weight,
    _dense_decision_weights,
    _seat_baselines,
    train_from_jsonl,
)


class TrainingTest(unittest.TestCase):
    def test_option_features_include_card_and_context_flags(self) -> None:
        obs = SimpleNamespace(
            select=SimpleNamespace(
                minCount=1,
                maxCount=1,
                option=[SimpleNamespace(type=OptionType.CARD, cardId=345)],
                context=SelectContext.SETUP_ACTIVE_POKEMON,
            ),
            current=SimpleNamespace(
                yourIndex=0,
                players=[
                    SimpleNamespace(
                        deckCount=40,
                        prize=[],
                        handCount=4,
                        active=[],
                        bench=[],
                        discard=[],
                    ),
                    SimpleNamespace(
                        deckCount=39,
                        prize=[],
                        handCount=5,
                        active=[],
                        bench=[],
                        discard=[],
                    ),
                ],
            ),
        )

        features = option_features(obs, obs.select.option[0])
        self.assertEqual(features["card_crustle"], 1.0)
        self.assertEqual(features["context_setup_active_pokemon"], 1.0)
        self.assertEqual(vectorize(features).shape, (len(FEATURE_NAMES),))

    def test_train_from_jsonl_learns_pairwise_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "self_play.jsonl"
            output_path = Path(tmp_dir) / "model.npz"
            decision = {
                "recordType": "decision",
                "selectedPlayerOutcome": 1.0,
                "options": [
                    {"selected": True, "features": {"card_crustle": 1.0}},
                    {"selected": False, "features": {"card_kyogre": 1.0}},
                ],
            }
            input_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")

            stats = train_from_jsonl(
                input_path=input_path,
                output_path=output_path,
                epochs=50,
                learning_rate=0.2,
                l2=0.0,
            )
            model = LinearActionModel.load(output_path)

            self.assertEqual(stats.pairs, 1)
            self.assertGreater(
                model.score({"card_crustle": 1.0}),
                model.score({"card_kyogre": 1.0}),
            )

    def test_seat_baselines_average_outcome_per_player(self) -> None:
        decisions = [
            {"playerIndex": 0, "selectedPlayerOutcome": -1.0},
            {"playerIndex": 0, "selectedPlayerOutcome": -1.0},
            {"playerIndex": 1, "selectedPlayerOutcome": 1.0},
            {"playerIndex": 1, "selectedPlayerOutcome": 1.0},
        ]
        baselines = _seat_baselines(decisions)
        self.assertEqual(baselines[0], -1.0)
        self.assertEqual(baselines[1], 1.0)

    def test_advantage_weight_removes_seat_bias(self) -> None:
        # Seat 1 wins every game, so a win there carries zero advantage while a
        # loss against the seat baseline is what actually signals a bad action.
        baselines = {0: -1.0, 1: 1.0}
        seat1_win = {"playerIndex": 1, "selectedPlayerOutcome": 1.0}
        seat0_win = {"playerIndex": 0, "selectedPlayerOutcome": 1.0}

        seat1_weight = _decision_weight(seat1_win, "advantage", baselines, temperature=1.0)
        seat0_weight = _decision_weight(seat0_win, "advantage", baselines, temperature=1.0)

        # The unexpected win (against a losing seat baseline) is weighted higher
        # than the expected win on the dominant seat.
        self.assertAlmostEqual(seat1_weight, 1.0, places=6)
        self.assertGreater(seat0_weight, seat1_weight)

    def test_train_from_jsonl_accepts_advantage_weighting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "self_play.jsonl"
            output_path = Path(tmp_dir) / "model.npz"
            decision = {
                "recordType": "decision",
                "playerIndex": 0,
                "selectedPlayerOutcome": 1.0,
                "options": [
                    {"selected": True, "features": {"card_crustle": 1.0}},
                    {"selected": False, "features": {"card_kyogre": 1.0}},
                ],
            }
            input_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")

            stats = train_from_jsonl(
                input_path=input_path,
                output_path=output_path,
                epochs=50,
                learning_rate=0.2,
                l2=0.0,
                outcome_weighting="advantage",
                temperature=0.5,
            )
            model = LinearActionModel.load(output_path)

            self.assertEqual(stats.pairs, 1)
            self.assertEqual(model.metadata["outcomeWeighting"], "advantage")
            self.assertEqual(model.metadata["temperature"], 0.5)

    def test_dense_weights_reward_prize_progress(self) -> None:
        # Same seat, two games: one where player0 takes prizes and wins, one where
        # player0 gives up prizes and loses. The winning trajectory must weigh more.
        decisions = [
            {"gameId": 0, "playerIndex": 0, "step": 0,
             "yourPrizeRemaining": 6, "opponentPrizeRemaining": 6, "selectedPlayerOutcome": 1.0},
            {"gameId": 0, "playerIndex": 0, "step": 2,
             "yourPrizeRemaining": 4, "opponentPrizeRemaining": 6, "selectedPlayerOutcome": 1.0},
            {"gameId": 1, "playerIndex": 0, "step": 0,
             "yourPrizeRemaining": 6, "opponentPrizeRemaining": 6, "selectedPlayerOutcome": -1.0},
            {"gameId": 1, "playerIndex": 0, "step": 2,
             "yourPrizeRemaining": 6, "opponentPrizeRemaining": 4, "selectedPlayerOutcome": -1.0},
        ]
        weights = _dense_decision_weights(decisions, gamma=0.99, temperature=2.0)
        winning = (weights[0] + weights[1]) / 2
        losing = (weights[2] + weights[3]) / 2
        self.assertGreater(winning, losing)

    def test_train_from_jsonl_accepts_dense_weighting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "self_play.jsonl"
            output_path = Path(tmp_dir) / "model.npz"
            decision = {
                "recordType": "decision",
                "gameId": 0,
                "playerIndex": 0,
                "step": 0,
                "yourPrizeRemaining": 4,
                "opponentPrizeRemaining": 6,
                "selectedPlayerOutcome": 1.0,
                "options": [
                    {"selected": True, "features": {"card_crustle": 1.0}},
                    {"selected": False, "features": {"card_kyogre": 1.0}},
                ],
            }
            input_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")

            stats = train_from_jsonl(
                input_path=input_path,
                output_path=output_path,
                epochs=20,
                learning_rate=0.2,
                l2=0.0,
                outcome_weighting="dense",
                gamma=0.95,
            )
            model = LinearActionModel.load(output_path)

            self.assertEqual(stats.pairs, 1)
            self.assertEqual(model.metadata["outcomeWeighting"], "dense")
            self.assertEqual(model.metadata["gamma"], 0.95)


if __name__ == "__main__":
    unittest.main()
