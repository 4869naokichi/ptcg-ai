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
from ptcg_ai.training.linear_model import LinearActionModel, train_from_jsonl


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


if __name__ == "__main__":
    unittest.main()
