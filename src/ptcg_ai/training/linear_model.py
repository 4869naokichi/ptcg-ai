from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ptcg_ai.training.features import FEATURE_NAMES
from ptcg_ai.training.logs import read_jsonl


@dataclass(frozen=True)
class TrainingStats:
    decisions: int
    pairs: int
    epochs: int
    loss: float
    accuracy: float
    output_path: Path


@dataclass(frozen=True)
class LinearActionModel:
    weights: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]

    def score(self, features: dict[str, float]) -> float:
        vector = np.array(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float32,
        )
        return float(vector @ self.weights)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights.astype(np.float32),
            feature_names=np.array(self.feature_names, dtype=object),
            metadata=json.dumps(self.metadata, ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: Path) -> "LinearActionModel":
        payload = np.load(path, allow_pickle=True)
        feature_names = tuple(str(name) for name in payload["feature_names"].tolist())
        metadata = json.loads(str(payload["metadata"].tolist()))
        return cls(
            weights=payload["weights"].astype(np.float32),
            feature_names=feature_names,
            metadata=metadata,
        )


def train_from_jsonl(
    input_path: Path,
    output_path: Path,
    epochs: int = 200,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    outcome_weighting: str = "imitation",
    temperature: float = 1.0,
) -> TrainingStats:
    records = read_jsonl(input_path)
    decisions = [record for record in records if record.get("recordType") == "decision"]
    pair_diffs, pair_weights = _build_pairs(
        decisions,
        outcome_weighting=outcome_weighting,
        temperature=temperature,
    )
    if not pair_diffs:
        raise ValueError(f"no training pairs found in {input_path}")

    x = np.stack(pair_diffs).astype(np.float32)
    sample_weights = np.array(pair_weights, dtype=np.float32)
    weights = np.zeros(x.shape[1], dtype=np.float32)

    for _ in range(epochs):
        logits = np.clip(x @ weights, -30.0, 30.0)
        probabilities = _sigmoid(logits)
        errors = (probabilities - 1.0) * sample_weights
        gradient = (errors[:, None] * x).mean(axis=0) + (l2 * weights)
        weights -= learning_rate * gradient.astype(np.float32)

    loss, accuracy = _loss_and_accuracy(x, sample_weights, weights, l2)
    model = LinearActionModel(
        weights=weights,
        feature_names=FEATURE_NAMES,
        metadata={
            "inputPath": str(input_path),
            "decisions": len(decisions),
            "pairs": len(pair_diffs),
            "epochs": epochs,
            "learningRate": learning_rate,
            "l2": l2,
            "outcomeWeighting": outcome_weighting,
            "temperature": temperature,
            "loss": loss,
            "accuracy": accuracy,
        },
    )
    model.save(output_path)
    return TrainingStats(
        decisions=len(decisions),
        pairs=len(pair_diffs),
        epochs=epochs,
        loss=loss,
        accuracy=accuracy,
        output_path=output_path,
    )


def _build_pairs(
    decisions: list[dict[str, Any]],
    outcome_weighting: str,
    temperature: float = 1.0,
) -> tuple[list[np.ndarray], list[float]]:
    pair_diffs: list[np.ndarray] = []
    pair_weights: list[float] = []
    feature_names = FEATURE_NAMES
    baselines = _seat_baselines(decisions) if outcome_weighting == "advantage" else {}

    for decision in decisions:
        outcome_weight = _decision_weight(
            decision,
            outcome_weighting,
            baselines=baselines,
            temperature=temperature,
        )

        selected = []
        unselected = []
        for option in decision.get("options", []):
            vector = np.array(
                [float(option.get("features", {}).get(name, 0.0)) for name in feature_names],
                dtype=np.float32,
            )
            if option.get("selected"):
                selected.append(vector)
            else:
                unselected.append(vector)

        for selected_vector in selected:
            for unselected_vector in unselected:
                diff = selected_vector - unselected_vector
                if not np.any(np.abs(diff) > 1e-8):
                    continue
                pair_diffs.append(diff)
                pair_weights.append(outcome_weight)

    return pair_diffs, pair_weights


def _seat_baselines(decisions: list[dict[str, Any]]) -> dict[int, float]:
    """Mean outcome per seat (playerIndex), used to remove first/second-player bias."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for decision in decisions:
        player_index = int(decision.get("playerIndex", -1))
        outcome = float(decision.get("selectedPlayerOutcome", 0.0))
        sums[player_index] = sums.get(player_index, 0.0) + outcome
        counts[player_index] = counts.get(player_index, 0) + 1
    return {index: sums[index] / counts[index] for index in sums if counts[index] > 0}


def _decision_weight(
    decision: dict[str, Any],
    outcome_weighting: str,
    baselines: dict[int, float],
    temperature: float,
) -> float:
    if outcome_weighting == "imitation":
        return 1.0

    if outcome_weighting == "winner":
        selected_player_outcome = float(decision.get("selectedPlayerOutcome", 0.0))
        if selected_player_outcome > 0:
            return 1.0
        if selected_player_outcome == 0:
            return 0.6
        return 0.35

    if outcome_weighting == "advantage":
        # Reward-weighted regression with a seat-conditioned baseline so the
        # dominant first/second-player effect does not drown out action quality.
        player_index = int(decision.get("playerIndex", -1))
        outcome = float(decision.get("selectedPlayerOutcome", 0.0))
        advantage = outcome - baselines.get(player_index, 0.0)
        scale = temperature if temperature > 1e-6 else 1e-6
        return float(np.exp(np.clip(advantage / scale, -3.0, 3.0)))

    raise ValueError(f"unknown outcome weighting: {outcome_weighting}")


def _loss_and_accuracy(
    x: np.ndarray,
    sample_weights: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> tuple[float, float]:
    logits = np.clip(x @ weights, -30.0, 30.0)
    probabilities = _sigmoid(logits)
    loss = -np.log(np.maximum(probabilities, 1e-8)) * sample_weights
    regularization = 0.5 * l2 * float(weights @ weights)
    accuracy = float(np.mean(logits > 0.0))
    return float(np.mean(loss) + regularization), accuracy


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))
