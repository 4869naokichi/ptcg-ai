from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ptcg_ai.training.features import FEATURE_NAMES
from ptcg_ai.training.linear_model import dense_advantages
from ptcg_ai.training.logs import read_jsonl


@dataclass(frozen=True)
class PolicyGradientStats:
    decisions: int
    samples: int
    epochs: int
    loss: float
    output_path: Path


@dataclass(frozen=True)
class MLPActionModel:
    """Small two-layer MLP that scores a single legal action.

    Inference is pure numpy so the submission stays dependency-free; only training
    needs the gradient machinery below.
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]

    def score(self, features: dict[str, float]) -> float:
        vector = np.array(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float32,
        )
        return float(self.score_matrix(vector[None, :])[0])

    def score_matrix(self, x: np.ndarray) -> np.ndarray:
        hidden = np.tanh(x @ self.w1 + self.b1)
        return (hidden @ self.w2 + self.b2).reshape(-1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w1=self.w1.astype(np.float32),
            b1=self.b1.astype(np.float32),
            w2=self.w2.astype(np.float32),
            b2=self.b2.astype(np.float32),
            feature_names=np.array(self.feature_names, dtype=object),
            metadata=json.dumps(self.metadata, ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: Path) -> "MLPActionModel":
        payload = np.load(path, allow_pickle=True)
        feature_names = tuple(str(name) for name in payload["feature_names"].tolist())
        metadata = json.loads(str(payload["metadata"].tolist()))
        return cls(
            w1=payload["w1"].astype(np.float32),
            b1=payload["b1"].astype(np.float32),
            w2=payload["w2"].astype(np.float32),
            b2=payload["b2"].astype(np.float32),
            feature_names=feature_names,
            metadata=metadata,
        )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def train_policy_gradient_from_jsonl(
    input_path: Path,
    output_path: Path,
    hidden: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.01,
    gamma: float = 0.99,
    seed: int = 4869,
) -> PolicyGradientStats:
    """REINFORCE on logged self-play with dense (prize-diff) advantages.

    Each decision is a categorical choice over its legal options via softmax(MLP(x)).
    The gradient is A * grad log pi(a|s) with A the shaped advantage; advantages are
    standardized for stability. Adam optimizer, pure numpy.
    """
    records = read_jsonl(input_path)
    decisions = [record for record in records if record.get("recordType") == "decision"]
    advantages = dense_advantages(decisions, gamma=gamma)

    samples = _build_samples(decisions, advantages)
    if not samples:
        raise ValueError(f"no policy-gradient samples found in {input_path}")

    advantage_values = np.array([sample[2] for sample in samples], dtype=np.float64)
    mean = float(advantage_values.mean())
    std = float(advantage_values.std()) or 1.0

    rng = np.random.default_rng(seed)
    dim = len(FEATURE_NAMES)
    params = {
        "w1": (rng.standard_normal((dim, hidden)) * (1.0 / np.sqrt(dim))),
        "b1": np.zeros(hidden),
        "w2": (rng.standard_normal((hidden, 1)) * (1.0 / np.sqrt(hidden))),
        "b2": np.zeros(1),
    }
    adam = {key: (np.zeros_like(value), np.zeros_like(value)) for key, value in params.items()}

    loss = 0.0
    step = 0
    for _ in range(epochs):
        grads = {key: np.zeros_like(value) for key, value in params.items()}
        loss = 0.0
        for x, selected, advantage in samples:
            normalized = (advantage - mean) / std
            hidden_pre = x @ params["w1"] + params["b1"]
            hidden_act = np.tanh(hidden_pre)
            scores = (hidden_act @ params["w2"] + params["b2"]).reshape(-1)
            probabilities = _softmax(scores)

            selected_mask = np.zeros(len(scores))
            selected_mask[selected] = 1.0
            log_probs = np.log(np.maximum(probabilities[selected], 1e-8))
            loss += float(-normalized * log_probs.sum())

            # d(-A * sum_S log pi)/d scores = -A * (selected_mask - |S| * pi)
            d_scores = -normalized * (selected_mask - len(selected) * probabilities)
            d_scores = d_scores.reshape(-1, 1)

            grads["w2"] += hidden_act.T @ d_scores
            grads["b2"] += d_scores.sum(axis=0)
            d_hidden = (d_scores @ params["w2"].T) * (1.0 - hidden_act**2)
            grads["w1"] += x.T @ d_hidden
            grads["b1"] += d_hidden.sum(axis=0)

        step += 1
        for key in params:
            grad = grads[key] / len(samples)
            moment1, moment2 = adam[key]
            moment1 = 0.9 * moment1 + 0.1 * grad
            moment2 = 0.999 * moment2 + 0.001 * (grad**2)
            adam[key] = (moment1, moment2)
            corrected1 = moment1 / (1.0 - 0.9**step)
            corrected2 = moment2 / (1.0 - 0.999**step)
            params[key] = params[key] - learning_rate * corrected1 / (np.sqrt(corrected2) + 1e-8)
        loss /= len(samples)

    model = MLPActionModel(
        w1=params["w1"].astype(np.float32),
        b1=params["b1"].astype(np.float32),
        w2=params["w2"].astype(np.float32),
        b2=params["b2"].astype(np.float32),
        feature_names=FEATURE_NAMES,
        metadata={
            "inputPath": str(input_path),
            "decisions": len(decisions),
            "samples": len(samples),
            "hidden": hidden,
            "epochs": epochs,
            "learningRate": learning_rate,
            "gamma": gamma,
            "loss": loss,
        },
    )
    model.save(output_path)
    return PolicyGradientStats(
        decisions=len(decisions),
        samples=len(samples),
        epochs=epochs,
        loss=loss,
        output_path=output_path,
    )


def _build_samples(
    decisions: list[dict[str, Any]],
    advantages: list[float],
) -> list[tuple[np.ndarray, list[int], float]]:
    samples: list[tuple[np.ndarray, list[int], float]] = []
    for decision, advantage in zip(decisions, advantages):
        options = decision.get("options", [])
        if len(options) < 2:
            continue
        selected = [index for index, option in enumerate(options) if option.get("selected")]
        if not selected:
            continue
        x = np.array(
            [
                [float(option.get("features", {}).get(name, 0.0)) for name in FEATURE_NAMES]
                for option in options
            ],
            dtype=np.float64,
        )
        samples.append((x, selected, float(advantage)))
    return samples
