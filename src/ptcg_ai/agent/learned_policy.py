from __future__ import annotations

from pathlib import Path

import numpy as np

from cg.api import SelectContext

from ptcg_ai.agent.policy import Policy
from ptcg_ai.agent.rule_based import RuleBasedPolicy
from ptcg_ai.training.features import option_features
from ptcg_ai.training.linear_model import LinearActionModel


class LearnedPolicy:
    """Legal-action scorer backed by a small trained linear model."""

    def __init__(self, model_path: Path, fallback: Policy | None = None) -> None:
        self.model = LinearActionModel.load(model_path)
        self.fallback = fallback or RuleBasedPolicy()

    def select(self, obs: object) -> list[int]:
        select = getattr(obs, "select")
        if select is None:
            raise ValueError("LearnedPolicy cannot choose the initial deck.")

        options = list(getattr(select, "option"))
        if not options or getattr(select, "maxCount") == 0:
            return []

        scores = [self.model.score(option_features(obs, option)) for option in options]
        target_count = self._target_count(obs, select, scores)
        if target_count <= 0:
            return []

        ranked_indexes = sorted(
            range(len(options)),
            key=lambda index: (scores[index], -index),
            reverse=True,
        )
        return ranked_indexes[:target_count]

    def _target_count(self, obs: object, select: object, scores: list[float]) -> int:
        if self.fallback is not None:
            try:
                return len(self.fallback.select(obs))
            except Exception:
                pass

        min_count = int(getattr(select, "minCount"))
        max_count = int(getattr(select, "maxCount"))
        if min_count == max_count:
            return min_count

        context = getattr(select, "context", None)
        if _is(context, SelectContext.SETUP_BENCH_POKEMON):
            return min(max_count, len([score for score in scores if score > 0.0]))

        if min_count == 0:
            return min(max_count, len([score for score in scores if score > 0.0]))

        return max_count


class StochasticLearnedPolicy(LearnedPolicy):
    """LearnedPolicy that samples actions via softmax for exploration during collection.

    Deterministic argmax self-play in a mirror never explores benching, so prizes
    never move and a dense (prize-based) reward stays invisible. Sampling with a
    temperature injects the variety the reward signal needs.
    """

    def __init__(
        self,
        model_path: Path,
        fallback: Policy | None = None,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(model_path=model_path, fallback=fallback)
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def select(self, obs: object) -> list[int]:
        if self.temperature <= 0.0:
            return super().select(obs)

        select = getattr(obs, "select")
        if select is None:
            raise ValueError("StochasticLearnedPolicy cannot choose the initial deck.")

        options = list(getattr(select, "option"))
        if not options or getattr(select, "maxCount") == 0:
            return []

        scores = [self.model.score(option_features(obs, option)) for option in options]
        target_count = self._target_count(obs, select, scores)
        if target_count <= 0:
            return []
        target_count = min(target_count, len(options))

        logits = np.array(scores, dtype=np.float64) / self.temperature
        probabilities = _softmax(logits)
        chosen = self.rng.choice(
            len(options),
            size=target_count,
            replace=False,
            p=probabilities,
        )
        return sorted(int(index) for index in chosen)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _is(value: object, enum_member: object) -> bool:
    if value is None:
        return False
    try:
        return int(value) == int(enum_member)
    except (TypeError, ValueError):
        return False
