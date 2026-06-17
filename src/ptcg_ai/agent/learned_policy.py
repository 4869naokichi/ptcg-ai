from __future__ import annotations

from pathlib import Path

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


def _is(value: object, enum_member: object) -> bool:
    if value is None:
        return False
    try:
        return int(value) == int(enum_member)
    except (TypeError, ValueError):
        return False
