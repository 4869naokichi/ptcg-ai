"""Evaluation helpers."""

from .metrics import EvaluationResult, GameResult

__all__ = ["EvaluationResult", "GameResult", "evaluate_self_play", "play_game"]


def __getattr__(name: str):
    if name in {"evaluate_self_play", "play_game"}:
        from .self_play import evaluate_self_play, play_game

        return {"evaluate_self_play": evaluate_self_play, "play_game": play_game}[name]
    raise AttributeError(name)
