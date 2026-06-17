from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameResult:
    winner: int
    steps: int
    error: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    games: int
    player0_wins: int
    player1_wins: int
    draws: int
    errors: int
    average_steps: float

    @property
    def player0_win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.player0_wins / self.games
