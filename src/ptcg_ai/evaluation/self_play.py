from __future__ import annotations

from collections.abc import Sequence

from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start

from ptcg_ai.agent.policy import Policy
from ptcg_ai.evaluation.metrics import EvaluationResult, GameResult


def play_game(
    deck0: Sequence[int],
    deck1: Sequence[int],
    policy0: Policy,
    policy1: Policy,
    max_steps: int = 1_000,
) -> GameResult:
    obs_dict = None
    try:
        obs_dict, start_data = battle_start(list(deck0), list(deck1))
        if obs_dict is None:
            return GameResult(
                winner=-1,
                steps=0,
                error=f"battle_start failed: player={start_data.errorPlayer}, type={start_data.errorType}",
            )

        policies = [policy0, policy1]
        for step in range(max_steps):
            obs = to_observation_class(obs_dict)
            current = obs.current
            if current is not None and current.result != -1:
                return GameResult(winner=current.result, steps=step)

            if current is None:
                return GameResult(winner=-1, steps=step, error="missing current state")

            select = obs.select
            if select is None:
                return GameResult(winner=-1, steps=step, error="unexpected deck selection")

            selection = policies[current.yourIndex].select(obs)
            obs_dict = battle_select(selection)

        final_obs = to_observation_class(obs_dict)
        final_result = final_obs.current.result if final_obs.current is not None else -1
        return GameResult(winner=final_result, steps=max_steps, error="max_steps reached")
    except Exception as exc:
        return GameResult(winner=-1, steps=0, error=repr(exc))
    finally:
        if obs_dict is not None:
            battle_finish()


def evaluate_self_play(
    deck: Sequence[int],
    policy0: Policy,
    policy1: Policy,
    games: int,
    max_steps: int = 1_000,
) -> EvaluationResult:
    results = [
        play_game(deck, deck, policy0, policy1, max_steps=max_steps)
        for _ in range(games)
    ]
    player0_wins = len([result for result in results if result.winner == 0 and result.error is None])
    player1_wins = len([result for result in results if result.winner == 1 and result.error is None])
    draws = len([result for result in results if result.winner == 2 and result.error is None])
    errors = len([result for result in results if result.error is not None])
    average_steps = sum(result.steps for result in results) / len(results) if results else 0.0
    return EvaluationResult(
        games=games,
        player0_wins=player0_wins,
        player1_wins=player1_wins,
        draws=draws,
        errors=errors,
        average_steps=average_steps,
    )
