from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start, visualize_data

from ptcg_ai.agent.policy import Policy


@dataclass(frozen=True)
class VisualizedGame:
    snapshots: list[dict]
    winner: int
    steps: int
    error: str | None = None


def capture_game(
    deck0: Sequence[int],
    deck1: Sequence[int],
    policy0: Policy,
    policy1: Policy,
    max_steps: int = 300,
) -> VisualizedGame:
    obs_dict = None
    winner = -1
    steps = 0
    error = None

    try:
        obs_dict, start_data = battle_start(list(deck0), list(deck1))
        if obs_dict is None:
            return VisualizedGame(
                snapshots=[],
                winner=-1,
                steps=0,
                error=f"battle_start failed: player={start_data.errorPlayer}, type={start_data.errorType}",
            )

        policies = [policy0, policy1]
        for steps in range(max_steps):
            obs = to_observation_class(obs_dict)
            current = obs.current
            if current is not None and current.result != -1:
                winner = current.result
                break
            if current is None:
                error = "missing current state"
                break
            if obs.select is None:
                error = "unexpected deck selection"
                break

            selection = policies[current.yourIndex].select(obs)
            obs_dict = battle_select(selection)
        else:
            error = "max_steps reached"

        final_obs = to_observation_class(obs_dict)
        if final_obs.current is not None:
            winner = final_obs.current.result

        return VisualizedGame(
            snapshots=_read_visualize_data(),
            winner=winner,
            steps=steps,
            error=error,
        )
    except Exception as exc:
        return VisualizedGame(
            snapshots=_safe_visualize_data(),
            winner=winner,
            steps=steps,
            error=repr(exc),
        )
    finally:
        if obs_dict is not None:
            battle_finish()


def _read_visualize_data() -> list[dict]:
    data = json.loads(visualize_data())
    if not isinstance(data, list):
        raise ValueError("visualize_data did not return a list")
    return data


def _safe_visualize_data() -> list[dict]:
    try:
        return _read_visualize_data()
    except Exception:
        return []
