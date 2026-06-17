from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start

from ptcg_ai.agent.policy import Policy
from ptcg_ai.training.features import option_card_id, option_features, target_card_id


@dataclass(frozen=True)
class LoggedSelfPlayGame:
    records: list[dict[str, Any]]
    winner: int
    steps: int
    error: str | None = None


def collect_logged_game(
    deck0: Sequence[int],
    deck1: Sequence[int],
    policy0: Policy,
    policy1: Policy,
    game_id: int,
    max_steps: int = 1_000,
    policy0_name: str = "player0",
    policy1_name: str = "player1",
) -> LoggedSelfPlayGame:
    obs_dict = None
    decision_records: list[dict[str, Any]] = []
    winner = -1
    steps = 0
    error = None

    try:
        obs_dict, start_data = battle_start(list(deck0), list(deck1))
        if obs_dict is None:
            error = f"battle_start failed: player={start_data.errorPlayer}, type={start_data.errorType}"
            return _finish_game(game_id, decision_records, winner, steps, error, policy0_name, policy1_name)

        policies = [policy0, policy1]
        policy_names = [policy0_name, policy1_name]
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

            player_index = current.yourIndex
            policy = policies[player_index]
            selected = policy.select(obs)
            decision_records.append(
                _decision_record(
                    game_id=game_id,
                    step=steps,
                    obs=obs,
                    selected=selected,
                    policy=policy,
                    policy_name=policy_names[player_index],
                )
            )
            obs_dict = battle_select(selected)
        else:
            error = "max_steps reached"

        final_obs = to_observation_class(obs_dict)
        if final_obs.current is not None:
            winner = final_obs.current.result

        return _finish_game(game_id, decision_records, winner, steps, error, policy0_name, policy1_name)
    except Exception as exc:
        error = repr(exc)
        return _finish_game(game_id, decision_records, winner, steps, error, policy0_name, policy1_name)
    finally:
        if obs_dict is not None:
            battle_finish()


def _finish_game(
    game_id: int,
    decision_records: list[dict[str, Any]],
    winner: int,
    steps: int,
    error: str | None,
    policy0_name: str,
    policy1_name: str,
) -> LoggedSelfPlayGame:
    for record in decision_records:
        player_index = int(record["playerIndex"])
        if error is not None or winner == 2 or winner == -1:
            outcome = 0.0
        elif winner == player_index:
            outcome = 1.0
        else:
            outcome = -1.0
        record["winner"] = winner
        record["selectedPlayerOutcome"] = outcome
        record["gameError"] = error

    summary = {
        "recordType": "game",
        "gameId": game_id,
        "winner": winner,
        "steps": steps,
        "error": error,
        "policy0": policy0_name,
        "policy1": policy1_name,
        "decisionCount": len(decision_records),
    }
    return LoggedSelfPlayGame(
        records=[*decision_records, summary],
        winner=winner,
        steps=steps,
        error=error,
    )


def _decision_record(
    game_id: int,
    step: int,
    obs: object,
    selected: list[int],
    policy: Policy,
    policy_name: str,
) -> dict[str, Any]:
    current = getattr(obs, "current")
    select = getattr(obs, "select")
    selected_set = set(selected)
    player_index = int(getattr(current, "yourIndex"))

    players = getattr(current, "players")
    your_prize_remaining = _prize_remaining(players[player_index])
    opponent_prize_remaining = _prize_remaining(players[1 - player_index])

    options = []
    for index, option in enumerate(getattr(select, "option")):
        score = _policy_score(policy, obs, option)
        options.append(
            {
                "index": index,
                "selected": index in selected_set,
                "type": _enum_name(getattr(option, "type", None)),
                "typeValue": _enum_value(getattr(option, "type", None)),
                "cardId": option_card_id(obs, option),
                "targetCardId": target_card_id(obs, option),
                "attackId": getattr(option, "attackId", None),
                "number": getattr(option, "number", None),
                "count": getattr(option, "count", None),
                "ruleScore": score,
                "features": option_features(obs, option),
            }
        )

    return {
        "recordType": "decision",
        "gameId": game_id,
        "step": step,
        "turn": int(getattr(current, "turn")),
        "turnActionCount": int(getattr(current, "turnActionCount")),
        "playerIndex": player_index,
        "yourPrizeRemaining": your_prize_remaining,
        "opponentPrizeRemaining": opponent_prize_remaining,
        "policy": policy_name,
        "selected": selected,
        "selectType": _enum_name(getattr(select, "type", None)),
        "selectTypeValue": _enum_value(getattr(select, "type", None)),
        "context": _enum_name(getattr(select, "context", None)),
        "contextValue": _enum_value(getattr(select, "context", None)),
        "minCount": int(getattr(select, "minCount")),
        "maxCount": int(getattr(select, "maxCount")),
        "options": options,
    }


def _prize_remaining(player: object) -> int:
    return len(getattr(player, "prize", []) or [])


def _policy_score(policy: Policy, obs: object, option: object) -> float | None:
    scorer = getattr(policy, "score_option", None)
    if scorer is None:
        return None
    try:
        return float(scorer(obs, option))
    except Exception:
        return None


def _enum_name(value: object) -> str | None:
    return getattr(value, "name", None)


def _enum_value(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
