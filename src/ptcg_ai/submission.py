from __future__ import annotations

from pathlib import Path

from cg.api import Observation, to_observation_class

from ptcg_ai.agent.rule_based import RuleBasedPolicy
from ptcg_ai.decks import ABOMASNOW_DECK

_POLICY = RuleBasedPolicy()


def read_deck_csv() -> list[int]:
    """Read the deck file from Kaggle, generated submission, or repo layout."""

    here = Path(__file__).resolve()
    candidates = [
        Path("deck.csv"),
        Path("/kaggle_simulations/agent/deck.csv"),
        here.parents[1] / "deck.csv",
        here.parents[2] / "submission" / "deck.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return [
                int(line.strip())
                for line in candidate.read_text().splitlines()
                if line.strip()
            ][:60]

    return list(ABOMASNOW_DECK)


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    return _POLICY.select(obs)
