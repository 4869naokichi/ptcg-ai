"""Agent policies."""

from .policy import Policy

__all__ = ["Policy", "RandomPolicy", "RuleBasedPolicy"]


def __getattr__(name: str):
    if name == "RandomPolicy":
        from .random_policy import RandomPolicy

        return RandomPolicy
    if name == "RuleBasedPolicy":
        from .rule_based import RuleBasedPolicy

        return RuleBasedPolicy
    raise AttributeError(name)
