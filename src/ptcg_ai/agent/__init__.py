"""Agent policies."""

from .policy import Policy

__all__ = ["LearnedPolicy", "Policy", "RandomPolicy", "RuleBasedPolicy"]


def __getattr__(name: str):
    if name == "RandomPolicy":
        from .random_policy import RandomPolicy

        return RandomPolicy
    if name == "RuleBasedPolicy":
        from .rule_based import RuleBasedPolicy

        return RuleBasedPolicy
    if name == "LearnedPolicy":
        from .learned_policy import LearnedPolicy

        return LearnedPolicy
    raise AttributeError(name)
