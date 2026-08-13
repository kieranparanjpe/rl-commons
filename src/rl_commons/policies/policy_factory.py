from __future__ import annotations
from typing import Any, Callable, Optional

from .policy import Policy


class PolicyFactory:

    _registry: dict[str, Callable[[int, int, Any], Policy]] = {}

    @classmethod
    def register(cls, policy_id: str, factory: Callable[[int, int, Any], Policy]) -> None:
        cls._registry[policy_id] = factory

    @classmethod
    def build_policy(cls, policy_id: str, obs_dimension: int, action_dimension: int,
                     config: Any = None) -> Policy:
        if policy_id not in cls._registry:
            raise ValueError(f"Policy not found: {policy_id}")
        return cls._registry[policy_id](obs_dimension, action_dimension, config)

    @classmethod
    def load_policy(cls, policy_id: str, path: str,
                    obs_dimension: Optional[int] = None, action_dimension: Optional[int] = None) -> Policy:
        checkpoint = Policy.load_checkpoint(path)

        obs_dim = obs_dimension if obs_dimension is not None else checkpoint.get("input_size")
        action_dim = action_dimension if action_dimension is not None else checkpoint.get("number_actions")

        policy = cls.build_policy(policy_id, int(obs_dim), int(action_dim), checkpoint.get("config"))

        policy_state_dict = checkpoint.get("policy")
        if policy_state_dict:
            policy.load_state_dict(policy_state_dict)
        policy.eval()

        return policy
