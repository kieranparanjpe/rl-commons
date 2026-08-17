from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Optional

import torch
from ml_commons.networks import SaveableNetwork
import ml_commons.stats  # noqa: F401 — registers NormalisationStats/numpy as safe globals for torch.load

class Policy(SaveableNetwork, torch.nn.Module):

    config: Any

    _registry: dict[str, Callable[[int, int, Any], Policy]] = {}

    def __init__(self, input_size : int, number_actions : int):
        super().__init__()
        self.input_size = input_size
        self._number_actions = number_actions

    @abstractmethod
    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        pass

    @abstractmethod
    def log_probability(self, action : torch.Tensor, distribution : torch.distributions.Distribution) -> torch.Tensor:
        pass

    def sample_action(self, distribution : torch.distributions.Distribution) -> torch.Tensor:
        return distribution.sample()

    def entropy(self, distribution : torch.distributions.Distribution) -> torch.Tensor:
        return distribution.entropy().sum(-1)

    def save(self, path, norm_stats=None) -> None:
        save_dict = {
            "policy": self.state_dict(),
            "config": self.config,
            "input_size": self.input_size,
            "number_actions": self._number_actions,
        }
        if norm_stats is not None:
            save_dict["norm_stats"] = norm_stats
        torch.save(save_dict, path)

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
    def load(cls, path, map_location="cpu", obs_dimension: Optional[int] = None,
              action_dimension: Optional[int] = None, *, policy_id: str, **kwargs):
        checkpoint = torch.load(path, map_location=map_location, weights_only=True) if path else {}

        obs_dim = obs_dimension if obs_dimension is not None else checkpoint["input_size"]
        action_dim = action_dimension if action_dimension is not None else checkpoint["number_actions"]

        policy = cls.build_policy(policy_id, int(obs_dim), int(action_dim), checkpoint.get("config"))

        policy_state_dict = checkpoint["policy"]
        policy.load_state_dict(policy_state_dict)
        policy.eval()

        return policy, checkpoint.get("norm_stats")
