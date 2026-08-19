from abc import ABC, abstractmethod
from typing import Optional

import torch

from .mdp_termination_state import MdpTerminationState


class Mdp(ABC):

    def __init__(self, device):
        self.device = device

    @property
    @abstractmethod
    def obs_dimension(self) -> int:
        """Returns the size of the observation."""
        pass

    @property
    @abstractmethod
    def action_dimension(self) -> int:
        """Returns the number of actions."""
        pass

    @property
    @abstractmethod
    def action_range(self) -> torch.Tensor:
        """Returns a tensor of shape:
        Continuous: [action_dimension, 2], with a min and max for each action.
        Discrete: [action_dimension]"""
        pass

    @property
    @abstractmethod
    def discrete(self) -> bool:
        """Returns whether the actions are continuous or discrete."""
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Resets the MDP and returns the initial observation.

        seed: if not None, reseeds the environment's RNG before reset -- use the same seed
        across two Mdp instances of the same task to make their hidden reset-time randomness
        (e.g. procedurally generated terrain, initial random impulses) identical."""
        pass

    @abstractmethod
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, MdpTerminationState]:
        """
        Advances the MDP obs by executing an action.
        Returns: (next_obs_tensor, reward_float, terminal_obs)
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the mdp environment.
        """
        pass
