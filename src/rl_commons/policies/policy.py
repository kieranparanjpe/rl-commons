from abc import ABC, abstractmethod
import torch

class Policy(ABC, torch.nn.Module):

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

    def save(self, path, config=None, norm_stats=None):
        save_dict = {
            "policy": self.state_dict(),
            "config": config,
            "input_size": self.input_size,
            "number_actions": self._number_actions,
        }
        if norm_stats is not None:
            save_dict["norm_stats"] = norm_stats
        torch.save(save_dict, path)

    @staticmethod
    def load_checkpoint(path, map_location="cpu"):
        return torch.load(path, map_location=map_location, weights_only=True) if path else {}
