from dataclasses import dataclass


@dataclass
class AlgorithmConfig:
    n_timesteps: int = 1_000_000
    lr: float = 3e-4
