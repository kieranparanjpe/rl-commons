from rl_commons.algorithms.algorithm_config import AlgorithmConfig


def test_defaults():
    config = AlgorithmConfig()
    assert config.n_timesteps == 1_000_000
    assert config.lr == 3e-4


def test_overrides():
    config = AlgorithmConfig(n_timesteps=100, lr=0.1)
    assert config.n_timesteps == 100
    assert config.lr == 0.1
