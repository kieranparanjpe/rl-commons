import numpy as np
import torch

from ml_commons.stats import NormalisationStats
from rl_commons.policies.policy import Policy


class _DummyPolicy(Policy):
    def __init__(self, input_size, number_actions, config=None):
        super().__init__(input_size, number_actions)
        self.config = config
        self.linear = torch.nn.Linear(input_size, number_actions)

    def forward(self, observation):
        return torch.distributions.Categorical(logits=self.linear(observation))

    def log_probability(self, action, distribution):
        return distribution.log_prob(action)


def test_sample_action_delegates_to_distribution():
    policy = _DummyPolicy(4, 2)
    dist = policy.forward(torch.zeros(1, 4))

    action = policy.sample_action(dist)

    assert action.shape == (1,)


def test_entropy_sums_last_dim():
    policy = _DummyPolicy(4, 2)
    dist = torch.distributions.Normal(torch.zeros(3, 2), torch.ones(3, 2))

    entropy = policy.entropy(dist)

    assert entropy.shape == (3,)
    assert torch.allclose(entropy, dist.entropy().sum(-1))


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    policy = _DummyPolicy(4, 2, config={"hidden_sizes": [4]})
    policy.obs_norm_stats = NormalisationStats(mean=np.zeros(4), var=np.ones(4))
    path = tmp_path / "policy.pth"

    policy.save(str(path))

    checkpoint = torch.load(str(path), weights_only=True)

    assert checkpoint["input_size"] == 4
    assert checkpoint["number_actions"] == 2
    assert checkpoint["config"] == {"hidden_sizes": [4]}
    assert np.array_equal(checkpoint["obs_norm_stats"].mean, np.zeros(4))
    assert set(checkpoint["policy"].keys()) == {"linear.weight", "linear.bias"}


def test_save_without_setting_norm_stats_saves_identity(tmp_path):
    policy = _DummyPolicy(4, 2)
    path = tmp_path / "policy.pth"

    policy.save(str(path))

    checkpoint = torch.load(str(path), weights_only=True)
    assert np.array_equal(checkpoint["obs_norm_stats"].mean, 0.0)
    assert np.array_equal(checkpoint["obs_norm_stats"].var, 1.0)
