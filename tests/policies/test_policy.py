import torch

from rl_commons.policies.policy import Policy


class _DummyPolicy(Policy):
    def __init__(self, input_size, number_actions):
        super().__init__(input_size, number_actions)
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
    policy = _DummyPolicy(4, 2)
    path = tmp_path / "policy.pth"

    policy.save(str(path), config={"hidden_sizes": [4]}, norm_stats={"obs_mean": torch.zeros(4)})

    checkpoint = Policy.load_checkpoint(str(path))

    assert checkpoint["input_size"] == 4
    assert checkpoint["number_actions"] == 2
    assert checkpoint["config"] == {"hidden_sizes": [4]}
    assert torch.equal(checkpoint["norm_stats"]["obs_mean"], torch.zeros(4))
    assert set(checkpoint["policy"].keys()) == {"linear.weight", "linear.bias"}


def test_save_without_norm_stats_omits_key(tmp_path):
    policy = _DummyPolicy(4, 2)
    path = tmp_path / "policy.pth"

    policy.save(str(path))

    checkpoint = Policy.load_checkpoint(str(path))
    assert "norm_stats" not in checkpoint


def test_load_checkpoint_with_falsy_path_returns_empty_dict():
    assert Policy.load_checkpoint("") == {}
    assert Policy.load_checkpoint(None) == {}
