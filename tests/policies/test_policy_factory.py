import torch

import pytest

from rl_commons.policies.policy import Policy
from rl_commons.policies.policy_factory import PolicyFactory


class _DummyPolicy(Policy):
    """Params don't depend on obs/action dims, so tests can freely vary those
    dims without triggering state_dict shape mismatches on load."""

    def __init__(self, input_size, number_actions, config=None):
        super().__init__(input_size, number_actions)
        self.config = config
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, observation):
        return torch.distributions.Categorical(logits=torch.zeros(observation.shape[0], self._number_actions))

    def log_probability(self, action, distribution):
        return distribution.log_prob(action)


@pytest.fixture(autouse=True)
def _use_isolated_registry(isolated_policy_registry):
    pass


def test_register_and_build_policy():
    PolicyFactory.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    policy = PolicyFactory.build_policy("dummy", 4, 2, config={"a": 1})

    assert isinstance(policy, _DummyPolicy)
    assert policy.input_size == 4
    assert policy._number_actions == 2
    assert policy.config == {"a": 1}


def test_build_policy_unknown_id_raises():
    with pytest.raises(ValueError):
        PolicyFactory.build_policy("nonexistent", 4, 2)


def test_load_policy_roundtrip(tmp_path):
    PolicyFactory.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    original = _DummyPolicy(4, 2, config={"a": 1})
    path = tmp_path / "policy.pth"
    original.save(str(path), config={"a": 1})

    loaded = PolicyFactory.load_policy("dummy", str(path))

    assert isinstance(loaded, _DummyPolicy)
    assert loaded.input_size == 4
    assert loaded._number_actions == 2
    assert loaded.config == {"a": 1}
    assert not loaded.training
    assert torch.equal(loaded.dummy_param, original.dummy_param)


def test_load_policy_dimension_override_takes_priority_over_checkpoint(tmp_path):
    PolicyFactory.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    original = _DummyPolicy(4, 2)
    path = tmp_path / "policy.pth"
    original.save(str(path))

    loaded = PolicyFactory.load_policy("dummy", str(path), obs_dimension=8, action_dimension=3)

    assert loaded.input_size == 8
    assert loaded._number_actions == 3


def test_load_policy_falls_back_to_checkpoint_dims_when_not_passed(tmp_path):
    PolicyFactory.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    original = _DummyPolicy(4, 2)
    path = tmp_path / "policy.pth"
    original.save(str(path))

    loaded = PolicyFactory.load_policy("dummy", str(path))

    assert loaded.input_size == 4
    assert loaded._number_actions == 2
