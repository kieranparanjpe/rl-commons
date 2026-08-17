import numpy as np
import torch

import pytest

from ml_commons.stats import NormalisationStats
from rl_commons.policies.policy import Policy


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
    Policy.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    policy = Policy.build_policy("dummy", 4, 2, config={"a": 1})

    assert isinstance(policy, _DummyPolicy)
    assert policy.input_size == 4
    assert policy._number_actions == 2
    assert policy.config == {"a": 1}


def test_build_policy_unknown_id_raises():
    with pytest.raises(ValueError):
        Policy.build_policy("nonexistent", 4, 2)


def test_load_by_policy_id_roundtrip(tmp_path):
    Policy.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    original = _DummyPolicy(4, 2, config={"a": 1})
    path = tmp_path / "policy.pth"
    original.save(str(path))

    loaded = Policy.load(str(path), policy_id="dummy")

    assert isinstance(loaded, _DummyPolicy)
    assert loaded.input_size == 4
    assert loaded._number_actions == 2
    assert loaded.config == {"a": 1}
    assert not loaded.training
    assert torch.equal(loaded.dummy_param, original.dummy_param)
    assert np.array_equal(loaded.obs_norm_stats.mean, 0.0)
    assert np.array_equal(loaded.obs_norm_stats.var, 1.0)


def test_load_by_policy_id_returns_saved_norm_stats(tmp_path):
    Policy.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    original = _DummyPolicy(4, 2)
    original.obs_norm_stats = NormalisationStats(mean=np.zeros(4), var=np.ones(4))
    path = tmp_path / "policy.pth"
    original.save(str(path))

    loaded = Policy.load(str(path), policy_id="dummy")

    assert np.array_equal(loaded.obs_norm_stats.mean, np.zeros(4))
    assert np.array_equal(loaded.obs_norm_stats.var, np.ones(4))


def test_load_migrates_legacy_norm_stats_dict(tmp_path):
    """Checkpoints saved after config/input_size/number_actions existed but before
    NormalisationStats existed stored norm_stats as a raw {"obs_mean": Tensor, "obs_var": Tensor}
    dict under the "norm_stats" key."""
    Policy.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    original = _DummyPolicy(4, 2)
    path = tmp_path / "policy.pth"
    torch.save({
        "policy": original.state_dict(),
        "config": None,
        "input_size": 4,
        "number_actions": 2,
        "norm_stats": {"obs_mean": torch.zeros(4), "obs_var": torch.ones(4)},
    }, str(path))

    loaded = Policy.load(str(path), policy_id="dummy")

    assert isinstance(loaded.obs_norm_stats, NormalisationStats)
    assert np.array_equal(loaded.obs_norm_stats.mean, np.zeros(4))
    assert np.array_equal(loaded.obs_norm_stats.var, np.ones(4))


def test_load_legacy_checkpoint_with_only_weights_and_norm_stats(tmp_path):
    """Pre-classmethod checkpoints (My_RL_Impl commit dafe693..dcf3639) only ever
    saved {"policy": state_dict, "norm_stats": {...}} -- no config/input_size/number_actions.
    Loading these requires the caller to pass obs_dimension/action_dimension explicitly."""
    Policy.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    original = _DummyPolicy(4, 2)
    path = tmp_path / "policy.pth"
    torch.save({
        "policy": original.state_dict(),
        "norm_stats": {"obs_mean": torch.zeros(4), "obs_var": torch.ones(4)},
    }, str(path))

    loaded = Policy.load(str(path), policy_id="dummy", obs_dimension=4, action_dimension=2)

    assert isinstance(loaded, _DummyPolicy)
    assert torch.equal(loaded.dummy_param, original.dummy_param)
    assert isinstance(loaded.obs_norm_stats, NormalisationStats)
    assert np.array_equal(loaded.obs_norm_stats.mean, np.zeros(4))
    assert np.array_equal(loaded.obs_norm_stats.var, np.ones(4))


def test_load_dimension_override_takes_priority_over_checkpoint(tmp_path):
    Policy.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    original = _DummyPolicy(4, 2)
    path = tmp_path / "policy.pth"
    original.save(str(path))

    loaded = Policy.load(str(path), policy_id="dummy", obs_dimension=8, action_dimension=3)

    assert loaded.input_size == 8
    assert loaded._number_actions == 3


def test_load_falls_back_to_checkpoint_dims_when_not_passed(tmp_path):
    Policy.register("dummy", lambda obs, act, cfg: _DummyPolicy(obs, act, cfg))

    original = _DummyPolicy(4, 2)
    path = tmp_path / "policy.pth"
    original.save(str(path))

    loaded = Policy.load(str(path), policy_id="dummy")

    assert loaded.input_size == 4
    assert loaded._number_actions == 2
