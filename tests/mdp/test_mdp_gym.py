import numpy as np
import torch
from ml_commons.stats import NormalisationStats

from rl_commons.mdp.mdp_config import MdpConfig
from rl_commons.mdp.mdp_gym import MdpGym
from rl_commons.mdp.mdp_termination_state import MdpTerminationState


def test_discrete_env_dimensions_and_step():
    mdp = MdpGym("CartPole-v1", mdp_config=MdpConfig(normalise_obs=False, normalise_reward=False))
    try:
        assert mdp.discrete is True
        assert mdp.obs_dimension == 4
        assert mdp.action_dimension == 2
        assert torch.equal(mdp.action_range, torch.zeros(2))

        obs = mdp.reset()
        assert obs.shape == (4,)
        assert obs.dtype == torch.float32

        next_obs, reward, terminal = mdp.step(torch.tensor(0))
        assert next_obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminal, MdpTerminationState)
    finally:
        mdp.close()


def test_continuous_env_action_range():
    mdp = MdpGym("Pendulum-v1", mdp_config=MdpConfig(normalise_obs=False, normalise_reward=False))
    try:
        assert mdp.discrete is False
        assert mdp.action_dimension == 1
        assert mdp.action_range.shape == (1, 2)
        assert torch.allclose(mdp.action_range, torch.tensor([[-2.0, 2.0]]))

        mdp.reset()
        next_obs, reward, terminal = mdp.step(torch.tensor([0.0]))
        assert next_obs.shape == (3,)
        assert isinstance(terminal, MdpTerminationState)
    finally:
        mdp.close()


def test_normalise_obs_enabled_exposes_running_stats():
    mdp = MdpGym("CartPole-v1", mdp_config=MdpConfig(normalise_obs=True, normalise_reward=False))
    try:
        mdp.reset()
        mdp.step(torch.tensor(0))

        stats = mdp.obs_rms_stats
        assert stats.mean.shape == (4,)
        assert stats.var.shape == (4,)
    finally:
        mdp.close()


def test_normalise_obs_disabled_has_no_running_stats():
    mdp = MdpGym("CartPole-v1", mdp_config=MdpConfig(normalise_obs=False, normalise_reward=False))
    try:
        assert np.array_equal(mdp.obs_rms_stats.mean, 0.0)
        assert np.array_equal(mdp.obs_rms_stats.var, 1.0)
    finally:
        mdp.close()


def test_injected_obs_rms_stats_are_preserved_not_recomputed():
    fixed_mean = np.array([1.0, 2.0, 3.0, 4.0])
    fixed_var = np.array([1.0, 1.0, 1.0, 1.0])

    mdp = MdpGym("CartPole-v1",
                mdp_config=MdpConfig(normalise_obs=True, normalise_reward=False),
                obs_rms_stats=NormalisationStats(mean=fixed_mean, var=fixed_var))
    try:
        mdp.reset()
        mdp.step(torch.tensor(0))

        stats = mdp.obs_rms_stats
        assert np.array_equal(stats.mean, fixed_mean)
        assert np.array_equal(stats.var, fixed_var)
    finally:
        mdp.close()
