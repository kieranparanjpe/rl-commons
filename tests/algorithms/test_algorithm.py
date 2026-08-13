import torch

from rl_commons.algorithms.algorithm import Algorithm
from rl_commons.policies.policy import Policy


class _DummyPolicy(Policy):
    def forward(self, observation):
        return torch.distributions.Categorical(logits=torch.zeros(observation.shape[0], self._number_actions))

    def log_probability(self, action, distribution):
        return distribution.log_prob(action)


class _DummyAlgorithm(Algorithm):
    def sample_action(self, obs):
        return torch.zeros(1), torch.zeros(1)

    def update_and_observe(self, initial_obs, next_obs, action, action_log_prob, reward, termination_state, timestep):
        return False


def test_algorithm_stores_constructor_args():
    policy = _DummyPolicy(4, 2)
    algo = _DummyAlgorithm({"lr": 0.1}, policy, obs_dimension=4, action_dimension=2, discrete=True)

    assert algo.hyperparameters == {"lr": 0.1}
    assert algo.policy is policy
    assert algo.obs_dimension == 4
    assert algo.action_dimension == 2
    assert algo.discrete is True
    assert algo.logger is None
    assert algo.device == torch.device("cpu")
