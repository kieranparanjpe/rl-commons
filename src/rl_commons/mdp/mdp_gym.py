from typing import cast

import torch
import gymnasium as gym
from ml_commons.stats import NormalisationStats
from . import envs  # noqa: F401 — registers custom environments with gymnasium

from rl_commons.log import BaseRecorder, NullRecorder
from .mdp_config import MdpConfig
from .mdp_base import Mdp
from .mdp_termination_state import MdpTerminationState


class MdpGym(Mdp):

    def __init__(self, environment_id: str, device: torch.device = torch.device('cpu'), render_mode=None,
                 recorder: BaseRecorder = NullRecorder(),
                 mdp_config: MdpConfig = MdpConfig(),
                 obs_rms_stats: NormalisationStats | None = None):
        super().__init__(device)

        self._norm_obs_wrapper: gym.wrappers.NormalizeObservation | None = None

        self._env = self.initialize_environment(environment_id, mdp_config, recorder, render_mode)

        if mdp_config.normalise_obs:
            self._env = gym.wrappers.NormalizeObservation(self._env)
            self._norm_obs_wrapper = self._env
            if obs_rms_stats is not None:
                self.enable_obs_normalization(obs_rms_stats)

        if mdp_config.normalise_reward:
            self._env = gym.wrappers.NormalizeReward(self._env, gamma=mdp_config.reward_norm_gamma)

    def initialize_environment(self, environment_id: str, mdp_config: MdpConfig, recorder: BaseRecorder, render_mode):
        if recorder.enabled:
            base_env = gym.make(environment_id, render_mode="rgb_array", **mdp_config.make_kwargs)
            recorder = cast(BaseRecorder, cast(object, recorder))
            return gym.wrappers.RecordVideo(
                base_env,
                video_folder=recorder.path,
                step_trigger=recorder.should_record
            )
        else:
            return gym.make(environment_id, render_mode=render_mode, **mdp_config.make_kwargs)

    def enable_obs_normalization(self, obs_rms_stats: NormalisationStats) -> None:
        """Wrap (or re-freeze) obs normalization on the already-constructed env, e.g. once a
        checkpoint's norm_stats become known after this MdpGym was built without them."""
        if self._norm_obs_wrapper is None:
            self._env = gym.wrappers.NormalizeObservation(self._env)
            self._norm_obs_wrapper = self._env
        self._norm_obs_wrapper.obs_rms.mean = obs_rms_stats.mean
        self._norm_obs_wrapper.obs_rms.var = obs_rms_stats.var
        self._norm_obs_wrapper.update_running_mean = False

    @property
    def obs_rms_stats(self) -> NormalisationStats:
        if self._norm_obs_wrapper is None:
            return NormalisationStats()
        return NormalisationStats(mean=self._norm_obs_wrapper.obs_rms.mean, var=self._norm_obs_wrapper.obs_rms.var)

    @property
    def action_range(self) -> torch.Tensor:
        if self.discrete:
            return torch.zeros(self.action_dimension, device=self.device)

        return torch.stack([
            torch.tensor(self._env.action_space.low, device=self.device),
            torch.tensor(self._env.action_space.high, device=self.device)
        ], dim=1)

    @property
    def obs_dimension(self) -> int:
        return self._env.observation_space.shape[0]

    @property
    def discrete(self) -> bool:
        return isinstance(self._env.action_space, gym.spaces.Discrete)

    @property
    def action_dimension(self) -> int:
        if self.discrete:
            return int(self._env.action_space.n)
        else:
            return int(self._env.action_space.shape[0])

    def reset(self) -> torch.Tensor:
        obs, _ = self._env.reset()
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, MdpTerminationState]:
        if self.discrete:
            raw_action = int(action.item())
        else:
            raw_action = action.cpu().numpy()

        next_obs, reward, terminated, truncated, _ = self._env.step(raw_action)

        terminal_state = MdpTerminationState.TERMINATED if terminated else (MdpTerminationState.TRUNCATED if truncated
                                                                            else MdpTerminationState.IN_PROGRESS)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        return next_obs_tensor, float(reward), terminal_state

    def close(self):
        self._env.close()
