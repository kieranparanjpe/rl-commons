from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from rl_commons.config.run_info import RunInfo
from rl_commons.log import WandBLogger, NullLogger, Recorder, NullRecorder
from rl_commons.mdp import MdpGym, MdpConfig


class BaseTrainer(ABC):

    def __init__(self, run_info: RunInfo, mdp_config: MdpConfig,
                 entity: str, project: str,
                 log_elements: dict, hyperparameters: dict,
                 logging: bool = True, record: bool = False,
                 total_timesteps: int | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._run_info = run_info

        print(f"\nRun: {self._run_info.run_name()}")

        self._logger = WandBLogger(run_info, entity, project, hyperparameters, log_elements) \
                       if logging else NullLogger()

        self._recorder = Recorder(
            run_info.local_folder_path("saved_videos"), 5, total_timesteps
        ) if (record and total_timesteps is not None) else NullRecorder()

        self._mdp = MdpGym(run_info.environment_id, self.device,
                           recorder=self._recorder, mdp_config=mdp_config)

    @abstractmethod
    def run(self):
        """Execute the main training loop."""
        pass
