from __future__ import annotations

from abc import ABC
from typing import Any

from ml_commons.config.run_info import RunInfo
from ml_commons.log import Logger
from ml_commons.execution.base_trainer import BaseTrainer as MLBaseTrainer
from rl_commons.log import Recorder, NullRecorder, WandBRecordableLogger, NullRecordableLogger
from rl_commons.mdp import MdpGym, MdpConfig


class BaseTrainer(MLBaseTrainer, ABC):

    def __init__(self, run_info: RunInfo, run_config: Any, mdp_config: MdpConfig,
                 entity: str, project: str, log_elements: dict,
                 logging: bool = True, record: bool = False,
                 total_timesteps: int | None = None):
        super().__init__(
            run_info=run_info,
            run_config=run_config,
            entity=entity,
            project=project,
            log_elements=log_elements,
            logging=logging,
        )

        self._recorder = Recorder(
            run_info.local_folder_path("saved_videos"), 5, total_timesteps
        ) if (record and total_timesteps is not None) else NullRecorder()

        self._mdp = MdpGym(run_info.task_id, self.device,
                           recorder=self._recorder, mdp_config=mdp_config)

    def _create_logger(self, run_info: RunInfo, entity: str, project: str,
                       hyperparameters: dict, elements: dict,
                       logging: bool) -> Logger:
        """Override to use recordable logger variants that support video upload."""
        if logging:
            return WandBRecordableLogger(run_info, entity, project,
                                         hyperparameters=hyperparameters,
                                         elements=elements)
        return NullRecordableLogger()
