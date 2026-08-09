from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ml_commons.config.run_info import RunInfo

import wandb

from ml_commons.log import WandBLogger, NullLogger
from rl_commons.log.recorder import BaseRecorder


class RecordableLogger(ABC):
    """Mixin for loggers that can upload recorded videos."""

    @abstractmethod
    def upload_videos(self, recorder: BaseRecorder):
        pass


class WandBRecordableLogger(WandBLogger, RecordableLogger):
    """WandBLogger with video upload support for RL training."""

    def upload_videos(self, recorder: BaseRecorder):
        if not recorder.enabled:
            return

        videos = [f"{recorder.path}/{video}" for video in os.listdir(recorder.path) if ".mp4" in video]
        for video in videos:
            step = int(video.split('-step-')[-1].split(".mp4")[0])
            self._run.log({
                "video/recording": wandb.Video(video, fps=30, format="mp4"),
                "global_step": step
            })


class NullRecordableLogger(NullLogger, RecordableLogger):
    """NullLogger with no-op video upload."""

    def upload_videos(self, recorder: BaseRecorder):
        pass
