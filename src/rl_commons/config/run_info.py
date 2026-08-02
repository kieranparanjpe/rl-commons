from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RunInfo:
    environment_id: str
    algorithm_id: str
    policy_id: str
    grid_index: int | None
    time: datetime

    def _env_and_time(self):
        return f"{self.environment_id}@{self.time:%Y-%m-%d-%H-%M-%S}"

    def group(self):
        return None if self.grid_index is None else self._env_and_time()

    def run_name(self) -> str:
        return self._env_and_time() if self.grid_index is None else f"{self._env_and_time()}_RUN-{self.grid_index}"

    def local_folder_path(self, folder_name: str) -> str:
        if self.grid_index is None:
            return f"{folder_name}/{self.environment_id}/{self.run_name()}"
        else:
            return f"{folder_name}/{self.environment_id}/{self.group()}/{self.run_name()}"
