from __future__ import annotations

from dataclasses import dataclass

from ml_commons.config import RunInfo


@dataclass
class RLRunInfo(RunInfo):
    policy_id: str = ""

    def tags(self) -> list[str]:
        return [self.algorithm_id, self.policy_id, self.task_id]
