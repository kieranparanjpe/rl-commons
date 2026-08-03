from __future__ import annotations

import torch
from datetime import datetime
from typing import Any, Callable

from rl_commons.config.run_info import RunInfo
from rl_commons.execution.base_trainer import BaseTrainer


def run_one(args: Any, run_config: Any, index: int | None, now: datetime,
            trainer_factory: Callable[[RunInfo, Any], BaseTrainer]) -> bool:
    """Standard single-run entry point for process-pool training.

    Sets subprocess-safe thread counts, constructs RunInfo from args,
    builds the trainer via trainer_factory, and calls trainer.run().

    Args:
        args: Parsed argparse namespace with at least: environment, algorithm, policy.
        run_config: Project-specific config for this run.
        index: Grid index if part of a grid search, else None.
        now: Timestamp used in the run name.
        trainer_factory: Callable(run_info, run_config) -> BaseTrainer.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    run_info = RunInfo(
        environment_id=args.environment,
        algorithm_id=args.algorithm,
        policy_id=args.policy,
        grid_index=index,
        time=now,
    )
    trainer_factory(run_info, run_config).run()
    return True
