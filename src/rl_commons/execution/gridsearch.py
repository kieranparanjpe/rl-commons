from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Callable


def gridsearch(run_one: Callable[[Any, int], Any], configs: list, max_parallel: int | None = None) -> None:
    """Run a list of configs in parallel using a process pool.

    Args:
        run_one: Callable with signature (config, index) -> Any. Called once per config.
        configs: List of configs to run.
        max_parallel: Max parallel workers. Defaults to min(cpu_count, 8).
    """
    n_parallel = min(os.cpu_count() or 1, 8) if max_parallel is None else max_parallel
    config_index = 0

    with ProcessPoolExecutor(max_workers=n_parallel) as pool:
        in_flight = set()

        for _ in range(min(n_parallel, len(configs))):
            in_flight.add(pool.submit(run_one, configs[config_index], config_index))
            config_index += 1

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                fut.result()
                if config_index < len(configs):
                    in_flight.add(pool.submit(run_one, configs[config_index], config_index))
                    config_index += 1
