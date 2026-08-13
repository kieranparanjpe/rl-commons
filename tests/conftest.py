from datetime import datetime
from unittest.mock import MagicMock

import pytest
import wandb

from rl_commons.config.run_info import RLRunInfo
from rl_commons.policies.policy_factory import PolicyFactory


@pytest.fixture
def sample_run_info() -> RLRunInfo:
    return RLRunInfo(
        task_id="CartPole-v1",
        algorithm_id="ppo",
        policy_id="categorical",
        grid_index=None,
        time=datetime(2026, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def mock_wandb(monkeypatch):
    mock_run = MagicMock()
    mock_init = MagicMock(return_value=mock_run)
    monkeypatch.setattr(wandb, "init", mock_init)
    return mock_init, mock_run


@pytest.fixture
def isolated_policy_registry():
    """PolicyFactory._registry is shared class state — snapshot/restore around
    tests that register/mutate it so they can't leak into each other."""
    original = dict(PolicyFactory._registry)
    PolicyFactory._registry.clear()
    yield
    PolicyFactory._registry.clear()
    PolicyFactory._registry.update(original)
