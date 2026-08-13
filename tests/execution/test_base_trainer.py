from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from rl_commons.config.run_info import RLRunInfo
from rl_commons.execution.base_trainer import BaseTrainer
from rl_commons.log.recordable_logger import NullRecordableLogger, WandBRecordableLogger
from rl_commons.log.recorder import NullRecorder, Recorder
from rl_commons.mdp.mdp_config import MdpConfig


class _DummyTrainer(BaseTrainer):
    def run(self):
        pass


@dataclass
class _RunConfig:
    lr: float = 0.1


def _run_info():
    return RLRunInfo(task_id="CartPole-v1", algorithm_id="ppo", policy_id="categorical",
                     grid_index=None, time=datetime(2026, 1, 1, 12, 0, 0))


@pytest.fixture
def fake_mdp_gym(monkeypatch):
    """Stub out MdpGym so trainer-construction tests don't need a real gym env."""
    fake = MagicMock()
    monkeypatch.setattr("rl_commons.execution.base_trainer.MdpGym", MagicMock(return_value=fake))
    return fake


def _make_trainer(**overrides):
    kwargs = dict(mdp_config=MdpConfig(), entity="me", project="proj",
                 log_elements={}, logging=False, record=False, total_timesteps=None)
    kwargs.update(overrides)
    return _DummyTrainer(_run_info(), _RunConfig(), **kwargs)


def test_create_logger_uses_recordable_variant_when_logging_enabled(fake_mdp_gym, mock_wandb):
    trainer = _make_trainer(logging=True)
    assert isinstance(trainer._logger, WandBRecordableLogger)


def test_create_logger_null_variant_when_logging_disabled(fake_mdp_gym):
    trainer = _make_trainer(logging=False)
    assert isinstance(trainer._logger, NullRecordableLogger)


def test_recorder_is_null_when_record_false(fake_mdp_gym):
    trainer = _make_trainer(record=False, total_timesteps=1000)
    assert isinstance(trainer._recorder, NullRecorder)


def test_recorder_is_null_when_total_timesteps_missing(fake_mdp_gym):
    trainer = _make_trainer(record=True, total_timesteps=None)
    assert isinstance(trainer._recorder, NullRecorder)


def test_recorder_is_real_when_record_true_and_total_timesteps_given(fake_mdp_gym):
    trainer = _make_trainer(record=True, total_timesteps=1000)
    assert isinstance(trainer._recorder, Recorder)


def test_mdp_is_wired_up_for_real_task():
    trainer = _DummyTrainer(_run_info(), _RunConfig(),
                            mdp_config=MdpConfig(normalise_obs=False, normalise_reward=False),
                            entity="me", project="proj", log_elements={}, logging=False)
    try:
        assert trainer._mdp.obs_dimension == 4
        assert trainer._mdp.action_dimension == 2
    finally:
        trainer._mdp.close()
