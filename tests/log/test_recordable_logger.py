from unittest.mock import MagicMock

import wandb

from rl_commons.log.recordable_logger import NullRecordableLogger, WandBRecordableLogger


class _StubRecorder:
    def __init__(self, path, enabled):
        self.path = path
        self.enabled = enabled


def test_upload_videos_noop_when_recorder_disabled(mock_wandb, sample_run_info, tmp_path):
    _, mock_run = mock_wandb
    logger = WandBRecordableLogger(sample_run_info, entity="me", project="proj",
                                   hyperparameters={}, elements={})

    logger.upload_videos(_StubRecorder(path=str(tmp_path), enabled=False))

    mock_run.log.assert_not_called()


def test_upload_videos_logs_each_mp4_with_parsed_step(mock_wandb, sample_run_info, monkeypatch, tmp_path):
    _, mock_run = mock_wandb
    (tmp_path / "run-step-1000.mp4").write_bytes(b"")
    (tmp_path / "run-step-2500.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore me")

    monkeypatch.setattr(wandb, "Video", MagicMock(return_value=MagicMock()))

    logger = WandBRecordableLogger(sample_run_info, entity="me", project="proj",
                                   hyperparameters={}, elements={})
    logger.upload_videos(_StubRecorder(path=str(tmp_path), enabled=True))

    assert mock_run.log.call_count == 2
    logged_steps = {call.args[0]["global_step"] for call in mock_run.log.call_args_list}
    assert logged_steps == {1000, 2500}


def test_null_recordable_logger_noop():
    NullRecordableLogger().upload_videos(_StubRecorder(path="unused", enabled=True))
