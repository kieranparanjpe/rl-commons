from rl_commons.log.recorder import NullRecorder, Recorder


def test_null_recorder_disabled_and_never_records():
    recorder = NullRecorder()
    assert recorder.enabled is False
    assert recorder.should_record(0) is False
    assert recorder.should_record(999999) is False


def test_recorder_enabled_and_records_at_thresholds():
    recorder = Recorder(path="videos", number_videos=3, total_timesteps=20000)
    assert recorder.enabled is True

    # threshold_0 = 0*(20000//2) - 5000 = -5000 -> step 0 > -5000 and new_episode -> records
    recorder.new_episode = True
    assert recorder.should_record(0) is True
    assert recorder._recorded_videos == 1

    # not a new episode -> never records, regardless of step
    recorder.new_episode = False
    assert recorder.should_record(999999) is False

    # threshold_1 = 1*(20000//2) - 5000 = 5000 -> step must exceed 5000
    recorder.new_episode = True
    assert recorder.should_record(5000) is False
    assert recorder.should_record(5001) is True
    assert recorder._recorded_videos == 2
