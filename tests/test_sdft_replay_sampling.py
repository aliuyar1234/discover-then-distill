import random

from llm_mhc_sdft_tttd.data.sdft_dataset import SDFTExample
from llm_mhc_sdft_tttd.training.sdft import ReplayBuffer, sample_batch_with_replay


def test_replay_sampling_uses_history_and_respects_capacity():
    replay = ReplayBuffer(max_size=2)
    rng = random.Random(0)

    first_batch = [
        SDFTExample(prompt="new-1", demonstration="demo-1"),
        SDFTExample(prompt="new-2", demonstration="demo-2"),
    ]
    out1 = sample_batch_with_replay(first_batch, replay=replay, replay_ratio=1.0, rng=rng)
    # replay is empty initially, so we must use stream examples.
    assert [x.prompt for x in out1] == ["new-1", "new-2"]
    assert len(replay) == 2

    second_batch = [
        SDFTExample(prompt="new-3", demonstration="demo-3"),
        SDFTExample(prompt="new-4", demonstration="demo-4"),
    ]
    out2 = sample_batch_with_replay(second_batch, replay=replay, replay_ratio=1.0, rng=rng)
    # replay_ratio=1 means we only sample from replay when available.
    assert all(x.prompt in {"new-1", "new-2"} for x in out2)
    # bounded replay keeps only latest 2 stream items after adding second batch.
    assert len(replay) == 2
    assert {x.prompt for x in replay.items} == {"new-3", "new-4"}
