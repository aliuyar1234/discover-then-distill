from llm_mhc_sdft_tttd.training.ttt_discover import Archive, puct_select_start_state


def test_puct_select_prefers_high_reward_initially():
    archive = Archive(max_size=10)
    lo = archive.add_state("low", 0.1)
    hi = archive.add_state("high", 0.9)
    chosen = puct_select_start_state(archive, c=1.0)
    assert chosen == hi


def test_puct_select_handles_single_state():
    archive = Archive(max_size=10)
    sid = archive.add_state("only", 0.5)
    chosen = puct_select_start_state(archive, c=2.0)
    assert chosen == sid
