from llm_mhc_sdft_tttd.training.sdft import should_revert_update


def test_gate_reverts_when_perplexity_regresses_too_much():
    before = {"ppl": 10.0, "probe_score": 0.6}
    after = {"ppl": 13.0, "probe_score": 0.58}
    assert should_revert_update(before, after, max_ppl_rel_increase=0.2, max_probe_score_drop=0.2)


def test_gate_reverts_when_probe_score_drops_too_much():
    before = {"ppl": 10.0, "probe_score": 0.8}
    after = {"ppl": 11.0, "probe_score": 0.5}
    assert should_revert_update(before, after, max_ppl_rel_increase=0.2, max_probe_score_drop=0.1)


def test_gate_accepts_when_regression_is_within_thresholds():
    before = {"ppl": 10.0, "probe_score": 0.8}
    after = {"ppl": 11.5, "probe_score": 0.75}
    assert not should_revert_update(before, after, max_ppl_rel_increase=0.2, max_probe_score_drop=0.1)


def test_gate_ignores_missing_metrics():
    before = {"ppl": None, "probe_score": 0.5}
    after = {"ppl": None, "probe_score": 0.45}
    assert not should_revert_update(before, after, max_ppl_rel_increase=0.1, max_probe_score_drop=0.1)
