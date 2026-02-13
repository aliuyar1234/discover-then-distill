import json

from llm_mhc_sdft_tttd.data.ttt_to_sdft import convert_discovery_logs_to_sdft_rows


def test_convert_discovery_logs_filters_and_maps_fields(tmp_path):
    p = tmp_path / "discoveries.jsonl"
    rows = [
        {"prompt": "p1", "best_state": "s1", "best_reward": 0.9},
        {"problem_description": "p2", "best_solution": "s2", "reward": 0.1},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = convert_discovery_logs_to_sdft_rows([str(p)], min_reward=0.5, default_prompt=None)
    assert out == [{"prompt": "p1", "demonstration": "s1"}]
